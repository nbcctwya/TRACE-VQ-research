"""Checklist batch05: preflight, train and exact standalone evaluation."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping

from trace_vq.cli.batch01 import reference_directory,cache_directory,gpu_profile,start_profile
from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.cli.next_screen import trainer
from trace_vq.cli.screening import ROOT,initialize,now,read_dataset,scalar,write_prediction
from trace_vq.data.frozen import atomic_json,daily_loader,prepare_frozen_cache,sha256,float_collate,DailyPositions
from trace_vq.models.batch05 import VARIANTS,Batch05Predictor
from trace_vq.training.resilient import AtomicEpochCheckpoint,BatchHeartbeat
from trace_vq.utils import apply_artifact_root


def make_config(market,variant,output,workers=2):
    settings=VARIANTS[variant];cfg=OmegaConf.load(reference_directory(market,settings['family'])/'effective.config.yaml')
    cfg.artifact_root=str(Path(output).resolve());cfg.train.seed=0;cfg.train.num_workers=workers
    cfg.train.wandb_mode='disabled';cfg.train.run_name=f'batch05_{market}_{variant}_s0'
    cfg.research=dict(variant=variant,batch=5,**copy.deepcopy(settings))
    if variant=='raven_context_experts':cfg.predictor.n_expert=3;cfg.predictor.k=1
    apply_artifact_root(cfg);result=OmegaConf.to_container(cfg,resolve=True)
    assert result['train']['num_epochs']==70 and result['train']['early_stopping']['patience']==15
    assert result['train']['early_stopping']['min_delta']==1e-5 and result['predictor']['target_day']==5
    checkpoint=Path(result['predictor']['saved_model']);assert checkpoint.is_file() and checkpoint.is_relative_to(ROOT)
    return result


def cached(model,config,split,args):
    data,path=read_dataset(config,split)
    cache=cache_directory(args.universe,VARIANTS[args.variant]['family'])/(split+'.npy')
    if not cache.is_file() or not cache.with_suffix('.json').is_file():raise FileNotFoundError(f'Existing Stage1 cache missing: {cache}')
    return prepare_frozen_cache(model,data,path,cache,config,args.device,args.num_workers)


def diagnostics(model,reset=False):
    moe=model.loadings.fusion.moe
    if reset and hasattr(moe,'reset_diagnostics'):moe.reset_diagnostics()
    return moe.diagnostics() if hasattr(moe,'diagnostics') else {}


def latest_checkpoint(output,recorder):
    for name in ('last.ckpt','last.previous.ckpt'):
        path=output/'checkpoints'/name
        if not path.exists():continue
        try:
            checkpoint=torch.load(path,map_location='cpu',weights_only=True)
            assert recorder.state_key in checkpoint['callbacks'],'Recoverable RNG missing'
            epoch=checkpoint['epoch'];del checkpoint
            return path,epoch
        except Exception as error:
            archive=output/'recovery';archive.mkdir(exist_ok=True)
            rejected=archive/(str(time.time_ns())+'_'+name);path.rename(rejected)
            atomic_json(rejected.with_suffix('.json'),dict(reason=str(error),updated=now()))
    return None,None


def finish_training(record,config,args,start):
    model=Batch05Predictor.load_from_checkpoint(record['checkpoint'],config=config,T_max=1,map_location='cpu',strict=True)
    assert_frozen(model,config);begin=time.perf_counter()
    for split in ('valid','test'):
        data,_=cached(model,config,split,args);diagnostics(model,True)
        metrics=write_prediction(model,data,config,args.device,args.output_dir/'training_evaluation',split,args.num_workers)
        record['validation_metrics' if split=='valid' else 'test_metrics']=metrics
        record['routing_'+split]=diagnostics(model)
        del data;gc.collect()
    record.update(status='complete',training_inference_seconds=time.perf_counter()-begin,
                  train_phase_seconds=time.perf_counter()-start,finished=now())
    atomic_json(args.output_dir/'training_result.json',record)


def train(args):
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    if (output/'training_result.json').exists():raise FileExistsError('Reuse completed training')
    initialize(0);phase_start=time.perf_counter()
    if (output/'training_fit.json').exists():
        config=OmegaConf.to_container(OmegaConf.load(output/'effective.config.yaml'),resolve=True)
        record=json.loads((output/'training_fit.json').read_text())
        assert sha256(record['checkpoint'])==record['checkpoint_sha256']
        finish_training(record,config,args,phase_start);return
    config=make_config(args.universe,args.variant,output,args.num_workers)
    model=Batch05Predictor(config,T_max=1)
    training,train_meta=cached(model,config,'train',args)
    validation,valid_meta=cached(model,config,'valid',args)
    train_loader=daily_loader(training,shuffle=True,num_workers=args.num_workers)
    valid_loader=daily_loader(validation,num_workers=args.num_workers)
    model.T_max=len(train_loader)*config['train']['num_epochs'];snapshot=copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot),output/'effective.config.yaml')
    initial_hash=sha256(config['predictor']['saved_model'])
    atomic_json(output/'input_manifest.json',dict(stage1_seed=42,stage2_seed=0,stage1_retrained=False,
        stage1_sha256=initial_hash,train=train_meta,valid=valid_meta,test_used_for_selection=False,
        selection='native validation target MSE plus unchanged auxiliary loss',runtime=dict(workers=args.num_workers)))
    best=AtomicEpochCheckpoint(dirpath=output/'checkpoints',monitor='val_loss',mode='min',save_top_k=1,save_last=True,
        save_on_train_epoch_end=True,filename='seed0-{epoch}-{val_loss:.6f}',enable_version_counter=False)
    early=config['train']['early_stopping'];stop=EarlyStopping(monitor='val_loss',mode='min',
        patience=early['patience'],min_delta=early['min_delta'],verbose=True)
    recorder=BatchHeartbeat(output,args.device);checkpoint,restored_epoch=latest_checkpoint(output,recorder)
    fit=trainer(config,output,[recorder,stop,best],args,0)
    counts=dict(total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    start=start_profile(args.device);recorder.fit_start=start
    fit.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader,ckpt_path=str(checkpoint) if checkpoint else None)
    seconds=recorder.previous_fit_seconds+time.perf_counter()-start;profile=gpu_profile(args.device)
    profile['gpu_peak_allocated_mib']=max(profile['gpu_peak_allocated_mib'],recorder.previous_peak_allocated)
    profile['gpu_peak_reserved_mib']=max(profile['gpu_peak_reserved_mib'],recorder.previous_peak_reserved)
    assert_frozen(model,config);assert sha256(config['predictor']['saved_model'])==initial_hash
    assert best.best_model_path
    record=dict(status='trained',universe=args.universe,variant=args.variant,stage1_seed=42,stage2_seed=0,
        stage1_retrained=False,stage1_unchanged=True,training_seconds=seconds,setup_seconds=start-phase_start,
        checkpoint=best.best_model_path,checkpoint_sha256=sha256(best.best_model_path),config=str(output/'effective.config.yaml'),
        completed_epochs=fit.current_epoch,best_val_loss=scalar(best.best_model_score),restored_epoch=restored_epoch,
        latest_checkpoint_epoch=json.loads((output/'checkpoints/latest_checkpoint.json').read_text())['epoch'],
        **profile,**counts)
    atomic_json(output/'training_fit.json',record)
    del model,fit,training,validation,train_loader,valid_loader;gc.collect();torch.cuda.empty_cache()
    finish_training(record,snapshot,args,phase_start)


def evaluate(args):
    initialize(0);start=start_profile(args.device);output=args.output_dir
    config=OmegaConf.to_container(OmegaConf.load(output/'effective.config.yaml'),resolve=True)
    record=json.loads((output/'training_result.json').read_text())
    model=Batch05Predictor.load_from_checkpoint(record['checkpoint'],config=config,T_max=1,map_location='cpu',strict=True)
    routing={}
    for split in ('valid','test'):
        data,_=cached(model,config,split,args);diagnostics(model,True)
        write_prediction(model,data,config,args.device,output/'evaluation',split,args.num_workers)
        pd.testing.assert_frame_equal(pd.read_pickle(output/'evaluation'/(split+'_predictions.pkl')),
            pd.read_pickle(output/'training_evaluation'/(split+'_predictions.pkl')),check_exact=True)
        routing[split]=diagnostics(model);del data;gc.collect()
    assert_frozen(model,config)
    atomic_json(output/'evaluation/completion.json',dict(status='complete',standalone_exact=True,
        seconds=time.perf_counter()-start,routing=routing,**gpu_profile(args.device)))


def preflight(args):
    checks=[];failures=[]
    for variant in VARIANTS:
        for market in ('csi300','csi800'):
            try:
                initialize(0);args.universe,args.variant=market,variant
                config=make_config(market,variant,args.output_dir/market/variant,0)
                model=Batch05Predictor(config,T_max=2).eval().to(args.device)
                data,_=cached(model,config,'valid',args);positions=DailyPositions(data).positions[20]
                batch={k:v.to(args.device) for k,v in float_collate(data[positions]).items()}
                with torch.no_grad():
                    parts,result=model.batch_prediction(batch)
                    h=model.encoder(model.revin(parts.stock_feature,mode='norm'))
                    assert torch.equal(h,batch['frozen_h']) and torch.equal(model.quantizer(h)[0],batch['frozen_zq'])
                model.train();model.log=lambda *a,**kw:None;model.log_dict=lambda *a,**kw:None
                begin=time.perf_counter();loss=model.training_step(batch,0);loss.backward()
                torch.cuda.synchronize();assert torch.isfinite(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4);optimizer.step()
                assert_frozen(model,config);model.eval()
                with torch.no_grad():expected=model.batch_prediction(batch)[1][0]
                restored=Batch05Predictor(copy.deepcopy(model.config),T_max=2).eval().to(args.device)
                restored.load_state_dict(model.state_dict(),strict=True)
                with torch.no_grad():assert torch.equal(expected,restored.batch_prediction(batch)[1][0])
                row=dict(universe=market,variant=variant,status='passed',stocks=len(positions),loss=float(loss.detach()),
                    cache_exact=True,frozen_stage1_exact=True,strict_restore_exact=True,finite_gradients=True,
                    full_day_train_step_seconds=time.perf_counter()-begin)
                checks.append(row);print(json.dumps(row),flush=True)
                del model,restored,data,batch,optimizer,result,loss
            except Exception as error:
                import traceback
                traceback.print_exc();failures.append(dict(universe=market,variant=variant,error=str(error)))
            gc.collect();torch.cuda.empty_cache()
            atomic_json(args.output_dir/'gpu_validation.json',dict(status='passed' if not failures else 'failed',checks=checks,failures=failures))
    if failures:raise RuntimeError('Preflight failures recorded')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','train','evaluate'])
    p.add_argument('--universe',choices=['csi300','csi800']);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--num-workers',type=int,default=2)
    args=p.parse_args();args.output_dir=args.output_dir.resolve()
    assert args.output_dir.is_relative_to(ROOT/'artifacts/research');args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.command!='preflight' and (not args.variant or not args.universe):p.error('Variant and universe required')
    globals()[args.command](args)


if __name__=='__main__':main()
