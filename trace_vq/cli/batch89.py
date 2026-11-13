"""Checklist batches08/09: preflight, train and exact standalone evaluation."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

from trace_vq.runtime import configure_runtime
configure_runtime()

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping

from trace_vq.cli.batch01 import reference_directory,cache_directory,gpu_profile,start_profile
from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.cli.next_screen import trainer
from trace_vq.cli.screening import ROOT,initialize,now,read_dataset,scalar,write_prediction
from trace_vq.data.frozen import atomic_json,daily_loader,prepare_frozen_cache,sha256,float_collate,DailyPositions
from trace_vq.models.batch89 import VARIANTS,Batch89Predictor,Batch89Stage1
from trace_vq.training.resilient import AtomicEpochCheckpoint,BatchHeartbeat
from trace_vq.utils import apply_artifact_root


def make_config(market,variant,output,workers=2,stage1_checkpoint=None,stage=2):
    settings=VARIANTS[variant];cfg=OmegaConf.load(reference_directory(market,'fsq')/'effective.config.yaml')
    cfg.artifact_root=str(Path(output).resolve());cfg.train.seed=42 if stage==1 else 0;cfg.train.num_workers=workers
    cfg.train.wandb_mode='disabled';cfg.train.run_name=f'batch{settings["batch"]:02d}_{market}_{variant}_stage{stage}'
    cfg.research=dict(variant=variant,**copy.deepcopy(settings))
    cfg.vqvae.predictor.pred_weight=settings.get('pred_weight',.0001)
    cfg.vqvae.quantizer.research_kind='batch89';cfg.vqvae.quantizer.research_levels=settings.get('levels',[8,8,8])
    cfg.vqvae.quantizer.batch89_settings=copy.deepcopy(settings)
    cfg.vqvae.quantizer.batch89_source_sha256={str(path):sha256(ROOT/path) for path in (
        'trace_vq/models/batch89_quantizers.py','trace_vq/models/batch89.py','trace_vq/data/batch89.py')}
    if stage1_checkpoint:cfg.predictor.saved_model=str(Path(stage1_checkpoint).resolve())
    apply_artifact_root(cfg);result=OmegaConf.to_container(cfg,resolve=True)
    assert result['train']['num_epochs']==70 and result['train']['early_stopping']['patience']==15
    assert result['train']['early_stopping']['min_delta']==1e-5 and result['predictor']['target_day']==5
    return result


def check_styles(data,config):
    if config['research']['quantizer']!='style':return
    for index,name in zip(config['research']['style_indices'],config['research']['style_names']):
        assert tuple(data.columns[index])==('feature',name),'Actual stock style columns unavailable'


def cached(model,config,split,args):
    data,path=read_dataset(config,split);check_styles(data,config)
    cache=args.output_dir/'cache'/(split+'.npy');cache.parent.mkdir(exist_ok=True)
    # A power loss between the atomic array rename and JSON write is repaired
    # only inside this run's own cache, preserving the rejected pair for audit.
    if cache.exists()!=cache.with_suffix('.json').exists():
        archive=cache.parent/('incomplete_'+split+'_'+str(time.time_ns()));archive.mkdir()
        for candidate in (cache,cache.with_suffix('.json')):
            if candidate.exists():candidate.rename(archive/candidate.name)
    result,metadata=prepare_frozen_cache(model,data,path,cache,config,args.device,args.num_workers)
    if args.variant in ('fsq_group','fsq_res'):
        components=cache.with_name(split+'_components.npy');marker=components.with_suffix('.json')
        identity=dict(frozen_cache_sha256=metadata['cache_sha256'],stage1_sha256=metadata['identity']['stage1_sha256'])
        if components.exists()!=marker.exists():
            archive=cache.parent/('incomplete_components_'+str(time.time_ns()));archive.mkdir()
            for candidate in (components,marker):
                if candidate.exists():candidate.rename(archive/candidate.name)
        if marker.exists():assert json.loads(marker.read_text())['identity']==identity
        else:
            temporary=components.with_suffix('.partial.npy')
            indices=np.lib.format.open_memmap(temporary,mode='w+',dtype=np.int16,shape=(len(result),2))
            model.eval().to(args.device)
            with torch.no_grad():
                for positions in DailyPositions(result).positions:
                    h=torch.from_numpy(result.values[positions,0].copy()).to(args.device)
                    model.quantizer(h);indices[positions]=model.quantizer.last_components.cpu().numpy()
            indices.flush();del indices;temporary.replace(components)
            atomic_json(marker,dict(identity=identity,shape=[len(result),2],dtype='int16',sha256=sha256(components),
                representation_layout='group0=z_q[:D/2],group1=z_q[D/2:]' if args.variant=='fsq_group' else 'coarse/fine integer indices; summed z_q in frozen cache'))
        metadata=dict(metadata,component_codes=json.loads(marker.read_text()))
    return result,metadata


def diagnostics(model,reset=False):
    return dict(variant=model.research['variant'])


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
    model=Batch89Predictor.load_from_checkpoint(record['checkpoint'],config=config,T_max=1,map_location='cpu',strict=True)
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
    config=make_config(args.universe,args.variant,output,args.num_workers,args.stage1_checkpoint)
    model=Batch89Predictor(config,T_max=1)
    training,train_meta=cached(model,config,'train',args)
    validation,valid_meta=cached(model,config,'valid',args)
    train_loader=daily_loader(training,shuffle=True,num_workers=args.num_workers)
    valid_loader=daily_loader(validation,num_workers=args.num_workers)
    model.T_max=len(train_loader)*config['train']['num_epochs'];snapshot=copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot),output/'effective.config.yaml')
    initial_hash=sha256(config['predictor']['saved_model'])
    atomic_json(output/'input_manifest.json',dict(stage1_seed=42,stage2_seed=0,stage1_retrained=True,
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
        stage1_retrained=True,stage1_unchanged=True,training_seconds=seconds,setup_seconds=start-phase_start,
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
    model=Batch89Predictor.load_from_checkpoint(record['checkpoint'],config=config,T_max=1,map_location='cpu',strict=True)
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


def train_stage1(args):
    from trace_vq.data.batch89 import PairedCanonicalDays
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    if (output/'training_result.json').exists():raise FileExistsError('Reuse completed Stage1')
    initialize(42);phase_start=time.perf_counter()
    config=make_config(args.universe,args.variant,output,args.num_workers,stage=1)
    train_data,train_path=read_dataset(config,'train');valid_data,valid_path=read_dataset(config,'valid')
    check_styles(train_data,config);check_styles(valid_data,config)
    if args.variant=='fsq_contrast':train_data=PairedCanonicalDays(train_data)
    train_loader=daily_loader(train_data,shuffle=True,num_workers=args.num_workers)
    valid_loader=daily_loader(valid_data,num_workers=args.num_workers)
    model=Batch89Stage1(config,T_max=len(train_loader)*config['train']['num_epochs'])
    OmegaConf.save(OmegaConf.create(config),output/'effective.config.yaml')
    atomic_json(output/'input_manifest.json',dict(stage=1,seed=42,train_sha256=sha256(train_path),valid_sha256=sha256(valid_path),
        test_used_for_training_or_selection=False,selection='validation reconstruction + intrinsic quantizer objective + declared prediction weight * native prediction MSE',
        extra_correlation_or_contrast_penalties_for_training_only=True,settings=config['research']))
    best=AtomicEpochCheckpoint(dirpath=output/'checkpoints',monitor='val_loss',mode='min',save_top_k=1,save_last=True,
        save_on_train_epoch_end=True,filename='seed42-{epoch}-{val_loss:.6f}',enable_version_counter=False)
    early=config['train']['early_stopping'];stop=EarlyStopping(monitor='val_loss',mode='min',
        patience=early['patience'],min_delta=early['min_delta'],verbose=True)
    recorder=BatchHeartbeat(output,args.device);checkpoint,restored_epoch=latest_checkpoint(output,recorder)
    fit=trainer(config,output,[recorder,stop,best],args,42)
    counts=dict(total_parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    start=start_profile(args.device);recorder.fit_start=start
    fit.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader,ckpt_path=str(checkpoint) if checkpoint else None)
    seconds=recorder.previous_fit_seconds+time.perf_counter()-start;profile=gpu_profile(args.device)
    profile['gpu_peak_allocated_mib']=max(profile['gpu_peak_allocated_mib'],recorder.previous_peak_allocated)
    profile['gpu_peak_reserved_mib']=max(profile['gpu_peak_reserved_mib'],recorder.previous_peak_reserved)
    assert best.best_model_path
    # Strictly restore the selected Stage1 before releasing it to Stage2.
    restored=Batch89Stage1.load_from_checkpoint(best.best_model_path,config=config,T_max=1,map_location='cpu',strict=True)
    atomic_json(output/'training_result.json',dict(status='complete',stage=1,universe=args.universe,variant=args.variant,stage1_seed=42,
        checkpoint=best.best_model_path,checkpoint_sha256=sha256(best.best_model_path),config=str(output/'effective.config.yaml'),
        completed_epochs=fit.current_epoch,best_val_loss=scalar(best.best_model_score),restored_epoch=restored_epoch,
        training_seconds=seconds,setup_seconds=start-phase_start,train_phase_seconds=time.perf_counter()-phase_start,
        stage1_retrained=True,strict_restore=True,finished=now(),**profile,**counts))


def preflight(args):
    from trace_vq.data.batch89 import PairedCanonicalDays
    checks=[];failures=[]
    for variant in VARIANTS:
        for market in ('csi300','csi800'):
            try:
                initialize(42);args.universe,args.variant=market,variant
                output=args.output_dir/market/variant;output.mkdir(parents=True,exist_ok=True)
                config=make_config(market,variant,output,0,stage=1)
                canonical,_=read_dataset(config,'valid');check_styles(canonical,config)
                data=PairedCanonicalDays(canonical) if variant=='fsq_contrast' else canonical
                positions=DailyPositions(data).positions[20];batch=float_collate(data[positions])
                batch={k:v.to(args.device) for k,v in batch.items()} if isinstance(batch,dict) else batch.to(args.device)
                model=Batch89Stage1(config,T_max=2).to(args.device).train()
                model.log=lambda *a,**k:None;model.log_dict=lambda *a,**k:None
                start=start_profile(args.device);loss=model.training_step(batch,0);loss.backward()
                assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.vqvae.spatial_encoder.parameters())
                optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4);optimizer.step();model.eval()
                checkpoint=output/'preflight_stage1.ckpt';torch.save(dict(state_dict=model.state_dict(),preflight_only=True),checkpoint)
                restored=Batch89Stage1(config,T_max=2).eval().to(args.device);restored.load_state_dict(model.state_dict(),strict=True)
                current=batch['canonical'] if isinstance(batch,dict) else batch
                from trace_vq.data.schema import unpack_batch
                parts=unpack_batch(current)
                with torch.no_grad():
                    a=model(parts.stock_feature,parts.prior_factor,parts.future_returns)[4]
                    b=restored(parts.stock_feature,parts.prior_factor,parts.future_returns)[4]
                    assert torch.equal(a,b)
                del model,restored,optimizer,loss;gc.collect();torch.cuda.empty_cache()
                config=make_config(market,variant,output,0,checkpoint)
                predictor=Batch89Predictor(config,T_max=2).to(args.device).eval()
                with torch.no_grad():
                    h=predictor.encoder(predictor.revin(parts.stock_feature,mode='norm'));z=predictor.quantizer(h)[0]
                frozen=dict(canonical=current,frozen_h=h,frozen_zq=z)
                predictor.train();predictor.log=lambda *a,**k:None;predictor.log_dict=lambda *a,**k:None
                loss=predictor.training_step(frozen,0);loss.backward()
                assert torch.isfinite(loss) and all(p.grad is None or torch.isfinite(p.grad).all() for p in predictor.parameters())
                optimizer=torch.optim.AdamW([p for p in predictor.parameters() if p.requires_grad],lr=1e-4);optimizer.step()
                assert_frozen(predictor,config);predictor.eval()
                with torch.no_grad():expected=predictor.batch_prediction(frozen)[1][0]
                restored=Batch89Predictor(copy.deepcopy(config),T_max=2).eval().to(args.device)
                restored.load_state_dict(predictor.state_dict(),strict=True)
                with torch.no_grad():assert torch.equal(expected,restored.batch_prediction(frozen)[1][0])
                torch.cuda.synchronize()
                row=dict(universe=market,variant=variant,status='passed',stocks=len(positions),stage1_step=True,stage2_step=True,
                    stage1_strict_restore=True,stage2_strict_restore=True,frozen_stage1=True,finite_gradients=True,
                    seconds=time.perf_counter()-start,**gpu_profile(args.device))
                checks.append(row);print(json.dumps(row),flush=True)
                del predictor,restored,optimizer,data,canonical,batch,parts,loss,h,z,frozen,current,a,b,expected
            except Exception as error:
                import traceback
                traceback.print_exc();failures.append(dict(universe=market,variant=variant,error=str(error)))
            gc.collect();torch.cuda.empty_cache()
            atomic_json(args.output_dir/'gpu_validation.json',dict(status='passed' if not failures else 'failed',checks=checks,failures=failures))
    if failures:raise RuntimeError('GPU preflight failures recorded; inspect per-variant errors')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['preflight','train_stage1','train','evaluate'])
    p.add_argument('--universe',choices=['csi300','csi800']);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--stage1-checkpoint',type=Path)
    p.add_argument('--device',default='cuda');p.add_argument('--num-workers',type=int,default=2)
    args=p.parse_args();args.output_dir=args.output_dir.resolve()
    assert args.output_dir.is_relative_to(ROOT/'artifacts/research');args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.command!='preflight' and (not args.variant or not args.universe):p.error('Variant and universe required')
    if args.command=='train':
        if not args.stage1_checkpoint or not args.stage1_checkpoint.is_file():p.error('Completed Stage1 checkpoint required')
        if not args.stage1_checkpoint.resolve().is_relative_to(ROOT/'artifacts/research'):p.error('Research-owned Stage1 checkpoint required')
        if torch.load(args.stage1_checkpoint,map_location='cpu',weights_only=True).get('preflight_only'):p.error('Preflight weights must never seed a formal fit')
    globals()[args.command](args)


if __name__=='__main__':main()


