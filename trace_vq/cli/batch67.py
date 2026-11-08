"""Checklist batches06/07: preflight, train and exact standalone evaluation."""
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
from trace_vq.models.batch67 import VARIANTS,Batch67Predictor
from trace_vq.training.resilient import AtomicEpochCheckpoint,BatchHeartbeat
from trace_vq.utils import apply_artifact_root


def make_config(market,variant,output,workers=2,cache_root=None):
    settings=VARIANTS[variant];cfg=OmegaConf.load(reference_directory(market,settings['family'])/'effective.config.yaml')
    cfg.artifact_root=str(Path(output).resolve());cfg.train.seed=0;cfg.train.num_workers=workers
    cfg.train.wandb_mode='disabled';cfg.train.run_name=f'batch{settings["batch"]:02d}_{market}_{variant}_s0'
    cache_root=Path(cache_root or ROOT/'artifacts/research/batch0607_20261107').resolve()
    cfg.research=dict(variant=variant,statistics_file=str(cache_root/'statistics'/(market+'.json')),
        history_root=str(ROOT/'artifacts/research/batch03_20261105/history'/market),memory_root=str(cache_root/'memory'/market),**copy.deepcopy(settings))
    apply_artifact_root(cfg);result=OmegaConf.to_container(cfg,resolve=True)
    assert result['train']['num_epochs']==70 and result['train']['early_stopping']['patience']==15
    assert result['train']['early_stopping']['min_delta']==1e-5 and result['predictor']['target_day']==5
    checkpoint=Path(result['predictor']['saved_model']);assert checkpoint.is_file() and checkpoint.is_relative_to(ROOT)
    return result


def cached(model,config,split,args):
    data,path=read_dataset(config,split)
    cache=cache_directory(args.universe,VARIANTS[args.variant]['family'])/(split+'.npy')
    if not cache.is_file() or not cache.with_suffix('.json').is_file():raise FileNotFoundError(f'Existing Stage1 cache missing: {cache}')
    result,metadata=prepare_frozen_cache(model,data,path,cache,config,args.device,args.num_workers)
    if args.variant=='hopfield_retrieval':
        from trace_vq.data.hopfield import HopfieldDataset
        result=HopfieldDataset(result,config['research']['memory_root'],config['research']['history_root'],split,
            [cache_directory(args.universe,'fsq')/(s+'.npy') for s in ('train','valid','test')])
        metadata=dict(metadata,causal_memory=json.loads((Path(config['research']['memory_root'])/'completion.json').read_text()))
    return result,metadata


def diagnostics(model,reset=False):
    return dict(variant=model.variant)


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
    model=Batch67Predictor.load_from_checkpoint(record['checkpoint'],config=config,T_max=1,map_location='cpu',strict=True)
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
    config=make_config(args.universe,args.variant,output,args.num_workers,args.cache_root)
    model=Batch67Predictor(config,T_max=1)
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
    model=Batch67Predictor.load_from_checkpoint(record['checkpoint'],config=config,T_max=1,map_location='cpu',strict=True)
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
                config=make_config(market,variant,args.output_dir/market/variant,0,args.cache_root)
                model=Batch67Predictor(config,T_max=2).eval().to(args.device)
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
                restored=Batch67Predictor(copy.deepcopy(model.config),T_max=2).eval().to(args.device)
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


def prepare(args):
    from trace_vq.data.fsq_history import index_hash
    from trace_vq.data.hopfield import build_candidates
    from trace_vq.data.schema import GROUP_SLICES
    initialize(0);checks=[];failures=[]
    (args.cache_root/'statistics').mkdir(parents=True,exist_ok=True)
    for market in ('csi300','csi800'):
        try:
            start=time.perf_counter();config=make_config(market,'fsq_soft',args.cache_root/'prepare'/market,0,args.cache_root)
            history=Path(config['research']['history_root']);old=json.loads((history/'completion.json').read_text())
            assert old['status']=='complete' and not old['labels_used']
            assert old['identity']['stage1_sha256']==sha256(config['predictor']['saved_model'])
            q=np.load(history/'coordinates.npy',mmap_mode='r');dates=np.load(history/'date_codes.npy',mmap_mode='r')
            for name in ('coordinates.npy','date_codes.npy'):
                assert old['files_sha256'][name]==sha256(history/name)
            end=old['offsets'][1];codes=(q[:end].astype(np.int64)*np.array([1,8,64])).sum(-1)
            stats=dict(train_code_counts=np.bincount(codes,minlength=512).tolist(),rows=end,
                train_only=True,index_sha256=old['index_sha256']['train'],coordinates_sha256=old['files_sha256']['coordinates.npy'])
            atomic_json(args.cache_root/'statistics'/(market+'.json'),stats)
            identity=dict(history_sha256=sha256(history/'completion.json'),implementation=sha256(ROOT/'trace_vq/data/hopfield.py'),
                cli_implementation=sha256(ROOT/'trace_vq/cli/batch67.py'),dataset_metadata={s:sha256(cache_directory(market,'fsq')/(s+'.json')) for s in ('train','valid','test')},
                candidates=16,bucket_capacity=8,label_horizon=5,rule='source_day_plus_5_strictly_before_query_day')
            output=args.cache_root/'memory'/market;marker=output/'completion.json'
            if marker.exists():
                completed=json.loads(marker.read_text());assert completed['identity']==identity,'Retrieval cache inputs changed'
                checks.append(dict(universe=market,status='passed',reused=True,spec=completed['spec']));continue
            if output.exists() and any(output.iterdir()):
                output.rename(output.with_name(market+'_interrupted_'+str(time.time_ns())))
            labels=[];calendars=[];indices=[]
            for split in ('train','valid','test'):
                data,path=read_dataset(config,split);idx=data.get_index();assert index_hash(idx)==old['index_sha256'][split]
                # Native endpoint lookup, verified against actual sampled windows.
                points=data.idx_arr[data.idx_map[:,0],data.idx_map[:,1]]
                assert np.isfinite(points).all()
                target=data.data_arr[points.astype(np.int64),GROUP_SLICES['label'].start+4].astype(np.float32)
                probe=np.linspace(0,len(data)-1,11,dtype=int)
                np.testing.assert_array_equal(target[probe],data[probe][:,-1,GROUP_SLICES['label'].start+4].astype(np.float32))
                labels.append(target);indices.append(idx.copy());calendars.append(data.market_frame.index.copy());del data;gc.collect()
            index=indices[0].append(indices[1:]);calendar=calendars[0].append(calendars[1:]).unique().sort_values()
            observed=index.get_level_values('datetime');calendar=calendar[(calendar>=observed.min())&(calendar<=observed.max())]
            np.testing.assert_array_equal(dates,calendar.get_indexer(observed))
            spec=build_candidates(dates,q,np.concatenate(labels),output,days=len(calendar))
            files={p.name:sha256(p) for p in output.glob('*.npy')}
            completed=dict(status='complete',identity=identity,spec=spec,files_sha256=files,seconds=time.perf_counter()-start)
            atomic_json(marker,completed);checks.append(dict(completed,universe=market,status='passed',reused=False))
            print(json.dumps(dict(universe=market,status='prepared',seconds=completed['seconds'],spec=spec)),flush=True)
            del q,dates,index,labels;gc.collect()
        except Exception as error:
            import traceback
            traceback.print_exc();failures.append(dict(universe=market,error=str(error)))
        atomic_json(args.cache_root/'prepare_validation.json',dict(checks=checks,failures=failures,status='passed' if not failures else 'failed'))
    if failures:raise RuntimeError('Some preparation failed; other variants remain eligible in preflight')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','preflight','train','evaluate'])
    p.add_argument('--cache-root',type=Path,default=ROOT/'artifacts/research/batch0607_20261107');p.add_argument('--universe',choices=['csi300','csi800']);p.add_argument('--variant',choices=VARIANTS)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--num-workers',type=int,default=2)
    args=p.parse_args();args.output_dir=args.output_dir.resolve();args.cache_root=args.cache_root.resolve()
    assert args.cache_root.is_relative_to(ROOT/'artifacts/research')
    assert args.output_dir.is_relative_to(ROOT/'artifacts/research');args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.command not in ('preflight','prepare') and (not args.variant or not args.universe):p.error('Variant and universe required')
    globals()[args.command](args)


if __name__=='__main__':main()
