"""Prepare, train and evaluate the prespecified batches03/04 only."""
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

from trace_vq.cli.batch01 import gpu_profile,start_profile
from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.cli.next_screen import callbacks,trainer
from trace_vq.cli.screening import ROOT,initialize,now,read_dataset,scalar,write_prediction
from trace_vq.data.frozen import atomic_json,daily_loader,prepare_frozen_cache,sha256,float_collate,DailyPositions
from trace_vq.data.fsq_history import FSQHistoryDataset,history_features,index_hash
from trace_vq.models.batch01 import CoordinateFSQ
from trace_vq.models.batch34 import BATCH03,BATCH04,VARIANTS,Batch34Predictor
from trace_vq.models.screening import ScreeningPredictor
from trace_vq.training.research_run import RecoverableEpochRecord
from trace_vq.utils import apply_artifact_root

PREVIOUS=ROOT/"artifacts/research/next_directions_20261027"


def reference(market,family):
    return PREVIOUS/market/"fsq" if family=="fsq" else ROOT/"artifacts/research/mechanism_screen_20261028"/market/"baseline"


def make_config(market,variant,output,history_root,workers=2):
    settings=VARIANTS[variant]
    cfg=OmegaConf.load(reference(market,settings["family"])/"effective.config.yaml")
    cfg.artifact_root=str(Path(output).resolve())
    cfg.train.seed=0;cfg.train.num_workers=workers;cfg.train.wandb_mode="disabled"
    cfg.train.run_name=f"batch{settings['batch']:02d}_{market}_{variant}_s0"
    cfg.research=dict(variant=variant,history_root=str(Path(history_root).resolve()),**copy.deepcopy(settings))
    if settings.get("routing")=="finegrained":
        cfg.predictor.n_expert=settings["experts"];cfg.predictor.k=settings["active_experts"]
    elif settings.get("routing")=="fsq_sort":
        cfg.predictor.n_expert=3;cfg.predictor.k=1
    elif settings.get("routing")=="all_experts":
        cfg.predictor.k=cfg.predictor.n_expert
    apply_artifact_root(cfg)
    cfg=OmegaConf.to_container(cfg,resolve=True)
    assert cfg["train"]["num_epochs"]==70 and cfg["train"]["early_stopping"]["patience"]==15
    assert cfg["train"]["early_stopping"]["min_delta"]==1e-5 and cfg["predictor"]["target_day"]==5
    source=Path(cfg["predictor"]["saved_model"])
    assert source.is_file() and source.resolve().is_relative_to(ROOT)
    return cfg


def old_cache(market,family,split):
    return PREVIOUS/"cache"/("fsq" if family=="fsq" else "legacy")/market/(split+".npy")


def cached(model,config,split,args,context=True):
    source,path=read_dataset(config,split)
    cache=old_cache(args.universe,VARIANTS[args.variant]["family"],split)
    if not cache.is_file() or not cache.with_suffix(".json").is_file():
        raise FileNotFoundError("Existing frozen Stage1 cache missing: "+str(cache))
    data,metadata=prepare_frozen_cache(model,source,path,cache,config,args.device,args.num_workers)
    if context and args.variant in BATCH03:
        data=FSQHistoryDataset(data,args.history_root/args.universe,split,args.variant)
    return data,metadata


def prepare(args):
    initialize(0)
    for market in ("csi300","csi800"):
        start=time.perf_counter()
        args.universe,args.variant=market,"code_delta"
        config=make_config(market,args.variant,args.output_dir/market,args.history_root,0)
        output=args.history_root/market
        identity=dict(stage1_sha256=sha256(config["predictor"]["saved_model"]),
                      implementation=sha256(ROOT/"trace_vq/data/fsq_history.py"),coordinate_api=sha256(ROOT/"trace_vq/models/batch01.py"),
                      frozen_metadata={s:sha256(old_cache(market,"fsq",s).with_suffix(".json")) for s in ("train","valid","test")},
                      prototype_decay=.99,hash_bins=4096)
        marker=output/"completion.json"
        if marker.exists():
            assert json.loads(marker.read_text())["identity"]==identity,"Historical cache implementation/input changed"
            print(f"REUSE completed FSQ history {market}",flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            output.rename(output.with_name(market+"_interrupted_"+str(time.time_ns())))
        output.mkdir(parents=True,exist_ok=True)
        model=ScreeningPredictor(config,T_max=1).eval().to(args.device)
        quantizer=CoordinateFSQ.from_frozen(model.quantizer).to(args.device)
        indices=[];values=[];calendars=[];metadata={};offsets=[0]
        for split in ("train","valid","test"):
            data,meta=cached(model,config,split,args,False)
            indices.append(data.get_index().copy(deep=True))
            calendars.append(data.canonical.market_frame.index.copy())
            values.append(np.load(old_cache(market,"fsq",split),mmap_mode="r"))
            offsets.append(offsets[-1]+len(data));metadata[split]=meta
            del data
            gc.collect()
        index=indices[0].append(indices[1:]);assert index.is_unique
        def read_h(positions):
            result=np.empty((len(positions),128),dtype=np.float32)
            for i,cache in enumerate(values):
                mask=(positions>=offsets[i])&(positions<offsets[i+1])
                result[mask]=cache[positions[mask]-offsets[i],0]
            return result
        q=np.empty((len(index),3),dtype=np.uint8)
        groups=pd.Series(np.arange(len(index))).groupby(index.get_level_values("datetime"),sort=True).indices
        with torch.no_grad():
            for day,rows in enumerate(groups.values(),1):
                h=torch.from_numpy(read_h(rows)).to(args.device)
                rounded=quantizer.coordinates(h)[1]
                assert torch.equal((rounded.long()*quantizer.basis).sum(-1),model.quantizer(h)[2][2])
                q[rows]=rounded.cpu().numpy().astype(np.uint8)
                if day%500==0: print(f"{market} historical coordinates {day}/{len(groups)} dates",flush=True)
        calendar=calendars[0].append(calendars[1:]).unique().sort_values()
        observed=index.get_level_values("datetime")
        calendar=calendar[(calendar>=observed.min())&(calendar<=observed.max())]
        train_mask=np.arange(len(index))<offsets[1]
        history_features(index,q,read_h,train_mask,calendar,output)
        assert_frozen(model,config)
        atomic_json(marker,dict(status="complete",identity=identity,offsets=offsets,
                    index_sha256={s:index_hash(idx) for s,idx in zip(("train","valid","test"),indices)},
                    train_only_transition_fit=True,labels_used=False,seconds=time.perf_counter()-start,
                    files_sha256={p.name:sha256(p) for p in output.glob("*.npy")}))
        print(f"{market} FSQ history complete in {time.perf_counter()-start:.1f}s",flush=True)
        del model,quantizer,indices,values,q,index
        gc.collect();torch.cuda.empty_cache()


def reset_routing(model):
    moe=model.loadings.fusion.moe
    if hasattr(moe,"reset_diagnostics"):moe.reset_diagnostics()


def routing_stats(model):
    moe=model.loadings.fusion.moe
    return moe.diagnostics() if hasattr(moe,"diagnostics") else {}


def finish_training(record,config,args,start):
    model=Batch34Predictor.load_from_checkpoint(record["checkpoint"],config=config,T_max=1,map_location="cpu",strict=True)
    assert_frozen(model,config)
    begin=time.perf_counter()
    for split in ("valid","test"):
        data,_=cached(model,config,split,args)
        reset_routing(model)
        metrics=write_prediction(model,data,config,args.device,args.output_dir/"training_evaluation",split,args.num_workers)
        record["validation_metrics" if split=="valid" else "test_metrics"]=metrics
        record["routing_"+split]=routing_stats(model)
        del data;gc.collect()
    record.update(status="complete",training_inference_seconds=time.perf_counter()-begin,
                  train_phase_seconds=time.perf_counter()-start,finished=now())
    atomic_json(args.output_dir/"training_result.json",record)


def train(args):
    output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    if (output/"training_result.json").exists():raise FileExistsError("Completed training must be reused")
    initialize(0);phase_start=time.perf_counter()
    if (output/"training_fit.json").exists():
        config=OmegaConf.to_container(OmegaConf.load(output/"effective.config.yaml"),resolve=True)
        record=json.loads((output/"training_fit.json").read_text())
        assert sha256(record["checkpoint"])==record["checkpoint_sha256"]
        finish_training(record,config,args,phase_start)
        return
    config=make_config(args.universe,args.variant,output,args.history_root,args.num_workers)
    model=Batch34Predictor(config,T_max=1)
    training,train_meta=cached(model,config,"train",args)
    validation,valid_meta=cached(model,config,"valid",args)
    train_loader=daily_loader(training,shuffle=True,num_workers=args.num_workers)
    valid_loader=daily_loader(validation,num_workers=args.num_workers)
    model.T_max=len(train_loader)*config["train"]["num_epochs"]
    snapshot=copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot),output/"effective.config.yaml")
    first_hash=sha256(config["predictor"]["saved_model"])
    atomic_json(output/"input_manifest.json",dict(stage1_seed=42,stage2_seed=0,stage1_retrained=False,
        stage1_sha256=first_hash,train=train_meta,valid=valid_meta,test_used_for_selection=False,
        selection="minimum validation target MSE plus available native auxiliary terms; no test selection"))
    best,stop=callbacks(config,output,0)
    recorder=RecoverableEpochRecord(output,args.device)
    fit=trainer(config,output,[best,stop,recorder],args,0)
    last=output/"checkpoints/last.ckpt"
    if last.exists():
        checkpoint=torch.load(last,map_location="cpu",weights_only=True)
        assert recorder.state_key in checkpoint["callbacks"],"Checkpoint lacks recoverable RNG state"
        del checkpoint
    counts=dict(total_parameters=sum(p.numel() for p in model.parameters()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    moe=model.loadings.fusion.moe
    if hasattr(moe,"parameter_counts"):counts.update(moe.parameter_counts())
    start=start_profile(args.device);recorder.fit_start=start
    fit.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader,ckpt_path=str(last) if last.exists() else None)
    seconds=recorder.previous_fit_seconds+time.perf_counter()-start
    profile=gpu_profile(args.device)
    profile["gpu_peak_allocated_mib"]=max(profile["gpu_peak_allocated_mib"],recorder.previous_peak_allocated)
    profile["gpu_peak_reserved_mib"]=max(profile["gpu_peak_reserved_mib"],recorder.previous_peak_reserved)
    assert_frozen(model,config);assert sha256(config["predictor"]["saved_model"])==first_hash
    assert best.best_model_path
    record=dict(status="trained",universe=args.universe,variant=args.variant,stage1_seed=42,stage2_seed=0,
                stage1_retrained=False,stage1_unchanged=True,training_seconds=seconds,setup_seconds=start-phase_start,
                checkpoint=best.best_model_path,checkpoint_sha256=sha256(best.best_model_path),
                config=str(output/"effective.config.yaml"),completed_epochs=fit.current_epoch,
                best_val_loss=scalar(best.best_model_score),**profile,**counts)
    atomic_json(output/"training_fit.json",record)
    del model,fit,training,validation,train_loader,valid_loader
    gc.collect();torch.cuda.empty_cache()
    finish_training(record,snapshot,args,phase_start)


def evaluate(args):
    initialize(0);start=start_profile(args.device)
    output=args.output_dir
    config=OmegaConf.to_container(OmegaConf.load(output/"effective.config.yaml"),resolve=True)
    record=json.loads((output/"training_result.json").read_text())
    model=Batch34Predictor.load_from_checkpoint(record["checkpoint"],config=config,T_max=1,map_location="cpu",strict=True)
    diagnostics={}
    for split in ("valid","test"):
        data,_=cached(model,config,split,args)
        reset_routing(model)
        write_prediction(model,data,config,args.device,output/"evaluation",split,args.num_workers)
        pd.testing.assert_frame_equal(pd.read_pickle(output/"evaluation"/(split+"_predictions.pkl")),
                                      pd.read_pickle(output/"training_evaluation"/(split+"_predictions.pkl")),check_exact=True)
        diagnostics[split]=routing_stats(model)
        del data;gc.collect()
    assert_frozen(model,config)
    atomic_json(output/"evaluation/completion.json",dict(status="complete",standalone_exact=True,
                seconds=time.perf_counter()-start,routing=diagnostics,**gpu_profile(args.device)))


def preflight(args):
    checks=[];failures=[]
    variants=BATCH03 if args.batch==3 else BATCH04
    for variant in variants:
        for market in ("csi300","csi800"):
            try:
                initialize(0);args.universe,args.variant=market,variant
                config=make_config(market,variant,args.output_dir/market/variant,args.history_root,0)
                model=Batch34Predictor(config,T_max=2).eval().to(args.device)
                data,_=cached(model,config,"valid",args)
                positions=DailyPositions(data).positions[20]
                batch={k:v.to(args.device) for k,v in float_collate(data[positions]).items()}
                with torch.no_grad():
                    parts,result=model.batch_prediction(batch)
                    h=model.encoder(model.revin(parts.stock_feature,mode="norm"))
                    assert torch.equal(h,batch["frozen_h"])
                    assert torch.equal(model.quantizer(h)[0],batch["frozen_zq"])
                model.train();model.log=lambda *a,**kw:None;model.log_dict=lambda *a,**kw:None
                loss=model.training_step(batch,0);loss.backward()
                assert torch.isfinite(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters() if p.requires_grad)
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
                optimizer.step();assert_frozen(model,config)
                model.eval()
                with torch.no_grad():expected=model.batch_prediction(batch)[1][0]
                restored=Batch34Predictor(copy.deepcopy(model.config),T_max=2).eval().to(args.device)
                restored.load_state_dict(model.state_dict(),strict=True)
                with torch.no_grad():assert torch.equal(expected,restored.batch_prediction(batch)[1][0])
                row=dict(universe=market,variant=variant,status="passed",stocks=len(positions),loss=float(loss.detach()),
                         cache_exact=True,frozen_stage1_exact=True,strict_restore_exact=True,finite_gradients=True)
                checks.append(row);print(json.dumps(row),flush=True)
                del model,restored,data,batch,optimizer,result,loss
            except Exception as error:
                import traceback
                traceback.print_exc();failures.append(dict(universe=market,variant=variant,error=str(error)))
            gc.collect();torch.cuda.empty_cache()
            atomic_json(args.output_dir/"gpu_validation.json",dict(status="passed" if not failures else "failed",checks=checks,failures=failures))
    if failures:raise RuntimeError("Preflight failures recorded")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("prepare","preflight","train","evaluate"))
    parser.add_argument("--batch",type=int,choices=(3,4),default=3)
    parser.add_argument("--universe",choices=("csi300","csi800"))
    parser.add_argument("--variant",choices=VARIANTS)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--history-root",type=Path,required=True)
    parser.add_argument("--device",default="cuda")
    parser.add_argument("--num-workers",type=int,default=2)
    args=parser.parse_args()
    for name in ("output_dir","history_root"):
        path=getattr(args,name).resolve()
        if not path.is_relative_to(ROOT/"artifacts/research"):raise ValueError("Research artifact root required")
        setattr(args,name,path)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    if args.command in ("train","evaluate") and (not args.variant or not args.universe):parser.error("model and market required")
    globals()[args.command](args)


if __name__=="__main__":main()
