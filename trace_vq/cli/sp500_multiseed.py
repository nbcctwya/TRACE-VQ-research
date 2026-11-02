"""Fixed TRACE-VQ SP500 protocol: Stage1 seed42, Stage2 seeds0-4.

Only orchestration changes; representation, prediction, losses, and optimizers
are the existing TRACE-VQ implementations. The baseline wrapper permits exact
frozen-feature reuse while preserving the original daily cross sections.
"""
import argparse
import copy
import gc
import hashlib
import json
import pickle
import time
from pathlib import Path

from trace_vq.runtime import configure_runtime
configure_runtime()

import numpy as np
import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from trace_vq.cli.screening import EpochRecord, initialize, now, scalar, write_prediction
from trace_vq.cli.next_screen import QuietFactorVQVAE, callbacks, trainer
from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.cli.stage2 import _apply_stage2_universe_preset
from trace_vq.data.frozen import atomic_json, daily_loader, prepare_frozen_cache, sha256
from trace_vq.data.schema import dataset_basename, unpack_batch, validate_columns, MARKET_INDICES
from trace_vq.models.screening import ScreeningPredictor
from trace_vq.training.prediction import GenerateReturn
from trace_vq.utils import apply_artifact_root

ROOT = Path(__file__).resolve().parents[2]
SEEDS = tuple(range(5))


def make_config(output, data_root, seed, stage1=None):
    if seed not in (*SEEDS, 42):
        raise ValueError("Only the prespecified seeds0-4 and Stage1seed42 are permitted")
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    with initialize_config_dir(config_dir=str(ROOT / "configs"), version_base="1.3"):
        cfg = compose(config_name="config", overrides=["experiment=full", "data.universe=sp500"])
    _apply_stage2_universe_preset(cfg, [])
    cfg.data.data_path = str(Path(data_root).resolve())
    cfg.predictor.saved_model = str(Path(stage1).resolve()) if stage1 else "NOT_USED_DURING_STAGE1"
    cfg.train.seed = seed
    cfg.train.num_workers = 2
    cfg.train.wandb_mode = "disabled"
    cfg.artifact_root = str(Path(output).resolve())
    apply_artifact_root(cfg)
    result = OmegaConf.to_container(cfg, resolve=True)
    result["research"] = {"variant": "baseline"}
    result["inference"] = {"matmul_precision": "high"}
    assert result["vqvae"]["encoder"]["type"] == "temporal-attention"
    assert result["predictor"]["shared_expert"] and result["predictor"]["quantization_confidence_adapter"]
    assert (result["predictor"]["n_expert"], result["predictor"]["k"],
            result["predictor"]["transformer"]["num_heads"], result["predictor"]["aux_weight"]) == (8, 4, 4, .001)
    assert result["train"]["num_epochs"] == 70 and result["predictor"]["target_day"] == 5
    return result


def read_dataset(config, split):
    path = Path(config["data"]["data_path"]) / "US" / f"{dataset_basename('sp500')}_{split}.pkl"
    with path.open("rb") as stream:
        dataset = pickle.load(stream)
    validate_columns(dataset.columns)
    if dataset.universe != "sp500" or tuple(dataset.market_indices) != MARKET_INDICES["sp500"]:
        raise ValueError("SP500 dataset provenance does not match")
    # Some old pickles retain cached DatetimeIndex reductions from the parent
    # array. Use the actual timestamp values for audit bounds.
    dates = dataset.get_index().get_level_values("datetime").to_numpy()
    begin, end = map(np.datetime64, config["data"][f"{split}_period"])
    if dates.min() < begin or dates.max() > end:
        raise ValueError(f"{split} rows extend outside the frozen data split")
    return dataset, path


def cached(model, config, split, args):
    dataset, path = read_dataset(config, split)
    return prepare_frozen_cache(model, dataset, path, args.root / "cache" / f"{split}.npy",
                                config, args.device, 2)


def audit_data(args):
    import yaml
    import qlib
    from qlib.data import D
    from qlib.data.dataset.processor import CSRankNorm
    from trace_vq.data.prepare import label_config
    source = ROOT.parent / "HVQ-Stock/dataset/processed/US"
    config = make_config(args.root / "audit", args.data_root, 42)
    original_cfg = ROOT.parent / "HVQ-Stock/dataset/2025_sp500.yaml"
    settings = yaml.safe_load(original_cfg.read_text())
    assert str(settings["data_handler_config"]["fit_start_time"])[:10] == "2009-01-01"
    assert str(settings["data_handler_config"]["fit_end_time"])[:10] == "2020-12-31"
    qlib.init(provider_uri=str(Path.home()/".qlib/qlib_data/us_data"), region="us", kernels=1,
              exp_manager={"class":"MLflowExpManager","module_path":"qlib.workflow.expm",
                           "kwargs":{"uri":str(args.root/"audit/qlib_runs"),"default_exp_name":"data_audit"}})
    reports = {}
    for split in ("train", "valid", "test"):
        src = source / f"sp500_20_h10_{split}.pkl"
        target = args.data_root / "US" / src.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.symlink_to(src.resolve())
        if target.resolve() != src.resolve():
            raise ValueError("Refusing to replace another dataset")
        digest = sha256(src)
        dataset, path = read_dataset(config, split)
        index = dataset.get_index()
        days = pd.DatetimeIndex(np.unique(index.get_level_values("datetime").to_numpy()))
        calendar = pd.DatetimeIndex(D.calendar(start_time=config["data"][f"{split}_period"][0],
                                              end_time=config["data"][f"{split}_period"][1]))
        assert days.equals(calendar) and index.is_unique
        date = days[len(days)//2]
        positions = np.flatnonzero(index.get_level_values("datetime").to_numpy() == date.to_datetime64())
        batch = dataset[positions]
        parts = unpack_batch(batch)
        assert np.isfinite(parts.stock_feature).all() and np.isfinite(parts.prior_factor).all()
        assert np.isfinite(parts.market_feature).all()
        assert np.all(parts.market_feature == parts.market_feature[:1])
        stocks = index.take(positions).get_level_values("instrument").tolist()
        expressions, names = label_config()
        raw = D.features(stocks, expressions, start_time=str(date.date()), end_time=str(date.date()))
        raw.columns = pd.MultiIndex.from_product([["label"], names])
        raw = raw.reorder_levels(["datetime", "instrument"]).sort_index()
        raw = raw.reindex(index.take(positions))
        if split != "test":
            raw = CSRankNorm(fields_group="label")(raw)
        expected = raw.to_numpy()
        actual = parts.future_returns
        finite = np.isfinite(expected) & np.isfinite(actual)
        assert finite[:,4].mean() > .9
        assert np.allclose(expected[finite], actual[finite], atol=2e-5, rtol=2e-5), split
        reports[split] = dict(source=str(src.resolve()), path=str(path), sha256=digest,
            bytes=src.stat().st_size, rows=len(dataset), days=len(days),
            stocks=int(index.get_level_values("instrument").nunique()), check_date_stocks=len(set(stocks)),
            start=str(days[0]), end=str(days[-1]), shape=list(dataset[0].shape),
            calendar_exact=True, raw_quote_label_check_date=str(date.date()), all_ten_labels_checked=True,
            label_preprocessing="raw" if split=="test" else "CSRankNorm",
            compared_label_values=int(finite.sum()), max_label_error=float(np.max(np.abs(expected[finite]-actual[finite]))),
            finite_stock_prior_market=True, shared_market_channel_exact=True, imported_by_read_only_usage_symlink=True)
        del dataset, index, batch, parts, raw, expected, actual
        gc.collect()
    atomic_json(args.root/"data_audit.json", dict(status="passed",splits=reports,
                original_handler_config=str(original_cfg),original_handler_sha256=sha256(original_cfg),
                label_formula="Ref($close,-5)/Ref($close,-1)-1",new_dataset_copies=0,
                train="2009-2020",valid="2021-2022",test="2023-2025"))


def stage1_check(args):
    config = make_config(args.root/"preflight_stage1", args.data_root, 42)
    initialize(42)
    source, _ = read_dataset(config, "train")
    batch = next(iter(daily_loader(source, num_workers=0))).to(args.device)
    model = QuietFactorVQVAE(config,T_max=70*3021).to(args.device).train()
    parts = unpack_batch(batch)
    result = model(parts.stock_feature, parts.prior_factor, parts.future_returns)
    loss = result[3]
    assert torch.isfinite(loss)
    loss.backward()
    gradients = {n:p.grad for n,p in model.named_parameters() if p.grad is not None}
    assert gradients and all(torch.isfinite(v).all() for v in gradients.values())
    assert any("spatial_encoder" in n and torch.count_nonzero(v) for n,v in gradients.items())
    atomic_json(args.root/"stage1_gpu_check.json",dict(status="passed",rows=len(batch),loss=scalar(loss),
                gradient_tensors=len(gradients),temporal_attention_gradient_nonzero=True,
                technical_check_only=True,scientific_fits=0))


def stage1_train(args):
    output=args.root/"stage1"
    if (output/"training_result.json").exists(): raise FileExistsError("Stage1 already complete")
    output.mkdir(parents=True,exist_ok=True)
    config=make_config(output,args.data_root,42)
    initialize(42)
    train_data,_=read_dataset(config,"train"); valid_data,_=read_dataset(config,"valid")
    train_loader=daily_loader(train_data,shuffle=True,num_workers=2)
    valid_loader=daily_loader(valid_data,num_workers=2)
    model=QuietFactorVQVAE(config,len(train_loader)*config["train"]["num_epochs"])
    OmegaConf.save(OmegaConf.create(config),output/"effective.config.yaml")
    best,stop=callbacks(config,output,42)
    fit=trainer(config,output,[best,stop,EpochRecord(output)],args,42)
    start=time.monotonic();fit.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader)
    if not best.best_model_path:raise RuntimeError("No Stage1 validation checkpoint")
    restored=QuietFactorVQVAE.load_from_checkpoint(best.best_model_path,config=copy.deepcopy(config),
                                                   T_max=1,map_location="cpu",strict=True)
    assert restored.config["train"]["seed"]==42
    atomic_json(output/"training_result.json",dict(status="complete",universe="sp500",stage1_seed=42,
        checkpoint=best.best_model_path,checkpoint_sha256=sha256(best.best_model_path),
        best_val_loss=scalar(best.best_model_score),completed_epochs=fit.current_epoch,
        elapsed_seconds=time.monotonic()-start,config=str(output/"effective.config.yaml"),
        strict_restore=True,test_used_for_training_or_selection=False,finished=now()))


def first_record(args):
    return json.loads((args.root/"stage1/training_result.json").read_text())


def second_config(args, seed=None):
    return make_config(args.root/f"seed{args.seed if seed is None else seed}",args.data_root,
                       args.seed if seed is None else seed,first_record(args)["checkpoint"])


def cache_all(args):
    config=second_config(args,0);initialize(0);model=ScreeningPredictor(config,T_max=1)
    records={}
    for split in ("train","valid","test"):
        data,records[split]=cached(model,config,split,args);del data;gc.collect()
    atomic_json(args.root/"cache/completion.json",dict(status="complete",splits=records,
                stage1_sha256=first_record(args)["checkpoint_sha256"]))


def validate_cache(args):
    checks=[]; hashes=[]
    for seed in SEEDS:
        config=second_config(args,seed);initialize(seed)
        model=ScreeningPredictor(config,T_max=100).to(args.device)
        hashes.append(hashlib.sha256(torch.cat([p.detach().cpu().flatten() for p in model.parameters()
                                               if p.requires_grad]).numpy().tobytes()).hexdigest())
        data,_=cached(model,config,"train",args)
        cached_batch=next(iter(daily_loader(data,num_workers=0)))
        cached_batch={k:v.to(args.device) for k,v in cached_batch.items()}
        frozen={n:v.detach().clone() for n,v in model.state_dict().items()
                if n.startswith(("encoder.","quantizer.","revin."))}
        model.train();initialize(500+seed)
        parts,result=model.batch_prediction(cached_batch);loss,_=model.training_loss(parts,result)
        loss.backward()
        grad={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True);initialize(500+seed)
        parts,result=model.batch_prediction(cached_batch["canonical"]);raw_loss,_=model.training_loss(parts,result)
        raw_loss.backward()
        assert torch.equal(loss,raw_loss)
        for n,p in model.named_parameters():
            if n in grad:assert torch.equal(grad[n],p.grad),n
        for n,v in frozen.items():assert torch.equal(v,model.state_dict()[n]),n
        native=GenerateReturn(copy.deepcopy(config),T_max=100).to(args.device)
        native.load_state_dict(model.state_dict(),strict=True);native.train();initialize(800+seed)
        parts=unpack_batch(cached_batch["canonical"])
        nresult=native(parts.stock_feature,parts.prior_factor)
        initialize(800+seed);wresult=model(parts.stock_feature,parts.prior_factor)
        assert all(torch.equal(a,b) for a,b in zip(nresult,wresult))
        checks.append(dict(seed=seed,cached_raw_loss_exact=True,all_stage2_gradients_exact=True,
                           native_forward_exact=True,stage1_unchanged=True,rows=len(parts.stock_feature)))
        del data,model,native,cached_batch,grad,frozen,parts,result,nresult,wresult,loss,raw_loss
        gc.collect();torch.cuda.empty_cache()
    assert len(set(hashes))==5
    atomic_json(args.root/"stage2_gpu_checks.json",dict(status="passed",checks=checks,
                five_distinct_initializations=True,initialization_sha256=hashes,technical_checks_only=True))


def stage2_train(args):
    output=args.root/f"seed{args.seed}"
    if (output/"training_result.json").exists():raise FileExistsError("Seed already completed")
    output.mkdir(parents=True,exist_ok=True);initialize(args.seed)
    config=second_config(args);model=ScreeningPredictor(config,T_max=1)
    training,tm=cached(model,config,"train",args);validation,vm=cached(model,config,"valid",args)
    train_loader=daily_loader(training,shuffle=True,num_workers=2);valid_loader=daily_loader(validation,num_workers=2)
    model.T_max=len(train_loader)*config["train"]["num_epochs"];snapshot=copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot),output/"effective.config.yaml")
    atomic_json(output/"input_manifest.json",dict(stage1_seed=42,stage2_seed=args.seed,
                stage1_sha256=first_record(args)["checkpoint_sha256"],train=tm,valid=vm,
                test_used_for_training_or_selection=False,selection="minimum native validation MSE plus auxiliary loss"))
    best,stop=callbacks(config,output,args.seed)
    fit=trainer(config,output,[best,stop,EpochRecord(output)],args,args.seed)
    start=time.monotonic();fit.fit(model,train_dataloaders=train_loader,val_dataloaders=valid_loader)
    if not best.best_model_path:raise RuntimeError("No Stage2 validation checkpoint")
    assert_frozen(model,config)
    record=dict(status="complete",universe="sp500",variant="TRACE-VQ",stage1_seed=42,stage2_seed=args.seed,
        stage1_unchanged=True,checkpoint=best.best_model_path,checkpoint_sha256=sha256(best.best_model_path),
        config=str(output/"effective.config.yaml"),completed_epochs=fit.current_epoch,
        best_val_loss=scalar(best.best_model_score),elapsed_seconds=time.monotonic()-start,finished=now())
    restored=ScreeningPredictor.load_from_checkpoint(best.best_model_path,config=snapshot,T_max=1,
                                                    map_location="cpu",strict=True)
    del model,fit,train_loader,training;gc.collect();torch.cuda.empty_cache()
    record["validation_metrics"]=write_prediction(restored,validation,snapshot,args.device,
                                                 output/"training_evaluation","valid",2)
    del validation,valid_loader;gc.collect()
    test,_=cached(restored,snapshot,"test",args)
    record["test_metrics"]=write_prediction(restored,test,snapshot,args.device,
                                           output/"training_evaluation","test",2)
    atomic_json(output/"training_result.json",record)


def evaluate(args):
    output=args.root/f"seed{args.seed}";initialize(args.seed)
    record=json.loads((output/"training_result.json").read_text())
    config=OmegaConf.to_container(OmegaConf.load(record["config"]),resolve=True)
    assert config["train"]["seed"]==args.seed and sha256(record["checkpoint"])==record["checkpoint_sha256"]
    model=ScreeningPredictor.load_from_checkpoint(record["checkpoint"],config=config,T_max=1,
                                                 map_location="cpu",strict=True)
    metrics={}
    for split in ("valid","test"):
        dataset,_=cached(model,config,split,args)
        metrics[split]=write_prediction(model,dataset,config,args.device,output/"evaluation",split,2)
        pd.testing.assert_frame_equal(pd.read_pickle(output/f"evaluation/{split}_predictions.pkl"),
                    pd.read_pickle(output/f"training_evaluation/{split}_predictions.pkl"),check_exact=True)
        del dataset;gc.collect()
    assert_frozen(model,config)
    atomic_json(output/"evaluation/completion.json",dict(status="complete",standalone_exact=True,
                stage1_unchanged=True,stage2_seed=args.seed,metrics=metrics))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["audit","stage1-check","stage1","cache","validate-cache","train","evaluate"])
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--seed",type=int,choices=SEEDS,default=0)
    parser.add_argument("--device",default="cuda")
    args=parser.parse_args();args.root=args.root.resolve();args.data_root=args.root/"data"
    if not args.root.is_relative_to(ROOT/"artifacts/research"):raise ValueError("Research output required")
    args.root.mkdir(parents=True,exist_ok=True)
    {"audit":audit_data,"stage1-check":stage1_check,"stage1":stage1_train,"cache":cache_all,
     "validate-cache":validate_cache,"train":stage2_train,"evaluate":evaluate}[args.command](args)


if __name__=="__main__":main()
