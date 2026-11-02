"""Isolated full-data training/evaluation for the single-seed research screen."""

import argparse
import copy
import gc
import json
import pickle
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

from trace_vq.runtime import configure_runtime
configure_runtime()

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from trace_vq.data.frozen import atomic_json, daily_loader, prepare_frozen_cache, sha256
from trace_vq.data.schema import dataset_basename
from trace_vq.evaluation.protocol_metrics import ranking_metrics
from trace_vq.models.screening import ScreeningPredictor, VARIANTS, variant_config
from trace_vq.utils import apply_artifact_root, seed_everything

ROOT = Path(__file__).resolve().parents[2]


def now():
    return datetime.now(timezone.utc).isoformat()


def stage1_checkpoint(universe):
    directory = (ROOT / "artifacts/028/run/checkpoints" if universe == "csi300" else
                 ROOT / "artifacts/csi800/s1seed42_s2seed0/stage1/checkpoints")
    paths = [path for path in directory.glob("*.ckpt") if path.name != "last.ckpt"]
    if len(paths) != 1 or "s42-" not in paths[0].name:
        raise RuntimeError(f"Expected one seed42 Stage1 checkpoint: {directory}")
    return paths[0].resolve()


def make_config(universe, variant, output):
    from trace_vq.cli.stage2 import _apply_stage2_universe_preset
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
    cfg = OmegaConf.load(ROOT / "configs/config.yaml")
    cfg.data.universe = universe
    cfg.data.data_path = str(ROOT / "dataset/processed")
    cfg.predictor.saved_model = str(stage1_checkpoint(universe))
    cfg.train.seed = 0
    cfg.train.wandb_mode = "disabled"
    cfg.artifact_root = str(Path(output).resolve())
    _apply_stage2_universe_preset(cfg, [])
    apply_artifact_root(cfg)
    result = variant_config(OmegaConf.to_container(cfg, resolve=True), variant)
    result["inference"] = {"matmul_precision": "high"}
    return result


def scalar(value):
    return float(value.detach().cpu()) if isinstance(value, torch.Tensor) else float(value)


class EpochRecord(Callback):
    def __init__(self, output):
        self.output = Path(output)
        self.start = time.monotonic()
        self.rows = []

    def on_validation_end(self, trainer, module):
        if trainer.sanity_checking:
            return
        metrics = {key: scalar(value) for key, value in trainer.callback_metrics.items()
                   if key in ("val_loss", "val_mse_loss", "Val_RIC", "Val_IC", "train_loss")}
        row = dict(epoch=trainer.current_epoch, global_step=trainer.global_step,
                   elapsed_seconds=time.monotonic() - self.start, updated=now(), **metrics)
        self.rows.append(row)
        atomic_json(self.output / "training_progress.json", {"epochs": self.rows, "latest": row})
        pd.DataFrame(self.rows).to_csv(self.output / "epoch_metrics.csv", index=False)
        print(json.dumps(row), flush=True)


def read_dataset(config, split):
    universe = config["data"]["universe"]
    path = Path(config["data"]["data_path"]) / "CN" / f"{dataset_basename(universe)}_{split}.pkl"
    with path.open("rb") as stream:
        dataset = pickle.load(stream)
    dates = dataset.get_index().get_level_values("datetime")
    begin, end = map(pd.Timestamp, config["data"][f"{split}_period"])
    if dates.min() < begin or dates.max() > end:
        raise ValueError(f"Dataset extends outside the fixed {split} period: {path}")
    return dataset, path


def cached_dataset(model, config, split, cache_root, device, workers):
    dataset, path = read_dataset(config, split)
    cache = Path(cache_root) / config["data"]["universe"] / f"{split}.npy"
    return prepare_frozen_cache(model, dataset, path, cache, config, device, workers)


@torch.no_grad()
def predict(model, dataset, config, device, workers=0):
    model.eval().to(device)
    loader = daily_loader(dataset, num_workers=workers)
    frames = []
    index = dataset.get_index()
    for positions, batch in zip(loader.sampler, loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        parts, result = model.batch_prediction(batch)
        scores = result[0].cpu().numpy().reshape(-1)
        labels = parts.target(config["predictor"]["target_day"]).cpu().numpy().reshape(-1)
        if not np.isfinite(scores).all():
            raise FloatingPointError("Non-finite inference scores")
        frames.append(pd.DataFrame({"score": scores, "label": labels}, index=index.take(positions)))
    frame = pd.concat(frames).sort_index()
    if len(frame) != len(index) or not frame.index.equals(index.sort_values()) or not frame.index.is_unique:
        raise RuntimeError("Inference lost, duplicated, or misassigned stock/date rows")
    metrics = ranking_metrics(frame.score.astype("float64"), frame.label.astype("float64"))
    return frame, metrics


def write_prediction(model, dataset, config, device, output, split, workers):
    frame, metrics = predict(model, dataset, config, device, workers)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_pickle(output / f"{split}_predictions.pkl")
    pd.Series(metrics, name="value").to_csv(output / f"{split}_metrics.csv")
    return metrics


def initialize(seed=0):
    torch.set_float32_matmul_precision("high")
    seed_everything(seed)
    pl.seed_everything(seed, workers=True)


def build_cache(args):
    initialize()
    config = make_config(args.universe, "baseline", args.output_dir)
    model = ScreeningPredictor(config, T_max=1)
    records = {}
    for split in ("train", "valid", "test"):
        dataset, metadata = cached_dataset(model, config, split, args.cache_root, args.device, args.num_workers)
        records[split] = metadata
        del dataset
        gc.collect()
    atomic_json(Path(args.cache_root) / args.universe / "completion.json", records)


def train(args):
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "training_result.json"
    if result_path.exists():
        raise FileExistsError(f"Refusing to overwrite completed training: {result_path}")
    if not (Path(args.cache_root) / args.universe / "completion.json").is_file():
        raise RuntimeError("Prepare all frozen caches in the separate cache phase before training")
    initialize()
    config = make_config(args.universe, args.variant, output)
    config["train"]["num_workers"] = args.num_workers
    model = ScreeningPredictor(config, T_max=1)
    train_data, train_metadata = cached_dataset(model, config, "train", args.cache_root, args.device, args.num_workers)
    valid_data, valid_metadata = cached_dataset(model, config, "valid", args.cache_root, args.device, args.num_workers)
    train_loader = daily_loader(train_data, shuffle=True, num_workers=args.num_workers)
    valid_loader = daily_loader(valid_data, num_workers=args.num_workers)
    model.T_max = len(train_loader) * config["train"]["num_epochs"]
    checkpoint = ModelCheckpoint(dirpath=output / "checkpoints", monitor="val_loss", mode="min",
                                 save_top_k=1, save_last=True, filename="seed0-{epoch}-{val_loss:.6f}")
    early = config["train"]["early_stopping"]
    stopper = EarlyStopping(monitor=early["monitor"], mode=early["mode"], min_delta=early["min_delta"],
                            patience=early["patience"], verbose=True)
    snapshot = copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot), output / "effective.config.yaml")
    atomic_json(output / "input_manifest.json", {"stage1_seed": 42, "stage2_seed": 0,
                "stage1_sha256": sha256(config["predictor"]["saved_model"]),
                "train": train_metadata, "valid": valid_metadata,
                "selection": "original target-day MSE plus original auxiliary loss, validation only"})
    logger = CSVLogger(save_dir=str(output), name="training", version="seed0")
    trainer = pl.Trainer(logger=logger, callbacks=[checkpoint, stopper, EpochRecord(output)],
                         max_epochs=config["train"]["num_epochs"], accelerator="gpu" if args.device.startswith("cuda") else "cpu",
                         devices=1, precision=config["train"]["precision"], deterministic=True,
                         gradient_clip_val=config["train"]["gradient_clip_val"], enable_progress_bar=False,
                         log_every_n_steps=50, default_root_dir=str(output))
    start = time.monotonic()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if not checkpoint.best_model_path:
        raise RuntimeError("No validation checkpoint was saved")
    # Stage1 is immutable for every variant and for every optimizer step.
    pretrained = torch.load(config["predictor"]["saved_model"], map_location="cpu", weights_only=False)["state_dict"]
    for name, tensor in model.state_dict().items():
        prefix = next((p for p in ("encoder.", "quantizer.", "revin.") if name.startswith(p)), None)
        if prefix:
            source_prefix = {"encoder.": "vqvae.spatial_encoder.", "quantizer.": "vqvae.quantizer.", "revin.": "vqvae.revin."}[prefix]
            if not torch.equal(tensor.detach().cpu(), pretrained[source_prefix + name[len(prefix):]]):
                raise RuntimeError(f"Frozen Stage1 tensor changed: {name}")
    del pretrained
    result = {"status": "complete", "finished": now(), "universe": args.universe, "variant": args.variant,
              "stage1_seed": 42, "stage2_seed": 0, "checkpoint": checkpoint.best_model_path,
              "checkpoint_sha256": sha256(checkpoint.best_model_path),
              "best_val_loss": scalar(checkpoint.best_model_score), "elapsed_seconds": time.monotonic() - start,
              "completed_epochs": int(trainer.current_epoch), "global_step": trainer.global_step,
              "stage1_unchanged": True, "cache_used": True, "config": str(output / "effective.config.yaml")}
    restored = ScreeningPredictor.load_from_checkpoint(checkpoint.best_model_path, config=snapshot, T_max=1,
                                                       map_location="cpu", strict=True)
    del trainer, model, train_loader, train_data
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    result["validation_metrics"] = write_prediction(restored, valid_data, snapshot, args.device,
                                                   output / "training_evaluation", "valid", args.num_workers)
    del valid_data, valid_loader
    gc.collect()
    test_data, _ = cached_dataset(restored, snapshot, "test", args.cache_root, args.device, args.num_workers)
    result["test_metrics"] = write_prediction(restored, test_data, snapshot, args.device,
                                             output / "training_evaluation", "test", args.num_workers)
    atomic_json(result_path, result)
    print(json.dumps(result, default=str), flush=True)


def evaluate(args):
    initialize()
    run = args.output_dir.resolve()
    config = OmegaConf.to_container(OmegaConf.load(run / "effective.config.yaml"), resolve=True)
    result = json.loads((run / "training_result.json").read_text())
    model = ScreeningPredictor.load_from_checkpoint(result["checkpoint"], config=config, T_max=1,
                                                    map_location="cpu", strict=True)
    report = {}
    for split in ("valid", "test"):
        dataset, _ = cached_dataset(model, config, split, args.cache_root, args.device, args.num_workers)
        report[split] = write_prediction(model, dataset, config, args.device, run / "evaluation", split, args.num_workers)
        actual = pd.read_pickle(run / "evaluation" / f"{split}_predictions.pkl")
        expected = pd.read_pickle(run / "training_evaluation" / f"{split}_predictions.pkl")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        del dataset
        gc.collect()
    atomic_json(run / "evaluation/completion.json", {"status": "complete", "standalone_exact": True, "metrics": report})


def adopt(args):
    """Reproduce an existing seed0 control with this execution path exactly."""
    initialize()
    output = args.output_dir.resolve()
    source = (args.native_run_dir or output).resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError("Baseline inputs must come from the independent research copy")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_result.json").exists():
        raise FileExistsError("Baseline training receipt already exists")
    checkpoints = [p for p in (source / "checkpoints").glob("*.ckpt") if p.name != "last.ckpt"]
    configs = list((source / "checkpoints").glob("*.config.yaml"))
    predictions = list((source / "res").rglob("0_best.pkl"))
    if not (len(checkpoints) == len(configs) == len(predictions) == 1):
        raise RuntimeError("Baseline must contain exactly one checkpoint, config, and seed0 prediction")
    config = OmegaConf.to_container(OmegaConf.load(configs[0]), resolve=True)
    if config["data"]["universe"] != args.universe or config["train"]["seed"] != 0:
        raise ValueError("Baseline universe or Stage2 seed mismatch")
    config["data"]["data_path"] = str(ROOT / "dataset/processed")
    config["predictor"]["saved_model"] = str(stage1_checkpoint(args.universe))
    config["research"] = {"variant": "baseline"}
    config["inference"] = {"matmul_precision": "high"}
    config["artifact_root"] = str(output)
    config["train"]["save_dir"] = str(output / "checkpoints")
    config["train"]["save_res"] = str(output / "res")
    checkpoint = checkpoints[0]
    if source != output:
        target = output / "checkpoints" / checkpoint.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, target)
        checkpoint = target
    OmegaConf.save(OmegaConf.create(config), output / "effective.config.yaml")
    model = ScreeningPredictor.load_from_checkpoint(checkpoint, config=config, T_max=1, map_location="cpu", strict=True)
    metrics = {}
    for split in ("valid", "test"):
        dataset, _ = cached_dataset(model, config, split, args.cache_root, args.device, args.num_workers)
        metrics[split] = write_prediction(model, dataset, config, args.device, output / "training_evaluation", split, args.num_workers)
        del dataset
        gc.collect()
    expected = pd.read_pickle(predictions[0])
    actual = pd.read_pickle(output / "training_evaluation/test_predictions.pkl")
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    callback = next(value for key, value in state.get("callbacks", {}).items() if "ModelCheckpoint" in key)
    result = {"status": "complete", "finished": now(), "universe": args.universe, "variant": "baseline",
              "stage1_seed": 42, "stage2_seed": 0, "checkpoint": str(checkpoint),
              "checkpoint_sha256": sha256(checkpoint), "best_val_loss": scalar(callback["best_model_score"]),
              "best_epoch": state["epoch"], "global_step_at_best": state["global_step"],
              "config": str(output / "effective.config.yaml"), "native_run": str(source),
              "native_prediction_sha256": sha256(predictions[0]), "native_prediction_exact": True,
              "validation_metrics": metrics["valid"], "test_metrics": metrics["test"],
              "historical_control_reused": source != output, "cache_used": True}
    atomic_json(output / "training_result.json", result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("cache", "train", "evaluate", "adopt"))
    parser.add_argument("--universe", choices=("csi300", "csi800"), required=True)
    parser.add_argument("--variant", choices=tuple(VARIANTS), default="baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--native-run-dir", type=Path)
    args = parser.parse_args()
    {"cache": build_cache, "train": train, "evaluate": evaluate, "adopt": adopt}[args.command](args)


if __name__ == "__main__":
    main()
