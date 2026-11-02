"""Train combinations using existing seed42 Stage1 weights and read-only caches."""
import argparse
import copy
import gc
import json
from pathlib import Path
import time

from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
import torch
from omegaconf import OmegaConf

from trace_vq.cli.screening import ROOT, EpochRecord, initialize, now, read_dataset, scalar, write_prediction
from trace_vq.cli.next_screen import callbacks, trainer
from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.data.frozen import atomic_json, daily_loader, prepare_frozen_cache, sha256
from trace_vq.models.combination_screen import REFERENCES, COMBINATIONS, family, settings
from trace_vq.models.next_screen import NextPredictor
from trace_vq.utils import apply_artifact_root

PREVIOUS = ROOT / "artifacts/research/next_directions_20261027"
MECHANISMS = ROOT / "artifacts/research/mechanism_screen_20261028"


def reference_directory(market, variant):
    if variant == "baseline":
        return MECHANISMS / market / variant
    if variant in ("market_route", "pairwise_loss"):
        return ROOT / "artifacts/research/innovation_screen_20261026" / market / variant
    return PREVIOUS / market / variant


def cache_directory(market, variant):
    return (PREVIOUS / "cache/fsq" if family(variant) == "fsq" else MECHANISMS / "cache") / market


def make_config(args):
    base = "fsq" if family(args.variant) == "fsq" else "baseline"
    config = OmegaConf.load(reference_directory(args.universe, base) / "effective.config.yaml")
    config.artifact_root = str(args.output_dir.resolve())
    apply_artifact_root(config)
    config.train.seed = 0
    config.train.num_workers = args.num_workers
    config.train.wandb_mode = "disabled"
    result = OmegaConf.to_container(config, resolve=True)
    result["research"] = settings(args.variant)
    assert result["train"]["num_epochs"] == 70 and result["predictor"]["target_day"] == 5
    assert result["train"]["early_stopping"]["patience"] == 15
    assert Path(result["predictor"]["saved_model"]).resolve().is_relative_to(ROOT)
    return result


def cached(model, config, split, args):
    source, path = read_dataset(config, split)
    cache = cache_directory(args.universe, args.variant) / (split + ".npy")
    if not cache.is_file() or not cache.with_suffix(".json").is_file():
        raise RuntimeError("Existing complete Stage1 cache is required; no implicit rebuild")
    return prepare_frozen_cache(model, source, path, cache, config, args.device, args.num_workers)


def train(args):
    output = args.output_dir
    if (output / "training_result.json").exists():
        raise FileExistsError("Refusing to overwrite completed combination training")
    output.mkdir(parents=True, exist_ok=True)
    initialize(0)
    config = make_config(args)
    model = NextPredictor(config, T_max=1)
    first_hash = sha256(config["predictor"]["saved_model"])
    training, train_metadata = cached(model, config, "train", args)
    validation, valid_metadata = cached(model, config, "valid", args)
    train_loader = daily_loader(training, shuffle=True, num_workers=args.num_workers)
    valid_loader = daily_loader(validation, num_workers=args.num_workers)
    model.T_max = len(train_loader) * config["train"]["num_epochs"]
    snapshot = copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot), output / "effective.config.yaml")
    atomic_json(output / "input_manifest.json", {
        "stage1_seed": 42, "stage2_seed": 0, "stage1_retrained": False,
        "stage1_sha256": first_hash, "train": train_metadata, "valid": valid_metadata,
        "test_used_for_training_or_selection": False,
        "selection": "minimum original target-day validation MSE plus native auxiliary loss",
        "existing_cache_root": str(cache_directory(args.universe, args.variant))})
    best, stop = callbacks(config, output, 0)
    fit = trainer(config, output, [best, stop, EpochRecord(output)], args, 0)
    start = time.monotonic()
    fit.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if not best.best_model_path:
        raise RuntimeError("No validation-selected checkpoint")
    assert_frozen(model, config)
    assert sha256(config["predictor"]["saved_model"]) == first_hash
    record = dict(status="complete", universe=args.universe, variant=args.variant,
                  stage1_seed=42, stage2_seed=0, stage1_retrained=False, stage1_unchanged=True,
                  checkpoint=best.best_model_path, checkpoint_sha256=sha256(best.best_model_path),
                  best_val_loss=scalar(best.best_model_score), completed_epochs=fit.current_epoch,
                  elapsed_seconds=time.monotonic() - start, config=str(output / "effective.config.yaml"), finished=now())
    restored = NextPredictor.load_from_checkpoint(best.best_model_path, config=snapshot, T_max=1,
                                                  map_location="cpu", strict=True)
    del fit, model, train_loader, training
    gc.collect(); torch.cuda.empty_cache()
    record["validation_metrics"] = write_prediction(restored, validation, snapshot, args.device,
                                                    output / "training_evaluation", "valid", args.num_workers)
    del validation, valid_loader
    gc.collect()
    test, _ = cached(restored, snapshot, "test", args)
    record["test_metrics"] = write_prediction(restored, test, snapshot, args.device,
                                              output / "training_evaluation", "test", args.num_workers)
    atomic_json(output / "training_result.json", record)


def evaluate(args):
    initialize(0)
    output = args.output_dir
    config = OmegaConf.to_container(OmegaConf.load(output / "effective.config.yaml"), resolve=True)
    record = json.loads((output / "training_result.json").read_text())
    model = NextPredictor.load_from_checkpoint(record["checkpoint"], config=config, T_max=1,
                                               map_location="cpu", strict=True)
    assert_frozen(model, config)
    metrics = {}
    for split in ("valid", "test"):
        dataset, _ = cached(model, config, split, args)
        metrics[split] = write_prediction(model, dataset, config, args.device, output / "evaluation", split, args.num_workers)
        actual = pd.read_pickle(output / "evaluation" / f"{split}_predictions.pkl")
        expected = pd.read_pickle(output / "training_evaluation" / f"{split}_predictions.pkl")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        del dataset
        gc.collect()
    assert_frozen(model, config)
    atomic_json(output / "evaluation/completion.json", dict(status="complete", standalone_exact=True, metrics=metrics))


def adopt(args):
    initialize(0)
    output = args.output_dir
    if (output / "training_result.json").exists():
        raise FileExistsError("Refusing to overwrite reference replay")
    output.mkdir(parents=True, exist_ok=True)
    config = make_config(args)
    source = reference_directory(args.universe, args.variant)
    record = json.loads((source / "completion.json").read_text())
    model = NextPredictor.load_from_checkpoint(record["checkpoint"], config=config, T_max=1,
                                               map_location="cpu", strict=True)
    assert_frozen(model, config)
    metrics = {}
    for split in ("valid", "test"):
        dataset, _ = cached(model, config, split, args)
        metrics[split] = write_prediction(model, dataset, config, args.device, output / "training_evaluation", split, args.num_workers)
        pd.testing.assert_frame_equal(pd.read_pickle(output / "training_evaluation" / f"{split}_predictions.pkl"),
                                      pd.read_pickle(source / "evaluation" / f"{split}_predictions.pkl"), check_exact=True)
        del dataset
        gc.collect()
    OmegaConf.save(OmegaConf.create(config), output / "effective.config.yaml")
    record.update(status="complete", universe=args.universe, variant=args.variant,
                  config=str(output / "effective.config.yaml"), stage1_retrained=False,
                  reference_source=str(source), reference_predictions_exact=True,
                  validation_metrics=metrics["valid"], test_metrics=metrics["test"])
    atomic_json(output / "training_result.json", record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate", "adopt"])
    parser.add_argument("--universe", choices=["csi300", "csi800"], required=True)
    parser.add_argument("--variant", choices=[*REFERENCES, *COMBINATIONS], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT / "artifacts/research"):
        raise ValueError("Explicit research artifact output required")
    {"train": train, "evaluate": evaluate, "adopt": adopt}[args.command](args)


if __name__ == "__main__":
    main()
