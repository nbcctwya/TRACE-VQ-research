"""Repeat frozen research configurations with explicit Stage2 seeds."""
import argparse
import copy
import gc
from pathlib import Path
import json
import time

from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
import torch
from omegaconf import OmegaConf

from trace_vq.cli import combination_screen as single
from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.cli.next_screen import callbacks, trainer
from trace_vq.cli.screening import ROOT, EpochRecord, initialize, now, scalar, write_prediction
from trace_vq.data.frozen import atomic_json, daily_loader, sha256
from trace_vq.models.next_screen import NextPredictor

VARIANTS = ("fsq_ranknet", "fsq", "ranknet")
SEEDS = (1, 2, 3, 4)


def tasks():
    """Finish both datasets/all seeds of the combination before either single."""
    return [(variant, market, seed) for variant in VARIANTS for seed in SEEDS
            for market in ("csi300", "csi800")]


def make_config(args):
    if args.variant not in VARIANTS or args.seed not in range(5):
        raise ValueError("Only frozen variants and prespecified Stage2 seeds0-4 are allowed")
    config = single.make_config(args)
    config["train"]["seed"] = args.seed
    return config


def train(args):
    output = args.output_dir
    if (output / "training_result.json").exists():
        raise FileExistsError("Completed training cannot be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    initialize(args.seed)
    config = make_config(args)
    model = NextPredictor(config, T_max=1)
    first_hash = sha256(config["predictor"]["saved_model"])
    training, train_metadata = single.cached(model, config, "train", args)
    validation, valid_metadata = single.cached(model, config, "valid", args)
    train_loader = daily_loader(training, shuffle=True, num_workers=args.num_workers)
    valid_loader = daily_loader(validation, num_workers=args.num_workers)
    model.T_max = len(train_loader) * config["train"]["num_epochs"]
    snapshot = copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot), output / "effective.config.yaml")
    atomic_json(output / "input_manifest.json", dict(stage1_seed=42, stage2_seed=args.seed,
        stage1_retrained=False, stage1_sha256=first_hash, train=train_metadata, valid=valid_metadata,
        test_used_for_training_or_selection=False,
        selection="minimum original target-day validation MSE plus native auxiliary loss",
        existing_cache_root=str(single.cache_directory(args.universe, args.variant))))
    best, stop = callbacks(config, output, args.seed)
    fit = trainer(config, output, [best, stop, EpochRecord(output)], args, args.seed)
    start = time.monotonic()
    fit.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if not best.best_model_path:
        raise RuntimeError("No validation-selected checkpoint")
    assert_frozen(model, config)
    assert sha256(config["predictor"]["saved_model"]) == first_hash
    record = dict(status="complete", universe=args.universe, variant=args.variant,
        stage1_seed=42, stage2_seed=args.seed, stage1_retrained=False, stage1_unchanged=True,
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
    test, _ = single.cached(restored, snapshot, "test", args)
    record["test_metrics"] = write_prediction(restored, test, snapshot, args.device,
                                               output / "training_evaluation", "test", args.num_workers)
    atomic_json(output / "training_result.json", record)


def evaluate(args):
    initialize(args.seed)
    output = args.output_dir
    config = OmegaConf.to_container(OmegaConf.load(output / "effective.config.yaml"), resolve=True)
    record = json.loads((output / "training_result.json").read_text())
    assert config["train"]["seed"] == args.seed == record["stage2_seed"]
    model = NextPredictor.load_from_checkpoint(record["checkpoint"], config=config, T_max=1,
                                                map_location="cpu", strict=True)
    assert_frozen(model, config)
    metrics = {}
    for split in ("valid", "test"):
        data, _ = single.cached(model, config, split, args)
        metrics[split] = write_prediction(model, data, config, args.device, output / "evaluation", split, args.num_workers)
        pd.testing.assert_frame_equal(pd.read_pickle(output / "evaluation" / f"{split}_predictions.pkl"),
            pd.read_pickle(output / "training_evaluation" / f"{split}_predictions.pkl"), check_exact=True)
        del data
        gc.collect()
    assert_frozen(model, config)
    atomic_json(output / "evaluation/completion.json", dict(status="complete", stage2_seed=args.seed,
                                                            standalone_exact=True, metrics=metrics))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--universe", choices=["csi300", "csi800"], required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT / "artifacts/research"):
        raise ValueError("Explicit research output is required")
    {"train": train, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
