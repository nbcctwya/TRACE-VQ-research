"""Isolated training, exact reference replay, and controls for screen three."""
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

from trace_vq.cli.screening import (ROOT, EpochRecord, initialize, make_config as base_config,
                                    now, read_dataset, scalar, write_prediction)
from trace_vq.cli.next_screen import callbacks, trainer
from trace_vq.data.frozen import atomic_json, daily_loader, prepare_frozen_cache, sha256
from trace_vq.data.mechanism_context import RepresentationHistory, StateBalancedDays, fit_state_balance, state_groups
from trace_vq.data.research_context import PairedDays
from trace_vq.models.mechanism_screen import MECHANISM_VARIANTS, MechanismPredictor


def make_config(args):
    config = base_config(args.universe, "baseline", args.output_dir)
    config["research"] = dict(variant=args.variant, **copy.deepcopy(MECHANISM_VARIANTS[args.variant]))
    config["train"]["num_workers"] = args.num_workers
    if config["research"].get("state_balanced"):
        settings = json.loads((args.context_root / "states" / f"{args.universe}.json").read_text())
        if settings["fit_splits"] != ["train"] or settings["test_used_for_fitting"]:
            raise RuntimeError("State thresholds must use training dates only")
        config["research"]["state_settings"] = settings
    return config


def cached(model, config, split, args):
    source, path = read_dataset(config, split)
    target = args.cache_root / args.universe / f"{split}.npy"
    if not target.is_file() or not target.with_suffix(".json").is_file():
        raise RuntimeError("Explicitly copy and audit the frozen native cache before training")
    dataset, metadata = prepare_frozen_cache(model, source, path, target, config, args.device, args.num_workers)
    settings = config["research"]
    if settings.get("history"):
        dataset = RepresentationHistory(dataset, args.context_root / "history", args.cache_root,
                                        args.universe, split, feature=settings.get("history_feature", "quantized"),
                                        mismatch=settings.get("history_mismatch", False))
    elif settings.get("paired_days") and split == "train":
        dataset = PairedDays(dataset)
    elif settings.get("state_balanced"):
        dataset = StateBalancedDays(dataset, settings["state_settings"])
    return dataset, metadata


def prepare_state(args):
    # This command never instantiates a predictor and only FITS training dates.
    config = base_config(args.universe, "baseline", args.output_dir)
    source, path = read_dataset(config, "train")
    dates = source.get_index().get_level_values("datetime").unique().sort_values()
    frame = source.market_frame
    training = frame.loc[dates]
    settings = fit_state_balance(training.iloc[:, 9].to_numpy(), training.iloc[:, 10].to_numpy())
    settings.update(universe=args.universe, training_source=str(path.resolve()),
                    training_source_sha256=sha256(path), fit_start=str(dates.min()), fit_end=str(dates.max()),
                    test_used_for_fitting=False, features=frame.columns[9:11].map(str).tolist())
    if not (dates.min() >= pd.Timestamp("2009-01-01") and dates.max() <= pd.Timestamp("2020-12-31")):
        raise RuntimeError("State fitting extends outside the fixed training period")
    output = args.context_root / "states"
    output.mkdir(parents=True, exist_ok=True)
    metadata = output / f"{args.universe}.json"
    calendar = output / f"{args.universe}_calendar.csv"
    selected = frame.loc[(frame.index >= pd.Timestamp("2009-01-01")) &
                         (frame.index <= pd.Timestamp("2025-12-31"))].iloc[:, [9, 10]].copy()
    selected.columns = ["trend20", "volatility20"]
    selected["group"] = state_groups(selected.trend20, selected.volatility20, settings["thresholds"])
    settings["calendar_role"] = "Held-out covariates assigned by frozen training thresholds for reporting; not threshold fitting."
    if metadata.exists():
        old = json.loads(metadata.read_text())
        if old != settings:
            raise RuntimeError("Frozen training-state settings changed")
        pd.testing.assert_frame_equal(pd.read_csv(calendar, index_col=0, parse_dates=True),
                                      selected, check_exact=False, check_freq=False, check_dtype=False)
        return
    temporary = calendar.with_suffix(".partial.csv")
    selected.to_csv(temporary)
    temporary.replace(calendar)
    atomic_json(metadata, settings)
    print(json.dumps(settings), flush=True)


def assert_frozen(model, config):
    first = torch.load(config["predictor"]["saved_model"], map_location="cpu", weights_only=False)["state_dict"]
    for name, tensor in model.state_dict().items():
        for prefix, original in [("encoder.", "vqvae.spatial_encoder."),
                                 ("quantizer.", "vqvae.quantizer."), ("revin.", "vqvae.revin.")]:
            if name.startswith(prefix) and not torch.equal(tensor.cpu(), first[original + name[len(prefix):]]):
                raise RuntimeError("Frozen Stage1 changed: " + name)


def train(args):
    initialize(0)
    output = args.output_dir
    if (output / "training_result.json").exists():
        raise FileExistsError("Refusing to overwrite a completed Stage2 fit")
    config = make_config(args)
    model = MechanismPredictor(config, T_max=1)
    train_data, training_metadata = cached(model, config, "train", args)
    valid_data, valid_metadata = cached(model, config, "valid", args)
    train_loader = daily_loader(train_data, shuffle=True, num_workers=args.num_workers)
    valid_loader = daily_loader(valid_data, num_workers=args.num_workers)
    model.T_max = len(train_loader) * config["train"]["num_epochs"]
    best, stop = callbacks(config, output, 0)
    snapshot = copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot), output / "effective.config.yaml")
    first_hash = sha256(config["predictor"]["saved_model"])
    atomic_json(output / "input_manifest.json", {"stage1_seed": 42, "stage2_seed": 0,
                "stage1_sha256": first_hash, "train": training_metadata, "valid": valid_metadata,
                "test_used_for_training_or_selection": False,
                "selection": "minimum original validation MSE plus original auxiliary loss"})
    fit = trainer(config, output, [best, stop, EpochRecord(output)], args, 0)
    start = time.monotonic()
    fit.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if not best.best_model_path:
        raise RuntimeError("No selected validation checkpoint")
    assert_frozen(model, config)
    if sha256(config["predictor"]["saved_model"]) != first_hash:
        raise RuntimeError("Stage1 checkpoint file changed")
    record = {"status": "complete", "universe": args.universe, "variant": args.variant,
              "stage1_seed": 42, "stage2_seed": 0, "stage1_retrained": False,
              "checkpoint": best.best_model_path, "checkpoint_sha256": sha256(best.best_model_path),
              "best_val_loss": scalar(best.best_model_score), "completed_epochs": fit.current_epoch,
              "elapsed_seconds": time.monotonic() - start, "stage1_unchanged": True,
              "config": str(output / "effective.config.yaml"), "finished": now()}
    restored = MechanismPredictor.load_from_checkpoint(best.best_model_path, config=snapshot, T_max=1,
                                                       map_location="cpu", strict=True)
    del model, fit, train_loader, train_data
    gc.collect()
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()
    record["validation_metrics"] = write_prediction(restored, valid_data, snapshot, args.device,
                                                     output / "training_evaluation", "valid", args.num_workers)
    del valid_data, valid_loader
    gc.collect()
    test_data, _ = cached(restored, snapshot, "test", args)
    record["test_metrics"] = write_prediction(restored, test_data, snapshot, args.device,
                                               output / "training_evaluation", "test", args.num_workers)
    atomic_json(output / "training_result.json", record)


def evaluate(args):
    initialize(0)
    output = args.output_dir
    config = OmegaConf.to_container(OmegaConf.load(output / "effective.config.yaml"), resolve=True)
    record = json.loads((output / "training_result.json").read_text())
    model = MechanismPredictor.load_from_checkpoint(record["checkpoint"], config=config, T_max=1,
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
    atomic_json(output / "evaluation/completion.json", {"status": "complete", "standalone_exact": True, "metrics": metrics})


def adopt(args):
    initialize(0)
    output = args.output_dir
    if (output / "training_result.json").exists():
        raise FileExistsError("Refusing to overwrite a reference replay")
    config = make_config(args)
    source = args.reference_dir.resolve()
    if not source.is_relative_to(ROOT / "artifacts"):
        raise ValueError("Reference must be inside this research copy")
    previous = json.loads((source / "completion.json").read_text())
    if sha256(previous["checkpoint"]) != previous["checkpoint_sha256"]:
        raise RuntimeError("Reference checkpoint changed")
    model = MechanismPredictor.load_from_checkpoint(previous["checkpoint"], config=config, T_max=1,
                                                     map_location="cpu", strict=True)
    assert_frozen(model, config)
    metrics = {}
    for split in ("valid", "test"):
        dataset, _ = cached(model, config, split, args)
        metrics[split] = write_prediction(model, dataset, config, args.device,
                                           output / "training_evaluation", split, args.num_workers)
        pd.testing.assert_frame_equal(pd.read_pickle(output / "training_evaluation" / f"{split}_predictions.pkl"),
                                      pd.read_pickle(source / "evaluation" / f"{split}_predictions.pkl"), check_exact=True)
        del dataset
        gc.collect()
    OmegaConf.save(OmegaConf.create(config), output / "effective.config.yaml")
    previous.update(status="complete", universe=args.universe, variant=args.variant,
                    config=str(output / "effective.config.yaml"), reference_source=str(source),
                    reference_predictions_exact=True, validation_metrics=metrics["valid"], test_metrics=metrics["test"],
                    stage1_retrained=False, stage1_unchanged=True, stage1_seed=42, stage2_seed=0)
    atomic_json(output / "training_result.json", previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-state", "train", "evaluate", "adopt"])
    parser.add_argument("--universe", choices=["csi300", "csi800"], required=True)
    parser.add_argument("--variant", choices=MECHANISM_VARIANTS, default="baseline")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    for name in ("output_dir", "cache_root", "context_root"):
        path = getattr(args, name).resolve()
        if not path.is_relative_to(ROOT / "artifacts/research"):
            raise ValueError("All experiment paths must remain inside research artifacts")
        setattr(args, name, path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    {"prepare-state": prepare_state, "train": train, "evaluate": evaluate, "adopt": adopt}[args.command](args)


if __name__ == "__main__":
    main()
