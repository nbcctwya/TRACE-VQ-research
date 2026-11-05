"""Execute only the prespecified Stage 2 variants from batch 01."""
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

from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.cli.next_screen import callbacks, trainer
from trace_vq.cli.screening import ROOT, EpochRecord, initialize, now, read_dataset, scalar, write_prediction
from trace_vq.data.frozen import atomic_json, daily_loader, prepare_frozen_cache, sha256, DailyPositions, float_collate
from trace_vq.models.batch01 import Batch01Predictor, VARIANTS
from trace_vq.utils import apply_artifact_root


def reference_directory(market, family):
    if family == "vq":
        return ROOT / "artifacts/research/mechanism_screen_20261028" / market / "baseline"
    return ROOT / "artifacts/research/next_directions_20261027" / market / "fsq"


def cache_directory(market, family):
    return ROOT / "artifacts/research/next_directions_20261027/cache" / ("legacy" if family == "vq" else "fsq") / market


def make_config(market, variant, output, workers=2):
    settings = VARIANTS[variant]
    if variant.startswith("tune800_") and market != "csi800":
        raise ValueError("G1 is restricted to CSI800")
    cfg = OmegaConf.load(reference_directory(market, settings["family"]) / "effective.config.yaml")
    cfg.artifact_root = str(Path(output).resolve())
    cfg.train.seed = 0
    cfg.train.num_workers = workers
    cfg.train.wandb_mode = "disabled"
    cfg.research = {"variant": variant, **copy.deepcopy(settings)}
    for key in ("n_expert", "aux_weight"):
        if key in settings:
            cfg.predictor[key] = settings[key]
    if "num_heads" in settings:
        cfg.predictor.transformer.num_heads = settings["num_heads"]
    if variant == "fsq_wo_ca":
        cfg.predictor.quantization_confidence_adapter = False
    if variant == "fsq_grid":
        cfg.predictor.stage2_latent_dim = 3
        cfg.research.grid_confidence_error = "normalized_coordinate_mse"
    apply_artifact_root(cfg)
    result = OmegaConf.to_container(cfg, resolve=True)
    assert result["train"]["num_epochs"] == 70
    assert result["train"]["early_stopping"]["patience"] == 15
    assert result["predictor"]["target_day"] == 5
    checkpoint = Path(result["predictor"]["saved_model"])
    if not checkpoint.is_relative_to(ROOT) or not checkpoint.is_file():
        raise FileNotFoundError("Existing research-copy Stage 1 required: " + str(checkpoint))
    return result


def cached(model, config, split, args):
    source, path = read_dataset(config, split)
    family = VARIANTS[args.variant]["family"]
    cache = cache_directory(args.universe, family) / f"{split}.npy"
    if not cache.exists() or not cache.with_suffix(".json").exists():
        raise FileNotFoundError("Existing frozen Stage 1 cache required: " + str(cache))
    return prepare_frozen_cache(model, source, path, cache, config, args.device, args.num_workers)


def gpu_profile(device):
    if not device.startswith("cuda"):
        return {"gpu_peak_allocated_mib": None, "gpu_peak_reserved_mib": None}
    torch.cuda.synchronize()
    return {"gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
            "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}


def start_profile(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    return time.perf_counter()


def train(args):
    initialize(0)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_result.json").exists():
        raise FileExistsError("Refusing to overwrite completed batch01 training")
    phase_start = time.perf_counter()
    config = make_config(args.universe, args.variant, output, args.num_workers)
    model = Batch01Predictor(config, T_max=1)
    training, training_metadata = cached(model, config, "train", args)
    validation, valid_metadata = cached(model, config, "valid", args)
    train_loader = daily_loader(training, shuffle=True, num_workers=args.num_workers)
    valid_loader = daily_loader(validation, num_workers=args.num_workers)
    model.T_max = len(train_loader) * config["train"]["num_epochs"]
    snapshot = copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot), output / "effective.config.yaml")
    first_hash = sha256(config["predictor"]["saved_model"])
    atomic_json(output / "input_manifest.json", {
        "stage1_seed": 42, "stage2_seed": 0, "stage1_retrained": False,
        "stage1_sha256": first_hash, "train": training_metadata, "valid": valid_metadata,
        "test_used_for_training_or_selection": False,
        "selection": "minimum original validation MSE plus original auxiliary loss",
        "existing_caches_read_only": True})
    best, stop = callbacks(config, output, 0)
    fit = trainer(config, output, [best, stop, EpochRecord(output)], args, 0)
    parameter_counts = {"total_parameters": sum(p.numel() for p in model.parameters()),
                        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    start = start_profile(args.device)
    fit.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    training_seconds = time.perf_counter() - start
    profile = gpu_profile(args.device)
    if not best.best_model_path:
        raise RuntimeError("No validation-selected checkpoint")
    assert_frozen(model, config)
    assert sha256(config["predictor"]["saved_model"]) == first_hash
    record = dict(status="trained", universe=args.universe, variant=args.variant,
                  stage1_seed=42, stage2_seed=0, stage1_retrained=False, stage1_unchanged=True,
                  checkpoint=best.best_model_path, checkpoint_sha256=sha256(best.best_model_path),
                  config=str(output / "effective.config.yaml"), best_val_loss=scalar(best.best_model_score),
                  completed_epochs=fit.current_epoch, training_seconds=training_seconds,
                  setup_seconds=start - phase_start, finished_training=now(),
                  **profile, **parameter_counts)
    atomic_json(output / "training_fit.json", record)
    del fit, model, train_loader, training
    gc.collect()
    torch.cuda.empty_cache()
    restored = Batch01Predictor.load_from_checkpoint(record["checkpoint"], config=snapshot, T_max=1,
                                                    map_location="cpu", strict=True)
    start = time.perf_counter()
    record["validation_metrics"] = write_prediction(restored, validation, snapshot, args.device,
                                                    output / "training_evaluation", "valid", args.num_workers)
    del validation, valid_loader
    gc.collect()
    test, _ = cached(restored, snapshot, "test", args)
    record["test_metrics"] = write_prediction(restored, test, snapshot, args.device,
                                              output / "training_evaluation", "test", args.num_workers)
    record.update(status="complete", training_inference_seconds=time.perf_counter() - start,
                  train_phase_seconds=time.perf_counter() - phase_start, finished=now())
    atomic_json(output / "training_result.json", record)
    print(json.dumps({k: v for k, v in record.items() if k not in {"validation_metrics", "test_metrics"}}), flush=True)


def evaluate(args):
    initialize(0)
    output = args.output_dir
    config = OmegaConf.to_container(OmegaConf.load(output / "effective.config.yaml"), resolve=True)
    record = json.loads((output / "training_result.json").read_text())
    phase_start = start_profile(args.device)
    model = Batch01Predictor.load_from_checkpoint(record["checkpoint"], config=config, T_max=1,
                                                 map_location="cpu", strict=True)
    assert_frozen(model, config)
    metrics = {}
    for split in ("valid", "test"):
        data, _ = cached(model, config, split, args)
        metrics[split] = write_prediction(model, data, config, args.device,
                                         output / "evaluation", split, args.num_workers)
        pd.testing.assert_frame_equal(pd.read_pickle(output / "evaluation" / f"{split}_predictions.pkl"),
                                      pd.read_pickle(output / "training_evaluation" / f"{split}_predictions.pkl"),
                                      check_exact=True)
        del data
        gc.collect()
    assert_frozen(model, config)
    atomic_json(output / "evaluation/completion.json", dict(status="complete", standalone_exact=True,
                metrics=metrics, evaluation_seconds=time.perf_counter() - phase_start,
                **gpu_profile(args.device)))


def preflight(args):
    """One full validation-day backward/restore, reusing each existing cache."""
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    checks, failures = [], []
    for market in ("csi300", "csi800"):
        # Load a canonical split once per market; no source file is modified.
        variants = [name for name in VARIANTS if market == "csi800" or not name.startswith("tune800_")]
        source = None
        for variant in variants:
            try:
                initialize(0)
                folder = output / market / variant
                folder.mkdir(parents=True, exist_ok=True)
                config = make_config(market, variant, folder, 0)
                model = Batch01Predictor(config, T_max=2).eval().to(args.device)
                if source is None:
                    source, source_path = read_dataset(config, "valid")
                    positions = DailyPositions(source).positions[0]
                cache = cache_directory(market, VARIANTS[variant]["family"]) / "valid.npy"
                frozen, _ = prepare_frozen_cache(model, source, source_path, cache, config, args.device, 0)
                batch = {key: value.to(args.device) for key, value in float_collate(frozen[positions]).items()}
                parts, result = model.batch_prediction(batch)
                with torch.no_grad():
                    h = model.encoder(model.revin(parts.stock_feature, mode="norm"))
                    z = model.quantizer(h)[0]
                    cache_exact = torch.equal(h, batch["frozen_h"]) and torch.equal(z, batch["frozen_zq"])
                if not cache_exact:
                    raise RuntimeError("Frozen cache differs from complete-day Stage 1 recomputation")
                initial = {k: t.clone() for k, t in model.state_dict().items()
                           if k.startswith(("encoder.", "quantizer.", "revin."))}
                model.train()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                optimizer.zero_grad(set_to_none=True)
                parts, result = model.batch_prediction(batch)
                loss, _ = model.training_loss(parts, result)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
                if variant == "gated_attn":
                    assert all(layer.head_gate.weight.grad is not None and layer.head_gate.weight.grad.abs().sum() > 0
                               for layer in model.loadings.temporal_transformer.transformer.layers)
                if variant == "fsq_ca_dim":
                    assert model.quantization_confidence_adapter.weight.grad.abs().sum() > 0
                optimizer.step()
                assert_frozen(model, config)
                assert all(torch.equal(t, model.state_dict()[k]) for k, t in initial.items())
                model.eval()
                with torch.no_grad():
                    expected = model.batch_prediction(batch)[1][0]
                saved = folder / "smoke_state.pt"
                torch.save(model.state_dict(), saved)
                restored = Batch01Predictor(copy.deepcopy(model.config), T_max=2).eval().to(args.device)
                restored.load_state_dict(torch.load(saved, map_location=args.device, weights_only=True), strict=True)
                with torch.no_grad():
                    actual = restored.batch_prediction(batch)[1][0]
                assert torch.equal(actual, expected)
                assert result[2].shape[-1] == (3 if variant == "fsq_grid" else 128)
                OmegaConf.save(OmegaConf.create(model.config), folder / "effective.config.yaml")
                row = dict(universe=market, variant=variant, status="passed", full_day_stocks=len(positions),
                           cache_exact=True, gradients_finite=True, frozen_stage1_exact=True,
                           strict_restore_exact=True, latent_dimension=result[3].shape[-1], loss=float(loss.detach()))
                checks.append(row)
                print(json.dumps(row), flush=True)
                del model, restored, optimizer, batch, result, frozen, initial
                gc.collect()
                torch.cuda.empty_cache()
            except Exception as error:
                import traceback
                traceback.print_exc()
                failures.append(dict(universe=market, variant=variant, error=str(error)))
            atomic_json(output / "gpu_validation.json", dict(status="passed" if not failures else "failed",
                        checks=checks, failures=failures))
        del source
        gc.collect()
    if failures:
        raise RuntimeError("Batch01 preflight failures: " + str(failures))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "preflight"))
    parser.add_argument("--universe", choices=("csi300", "csi800"))
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT / "artifacts/research"):
        raise ValueError("An explicit output under artifacts/research is required")
    if args.command != "preflight" and (not args.universe or not args.variant):
        parser.error("--universe and --variant are required")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
