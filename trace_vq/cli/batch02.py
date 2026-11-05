"""Checklist 09–10: existing paired-day implementation, VQ repeats and FSQ."""
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

from trace_vq.cli.batch01 import gpu_profile, start_profile
from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.cli.next_screen import callbacks, trainer
from trace_vq.cli.screening import ROOT, initialize, now, read_dataset, scalar, write_prediction
from trace_vq.data.frozen import atomic_json, daily_loader, prepare_frozen_cache, sha256, float_collate
from trace_vq.data.research_context import PairedDays
from trace_vq.models.next_screen import NEXT_VARIANTS, NextPredictor
from trace_vq.training.research_run import RecoverableEpochRecord
from trace_vq.utils import apply_artifact_root

PREVIOUS = ROOT / "artifacts/research/next_directions_20261027"
VARIANTS = ("turnover_control", "turnover", "fsq_turnover_control", "fsq_turnover")


def family(variant):
    if variant not in VARIANTS:
        raise ValueError(variant)
    return "fsq" if variant.startswith("fsq_") else "vq"


def reference_directory(market, variant):
    return PREVIOUS / market / variant


def cache_directory(market, variant):
    return PREVIOUS / "cache" / ("fsq" if family(variant) == "fsq" else "legacy") / market


def settings(variant):
    result = copy.deepcopy(NEXT_VARIANTS["turnover_control" if variant.endswith("control") else "turnover"])
    if family(variant) == "fsq":
        result.update(copy.deepcopy(NEXT_VARIANTS["fsq"]))
    return dict(variant=variant, **result)


def make_config(market, variant, seed, output, workers=2):
    base = "fsq" if family(variant) == "fsq" else variant
    cfg = OmegaConf.load(reference_directory(market, base) / "effective.config.yaml")
    cfg.artifact_root = str(Path(output).resolve())
    cfg.train.seed = seed
    cfg.train.num_workers = workers
    cfg.train.wandb_mode = "disabled"
    cfg.train.run_name = f"batch02_{market}_{variant}_s{seed}"
    cfg.research = settings(variant)
    apply_artifact_root(cfg)
    config = OmegaConf.to_container(cfg, resolve=True)
    assert config["train"]["num_epochs"] == 70
    assert config["train"]["early_stopping"] == dict(monitor="val_loss", min_delta=1e-5,
                                                     patience=15, verbose=True, mode="min")
    assert config["research"]["pairwise_weight"] == 0.1 and config["research"]["paired_days"]
    assert config["research"]["turnover_weight"] == (0 if variant.endswith("control") else .05)
    checkpoint = Path(config["predictor"]["saved_model"])
    assert checkpoint.is_file() and checkpoint.resolve().is_relative_to(ROOT)
    return config


def cached(model, config, split, args, paired=True):
    source, path = read_dataset(config, split)
    target = cache_directory(args.universe, args.variant) / f"{split}.npy"
    if not target.is_file() or not target.with_suffix(".json").is_file():
        raise FileNotFoundError("Existing complete Stage1 cache required: " + str(target))
    dataset, metadata = prepare_frozen_cache(model, source, path, target, config, args.device, args.num_workers)
    return (PairedDays(dataset) if paired and split == "train" else dataset), metadata


def train(args):
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / "training_result.json").exists():
        raise FileExistsError("Completed training must be reused")
    initialize(args.seed)
    phase_start = time.perf_counter()
    if (output / "training_fit.json").exists():
        record = json.loads((output / "training_fit.json").read_text())
        snapshot = OmegaConf.to_container(OmegaConf.load(output / "effective.config.yaml"), resolve=True)
        assert (record["universe"], record["variant"], record["stage2_seed"]) == (args.universe, args.variant, args.seed)
        assert sha256(record["checkpoint"]) == record["checkpoint_sha256"]
        record["inference_recovered_after_completed_fit"] = True
        finish_training(record, snapshot, args, phase_start)
        return
    config = make_config(args.universe, args.variant, args.seed, output, args.num_workers)
    model = NextPredictor(config, T_max=1)
    training, train_metadata = cached(model, config, "train", args)
    validation, valid_metadata = cached(model, config, "valid", args)
    train_loader = daily_loader(training, shuffle=True, num_workers=args.num_workers)
    valid_loader = daily_loader(validation, num_workers=args.num_workers)
    model.T_max = len(train_loader) * config["train"]["num_epochs"]
    snapshot = copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot), output / "effective.config.yaml")
    first_hash = sha256(config["predictor"]["saved_model"])
    atomic_json(output / "input_manifest.json", dict(
        stage1_seed=42, stage2_seed=args.seed, stage1_retrained=False, stage1_sha256=first_hash,
        train=train_metadata, valid=valid_metadata, existing_caches_read_only=True,
        test_used_for_training_or_selection=False,
        selection="minimum original validation MSE plus auxiliary; pairwise/turnover penalties training only",
        alignment="complete current day and immediately preceding training-market day, same instrument",
        zero_penalty_control="executes the identical second forward and standardized penalty computation"))
    best, stop = callbacks(config, output, args.seed)
    recorder = RecoverableEpochRecord(output, args.device)
    fit = trainer(config, output, [best, stop, recorder], args, args.seed)
    last = output / "checkpoints/last.ckpt"
    if last.exists() and not args.resume:
        raise RuntimeError("Incomplete run exists; inspect or pass --resume with audited RNG checkpoint")
    if args.resume and last.exists():
        checkpoint = torch.load(last, map_location="cpu", weights_only=False)
        if recorder.state_key not in checkpoint.get("callbacks", {}):
            raise RuntimeError("Incomplete checkpoint has no complete RNG state; archive rather than silently resume")
        del checkpoint
    counts = dict(total_parameters=sum(p.numel() for p in model.parameters()),
                  trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    start = start_profile(args.device)
    recorder.fit_start = start
    fit.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader,
            ckpt_path=str(last) if args.resume and last.exists() else None)
    elapsed = recorder.previous_fit_seconds + time.perf_counter() - start
    profile = gpu_profile(args.device)
    profile["gpu_peak_allocated_mib"] = max(profile["gpu_peak_allocated_mib"], recorder.previous_peak_allocated)
    profile["gpu_peak_reserved_mib"] = max(profile["gpu_peak_reserved_mib"], recorder.previous_peak_reserved)
    assert best.best_model_path
    assert_frozen(model, config)
    assert sha256(config["predictor"]["saved_model"]) == first_hash
    record = dict(status="trained", universe=args.universe, variant=args.variant, stage1_seed=42,
                  stage2_seed=args.seed, stage1_retrained=False, stage1_unchanged=True,
                  checkpoint=best.best_model_path, checkpoint_sha256=sha256(best.best_model_path),
                  config=str(output / "effective.config.yaml"), best_val_loss=scalar(best.best_model_score),
                  completed_epochs=fit.current_epoch, training_seconds=elapsed,
                  setup_seconds=start-phase_start, resumed_from=str(last) if args.resume and last.exists() else None,
                  finished_training=now(), **profile, **counts)
    atomic_json(output / "training_fit.json", record)
    del fit, model, training, train_loader, validation, valid_loader
    gc.collect()
    torch.cuda.empty_cache()
    finish_training(record, snapshot, args, phase_start)


def finish_training(record, snapshot, args, phase_start):
    output = args.output_dir
    start = time.perf_counter()
    restored = NextPredictor.load_from_checkpoint(record["checkpoint"], config=snapshot, T_max=1,
                                                  map_location="cpu", strict=True)
    assert_frozen(restored, snapshot)
    validation, _ = cached(restored, snapshot, "valid", args)
    record["validation_metrics"] = write_prediction(restored, validation, snapshot, args.device,
                                                     output / "training_evaluation", "valid", args.num_workers)
    del validation
    gc.collect()
    test, _ = cached(restored, snapshot, "test", args)
    record["test_metrics"] = write_prediction(restored, test, snapshot, args.device,
                                               output / "training_evaluation", "test", args.num_workers)
    record.update(status="complete", training_inference_seconds=time.perf_counter()-start,
                  train_phase_seconds=time.perf_counter()-phase_start, finished=now())
    atomic_json(output / "training_result.json", record)
    print(json.dumps(record), flush=True)


def evaluate(args):
    initialize(args.seed)
    output = args.output_dir
    config = OmegaConf.to_container(OmegaConf.load(output / "effective.config.yaml"), resolve=True)
    record = json.loads((output / "training_result.json").read_text())
    start = start_profile(args.device)
    model = NextPredictor.load_from_checkpoint(record["checkpoint"], config=config, T_max=1,
                                              map_location="cpu", strict=True)
    metrics = {}
    for split in ("valid", "test"):
        dataset, _ = cached(model, config, split, args)
        metrics[split] = write_prediction(model, dataset, config, args.device,
                                           output / "evaluation", split, args.num_workers)
        pd.testing.assert_frame_equal(pd.read_pickle(output / "evaluation" / f"{split}_predictions.pkl"),
                                      pd.read_pickle(output / "training_evaluation" / f"{split}_predictions.pkl"),
                                      check_exact=True)
        del dataset
        gc.collect()
    assert_frozen(model, config)
    atomic_json(output / "evaluation/completion.json", dict(status="complete", standalone_exact=True,
                metrics=metrics, evaluation_seconds=time.perf_counter()-start, **gpu_profile(args.device)))


def audit_reference(args):
    """Replay existing VQ seed0 scores exactly; keep old outputs read-only."""
    assert args.seed == 0 and family(args.variant) == "vq"
    initialize(0)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source = reference_directory(args.universe, args.variant)
    old = json.loads((source / "training_result.json").read_text())
    assert sha256(old["checkpoint"]) == old["checkpoint_sha256"]
    config = make_config(args.universe, args.variant, 0, output, args.num_workers)
    OmegaConf.save(OmegaConf.create(config), output / "effective.config.yaml")
    start = start_profile(args.device)
    model = NextPredictor.load_from_checkpoint(old["checkpoint"], config=config, T_max=1,
                                              map_location="cpu", strict=True)
    for split in ("valid", "test"):
        data, _ = cached(model, config, split, args)
        write_prediction(model, data, config, args.device, output / "evaluation", split, args.num_workers)
        pd.testing.assert_frame_equal(pd.read_pickle(output / "evaluation" / f"{split}_predictions.pkl"),
                                      pd.read_pickle(source / "evaluation" / f"{split}_predictions.pkl"),
                                      check_exact=True)
        del data
        gc.collect()
    assert_frozen(model, config)
    atomic_json(output / "reference_audit.json", dict(status="passed", reference_predictions_exact=True,
                reference_source=str(source), checkpoint_sha256=old["checkpoint_sha256"],
                config=str(output / "effective.config.yaml"), seconds=time.perf_counter()-start,
                gpu_profile_role="current reference replay only; not historical training peak",
                **gpu_profile(args.device)))


def preflight(args):
    """Four paired objectives per market; full cross-section, finite backward."""
    checks, failures = [], []
    for market in ("csi300", "csi800"):
        for variant in VARIANTS:
            model = data = batch = optimizer = None
            try:
                initialize(0)
                args.universe, args.variant = market, variant
                folder = args.output_dir / market / variant
                folder.mkdir(parents=True, exist_ok=True)
                config = make_config(market, variant, 0, folder, 0)
                model = NextPredictor(config, T_max=2).to(args.device)
                data, _ = cached(model, config, "train", args)
                day = 1
                positions = data.days[day]
                raw = data[positions]
                index = data.get_index()
                previous = data.days[day-1]
                assert (index[positions[raw["align_current"]]].get_level_values("instrument") ==
                        index[previous[raw["align_previous"]]].get_level_values("instrument")).all()
                assert index[previous].get_level_values("datetime").max() < index[positions].get_level_values("datetime").min()
                batch = {k:v.to(args.device) for k,v in float_collate(raw).items()}
                model.eval()
                with torch.no_grad():
                    parts, result = model.batch_prediction(batch)
                    h = model.encoder(model.revin(parts.stock_feature, mode="norm"))
                    assert torch.equal(h, batch["frozen_h"])
                    assert torch.equal(model.quantizer(h)[0], batch["frozen_zq"])
                model.train()
                model.log = lambda *a, **kw: None
                model.log_dict = lambda *a, **kw: None
                loss = model.training_step(batch, 0)
                loss.backward()
                assert torch.isfinite(loss)
                assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters() if p.requires_grad)
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                optimizer.step()
                assert_frozen(model, config)
                model.eval()
                with torch.no_grad():
                    expected = model.batch_prediction(batch)[1][0]
                restored = NextPredictor(copy.deepcopy(model.config), T_max=2).to(args.device).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    assert torch.equal(expected, restored.batch_prediction(batch)[1][0])
                row = dict(universe=market, variant=variant, full_day_stocks=len(positions),
                           aligned_stocks=len(raw["align_current"]), status="passed", loss=float(loss.detach()),
                           cache_exact=True, frozen_stage1_exact=True, finite_gradients=True, strict_restore_exact=True)
                checks.append(row)
                print(json.dumps(row), flush=True)
                del restored
            except Exception as error:
                import traceback
                traceback.print_exc()
                failures.append(dict(universe=market, variant=variant, error=str(error)))
            finally:
                del model, data, batch, optimizer
                gc.collect()
                torch.cuda.empty_cache()
            atomic_json(args.output_dir / "gpu_validation.json", dict(status="passed" if not failures else "failed",
                                                                      checks=checks, failures=failures))
    if failures:
        raise RuntimeError("Batch02 preflight failures")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "preflight", "audit_reference"))
    parser.add_argument("--universe", choices=("csi300", "csi800"))
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int, default=0, choices=range(5))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT / "artifacts/research"):
        raise ValueError("Explicit research artifact output required")
    if args.command != "preflight" and (not args.universe or not args.variant):
        parser.error("universe and variant required")
    globals()[args.command](args)


if __name__ == "__main__":
    main()
