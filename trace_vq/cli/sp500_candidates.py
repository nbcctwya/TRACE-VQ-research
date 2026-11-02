"""SP500 repetitions of the frozen FSQ, RankNet and combined experiments.

Common settings come from the completed SP500 TRACE-VQ run, including the
separate Stage 1 configuration. Only the previously screened components differ.
"""
import argparse
import copy
import gc
import hashlib
import json
import time
from pathlib import Path

from trace_vq.runtime import configure_runtime
configure_runtime()

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Callback

from trace_vq.cli.mechanism_screen import assert_frozen
from trace_vq.cli.next_screen import QuietFactorVQVAE, callbacks, trainer
from trace_vq.cli.screening import EpochRecord, initialize, now, scalar, write_prediction
from trace_vq.cli.sp500_multiseed import read_dataset
from trace_vq.data.frozen import atomic_json, daily_loader, prepare_frozen_cache, sha256
from trace_vq.data.schema import unpack_batch
from trace_vq.models.combination_screen import family, settings
from trace_vq.models.next_screen import NextPredictor
from trace_vq.utils import apply_artifact_root

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "artifacts/research/sp500_multiseed_20261031"
VARIANTS = ("fsq", "ranknet", "fsq_ranknet")
SEEDS = tuple(range(5))


def tasks():
    return [(variant, seed) for seed in SEEDS for variant in VARIANTS]


def common_config(config):
    """Retain every hyperparameter except the explicitly varied components."""
    result = copy.deepcopy(config)
    result.pop("artifact_root", None)
    result.pop("research", None)
    for key in ("seed", "save_dir", "save_res"):
        result["train"].pop(key, None)
    result["predictor"].pop("saved_model", None)
    for key in ("research_kind", "research_levels"):
        result["vqvae"]["quantizer"].pop(key, None)
    return result


def stage1_record(root, variant):
    folder = root / "stage1_fsq" if family(variant) == "fsq" else BASELINE / "stage1"
    return json.loads((folder / "training_result.json").read_text())


def make_config(root, variant, seed, stage1=False):
    if variant not in VARIANTS or seed not in ((42,) if stage1 else SEEDS):
        raise ValueError("Only the three frozen variants and prespecified stage seeds are allowed")
    if stage1 and variant != "fsq":
        raise ValueError("Only one shared FSQ Stage1 is newly trained")
    reference = BASELINE / ("stage1" if stage1 else "seed0") / "effective.config.yaml"
    original = OmegaConf.to_container(OmegaConf.load(reference), resolve=True)
    output = root / "stage1_fsq" if stage1 else root / variant / f"seed{seed}"
    cfg = OmegaConf.create(original)
    cfg.artifact_root = str(output.resolve())
    apply_artifact_root(cfg)
    cfg.train.seed = seed
    result = OmegaConf.to_container(cfg, resolve=True)
    result["research"] = settings(variant)
    if family(variant) == "fsq":
        result["vqvae"]["quantizer"].update(research_kind="fsq", research_levels=[8, 8, 8])
    if not stage1:
        if family(variant) == "native":
            result["predictor"]["saved_model"] = stage1_record(root, variant)["checkpoint"]
        elif (root / "stage1_fsq/training_result.json").exists():
            result["predictor"]["saved_model"] = stage1_record(root, variant)["checkpoint"]
        else:
            result["predictor"]["saved_model"] = str(root / "stage1_fsq/checkpoints/VALIDATION_SELECTED.ckpt")
    if common_config(result) != common_config(original):
        raise RuntimeError("A common SP500 hyperparameter changed")
    return result


def cache_path(root, variant, split):
    directory = root / "cache/fsq" if family(variant) == "fsq" else BASELINE / "cache"
    return directory / f"{split}.npy"


def cached(model, config, split, args):
    source, path = read_dataset(config, split)
    target = cache_path(args.root, args.variant, split)
    if args.command != "cache" and not target.with_suffix(".json").exists():
        raise RuntimeError("Complete the explicit cache phase before training")
    return prepare_frozen_cache(model, source, path, target, config, args.device, 2)


class Pulse(Callback):
    """Write progress during long epochs without changing model RNG or losses."""
    def __init__(self, output):
        self.path = output / "heartbeat.json"

    def write(self, fit, phase, batch=None, loss=None):
        atomic_json(self.path, dict(updated=now(), phase=phase, epoch=fit.current_epoch,
                                   global_step=fit.global_step, batch=batch, loss=loss))

    def on_fit_start(self, trainer, pl_module):
        self.write(trainer, "fit_start")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx % 200 == 0:
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if loss is not None and not torch.isfinite(loss).all():
                raise FloatingPointError("Non-finite training loss")
            self.write(trainer, "train", batch_idx, scalar(loss) if loss is not None else None)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx % 200 == 0:
            self.write(trainer, "validation", batch_idx)

    def on_fit_end(self, trainer, pl_module):
        self.write(trainer, "fit_complete")


def fit_model(model, config, output, args, train_loader, valid_loader):
    best, stop = callbacks(config, output, config["train"]["seed"])
    fit = trainer(config, output, [best, stop, EpochRecord(output), Pulse(output)], args, config["train"]["seed"])
    last = output / "checkpoints/last.ckpt"
    start = time.monotonic()
    fit.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader,
            ckpt_path=str(last) if last.exists() else None)
    if not best.best_model_path:
        raise RuntimeError("No validation-selected checkpoint")
    return fit, best, time.monotonic() - start


def check_stage1(args):
    initialize(42)
    config = make_config(args.root, "fsq", 42, stage1=True)
    source, _ = read_dataset(config, "train")
    batch = next(iter(daily_loader(source))).to(args.device)
    parts = unpack_batch(batch)
    model = QuietFactorVQVAE(config, T_max=3021 * 70).to(args.device).train()
    loss = model(parts.stock_feature, parts.prior_factor, parts.future_returns)[3]
    if not torch.isfinite(loss):
        raise FloatingPointError("Non-finite FSQ Stage1 loss")
    loss.backward()
    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert grads and all(torch.isfinite(g).all() for g in grads.values())
    for fragment in ("spatial_encoder", "quantizer.to_grid", "quantizer.from_grid"):
        assert any(fragment in n and torch.count_nonzero(g) for n, g in grads.items()), fragment
    atomic_json(args.root / "stage1_gpu_check.json", dict(status="passed", seed=42,
                rows=len(batch), loss=scalar(loss), all_gradients_finite=True,
                encoder_and_both_fsq_projections_have_gradients=True, technical_check_only=True))


def train_stage1(args):
    output = args.root / "stage1_fsq"
    if (output / "training_result.json").exists():
        raise FileExistsError("FSQ Stage1 is already complete")
    output.mkdir(parents=True, exist_ok=True)
    config = make_config(args.root, "fsq", 42, stage1=True)
    initialize(42)
    train_data, train_path = read_dataset(config, "train")
    valid_data, valid_path = read_dataset(config, "valid")
    training = daily_loader(train_data, shuffle=True, num_workers=2)
    validation = daily_loader(valid_data, num_workers=2)
    model = QuietFactorVQVAE(config, len(training) * config["train"]["num_epochs"])
    OmegaConf.save(OmegaConf.create(config), output / "effective.config.yaml")
    atomic_json(output / "input_manifest.json", dict(stage1_seed=42, train_sha256=sha256(train_path),
                valid_sha256=sha256(valid_path), test_used_for_training_or_selection=False))
    fit, best, elapsed = fit_model(model, config, output, args, training, validation)
    restored = QuietFactorVQVAE.load_from_checkpoint(best.best_model_path, config=copy.deepcopy(config),
                                                    T_max=1, map_location="cpu", strict=True)
    assert restored.config["train"]["seed"] == 42
    atomic_json(output / "training_result.json", dict(status="complete", universe="sp500", variant="fsq",
                stage1_seed=42, checkpoint=best.best_model_path, checkpoint_sha256=sha256(best.best_model_path),
                best_val_loss=scalar(best.best_model_score), completed_epochs=fit.current_epoch,
                elapsed_seconds=elapsed, config=str(output / "effective.config.yaml"), strict_restore=True,
                shared_by=["fsq", "fsq_ranknet"], test_used_for_training_or_selection=False, finished=now()))


def build_cache(args):
    initialize(0)
    args.variant = "fsq"
    config = make_config(args.root, "fsq", 0)
    model = NextPredictor(config, T_max=1)
    records = {}
    for split in ("train", "valid", "test"):
        data, records[split] = cached(model, config, split, args)
        del data
        gc.collect()
    atomic_json(args.root / "cache/fsq/completion.json", dict(status="complete", splits=records,
                stage1_sha256=stage1_record(args.root, "fsq")["checkpoint_sha256"]))


def check_stage2(args):
    config = make_config(args.root, "ranknet", 0)
    source, path = read_dataset(config, "train")
    checks, initializations = [], {}
    for variant in VARIANTS:
        hashes = []
        for seed in SEEDS:
            initialize(seed)
            cfg = make_config(args.root, variant, seed)
            model = NextPredictor(cfg, T_max=100).to(args.device)
            vector = torch.cat([p.detach().cpu().flatten() for p in model.parameters() if p.requires_grad])
            hashes.append(hashlib.sha256(vector.numpy().tobytes()).hexdigest())
            data, _ = prepare_frozen_cache(model, source, path, cache_path(args.root, variant, "train"),
                                           cfg, args.device, 2)
            batch = {k: v.to(args.device) for k, v in next(iter(daily_loader(data))).items()}
            model.train()
            initialize(10000 + seed)
            parts, result = model.batch_prediction(batch)
            loss, terms = model.training_loss(parts, result)
            assert bool("ranknet" in terms) == bool(settings(variant).get("ranknet_weight"))
            loss.backward()
            grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
            assert grads and all(torch.isfinite(g).all() for g in grads.values())
            model.zero_grad(set_to_none=True)
            initialize(10000 + seed)
            raw_parts, raw_result = model.batch_prediction(batch["canonical"])
            raw_loss, _ = model.training_loss(raw_parts, raw_result)
            assert torch.equal(loss, raw_loss), (variant, seed, "cached/raw loss")
            raw_loss.backward()
            for n, p in model.named_parameters():
                if n in grads:
                    assert torch.equal(grads[n], p.grad), (variant, seed, n)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["learning_rate"], weight_decay=1e-5)
            optimizer.step()
            assert_frozen(model, cfg)
            assert all(not p.requires_grad for n, p in model.named_parameters()
                       if n.startswith(("encoder.", "quantizer.", "revin.")))
            checks.append(dict(variant=variant, seed=seed, cached_raw_loss_and_gradients_exact=True,
                               all_gradients_finite=True, stage1_unchanged_after_optimizer_step=True))
            del model, data, batch, parts, result, terms, grads, raw_parts, raw_result, loss, raw_loss, optimizer, vector
            gc.collect()
            torch.cuda.empty_cache()
        assert len(set(hashes)) == 5
        initializations[variant] = hashes
    assert initializations["fsq"] == initializations["ranknet"] == initializations["fsq_ranknet"]
    atomic_json(args.root / "stage2_gpu_checks.json", dict(status="passed", checks=checks,
                initialization_sha256=initializations, same_seed_common_initializations_exact=True,
                five_distinct_initializations_per_variant=True, technical_checks_only=True))


def train_stage2(args):
    output = args.root / args.variant / f"seed{args.seed}"
    if (output / "training_result.json").exists():
        raise FileExistsError("Stage2 training is already complete")
    output.mkdir(parents=True, exist_ok=True)
    initialize(args.seed)
    config = make_config(args.root, args.variant, args.seed)
    model = NextPredictor(config, T_max=1)
    training, tm = cached(model, config, "train", args)
    validation, vm = cached(model, config, "valid", args)
    train_loader = daily_loader(training, shuffle=True, num_workers=2)
    valid_loader = daily_loader(validation, num_workers=2)
    model.T_max = len(train_loader) * config["train"]["num_epochs"]
    snapshot = copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot), output / "effective.config.yaml")
    first_hash = sha256(config["predictor"]["saved_model"])
    atomic_json(output / "input_manifest.json", dict(stage1_seed=42, stage2_seed=args.seed,
                stage1_sha256=first_hash, train=tm, valid=vm, test_used_for_training_or_selection=False,
                selection="minimum native validation target-day MSE plus auxiliary loss; excludes RankNet"))
    fit, best, elapsed = fit_model(model, config, output, args, train_loader, valid_loader)
    assert_frozen(model, config)
    assert sha256(config["predictor"]["saved_model"]) == first_hash
    record = dict(status="complete", universe="sp500", variant=args.variant, stage1_seed=42,
                  stage2_seed=args.seed, stage1_unchanged=True, stage1_sha256=first_hash,
                  checkpoint=best.best_model_path, checkpoint_sha256=sha256(best.best_model_path),
                  best_val_loss=scalar(best.best_model_score), completed_epochs=fit.current_epoch,
                  elapsed_seconds=elapsed, config=str(output / "effective.config.yaml"), finished=now())
    restored = NextPredictor.load_from_checkpoint(best.best_model_path, config=snapshot, T_max=1,
                                                 map_location="cpu", strict=True)
    del fit, model, train_loader, training
    gc.collect()
    torch.cuda.empty_cache()
    record["validation_metrics"] = write_prediction(restored, validation, snapshot, args.device,
                                                     output / "training_evaluation", "valid", 2)
    del validation, valid_loader
    gc.collect()
    test, _ = cached(restored, snapshot, "test", args)
    record["test_metrics"] = write_prediction(restored, test, snapshot, args.device,
                                               output / "training_evaluation", "test", 2)
    atomic_json(output / "training_result.json", record)


def evaluate(args):
    initialize(args.seed)
    output = args.root / args.variant / f"seed{args.seed}"
    record = json.loads((output / "training_result.json").read_text())
    config = OmegaConf.to_container(OmegaConf.load(record["config"]), resolve=True)
    assert config == make_config(args.root, args.variant, args.seed)
    assert sha256(record["checkpoint"]) == record["checkpoint_sha256"]
    model = NextPredictor.load_from_checkpoint(record["checkpoint"], config=config, T_max=1,
                                               map_location="cpu", strict=True)
    metrics = {}
    for split in ("valid", "test"):
        data, _ = cached(model, config, split, args)
        metrics[split] = write_prediction(model, data, config, args.device, output / "evaluation", split, 2)
        pd.testing.assert_frame_equal(pd.read_pickle(output / f"evaluation/{split}_predictions.pkl"),
                                      pd.read_pickle(output / f"training_evaluation/{split}_predictions.pkl"), check_exact=True)
        del data
        gc.collect()
    assert_frozen(model, config)
    atomic_json(output / "evaluation/completion.json", dict(status="complete", standalone_exact=True,
                stage1_unchanged=True, stage2_seed=args.seed, variant=args.variant, metrics=metrics))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check-stage1", "stage1", "cache", "check-stage2", "train", "evaluate"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, default="fsq")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    args.root = args.root.resolve()
    if not args.root.is_relative_to(ROOT / "artifacts/research"):
        raise ValueError("New output must stay in research artifacts")
    args.root.mkdir(parents=True, exist_ok=True)
    {"check-stage1": check_stage1, "stage1": train_stage1, "cache": build_cache,
     "check-stage2": check_stage2, "train": train_stage2, "evaluate": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
