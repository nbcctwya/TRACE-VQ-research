"""Audit imported artifacts and compare migrated models with frozen HVQ code.

Default: exact frozen-source parity on the complete CSI300 test set for all
four experiments, seeds 0-4, using GPU when available. Report cross-hardware
historical differences separately at each run's effective matrix precision. Use
--sample-only --device cpu for exact CPU source parity without claiming that
CPU recomputation reproduces historical CUDA predictions.
"""

import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from import_hvq_experiments import git, sha256, write_json
from trace_vq.data.loaders import DailyBatchSamplerRandom
from trace_vq.data.schema import unpack_batch
from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
from trace_vq.training.prediction import GenerateReturn
from trace_vq.training.representation import FactorVQVAE

METRICS = ("IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "MDD", "Sharpe", "Sortino", "Calmar")


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def fingerprint(dataset, split):
    """The exact Phase 3 fingerprint: every sampled window and its metadata."""
    index = dataset.get_index()
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(index, index=True).to_numpy(dtype="<u8").tobytes())
    metadata = {"n": len(dataset), "index_names": list(index.names),
                "columns": [list(x) if isinstance(x, tuple) else str(x) for x in dataset.columns],
                "pandas": pd.__version__, "sample_shape": list(dataset[0].shape),
                "dtype": str(dataset[0].dtype), "split": split}
    h.update(identity(metadata).encode())
    for start in range(0, len(dataset), 128):
        array = np.asarray(dataset[list(range(start, min(start + 128, len(dataset))))], dtype="<f8").copy()
        array[array == 0] = 0
        array[np.isnan(array)] = np.nan
        h.update(array.tobytes(order="C"))
    return {"sha256": h.hexdigest(), **metadata}


def equal_state(actual, expected):
    if set(actual) != set(expected):
        raise AssertionError(f"State keys differ: {set(actual) ^ set(expected)}")
    for key in actual:
        torch.testing.assert_close(actual[key].cpu(), expected[key].cpu(), rtol=0, atol=0, equal_nan=True, msg=key)


def equal_outputs(actual, expected):
    if len(actual) != len(expected):
        raise AssertionError("Output tuple lengths differ")
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0, equal_nan=True)


def check_config(config, original, info):
    expected = copy.deepcopy(original)
    expected["vqvae"]["encoder"]["type"] = info["encoder"]
    expected["predictor"]["quantization_confidence_adapter"] = info["confidence_adapter"]
    expected["predictor"]["shared_expert"] = info["shared_expert"]
    assert config["vqvae"] == expected["vqvae"], "Stage 1 hyperparameters changed"
    for node in (config, expected):
        node["predictor"].pop("saved_model", None)
    assert config["predictor"] == expected["predictor"], "Stage 2 hyperparameters changed"
    ignored = {"save_dir", "save_res", "project_name", "wandb_mode"}
    assert {k: v for k, v in config["train"].items() if k not in ignored} == {
        k: v for k, v in expected["train"].items() if k not in ignored
    }, "Training hyperparameters changed"
    for key in ("universe", "window_size", "step", "train_period", "valid_period", "test_period"):
        assert config["data"][key] == expected["data"][key], f"Data convention changed: {key}"


def verify_training_groups(manifest):
    from hydra import compose, initialize_config_dir
    from trace_vq.cli.stage2 import _apply_stage2_universe_preset
    report = {}
    for info in manifest["experiments"].values():
        overrides = [f"experiment={info['variant']}"]
        with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
            config = compose(config_name="config", overrides=overrides)
        _apply_stage2_universe_preset(config, overrides)
        config = OmegaConf.to_container(config, resolve=True)
        original = json.loads((ROOT / info["runs"]["0"]["original_config"]).read_text())
        checked = copy.deepcopy(config)
        inactive = []
        if info["encoder"] == "gru" and "temporal_dropout" not in original["vqvae"]["encoder"]:
            # This inherited attention-only key is never read by the GRU factory.
            checked["vqvae"]["encoder"].pop("temporal_dropout", None)
            inactive.append("vqvae.encoder.temporal_dropout")
        check_config(checked, original, info)
        assert (ROOT / config["predictor"]["saved_model"]).resolve() == (ROOT / info["stage1_checkpoint"]).resolve()
        assert (ROOT / config["data"]["data_path"]).resolve() == Path(manifest["dataset_root"])
        report[info["variant"]] = {"active_hyperparameters_exact": True,
                                    "stage1_and_shared_data_paths_correct": True,
                                    "inactive_inherited_keys": inactive}
    return report


def verify_results(manifest):
    seed = pd.read_csv(ROOT / "results/metrics/seed_metrics.csv", dtype={"model": str})
    aggregate = pd.read_csv(ROOT / "results/metrics/aggregate_metrics.csv", dtype={"model": str})
    ensemble = pd.read_csv(ROOT / "results/metrics/ensemble_metrics.csv", dtype={"model": str})
    assert len(seed) == 20 and set(seed["model"]) == set(manifest["experiments"])
    assert not seed.duplicated(["market", "model", "seed"]).any()
    result = {}
    for eid, info in manifest["experiments"].items():
        frames = []
        for number in range(5):
            # Formal HVQ load_prediction converts to float64 before averaging
            # seed scores. Averaging the raw float32 pickles changes rounding.
            frame = pd.read_pickle(ROOT / info["runs"][str(number)]["prediction"])[["score", "label"]].astype(float).sort_index()
            frames.append(frame)
            row = seed[(seed["model"] == eid) & (seed["seed"] == number)].iloc[0]
            computed = ranking_metrics(frame["score"], frame["label"])
            for name, value in computed.items():
                np.testing.assert_allclose(value, row[name], rtol=1e-12, atol=1e-12)
            assert (ROOT / row["pred_path_or_ckpt_path"]).is_file()
        group = seed[seed["model"] == eid]
        ag = aggregate[aggregate["model"] == eid].iloc[0]
        for name in METRICS:
            np.testing.assert_allclose(ag[f"{name}_mean"], group[name].mean(), rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(ag[f"{name}_std"], group[name].std(ddof=1), rtol=1e-12, atol=1e-12)
        scores = pd.concat([frame["score"].rename(i) for i, frame in enumerate(frames)], axis=1, join="inner").mean(axis=1)
        erow = ensemble[ensemble["model"] == eid].iloc[0]
        assert erow["ensemble_method"] == "avg_none"
        for name, value in ranking_metrics(scores, frames[0]["label"].reindex(scores.index)).items():
            np.testing.assert_allclose(value, erow[name], rtol=1e-12, atol=1e-12)
        curve = pd.read_csv(ROOT / "results/curves/ensemble" / f"csi300_{eid}.csv")
        np.testing.assert_allclose(curve["daily_ret_net"], curve["daily_ret_gross"] - curve["cost"], rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(curve["nav"], (1 + curve["daily_ret_net"]).cumprod(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(curve["bench_nav"], (1 + curve["bench_ret"]).cumprod(), rtol=1e-12, atol=1e-12)
        for name, value in portfolio_metrics(curve["daily_ret_net"]).items():
            np.testing.assert_allclose(value, erow[name], rtol=1e-12, atol=1e-12)
        result[eid] = {"seed_ranking_metrics": "PASS", "aggregate_mean_std": "PASS",
                       "ensemble_ranking_metrics": "PASS", "ensemble_curve_and_portfolio_metrics": "PASS"}
    return result


def summarize_historical_differences(manifest, device):
    archived_metrics = pd.read_csv(ROOT / "results/metrics/seed_metrics.csv", dtype={"model": str})
    metrics, differences = [], []
    for eid, info in manifest["experiments"].items():
        for seed in range(5):
            archived = pd.read_pickle(ROOT / info["runs"][str(seed)]["prediction"])
            current = pd.read_pickle(ROOT / "artifacts/verification" / f"{eid}_seed{seed}_{device}_test.pkl")
            pd.testing.assert_index_equal(current.index, archived.index)
            np.testing.assert_array_equal(current["label"], archived["label"])
            outside = ~np.isclose(current["score"], archived["score"], atol=1e-5, rtol=1e-5, equal_nan=True)
            changed = pd.DataFrame({"model": eid, "variant": info["variant"], "seed": seed,
                "score_archived": archived.loc[outside, "score"], "score_current": current.loc[outside, "score"]})
            changed["abs_error"] = (changed["score_current"] - changed["score_archived"]).abs()
            differences.append(changed.reset_index())
            original = archived_metrics[(archived_metrics["model"] == eid) & (archived_metrics["seed"] == seed)].iloc[0]
            computed = ranking_metrics(current["score"].astype(float), current["label"].astype(float))
            row = {"model": eid, "variant": info["variant"], "seed": seed}
            for name, value in computed.items():
                row.update({f"{name}_current": value, f"{name}_archived": original[name], f"{name}_delta": value - original[name]})
            metrics.append(row)
    folder = ROOT / "results/diagnostics"
    folder.mkdir(parents=True, exist_ok=True)
    pd.concat(differences, ignore_index=True).to_csv(folder / "cross_hardware_score_differences.csv", index=False)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(folder / "cross_hardware_metrics.csv", index=False)
    return {"outside_original_score_tolerance": sum(len(frame) for frame in differences),
            "max_abs_ranking_metric_delta": {name: float(metrics[f"{name}_delta"].abs().max()) for name in computed},
            "score_details": "results/diagnostics/cross_hardware_score_differences.csv",
            "metric_details": "results/diagnostics/cross_hardware_metrics.csv"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-only", action="store_true", help="Skip both complete-test prediction comparisons")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--require-historical-close", action="store_true",
                        help="Fail also on cross-hardware differences against archived predictions")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/migration/consistency_report.json")
    args = parser.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is not available; use --sample-only --device cpu for source parity")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)
    manifest = json.loads((ROOT / "docs/migration/experiments_manifest.json").read_text())
    source = Path(manifest["source_root"])
    report = {"passed": False, "device": device, "torch": torch.__version__, "experiments": {},
              "scope": "sampled source parity" if args.sample_only else "exact full-test source parity + cross-hardware historical comparison",
              "historical_score_tolerance": {"atol": 1e-5, "rtol": 1e-5},
              "new_training": False, "new_backtest": False}
    if device == "cuda":
        report["gpu"] = torch.cuda.get_device_name(0)
    start = time.monotonic()
    try:
        for record in manifest["copied_files"]:
            target = ROOT / record["target"]
            if "symlink" in record:
                assert target.is_symlink() and os.readlink(target) == record["symlink"]
                assert target.exists(), f"Broken imported link: {target}"
            else:
                assert sha256(target) == record["sha256"], record["target"]
                assert sha256(source / record["source"]) == record["sha256"], record["source"]
        for record in manifest["filtered_results"]:
            assert sha256(ROOT / record["target"]) == record["sha256"]
        report["artifact_checksums"] = {"passed": True, "files_and_links": len(manifest["copied_files"]),
                                        "bytes": manifest["copied_bytes"]}
        report["training_groups"] = verify_training_groups(manifest)
        assert not (ROOT / "dataset/processed").exists(), "Datasets were unexpectedly copied"
        data_root = Path(os.environ.get("TRACE_VQ_DATA_ROOT", manifest["dataset_root"]))
        samples = []
        sampled_dates = []
        # Compare complete cross-sections: cross-asset attention depends on all stocks in a day.
        for split in ("valid", "test"):
            with (data_root / "CN" / f"csi300_20_h10_{split}.pkl").open("rb") as stream:
                dataset = pickle.load(stream)
            days = list(DailyBatchSamplerRandom(dataset))
            for position in (0, len(days) // 2, len(days) - 1):
                batch = torch.as_tensor(dataset[days[position]]).float()
                samples.append(batch)
                sampled_dates.append({"split": split, "date": str(dataset.get_index()[days[position][0]][0]),
                                      "stocks": len(days[position])})
            if split == "valid":
                del dataset, days
                gc.collect()
        # Historical Phase 3 receipts contain the fingerprint of every actual test window.
        observed = fingerprint(dataset, "test")
        expected = None
        for info in manifest["experiments"].values():
            evidence = ROOT / Path(info["runs"]["1"]["checkpoint"]).parents[1] / "training_evidence.json"
            recorded = json.loads(evidence.read_text())["data_fingerprint"]["test"]
            assert observed == recorded, f"Shared test dataset differs from historical training: {info['variant']}"
            expected = recorded
        report["shared_test_dataset"] = {"passed": True, "path": str(data_root / "CN/csi300_20_h10_test.pkl"),
                                         "sha256_of_sampled_windows": observed["sha256"], "samples": observed["n"]}
        report["sampled_cross_sections"] = sampled_dates
        print(f"Historical test fingerprint matches; {len(days)} trading days, {len(dataset):,} rows", flush=True)
        models = []
        source_full = {}
        with tempfile.TemporaryDirectory(prefix="trace-vq-consistency-") as temporary:
            temporary = Path(temporary)
            for eid, info in manifest["experiments"].items():
                assert sha256(ROOT / info["raw_config"]) == info["reference_files"]["configs/config.yaml"]
                for relative, digest in info["reference_files"].items():
                    assert sha256(ROOT / info["reference_root"] / relative) == digest
                configs = []
                checkpoints = []
                for seed in range(5):
                    run = info["runs"][str(seed)]
                    config = OmegaConf.to_container(OmegaConf.load(ROOT / run["config"]), resolve=True)
                    original = json.loads((ROOT / run["original_config"]).read_text())
                    check_config(copy.deepcopy(config), original, info)
                    configs.append(config)
                    checkpoints.append(str(ROOT / run["checkpoint"]))
                request = temporary / f"{eid}-request.pt"
                reference = temporary / f"{eid}-reference.pt"
                torch.save({"configs": configs, "checkpoints": checkpoints, "samples": samples, "device": device,
                            "full_data_path": str(data_root / "CN/csi300_20_h10_test.pkl") if not args.sample_only else None}, request)
                log = ROOT / "artifacts/verification" / f"source_{eid}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                with log.open("w") as stream:
                    subprocess.run([sys.executable, str(ROOT / "scripts/_hvq_reference.py"),
                                    "--reference", str(ROOT / info["reference_root"]),
                                    "--input", str(request), "--output", str(reference)],
                                   cwd=temporary, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                oracle = torch.load(reference, map_location="cpu", weights_only=False)
                if not args.sample_only:
                    source_full[eid] = {"index": oracle["full_index"],
                                        "scores": {seed: run["full_scores"] for seed, run in oracle["runs"].items()}}
                torch.manual_seed(42)
                first = FactorVQVAE(copy.deepcopy(configs[0]), T_max=1)
                equal_state(first.state_dict(), oracle["stage1_initial"])
                state = torch.load(configs[0]["predictor"]["saved_model"], map_location="cpu", weights_only=False)["state_dict"]
                first.load_state_dict(state, strict=True)
                first.to(device).eval()
                torch.set_float32_matmul_precision(configs[0].get("inference", {}).get("matmul_precision", "high"))
                parts = unpack_batch(samples[0].to(device))
                with torch.no_grad():
                    equal_outputs(first(parts.stock_feature, parts.prior_factor, parts.future_returns)[:5], oracle["stage1_outputs"])
                experiment = {"variant": info["variant"], "source_commit": info["commit"],
                              "stage1_strict_load": "PASS", "stage1_initialization_and_forward_exact": True, "runs": {}}
                for seed, config in enumerate(configs):
                    torch.set_float32_matmul_precision(config.get("inference", {}).get("matmul_precision", "high"))
                    torch.manual_seed(seed)
                    model = GenerateReturn(copy.deepcopy(config), T_max=1)
                    expected_run = oracle["runs"][str(seed)]
                    equal_state(model.state_dict(), expected_run["initial"])
                    state = torch.load(checkpoints[seed], map_location="cpu", weights_only=False)["state_dict"]
                    model.load_state_dict(state, strict=True)
                    model.to(device)
                    frozen = torch.load(config["predictor"]["saved_model"], map_location="cpu", weights_only=False)["state_dict"]
                    for name, prefix in (("encoder", "vqvae.spatial_encoder."), ("quantizer", "vqvae.quantizer."), ("revin", "vqvae.revin.")):
                        module = getattr(model, name)
                        equal_state(module.state_dict(), {k[len(prefix):]: v for k, v in frozen.items() if k.startswith(prefix)})
                        assert all(not p.requires_grad for p in module.parameters())
                    model.eval()
                    with torch.no_grad():
                        for batch, output in zip(samples, expected_run["outputs"]):
                            parts = unpack_batch(batch.to(device))
                            equal_outputs(model(parts.stock_feature, parts.prior_factor), output)
                    run_report = {"strict_load": "PASS", "initialization_exact": True,
                                  "six_cross_section_outputs_exact": True, "frozen_stage1_exact": True,
                                  "matmul_precision": config.get("inference", {}).get("matmul_precision", "high")}
                    if seed == 0:
                        model.train()
                        torch.manual_seed(123)
                        parts = unpack_batch(samples[0].to(device))
                        outputs = model(parts.stock_feature, parts.prior_factor)
                        loss = model.rank_loss(outputs[0], parts.target(config["predictor"]["target_day"])) + model.aux_weight * outputs[-1]
                        loss.backward()
                        equal_outputs((loss,), (expected_run["training_loss"],))
                        for name, parameter in model.named_parameters():
                            gradient = expected_run["gradients"][name]
                            assert (parameter.grad is None) == (gradient is None), name
                            if gradient is not None:
                                equal_outputs((parameter.grad,), (gradient,))
                        optimizers, _ = model.configure_optimizers()
                        optimizers[0].step()
                        equal_state(model.state_dict(), expected_run["after_step"])
                        # Restore the imported checkpoint for historical inference.
                        model.load_state_dict(state, strict=True)
                        model.zero_grad(set_to_none=True)
                        run_report["loss_gradients_and_optimizer_step_exact"] = True
                    model.eval()
                    models.append((eid, seed, model, []))
                    experiment["runs"][str(seed)] = run_report
                report["experiments"][eid] = experiment
                print(f"{eid} {info['label']}: all 5 checkpoints strict; source initialization/outputs/training update exact", flush=True)
                del oracle, first, state, expected_run
                gc.collect()
            if not args.sample_only:
                labels = []
                positions = []
                inference_start = time.monotonic()
                with torch.no_grad():
                    for number, indices in enumerate(days, 1):
                        parts = unpack_batch(torch.as_tensor(dataset[indices]).float().to(device))
                        labels.append(parts.target(5).cpu().numpy())
                        positions.append(indices)
                        for eid, seed, model, predictions in models:
                            precision = manifest["experiments"][eid]["runs"][str(seed)].get("matmul_precision", "high")
                            torch.set_float32_matmul_precision(precision)
                            predictions.append(model(parts.stock_feature, parts.prior_factor)[0].cpu().numpy())
                        if number % 100 == 0 or number == len(days):
                            print(f"Full test inference: {number}/{len(days)} days × 20 models, {time.monotonic() - inference_start:.1f}s", flush=True)
                index = dataset.get_index()[np.concatenate(positions)]
                label = np.concatenate(labels)
                for eid, seed, model, predictions in models:
                    imported = pd.read_pickle(ROOT / manifest["experiments"][eid]["runs"][str(seed)]["prediction"])
                    actual = pd.DataFrame({"score": np.concatenate(predictions), "label": label}, index=index).reindex(imported.index)
                    prediction_path = ROOT / "artifacts/verification" / f"{eid}_seed{seed}_{device}_test.pkl"
                    actual.to_pickle(prediction_path)
                    assert len(actual) == len(dataset) == len(imported)
                    assert index.is_unique and index.difference(imported.index).empty
                    np.testing.assert_array_equal(actual["label"].to_numpy(), imported["label"].to_numpy())
                    reference_scores = pd.Series(source_full[eid]["scores"][str(seed)],
                                                 index=source_full[eid]["index"]).reindex(imported.index)
                    np.testing.assert_array_equal(actual["score"].to_numpy(), reference_scores.to_numpy())
                    source_delta = (actual["score"] - reference_scores).abs()
                    delta = (actual["score"] - imported["score"]).abs()
                    close = np.isclose(actual["score"], imported["score"], rtol=1e-5, atol=1e-5, equal_nan=True)
                    report["experiments"][eid]["runs"][str(seed)]["historical_test_predictions"] = {
                        "source_migration_exact": True, "source_max_abs_error": float(source_delta.max()),
                        "historical_close_at_original_tolerance": bool(close.all()),
                        "outside_original_tolerance": int((~close).sum()),
                        "rows": len(actual), "days": len(days), "labels_exact": True,
                        "max_abs_score_error": float(delta.max()), "mean_abs_score_error": float(delta.mean()),
                        "median_abs_score_error": float(delta.median()), "p99_abs_score_error": float(delta.quantile(0.99)),
                    }
                    print(f"{eid}/seed{seed}: all {len(actual):,} source/migrated predictions EXACT; "
                          f"historical max error {delta.max():.3g}, outside 1e-5 tolerance {int((~close).sum())}", flush=True)
                comparisons = [run["historical_test_predictions"] for exp in report["experiments"].values() for run in exp["runs"].values()]
                report["historical_reprediction_strict_passed"] = all(item["historical_close_at_original_tolerance"] for item in comparisons)
                report["full_source_migration_exact"] = True
                report["historical_comparison_note"] = (
                    "Copied predictions are byte-identical to HVQ. Source and migrated code give identical full-test outputs "
                    "on this device. Historical Phase 3 used RTX 2080 Ti/cu128; current RTX 5060 Ti/cu129 can change "
                    "hard VQ/routing choices at numerical boundaries. Cross-hardware deviations are reported, not hidden or retoleranced."
                )
        report["historical_results"] = verify_results(manifest)
        print("20 seed ranking metrics, aggregates and 4 ensemble metrics/curves match", flush=True)
        if not args.sample_only:
            report["cross_hardware_diagnostics"] = summarize_historical_differences(manifest, device)
        for relative, digest in manifest["source_tracked_hashes"].items():
            assert sha256(source / relative) == digest, relative
        for relative, recorded in manifest["datasets"].items():
            assert sha256(source / relative) == recorded["sha256"], relative
        assert git(source, "rev-parse", "HEAD").decode().strip() == manifest["source_head"]
        assert git(source, "status", "--porcelain").decode() == manifest["source_status"]
        report["source_repository_and_six_datasets_unchanged"] = True
        report["passed"] = True
        if args.require_historical_close and not args.sample_only and not report["historical_reprediction_strict_passed"]:
            report["passed"] = False
            raise AssertionError("Strict archived-prediction equality failed across GPU environments; see per-run comparisons")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic() - start
        write_json(args.report, report)
        write_json(ROOT / "results/diagnostics/migration_consistency.json", report)
        print(f"Report saved: {args.report}; passed={report['passed']}", flush=True)


if __name__ == "__main__":
    main()
