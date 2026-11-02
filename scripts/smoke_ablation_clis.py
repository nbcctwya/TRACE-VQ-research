"""Short CPU/CUDA CLI smoke using synthetic data or existing real windows."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()

import numpy as np
import pandas as pd

from trace_vq.data.market import CanonicalSampler, market_feature_config
from trace_vq.data.schema import GROUP_DIMS, MARKET_INDICES, TOTAL_DIM, dataset_basename
from scripts._smoke_data import sample_real_windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--data-path", type=Path, help="Existing processed data; sample complete days")
    parser.add_argument("--days", type=int, default=2, help="Trading days per real split (at least 2)")
    parser.add_argument("--skip-imported", action="store_true", help="For code-only deployments without historical artifacts")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/migration/cli_smoke_report.json")
    args = parser.parse_args()
    if args.days < 2:
        parser.error("--days must be at least 2 for daily ICIR")
    if args.data_path:
        args.data_path = args.data_path.expanduser().resolve()
    report = {"passed": False, "device": args.device,
              "data_kind": "real" if args.data_path else "synthetic", "stage1": {}, "stage2": {}}
    source_stats = {}
    logs = ROOT / "artifacts/verification/cli_smoke"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", WANDB_MODE="disabled")

    def run(arguments, name):
        with (logs / f"{name}.log").open("w") as stream:
            result = subprocess.run([sys.executable, *arguments], cwd=ROOT, env=env,
                                    stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {logs / (name + '.log')}")

    def entrypoint(variant, command):
        if variant == "full":
            return ["-m", f"trace_vq.cli.{command}"]
        return [str(ROOT / "ablation" / f"{variant}.py"), command]

    def training_entrypoint(variant, command):
        return entrypoint(variant, command) + ([f"experiment={variant}"]
                                              if variant == "full" else [])

    try:
        with tempfile.TemporaryDirectory(prefix="trace-vq-cli-smoke-") as tmp:
            tmp = Path(tmp)
            data_root = tmp / "data"
            (data_root / "CN").mkdir(parents=True)
            if not args.data_path:
                dates = pd.bdate_range("2023-01-02", periods=25, name="datetime")
                index = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(8)]], names=["datetime", "instrument"])
                columns = pd.MultiIndex.from_tuples([(group, f"{group}_{i}") for group, width in GROUP_DIMS.items() for i in range(width)])
                rng = np.random.default_rng(314)
                frame = pd.DataFrame(rng.normal(size=(len(index), TOTAL_DIM)), index=index, columns=columns)
                market = pd.DataFrame(rng.normal(size=(len(dates), 63)), index=dates,
                                      columns=pd.MultiIndex.from_product([["market"], market_feature_config(MARKET_INDICES["csi300"])[1]]))
                sampler = CanonicalSampler(frame, market, "csi300", dates[20], dates[-1])
            report["splits"] = {}
            for split in ("train", "valid", "test"):
                if args.data_path:
                    source = args.data_path / "CN" / f"{dataset_basename('csi300')}_{split}.pkl"
                    stat = source.stat()
                    source_stats[source] = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
                    sampler, info = sample_real_windows(source, args.days)
                    info["sampled_windows_sha256"] = hashlib.sha256(sampler.windows.tobytes()).hexdigest()
                    report["splits"][split] = info
                    print(f"Real {split}: {len(sampler)} rows, {args.days} complete trading days", flush=True)
                else:
                    report["splits"][split] = {"sampled_rows": len(sampler), "days": 5}
                with (data_root / "CN" / f"{dataset_basename('csi300')}_{split}.pkl").open("wb") as stream:
                    pickle.dump(sampler, stream)
            accelerator = "gpu" if args.device == "cuda" else "cpu"
            common = [f"data.data_path={data_root}", f"+train.accelerator={accelerator}", f"train.device={args.device}",
                      "train.num_epochs=1", "train.num_workers=0", "train.wandb_mode=disabled"]
            stage1 = {}
            for variant in ("full", "wo_ta"):
                artifacts = tmp / f"stage1_{variant}"
                run([*training_entrypoint(variant, "stage1"), *common,
                     f"artifact_root={artifacts}", f"train.run_name=smoke_{variant}",
                     f"hydra.run.dir={artifacts}/hydra"], f"stage1_{variant}")
                best, = [p for p in (artifacts / "checkpoints").glob("*.ckpt") if p.name != "last.ckpt"]
                stage1[variant] = best
                report["stage1"][variant] = "PASS"
                print(f"Stage 1 {variant}: one epoch and checkpoint PASS", flush=True)
            for variant in ("full", "wo_ta", "wo_ca", "wo_sr"):
                artifacts = tmp / f"stage2_{variant}"
                checkpoint1 = stage1["wo_ta" if variant == "wo_ta" else "full"]
                run([*training_entrypoint(variant, "stage2"), *common,
                     f"artifact_root={artifacts}", f"predictor.saved_model='{checkpoint1}'",
                     f"hydra.run.dir={artifacts}/hydra"], f"stage2_{variant}")
                checkpoint2, = (artifacts / "checkpoints").glob("*.ckpt")
                config, = (artifacts / "checkpoints").glob("*.config.yaml")
                training_prediction, = (artifacts / "res").glob("*/*_best.pkl")
                evaluation = tmp / f"evaluation_{variant}"
                run([*entrypoint(variant, "evaluate"), "--checkpoint", str(checkpoint2),
                      "--stage1-checkpoint", str(checkpoint1), "--config", str(config),
                     "--data-path", str(data_root), "--device", args.device,
                     "--output-dir", str(evaluation)], f"evaluate_{variant}")
                pd.testing.assert_frame_equal(pd.read_pickle(training_prediction),
                                              pd.read_pickle(evaluation / "test_predictions.pkl"), check_exact=True)
                evaluated = pd.read_pickle(evaluation / "test_predictions.pkl")
                if not np.isfinite(evaluated["score"]).all():
                    raise RuntimeError(f"Non-finite predictions for {variant}")
                selector_status = "not requested: code-only deployment" if args.skip_imported else "PASS"
                if not args.skip_imported:
                    imported = tmp / f"imported_{variant}"
                    selector = ["--experiment", variant] if variant == "full" else []
                    run([*entrypoint(variant, "evaluate"), *selector, "--seed", "0",
                         "--data-path", str(data_root), "--device", args.device,
                         "--output-dir", str(imported)], f"imported_{variant}")
                    imported_prediction = pd.read_pickle(imported / "test_predictions.pkl")
                    assert len(imported_prediction) == len(sampler)
                    assert np.isfinite(imported_prediction["score"]).all()
                report["stage2"][variant] = {"one_epoch_and_checkpoint": "PASS",
                    "standalone_predictions_exact": True, "imported_experiment_selector": selector_status,
                    "prediction_rows": len(evaluated), "scores_finite": True,
                    "entrypoint": f"ablation/{variant}.py" if variant != "full" else "trace_vq.cli"}
                print(f"Stage 2 {variant}: one epoch and exact standalone evaluation PASS", flush=True)
            for source, before in source_stats.items():
                stat = source.stat()
                if before != (stat.st_size, stat.st_mtime_ns, stat.st_ino):
                    raise RuntimeError(f"Source dataset changed during the smoke: {source}")
            report["source_dataset_files_unchanged"] = True
        report["temporary_data_and_checkpoints_removed"] = True
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        path = args.report.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
