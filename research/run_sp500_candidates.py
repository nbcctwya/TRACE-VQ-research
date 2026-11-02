"""Resumable SP500 candidate runs, with a persistent process/heartbeat monitor."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()

from trace_vq.cli.screening import now
from trace_vq.cli.sp500_candidates import BASELINE, VARIANTS, SEEDS, make_config, common_config, tasks
from trace_vq.data.frozen import atomic_json, sha256

DEFAULT_ROOT = ROOT / "artifacts/research/sp500_candidates_multiseed_20261031"


def read(path):
    return json.loads(Path(path).read_text())


def source_identity():
    paths = [p for folder in ("trace_vq", "configs", "tests") for p in (ROOT / folder).rglob("*")
             if p.is_file() and p.suffix in (".py", ".yaml")]
    paths += [Path(__file__), ROOT / "research/run.sh", ROOT / "research/verify_baseline.py", ROOT / "AGENTS.md"]
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(paths))}


class Pipeline:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = (root / "pipeline.lock").open("a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.source = source_identity()
        self.path = root / "progress.json"
        self.state = read(self.path) if self.path.exists() else dict(started=now(), phases={}, runs={})
        if self.state.get("source_sha256", self.source) != self.source:
            raise RuntimeError("Frozen experiment source changed; inspect before resuming")
        self.state.update(pid=os.getpid(), source_sha256=self.source)
        self.env = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                        OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
        plan = dict(universe="sp500", variants=list(VARIANTS), stage1_seed=42, stage2_seeds=list(SEEDS),
                    tasks=tasks(), native_stage1_source=str(BASELINE / "stage1"),
                    fsq_stage1_shared_by=["fsq", "fsq_ranknet"], fsq_levels=[8, 8, 8], ranknet_weight=0.1,
                    new_stage1_runs=1, new_stage2_runs=15, baseline_results_reused=5, parallel_fits=1,
                    common_stage1=common_config(make_config(root, "fsq", 42, stage1=True)),
                    common_stage2=common_config(make_config(root, "ranknet", 0)),
                    selection="minimum validation loss; original target-day MSE plus native auxiliary loss",
                    all_prespecified_seeds_run_regardless_of_test_results=True,
                    independent_seed_portfolios=True, prediction_ensemble=False,
                    native_cache_reused_without_copy=True, source_sha256=self.source)
        plan = json.loads(json.dumps(plan))
        path = root / "plan.json"
        if path.exists() and read(path) != plan:
            raise RuntimeError("Prespecified SP500 plan changed")
        if not path.exists():
            atomic_json(path, plan)
        for name, digest in self.source.items():
            target = root / "source_snapshot" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes((ROOT / name).read_bytes())
            if sha256(target) != digest:
                raise RuntimeError("Source snapshot mismatch: " + name)

    def save(self):
        self.state["updated"] = now()
        atomic_json(self.path, self.state)

    def monitor(self, phase=None, process=None, folder=None):
        pulse = folder / "heartbeat.json" if folder else None
        epoch = folder / "training_progress.json" if folder else None
        heartbeat = read(pulse) if pulse and pulse.exists() else None
        metrics = read(epoch).get("latest") if epoch and epoch.exists() else None
        age = time.time() - pulse.stat().st_mtime if pulse and pulse.exists() else None
        phases = self.state["phases"]
        completed = len([v for v in self.state["runs"].values() if v.get("status") == "complete"])
        status = dict(updated=now(), supervisor_pid=os.getpid(), status=self.state.get("status"),
                      current_phase=phase or self.state.get("current_phase"),
                      child_pid=process.pid if process else None, child_alive=process.poll() is None if process else None,
                      completed_candidate_runs=completed, planned_candidate_runs=15,
                      completed_training_runs=sum(k.endswith(".train") and v.get("status") == "complete"
                                                  for k, v in phases.items()),
                      heartbeat=heartbeat, latest_validation=metrics, heartbeat_age_seconds=age,
                      stalled_warning=bool(age is not None and age > 900),
                      failures=[k for k, v in phases.items() if v.get("status") == "failed"])
        atomic_json(self.root / "monitor_status.json", status)
        with (self.root / "monitor_history.jsonl").open("a") as stream:
            stream.write(json.dumps(status, ensure_ascii=False) + "\n")

    def phase(self, name, arguments, log=None, folder=None):
        if source_identity() != self.source:
            raise RuntimeError("Frozen source changed before phase " + name)
        previous = self.state["phases"].get(name, {})
        if previous.get("status") == "complete":
            return
        log = log or self.root / (name + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        entry = dict(status="running", started=now(), command=[sys.executable, *arguments], log=str(log),
                     attempts=previous.get("attempts", []))
        self.state["phases"][name] = entry
        self.state.update(status="running", current_phase=name)
        self.save()
        for retry in range(3):
            print(f"[{now()}] {name} starting (attempt {len(entry['attempts']) + 1})", flush=True)
            attempt = dict(started=now())
            entry["attempts"].append(attempt)
            with log.open("a") as stream:
                process = subprocess.Popen(entry["command"], cwd=ROOT, env=self.env, stdout=stream,
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
                entry["pid"] = process.pid
                attempt["pid"] = process.pid
                self.save()
                while process.poll() is None:
                    self.monitor(name, process, folder)
                    time.sleep(15)
                code = process.returncode
            attempt.update(returncode=code, finished=now())
            entry.update(returncode=code, finished=now(), status="complete" if code == 0 else "failed")
            self.save()
            if code == 0:
                print(f"[{now()}] {name} complete", flush=True)
                return
            tail = log.read_text(errors="replace")[-12000:]
            transient = code in (-9, -11) or re.search(
                r"CUDA out of memory|DataLoader worker.*killed|Bus error|ConnectionResetError", tail)
            if not transient or retry == 2:
                raise RuntimeError(f"{name} failed with {code}; see {log}")
            print(f"[{now()}] transient failure; retry with saved training checkpoint", flush=True)
            entry["status"] = "retrying"
            self.save()
            time.sleep(15)

    def cli(self, command, variant="fsq", seed=0):
        return ["-m", "trace_vq.cli.sp500_candidates", command, "--root", str(self.root),
                "--variant", variant, "--seed", str(seed)]

    def audit_inputs(self):
        path = self.root / "input_integrity_before.json"
        if path.exists():
            return read(path)["files"]
        if read(BASELINE / "completion.json")["status"] != "complete":
            raise RuntimeError("Complete TRACE-VQ SP500 baseline required")
        audit = read(BASELINE / "data_audit.json")
        assert audit["status"] == "passed"
        expected = {v["source"]: v["sha256"] for v in audit["splits"].values()}
        expected[audit["original_handler_config"]] = audit["original_handler_sha256"]
        first = read(BASELINE / "stage1/training_result.json")
        expected[first["checkpoint"]] = first["checkpoint_sha256"]
        paths = [BASELINE / "completion.json", BASELINE / "data_audit.json",
                 BASELINE / "stage1/effective.config.yaml", BASELINE / "stage1/training_result.json"]
        for split in ("train", "valid", "test"):
            folder = BASELINE / "cache"
            record = read(folder / f"{split}.json")
            assert record["source_sha256"] == audit["splits"][split]["sha256"]
            expected[str(folder / f"{split}.npy")] = record["cache_sha256"]
            paths.append(folder / f"{split}.json")
        for seed in SEEDS:
            folder = BASELINE / f"seed{seed}"
            record = read(folder / "training_result.json")
            expected[record["checkpoint"]] = record["checkpoint_sha256"]
            for relative in ("effective.config.yaml", "training_result.json", "completion.json",
                             "evaluation/completion.json", "evaluation/test_predictions.pkl",
                             "backtest/portfolio_return.csv", "backtest/backtest_config.json"):
                paths.append(folder / relative)
        for p in paths:
            expected[str(p)] = sha256(p)
        for p, digest in expected.items():
            if sha256(p) != digest:
                raise RuntimeError("Baseline input checksum mismatch: " + str(p))
        atomic_json(path, dict(status="passed", files=expected, baseline=str(BASELINE), finished=now()))
        configs = {f"{v}.seed{s}": make_config(self.root, v, s) for v, s in tasks()}
        atomic_json(self.root / "configuration_parity.json", dict(status="passed", stage2_configs=configs,
                    stage1_config=make_config(self.root, "fsq", 42, stage1=True),
                    all_shared_hyperparameters_exact=True, only_component_and_seed_output_differences=True))
        return expected

    def summarize(self):
        import pandas as pd
        from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
        rows = []
        for seed in SEEDS:
            r = read(BASELINE / f"seed{seed}/completion.json")
            rows.append(dict(variant="baseline", seed=seed, **r["test_metrics"]))
        for r in self.state["runs"].values():
            rows.append(dict(variant=r["variant"], seed=r["stage2_seed"], **r["test_metrics"]))
        table = pd.DataFrame(rows).sort_values(["variant", "seed"])
        table.to_csv(self.root / "seed_metrics.csv", index=False)
        metrics = [c for c in table.select_dtypes(include="number").columns if c != "seed"]
        summary = table.groupby("variant")[metrics].agg(["mean", "std", "count"])
        summary.columns = ["_".join(c) for c in summary.columns]
        summary.to_csv(self.root / "mean_std.csv")
        deltas = []
        base = table[table.variant == "baseline"].set_index("seed")
        for _, row in table[table.variant != "baseline"].iterrows():
            for metric in ["RankIC", "IC", "AR", "Sharpe", "MDD"]:
                deltas.append(dict(variant=row.variant, seed=row.seed, metric=metric,
                                   candidate=row[metric], baseline=base.loc[row.seed, metric],
                                   delta=row[metric] - base.loc[row.seed, metric]))
        if deltas:
            pd.DataFrame(deltas).to_csv(self.root / "matched_seed_deltas.csv", index=False)
        return table, summary

    def aggregate(self, variant, seed):
        import numpy as np
        import pandas as pd
        from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
        folder = self.root / variant / f"seed{seed}"
        r = read(folder / "training_result.json")
        evaluation = read(folder / "evaluation/completion.json")
        assert evaluation["standalone_exact"] and evaluation["stage1_unchanged"]
        config = read(folder / "backtest/backtest_config.json")
        assert config == read(BASELINE / f"seed{seed}/backtest/backtest_config.json")
        prediction = pd.read_pickle(folder / "evaluation/test_predictions.pkl")
        baseline = pd.read_pickle(BASELINE / f"seed{seed}/evaluation/test_predictions.pkl")
        assert prediction.index.equals(baseline.index)
        pd.testing.assert_series_equal(prediction.label, baseline.label, check_exact=True)
        returns = pd.read_csv(folder / "backtest/portfolio_return.csv", index_col=0, parse_dates=True)
        assert len(returns) == prediction.index.get_level_values("datetime").nunique() == 752
        assert np.allclose(returns["return"], returns.gross_return - returns.cost, rtol=0, atol=1e-14)
        metrics = ranking_metrics(prediction.score.astype("float64"), prediction.label.astype("float64"))
        metrics.update(portfolio_metrics(returns["return"]))
        r.update(status="complete", test_metrics=metrics, standalone_exact=True,
                 backtest_config=config, backtest_configuration_exact_to_baseline=True,
                 test_indices_and_labels_exact_to_baseline=True, run_dir=str(folder), completed=now())
        atomic_json(folder / "completion.json", r)
        self.state["runs"][f"{variant}.seed{seed}"] = r
        self.save()
        self.summarize()
        print(f"[{now()}] RESULT {variant}.seed{seed}: {json.dumps(metrics)}", flush=True)

    def run(self, prepare_only=False):
        self.state.update(status="preparing", current_phase="input_audit")
        self.save()
        inputs = self.audit_inputs()
        self.summarize()
        if prepare_only:
            self.state.update(status="prepared", current_phase=None)
            self.save()
            self.monitor()
            return
        try:
            self.phase("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"], self.root / "all_tests.log")
            self.phase("integrity_before", ["research/verify_baseline.py", "--full"], self.root / "integrity_before.json")
            assert read(self.root / "integrity_before.json")["source_unchanged"]
            self.phase("stage1_preflight", self.cli("check-stage1"))
            self.phase("stage1_fsq", self.cli("stage1"), self.root / "stage1_fsq/train.log", self.root / "stage1_fsq")
            self.phase("cache_fsq", self.cli("cache"))
            self.phase("stage2_preflight", self.cli("check-stage2"))
            for variant, seed in tasks():
                key = f"{variant}.seed{seed}"
                if self.state["runs"].get(key, {}).get("status") == "complete":
                    continue
                folder = self.root / variant / f"seed{seed}"
                self.phase(key + ".train", self.cli("train", variant, seed), folder / "train.log", folder)
                self.phase(key + ".evaluate", self.cli("evaluate", variant, seed), folder / "evaluate.log")
                self.phase(key + ".backtest", ["-m", "trace_vq.evaluation.backtest", "--universe", "sp500",
                           "--quote-universe", "all", "--pred_path", str(folder / "evaluation/test_predictions.pkl"),
                           "--output_dir", str(folder / "backtest")], folder / "backtest.log")
                self.aggregate(variant, seed)
            assert len(self.state["runs"]) == 15
            self.phase("integrity_after", ["research/verify_baseline.py", "--full"], self.root / "integrity_after.json")
            assert read(self.root / "integrity_after.json")["source_unchanged"]
            unchanged = {p: sha256(p) == h for p, h in inputs.items()}
            assert all(unchanged.values()) and source_identity() == self.source
            atomic_json(self.root / "input_integrity_after.json", dict(status="passed", files=unchanged))
            table, summary = self.summarize()
            atomic_json(self.root / "completion.json", dict(status="complete", finished=now(), universe="sp500",
                        variants=list(VARIANTS), stage1_seed=42, stage2_seeds=list(SEEDS), new_stage1_runs=1,
                        new_stage2_runs=15, reused_baseline_results=5, all_standalone_predictions_exact=True,
                        common_hyperparameters_exact_to_baseline=True, baseline_inputs_unchanged=True,
                        original_unchanged=True, prediction_ensemble=False,
                        summary=summary.reset_index().to_dict(orient="records")))
            self.state.update(status="complete", current_phase=None, finished=now())
            self.save()
            self.monitor()
        except BaseException as error:
            self.state.update(status="failed", error=str(error), traceback=traceback.format_exc())
            self.save()
            self.monitor()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_relative_to(ROOT / "artifacts/research"):
        raise ValueError("Research output required")
    if args.status:
        print(json.dumps(read(root / "monitor_status.json"), ensure_ascii=False, indent=2))
        return
    Pipeline(root).run(args.prepare_only)


if __name__ == "__main__":
    main()
