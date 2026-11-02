"""Run and retain a seed42/seed0 research screen under the existing protocol.

All initial variants receive the full existing training budget. Optional
combinations are selected using validation results only. Test predictions and
backtests are retained for every successful model, including worse models.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
from trace_vq.data.frozen import atomic_json, sha256
from trace_vq.evaluation.protocol_metrics import portfolio_metrics, ranking_metrics
from trace_vq.models.screening import INITIAL_VARIANTS, VARIANTS


PROTOCOL = {
    "stage1_seed": 42, "stage2_seed": 0,
    "train_period": ["2009-01-01", "2020-12-31"],
    "valid_period": ["2021-01-01", "2022-12-31"],
    "test_period": ["2023-01-01", "2025-12-31"],
    "target_day": 5, "target_formula": "Ref($close,-5)/Ref($close,-1)-1",
    "max_epochs": 70, "early_stopping_patience": 15, "early_stopping_min_delta": 0.00001,
    "checkpoint_selection": "minimum original target-day validation MSE + original auxiliary loss",
    "additional_losses": "training only; original validation selection objective retained",
    "quote_universe": "all", "topk": 30, "drop": 5, "risk_degree": 0.95,
    "account": 100000000.0, "deal_price": "close", "open_cost": 0.0005,
    "close_cost": 0.0015, "min_cost": 0.0,
    "metrics": "Baseline Results Protocol v1.0; net simple returns -> log returns; ddof=1; 252 days",
    "matmul_precision": "high", "training_precision": 32,
    "historical_behavior": "Codebook decay, routing implementation, labels and dataset splits preserved",
    "combination_selection": "validation RankIC only; never optimize a coefficient against test results",
    "interpretation": "Single-seed exploratory comparisons; shared test period has been inspected before",
}
IDEAS = {"baseline": [], "market_route": [12], "market_direct": [12], "ic_loss": [26],
         "pairwise_loss": [26], "multihorizon": [27], "sparse_prior": [17, 21],
         "market_route_ic": [12, 26], "market_route_multihorizon": [12, 27]}


def now():
    return datetime.now(timezone.utc).isoformat()


def code_identity():
    files = list((ROOT / "trace_vq").rglob("*.py")) + list((ROOT / "configs").rglob("*.yaml"))
    files.extend([Path(__file__).resolve(), ROOT / "research/run.sh"])
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(files)}


class Screen:
    def __init__(self, args):
        self.args = args
        self.output = args.output_dir.resolve()
        if not self.output.is_relative_to(ROOT / "artifacts/research"):
            raise ValueError("All screen outputs must be under this copy's artifacts/research directory")
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "screen.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.mutex = threading.RLock()
        self.market_locks = {name: threading.Semaphore(1) for name in args.universes}
        self.cache = self.output / "cache"
        self.state_path = self.output / "progress.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "started": now(), "phases": {}, "runs": {},
        }
        self.identity = code_identity()
        plan = {"protocol": PROTOCOL, "universe": args.universes,
                "initial_variants": {name: VARIANTS[name] for name in INITIAL_VARIANTS},
                "optional_combinations": ["market_route_ic", "market_route_multihorizon"],
                "source_sha256": self.identity,
                "retention": "Keep every result/config/checkpoint/log; label promising/mixed/underperforming explicitly",
                "selection_rule": "Promising requires better validation RankIC, test RankIC and net Sharpe; MDD loss <=2 percentage points",
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()}
        path = self.output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise RuntimeError("Plan or source changed; use a new run directory instead of mixing experiments")
        atomic_json(path, plan)
        self.state.update(status="running", pid=os.getpid(), python=sys.executable)
        self.save()

    def save(self):
        with self.mutex:
            self.state["updated"] = now()
            atomic_json(self.state_path, self.state)

    def event(self, message):
        with self.mutex:
            print(f"[{now()}] {message}", flush=True)
            with (self.output / "events.jsonl").open("a") as stream:
                stream.write(json.dumps({"time": now(), "message": message}) + "\n")

    def phase(self, key, command, log, required=()):
        with self.mutex:
            previous = self.state["phases"].get(key, {})
            if previous.get("status") == "complete" and all(Path(path).is_file() for path in required):
                self.event(f"reuse {key}")
                return
            if code_identity() != self.identity:
                raise RuntimeError("Research source changed during the screen")
            history = previous.get("previous_attempts", [])
            if previous:
                history = history + [{k: v for k, v in previous.items() if k != "previous_attempts"}]
            entry = {"status": "running", "started": now(), "command": command, "log": str(log),
                     "previous_attempts": history}
            self.state["phases"][key] = entry
            self.save()
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        self.event(f"start {key}")
        started = time.monotonic()
        environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", WANDB_MODE="disabled")
        with Path(log).open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            with self.mutex:
                entry["pid"] = process.pid
                self.save()
            code = process.wait()
        with self.mutex:
            entry.update(status="complete" if code == 0 else "failed", finished=now(), returncode=code,
                         elapsed_seconds=time.monotonic() - started)
            self.save()
        self.event(f"{'complete' if code == 0 else 'failed'} {key}")
        if code:
            raise RuntimeError(f"{key} exited {code}; inspect {log}")
        if not all(Path(path).is_file() for path in required):
            raise RuntimeError(f"{key} exited successfully but required output is missing")

    def cli(self, command, universe, variant, output):
        return [sys.executable, "-m", "trace_vq.cli.screening", command, "--universe", universe,
                "--variant", variant, "--output-dir", str(output), "--cache-root", str(self.cache),
                "--device", "cuda", "--num-workers", str(self.args.num_workers)]

    def integrity(self, label):
        self.phase(f"integrity.{label}", [sys.executable, "research/verify_baseline.py", "--full"],
                   self.output / f"integrity_{label}.log")
        original = json.loads((ROOT / "research/logs/source-integrity.json").read_text())
        if not original["source_unchanged"] or not original["full_hash_check"]:
            raise RuntimeError("Original TRACE-VQ integrity check failed")
        atomic_json(self.output / f"integrity_{label}.json", original)

    def prepare(self):
        self.integrity("before")
        for universe in self.args.universes:
            self.phase(f"cache.{universe}", self.cli("cache", universe, "baseline", self.cache / universe),
                       self.output / f"cache_{universe}.log", [self.cache / universe / "completion.json"])

    def wait_for_control(self):
        output = self.output / "csi300/baseline"
        if (output / "training_result.json").exists():
            return
        log = output / "train.log"
        if not log.exists():
            raise RuntimeError("The independently launched CSI300 seed0 control is missing")
        self.event("wait for the already-running native CSI300 control")
        while True:
            with log.open("rb") as stream:
                stream.seek(max(0, log.stat().st_size - 65536))
                tail = stream.read().decode(errors="replace")
            if "Results saved to" in tail:
                return
            if "Traceback (most recent call last)" in tail:
                raise RuntimeError(f"Native CSI300 control failed: {log}")
            time.sleep(15)

    def backtest_command(self, universe, run):
        return [sys.executable, "-m", "trace_vq.evaluation.backtest", "--pred_path",
                str(run / "evaluation/test_predictions.pkl"), "--universe", universe,
                "--quote-universe", "all", "--start_time", "2023-01-01", "--end_time", "2025-12-31",
                "--topk", "30", "--drop", "5", "--open_cost", "0.0005", "--close_cost", "0.0015",
                "--min_cost", "0", "--account", "100000000", "--output_dir", str(run / "backtest")]

    def run_model(self, universe, variant):
        run = self.output / universe / variant
        run.mkdir(parents=True, exist_ok=True)
        key = f"{universe}.{variant}"
        with self.market_locks[universe]:
            try:
                if variant == "baseline":
                    command = self.cli("adopt", universe, variant, run)
                    if universe == "csi800":
                        command += ["--native-run-dir", str(ROOT / "artifacts/csi800/s1seed42_s2seed0/stage2")]
                    self.phase(f"{key}.train", command, run / "adopt.log", [run / "training_result.json"])
                else:
                    self.phase(f"{key}.train", self.cli("train", universe, variant, run),
                               run / "train.log", [run / "training_result.json"])
                self.phase(f"{key}.evaluate", self.cli("evaluate", universe, variant, run),
                           run / "evaluate.log", [run / "evaluation/completion.json"])
                self.phase(f"{key}.backtest", self.backtest_command(universe, run), run / "backtest.log",
                           [run / "backtest/portfolio_return.csv", run / "backtest/backtest_config.json"])
                metadata = json.loads((run / "backtest/backtest_config.json").read_text())
                if (metadata["quote_universe"], metadata["strategy"]["topk"], metadata["strategy"]["n_drop"],
                    metadata["exchange"]["open_cost"], metadata["exchange"]["close_cost"], metadata["account"],
                    metadata["start_time"], metadata["end_time"]) != (
                    "all", 30, 5, 0.0005, 0.0015, 100000000.0, "2023-01-01", "2025-12-31"):
                    raise RuntimeError("Backtest settings differ from the fixed common protocol")
                result = json.loads((run / "training_result.json").read_text())
                predictions = pd.read_pickle(run / "evaluation/test_predictions.pkl")
                test_metrics = ranking_metrics(predictions.score.astype("float64"), predictions.label.astype("float64"))
                returns = pd.read_csv(run / "backtest/portfolio_return.csv", index_col=0, parse_dates=True)["return"]
                test_metrics.update(portfolio_metrics(returns))
                pd.Series(test_metrics, name=variant).to_csv(run / "protocol_metrics.csv")
                result.update(test_metrics=test_metrics, standalone_exact=True, backtest_config=metadata,
                              status="complete", ideas=IDEAS[variant], completed=now())
                atomic_json(run / "completion.json", result)
                with self.mutex:
                    self.state["runs"][key] = result
                    self.save()
                    self.summarize()
                self.event(f"RESULT {key}: valid RankIC={result['validation_metrics']['RankIC']:.6f}, "
                           f"test RankIC={test_metrics['RankIC']:.6f}, net Sharpe={test_metrics['Sharpe']:.6f}")
                return result
            except Exception as error:
                failure = {"status": "failed", "universe": universe, "variant": variant,
                           "ideas": IDEAS[variant], "error": str(error), "traceback": traceback.format_exc(), "time": now()}
                atomic_json(run / "failure.json", failure)
                with self.mutex:
                    self.state["runs"][key] = failure
                    self.save()
                    self.summarize()
                self.event(f"FAILED {key}: {error}")
                return failure

    def summarize(self):
        rows = []
        for key, result in self.state["runs"].items():
            row = {"universe": result["universe"], "variant": result["variant"], "seed": 0,
                   "ideas": ",".join(map(str, result.get("ideas", []))), "status": result["status"]}
            if result["status"] == "complete":
                row.update({f"valid_{key}": value for key, value in result["validation_metrics"].items()})
                row.update(result["test_metrics"])
                row.update(best_val_loss=result["best_val_loss"], best_epoch=result.get("best_epoch"),
                           elapsed_seconds=result.get("elapsed_seconds"))
                baseline = self.state["runs"].get(f"{row['universe']}.baseline", {})
                if row["variant"] == "baseline":
                    row["decision"] = "control"
                elif baseline.get("status") == "complete":
                    for metric in ("RankIC", "Sharpe", "AR", "MDD"):
                        row[f"delta_{metric}"] = row[metric] - baseline["test_metrics"][metric]
                    row["delta_valid_RankIC"] = row["valid_RankIC"] - baseline["validation_metrics"]["RankIC"]
                    if row["delta_valid_RankIC"] > 0 and row["delta_RankIC"] > 0 and row["delta_Sharpe"] > 0 and row["delta_MDD"] >= -0.02:
                        row["decision"] = "promising"
                    elif row["delta_valid_RankIC"] <= 0 and row["delta_RankIC"] <= 0 and row["delta_Sharpe"] <= 0:
                        row["decision"] = "underperforming"
                    else:
                        row["decision"] = "mixed"
                else:
                    row["decision"] = "awaiting_control"
            else:
                row.update(decision="technical_failure", error=result.get("error"))
            rows.append(row)
        if rows:
            frame = pd.DataFrame(rows).sort_values(["universe", "variant"])
            frame.to_csv(self.output / "leaderboard.csv", index=False)
            atomic_json(self.output / "decisions.json", rows)

    def combinations(self):
        selected = []
        for universe in self.args.universes:
            runs = self.state["runs"]
            base = runs.get(f"{universe}.baseline", {})
            route = runs.get(f"{universe}.market_route", {})
            if base.get("status") != "complete" or route.get("status") != "complete":
                continue
            threshold = base["validation_metrics"]["RankIC"]
            if route["validation_metrics"]["RankIC"] <= threshold:
                continue
            candidates = []
            for objective, combination in (("ic_loss", "market_route_ic"), ("multihorizon", "market_route_multihorizon")):
                result = runs.get(f"{universe}.{objective}", {})
                if result.get("status") == "complete" and result["validation_metrics"]["RankIC"] > threshold:
                    candidates.append((result["validation_metrics"]["RankIC"], combination))
            if candidates:
                selected.append((universe, max(candidates)[1]))
        atomic_json(self.output / "combination_selection.json", {"based_on": "validation RankIC only", "selected": selected})
        return selected

    def execute(self):
        self.prepare()
        if "csi300" in self.args.universes:
            self.wait_for_control()
        for universe in self.args.universes:
            self.run_model(universe, "baseline")
        tasks = [(universe, variant) for variant in INITIAL_VARIANTS if variant != "baseline"
                 for universe in self.args.universes]
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            futures = [pool.submit(self.run_model, universe, variant) for universe, variant in tasks]
            for future in as_completed(futures):
                future.result()
        for universe, variant in self.combinations():
            self.run_model(universe, variant)
        self.integrity("after")
        data_ok = {}
        for universe in self.args.universes:
            for split in ("train", "valid", "test"):
                metadata = json.loads((self.cache / universe / f"{split}.json").read_text())
                data_ok[f"{universe}.{split}"] = sha256(metadata["identity"]["source"]) == metadata["source_sha256"]
        atomic_json(self.output / "dataset_integrity_after.json", data_ok)
        if not all(data_ok.values()):
            raise RuntimeError("A source dataset changed during the screen")
        failures = [key for key, value in self.state["runs"].items() if value["status"] != "complete"]
        self.state.update(status="complete" if not failures else "completed_with_failures", finished=now(), failed_runs=failures)
        self.save()
        self.summarize()
        self.event(f"screen finished: {self.state['status']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/research/innovation_screen_20261026")
    parser.add_argument("--universes", nargs="+", choices=("csi300", "csi800"), default=["csi300", "csi800"])
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    screen = Screen(args)
    try:
        screen.execute()
    except BaseException as error:
        screen.state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        screen.save()
        raise


if __name__ == "__main__":
    main()
