"""Frozen-plan serial single-seed combinations with exact single-component controls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()
import pandas as pd
from trace_vq.cli.combination_screen import cache_directory, reference_directory
from trace_vq.data.frozen import atomic_json, sha256
from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
from trace_vq.models.combination_screen import COMBINATIONS, COMPONENTS, REFERENCES, PHASES, PARENTS
from research.run_innovation_screen import PROTOCOL


def now():
    return datetime.now(timezone.utc).isoformat()


def identity():
    paths = list((ROOT / "trace_vq").rglob("*.py")) + list((ROOT / "configs").rglob("*.yaml"))
    paths += list((ROOT / "tests").glob("test_*.py")) + list((ROOT / "research").glob("*.py"))
    paths += [ROOT / "AGENTS.md", ROOT / "research/run.sh"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(paths))}


def inputs():
    expected = {}
    for market in ["csi300", "csi800"]:
        for variant in ["baseline", "fsq"]:
            for split in ["train", "valid", "test"]:
                cache = cache_directory(market, variant) / (split + ".npy")
                metadata_path = cache.with_suffix(".json")
                metadata = json.loads(metadata_path.read_text())
                expected[str(cache)] = metadata["cache_sha256"]
                expected[str(metadata_path)] = sha256(metadata_path)
                expected[metadata["identity"]["source"]] = metadata["source_sha256"]
                expected[metadata["identity"]["stage1_checkpoint"]] = metadata["identity"]["stage1_sha256"]
        for variant in REFERENCES:
            folder = reference_directory(market, variant)
            receipt = json.loads((folder / "completion.json").read_text())
            expected[receipt["checkpoint"]] = receipt["checkpoint_sha256"]
            for name in ["completion.json", "effective.config.yaml", "evaluation/valid_predictions.pkl",
                         "evaluation/test_predictions.pkl", "backtest/portfolio_return.csv"]:
                path = folder / name
                expected[str(path)] = sha256(path)
    return expected


class Screen:
    def __init__(self, args):
        self.args, self.output = args, args.output_dir.resolve()
        if not self.output.is_relative_to(ROOT / "artifacts/research"):
            raise ValueError("Explicit research output directory required")
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "screen.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.source = identity()
        protocol = dict(PROTOCOL, stage1_retrained=False,
                        combination_selection="Seven combinations fixed before training; no test-driven tuning")
        self.plan = dict(protocol=protocol, universes=["csi300", "csi800"], phases=PHASES,
                         components=COMPONENTS, combinations=COMBINATIONS, parents=PARENTS,
                         references=REFERENCES, new_stage2_runs=14, new_stage1_runs=0,
                         reference_replays=10, num_workers=args.num_workers, parallel_fits=1,
                         source_sha256=self.source, input_sha256=inputs(),
                         existing_caches_reused_read_only=True, new_large_cache_copies=0,
                         omitted="RankNet and ReLU pairwise are alternative losses; not combined together",
                         retention="Keep all configs, weights, predictions and outcomes; no default adoption",
                         limitations="Single trained seed; previously inspected shared test; temporal intervals exclude training randomness")
        self.plan = json.loads(json.dumps(self.plan))
        path = self.output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != self.plan:
            raise RuntimeError("Frozen plan/source/inputs changed")
        atomic_json(path, self.plan)
        for name in self.source:
            target = self.output / "source_snapshot" / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
        self.path = self.output / "progress.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {
            "started": now(), "phases": {}, "runs": {}, "direction_status": {}}
        self.state.update(status="running", pid=os.getpid(), python=sys.executable)
        self.save()

    def save(self):
        self.state["updated"] = now(); atomic_json(self.path, self.state)

    def event(self, message):
        print(f"[{now()}] {message}", flush=True)

    def phase(self, key, command, log, required=()):
        prior = self.state["phases"].get(key, {})
        if prior.get("status") == "complete" and all(Path(p).is_file() for p in required):
            return
        if identity() != self.source:
            raise RuntimeError("Source changed during the frozen experiment")
        entry = dict(status="running", started=now(), command=command, log=str(log),
                     previous_attempts=prior.get("previous_attempts", []) + ([prior] if prior else []))
        self.state["phases"][key] = entry; self.save()
        log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", WANDB_MODE="disabled")
        self.event("start " + key); start = time.monotonic()
        with log.open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            entry["pid"] = process.pid; self.save(); code = process.wait()
        entry.update(status="complete" if code == 0 else "failed", returncode=code,
                     finished=now(), elapsed_seconds=time.monotonic() - start)
        self.save(); self.event(entry["status"] + " " + key)
        if code or not all(Path(p).is_file() for p in required):
            raise RuntimeError(key + " failed; inspect " + str(log))

    def cli(self, command, market, variant, output):
        return [sys.executable, "-m", "trace_vq.cli.combination_screen", command,
                "--universe", market, "--variant", variant, "--output-dir", str(output),
                "--num-workers", str(self.args.num_workers), "--device", "cuda"]

    def integrity(self, label):
        self.phase("integrity." + label, [sys.executable, "research/verify_baseline.py", "--full"],
                   self.output / f"integrity_{label}.log")
        record = json.loads((ROOT / "research/logs/source-integrity.json").read_text())
        assert record["source_unchanged"] and record["full_hash_check"]
        atomic_json(self.output / f"integrity_{label}.json", record)
        checked = {name: sha256(name) == digest for name, digest in self.plan["input_sha256"].items()}
        if not all(checked.values()):
            raise RuntimeError("Existing data/cache/reference artifact changed")
        atomic_json(self.output / f"input_integrity_{label}.json", dict(status="passed", files=checked))

    def backtest(self, market, output):
        return [sys.executable, "-m", "trace_vq.evaluation.backtest", "--pred_path", str(output / "evaluation/test_predictions.pkl"),
                "--universe", market, "--quote-universe", "all", "--start_time", "2023-01-01", "--end_time", "2025-12-31",
                "--topk", "30", "--drop", "5", "--open_cost", "0.0005", "--close_cost", "0.0015", "--min_cost", "0",
                "--account", "100000000", "--output_dir", str(output / "backtest")]

    def run_model(self, market, variant):
        output = self.output / market / variant; output.mkdir(parents=True, exist_ok=True)
        key = market + "." + variant
        if self.state["runs"].get(key, {}).get("status") == "complete":
            return
        self.phase(key + ".train", self.cli("adopt" if variant in REFERENCES else "train", market, variant, output),
                   output / "train.log", [output / "training_result.json"])
        self.phase(key + ".evaluate", self.cli("evaluate", market, variant, output), output / "evaluate.log",
                   [output / "evaluation/completion.json"])
        self.phase(key + ".backtest", self.backtest(market, output), output / "backtest.log",
                   [output / "backtest/portfolio_return.csv", output / "backtest/backtest_config.json"])
        record = json.loads((output / "training_result.json").read_text())
        config = json.loads((output / "backtest/backtest_config.json").read_text())
        strategy, exchange = config["strategy"], config["exchange"]
        assert config["quote_universe"] == "all" and strategy["topk"] == 30 and strategy["n_drop"] == 5
        assert strategy["risk_degree"] == .95 and config["account"] == 1e8
        assert exchange["open_cost"] == .0005 and exchange["close_cost"] == .0015 and exchange["min_cost"] == 0
        assert exchange["deal_price"] == "close"
        assert (config["start_time"], config["end_time"]) == ("2023-01-01", "2025-12-31")
        prediction = pd.read_pickle(output / "evaluation/test_predictions.pkl")
        metrics = ranking_metrics(prediction.score.astype("float64"), prediction.label.astype("float64"))
        returns = pd.read_csv(output / "backtest/portfolio_return.csv", index_col=0, parse_dates=True)["return"]
        metrics.update(portfolio_metrics(returns))
        record.update(status="complete", standalone_exact=True, test_metrics=metrics, backtest_config=config,
                      replayed_reference=variant in REFERENCES, parents=PARENTS.get(variant, ["baseline"]),
                      primary_reference=PARENTS.get(variant, ["baseline"])[0], completed=now())
        atomic_json(output / "completion.json", record)
        self.state["runs"][key] = record; self.save(); self.summarize()
        self.event(f"RESULT {key}: validRIC={record['validation_metrics']['RankIC']:.6f}, testRIC={metrics['RankIC']:.6f}, AR={metrics['AR']:.6f}, Sharpe={metrics['Sharpe']:.6f}, MDD={metrics['MDD']:.6f}")

    def summarize(self):
        rows = []
        for key, record in self.state["runs"].items():
            if record["status"] != "complete":
                continue
            market, variant = record["universe"], record["variant"]
            row = dict(universe=market, variant=variant, status="complete", seed=0,
                       elapsed_seconds=record.get("elapsed_seconds"), completed_epochs=record.get("completed_epochs"),
                       **record["test_metrics"], **{"valid_" + k: v for k, v in record["validation_metrics"].items()},
                       replayed_reference=variant in REFERENCES, primary_reference=record["primary_reference"],
                       role="replayed_reference" if variant in REFERENCES else "combination")
            epoch = re.search(r"epoch=(\d+)", Path(record["checkpoint"]).name)
            row["best_epoch"] = int(epoch[1]) if epoch else None
            for label, control in [("baseline", "baseline"), ("primary", row["primary_reference"])]:
                parent = self.state["runs"].get(market + "." + control)
                if parent:
                    for metric in ["RankIC", "AR", "Sharpe", "MDD"]:
                        row[f"delta_{label}_{metric}"] = row[metric] - parent["test_metrics"][metric]
                    row[f"delta_{label}_valid_RankIC"] = row["valid_RankIC"] - parent["validation_metrics"]["RankIC"]
            if variant == "baseline":
                row["decision"] = "control"
            elif row.get("delta_baseline_valid_RankIC", -1) > 0 and row.get("delta_baseline_RankIC", -1) > 0 and row.get("delta_baseline_Sharpe", -1) > 0 and row.get("delta_baseline_MDD", -1) >= -.02:
                row["decision"] = "promising"
            else:
                row["decision"] = "mixed" if row.get("delta_baseline_RankIC", -1) > 0 or row.get("delta_baseline_Sharpe", -1) > 0 else "underperforming"
            rows.append(row)
        if rows:
            table = pd.DataFrame(rows).sort_values(["universe", "variant"])
            target = self.output / "leaderboard.csv"; temp = target.with_suffix(".partial.csv")
            table.to_csv(temp, index=False); temp.replace(target)

    def execute(self):
        gpu = json.loads((self.output / "gpu_validation.json").read_text())
        smoke = json.loads((self.output / "smoke_validation.json").read_text())
        assert gpu["status"] == "passed" and len(gpu["checks"]) == 24
        assert smoke["status"] == "passed" and smoke["technical_smoke_fits"] == 4
        self.integrity("before")
        for variant in REFERENCES:
            for market in ["csi300", "csi800"]:
                self.run_model(market, variant)
        for direction, variants in PHASES:
            self.state["current_direction"] = direction; self.state["direction_status"][direction] = "running"; self.save()
            for variant in variants:
                for market in ["csi300", "csi800"]:
                    self.run_model(market, variant)
            self.state["direction_status"][direction] = "complete"; self.save()
        self.integrity("after")
        assert identity() == self.source
        self.state.update(status="complete", finished=now(), failed_runs=[]); self.save(); self.summarize()
        self.event("All fourteen new combinations and ten reference replays complete")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/research/combination_screen_20261029")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args(); screen = Screen(args)
    try:
        screen.execute()
    except BaseException as error:
        screen.state.update(status="failed", error=str(error), traceback=traceback.format_exc()); screen.save()
        screen.event("Combination screen failed: " + str(error)); raise


if __name__ == "__main__":
    main()
