"""Sequential, frozen-plan Stage2 mechanism screen with exact archived controls."""
import argparse
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()
import pandas as pd
from trace_vq.data.frozen import atomic_json, sha256
from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
from trace_vq.models.mechanism_screen import MECHANISM_VARIANTS, PHASES, REFERENCES, PRIMARY
from research.run_innovation_screen import PROTOCOL


def now():
    return datetime.now(timezone.utc).isoformat()


def identity():
    paths = list((ROOT / "trace_vq").rglob("*.py")) + list((ROOT / "configs").rglob("*.yaml"))
    paths += list((ROOT / "tests").glob("test_*.py"))
    paths += [Path(__file__).resolve(), ROOT / "research/run.sh", ROOT / "research/run_next_screen.py",
              ROOT / "research/run_innovation_screen.py", ROOT / "AGENTS.md"]
    return {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(paths))}


class Screen:
    def __init__(self, args):
        self.args, self.output = args, args.output_dir.resolve()
        if not self.output.is_relative_to(ROOT / "artifacts/research"):
            raise ValueError("All outputs must remain in research artifacts")
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "screen.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.source = identity()
        context_files = [self.output / "context" / section / market / name
                         for section, name in [("history", "positions.npy"), ("history", "completion.json")]
                         for market in ("csi300", "csi800")]
        context_files += [self.output / "context/states" / (market + suffix)
                          for market in ("csi300", "csi800") for suffix in (".json", "_calendar.csv")]
        self.context_identity = {str(path.relative_to(self.output)): sha256(path) for path in context_files}
        protocol = dict(PROTOCOL)
        protocol.update(stage1_retrained=False, combination_selection="No adaptive combinations or test-driven parameter changes",
                        historical_behavior="Original Stage1 and Stage2 behavior unchanged in references")
        plan = {"protocol": protocol, "universes": ["csi300", "csi800"], "phases": PHASES,
                "variants": MECHANISM_VARIANTS, "references": REFERENCES, "primary_comparisons": PRIMARY,
                "additional_comparisons": {"peer_code": ["peer_self"], "beta_smooth": ["history_gru"]},
                "new_stage2_runs": 14, "new_stage1_runs": 0, "replayed_references": 6,
                "workers": 1, "data_workers": args.num_workers, "source_sha256": self.source,
                "context_sha256": self.context_identity,
                "previous_dir": str(args.previous_dir.resolve()),
                "state_balance": {"trend_quantile": .5, "volatility_quantile": .8, "weight_bounds": [.5, 2.],
                                  "fit_period": "training dates only; mean training loss weight1"},
                "relation_semantics": "Stage1 already has global cross-asset attention. Stage2 mean of continuous h, excluding self, within exact native code IDs.",
                "retention": "Preserve all outcomes, controls, weights and failed attempts; do not adopt defaults automatically",
                "selection_rule": "Baseline point criterion: valid/test RankIC and net Sharpe improve; MDD deterioration <=2pp. Also report matched controls.",
                "interpretation": "Exploratory shared-test single-seed comparisons; no novelty or robustness guarantee"}
        plan = json.loads(json.dumps(plan))
        path = self.output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise RuntimeError("Frozen source/plan changed; choose a new run directory")
        atomic_json(path, plan)
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
        self.state["updated"] = now()
        atomic_json(self.path, self.state)

    def event(self, message):
        print(f"[{now()}] {message}", flush=True)
        with (self.output / "events.jsonl").open("a") as stream:
            stream.write(json.dumps({"time": now(), "message": message}) + "\n")

    def phase(self, key, command, log, required=()):
        prior = self.state["phases"].get(key, {})
        if prior.get("status") == "complete" and all(Path(p).is_file() for p in required):
            return
        if identity() != self.source:
            raise RuntimeError("Source changed during execution")
        entry = {"status": "running", "started": now(), "command": command, "log": str(log),
                 "previous_attempts": prior.get("previous_attempts", []) + ([prior] if prior else [])}
        self.state["phases"][key] = entry; self.save()
        log = Path(log); log.parent.mkdir(parents=True, exist_ok=True)
        self.event("start " + key)
        environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
                           OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", WANDB_MODE="disabled")
        start = time.monotonic()
        with log.open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
            entry["pid"] = process.pid; self.save()
            code = process.wait()
        entry.update(status="complete" if code == 0 else "failed", returncode=code,
                     finished=now(), elapsed_seconds=time.monotonic() - start)
        self.save(); self.event(entry["status"] + " " + key)
        if code or not all(Path(p).is_file() for p in required):
            raise RuntimeError(key + " failed; see " + str(log))

    def cli(self, command, universe, variant, output):
        return [sys.executable, "-m", "trace_vq.cli.mechanism_screen", command,
                "--universe", universe, "--variant", variant, "--output-dir", str(output),
                "--cache-root", str(self.output / "cache"), "--context-root", str(self.output / "context"),
                "--num-workers", str(self.args.num_workers), "--device", "cuda"]

    def integrity(self, label):
        self.phase("integrity." + label, [sys.executable, "research/verify_baseline.py", "--full"],
                   self.output / f"integrity_{label}.log")
        record = json.loads((ROOT / "research/logs/source-integrity.json").read_text())
        if not record["source_unchanged"] or not record["full_hash_check"]:
            raise RuntimeError("Original TRACE-VQ changed")
        atomic_json(self.output / f"integrity_{label}.json", record)

    def audit_inputs(self, label):
        records, history = {}, {}
        for market in ("csi300", "csi800"):
            for split in ("train", "valid", "test"):
                metadata = json.loads((self.output / "cache" / market / f"{split}.json").read_text())
                cache = self.output / "cache" / market / f"{split}.npy"
                original = self.args.previous_dir / "cache/legacy" / market / f"{split}.npy"
                row = {"source_unchanged": sha256(metadata["identity"]["source"]) == metadata["source_sha256"],
                       "cache_unchanged": sha256(cache) == metadata["cache_sha256"],
                       "copied_cache_matches_previous": sha256(original) == metadata["cache_sha256"],
                       "stage1_unchanged": sha256(metadata["identity"]["stage1_checkpoint"]) == metadata["identity"]["stage1_sha256"]}
                if not all(row.values()):
                    raise RuntimeError("Input identity changed: " + market + "." + split)
                records[market + "." + split] = row
            current = self.output / "context/history" / market / "positions.npy"
            previous = self.args.previous_dir / "history" / market / "positions.npy"
            history[market] = sha256(current) == sha256(previous)
            if not history[market]:
                raise RuntimeError("Causal history positions changed")
        context = {name: sha256(self.output / name) == digest for name, digest in self.context_identity.items()}
        if not all(context.values()):
            raise RuntimeError("Frozen context/statistics changed")
        atomic_json(self.output / f"input_integrity_{label}.json", {"datasets_and_caches": records,
                                                               "history_positions": history, "context_unchanged": context})

    def backtest(self, market, run):
        return [sys.executable, "-m", "trace_vq.evaluation.backtest", "--pred_path", str(run / "evaluation/test_predictions.pkl"),
                "--universe", market, "--quote-universe", "all", "--start_time", "2023-01-01", "--end_time", "2025-12-31",
                "--topk", "30", "--drop", "5", "--open_cost", "0.0005", "--close_cost", "0.0015", "--min_cost", "0",
                "--account", "100000000", "--output_dir", str(run / "backtest")]

    def run_model(self, market, variant):
        directory = self.output / market / variant
        directory.mkdir(parents=True, exist_ok=True)
        key = market + "." + variant
        if self.state["runs"].get(key, {}).get("status") == "complete" and (directory / "completion.json").exists():
            return
        try:
            command = self.cli("train", market, variant, directory)
            if variant in REFERENCES:
                command = self.cli("adopt", market, variant, directory) + ["--reference-dir", str(self.args.previous_dir / market / variant)]
            self.phase(key + ".train", command, directory / "train.log", [directory / "training_result.json"])
            self.phase(key + ".evaluate", self.cli("evaluate", market, variant, directory), directory / "evaluate.log",
                       [directory / "evaluation/completion.json"])
            self.phase(key + ".backtest", self.backtest(market, directory), directory / "backtest.log",
                       [directory / "backtest/portfolio_return.csv", directory / "backtest/backtest_config.json"])
            record = json.loads((directory / "training_result.json").read_text())
            settings = json.loads((directory / "backtest/backtest_config.json").read_text())
            strategy, exchange = settings["strategy"], settings["exchange"]
            assert settings["quote_universe"] == "all" and strategy["topk"] == 30 and strategy["n_drop"] == 5
            assert strategy["risk_degree"] == .95 and exchange["open_cost"] == .0005 and exchange["close_cost"] == .0015
            assert exchange["min_cost"] == 0 and settings["account"] == 1e8
            assert (settings["start_time"], settings["end_time"]) == ("2023-01-01", "2025-12-31")
            predictions = pd.read_pickle(directory / "evaluation/test_predictions.pkl")
            metrics = ranking_metrics(predictions.score.astype("float64"), predictions.label.astype("float64"))
            returns = pd.read_csv(directory / "backtest/portfolio_return.csv", index_col=0, parse_dates=True)["return"]
            metrics.update(portfolio_metrics(returns))
            record.update(status="complete", standalone_exact=True, test_metrics=metrics, backtest_config=settings,
                          replayed_reference=variant in REFERENCES, primary_reference=PRIMARY[variant], completed=now())
            atomic_json(directory / "completion.json", record)
            self.state["runs"][key] = record; self.save(); self.summarize()
            self.event(f'RESULT {key}: valid RIC={record["validation_metrics"]["RankIC"]:.6f}, '
                       f'test RIC={metrics["RankIC"]:.6f}, net Sharpe={metrics["Sharpe"]:.6f}, MDD={metrics["MDD"]:.6f}')
        except BaseException as error:
            failure = {"status": "technical_failure", "universe": market, "variant": variant,
                       "error": str(error), "traceback": traceback.format_exc(), "finished": now()}
            atomic_json(directory / "failure.json", failure)
            self.state["runs"][key] = failure; self.save(); self.summarize()
            raise

    def summarize(self):
        rows = []
        for record in self.state["runs"].values():
            row = {key: record.get(key) for key in ("universe", "variant", "status", "elapsed_seconds", "completed_epochs")}
            if row["status"] == "complete":
                row.update(seed=0, **record["test_metrics"])
                row.update({"valid_" + key: value for key, value in record["validation_metrics"].items()})
                row.update(primary_reference=record["primary_reference"], replayed_reference=record["replayed_reference"])
                epoch = re.search(r"epoch=(\d+)", Path(record["checkpoint"]).name)
                row["best_epoch"] = int(epoch[1]) if epoch else None
                for label, reference in [("baseline", "baseline"), ("primary", record["primary_reference"])]:
                    control = self.state["runs"].get(row["universe"] + "." + reference, {})
                    if control.get("status") == "complete":
                        for metric in ("RankIC", "Sharpe", "AR", "MDD"):
                            row[f"delta_{label}_{metric}"] = row[metric] - control["test_metrics"][metric]
                        row[f"delta_{label}_valid_RankIC"] = row["valid_RankIC"] - control["validation_metrics"]["RankIC"]
                if row["variant"] in REFERENCES:
                    row["role"] = "replayed_reference"
                elif row["variant"] == "history_mismatched":
                    row["role"] = "negative_control"
                elif row["variant"] in ("history_continuous", "peer_self", "peer_market"):
                    row["role"] = "competitive_control"
                else:
                    row["role"] = "new_direction"
                if row["variant"] == "baseline":
                    row["decision"] = "control"
                elif row.get("delta_baseline_RankIC", -1) > 0 and row.get("delta_baseline_valid_RankIC", -1) > 0 and \
                     row.get("delta_baseline_Sharpe", -1) > 0 and row.get("delta_baseline_MDD", -1) >= -.02:
                    row["decision"] = "promising"
                else:
                    row["decision"] = "mixed" if row.get("delta_baseline_Sharpe", -1) > 0 or row.get("delta_baseline_RankIC", -1) > 0 else "underperforming"
            rows.append(row)
        if rows:
            table = pd.DataFrame(rows).sort_values(["universe", "variant"])
            temporary = self.output / "leaderboard.partial.csv"
            table.to_csv(temporary, index=False); temporary.replace(self.output / "leaderboard.csv")
            atomic_json(self.output / "decisions.json", rows)

    def execute(self):
        proof = json.loads((self.output / "gpu_validation.json").read_text())
        smoke = json.loads((self.output / "smoke_validation.json").read_text())
        if proof["status"] != "passed" or not proof["zero_control_loss_and_gradients_exact"]:
            raise RuntimeError("Real-day GPU and exact-control checks are required")
        if smoke["status"] != "passed" or smoke["technical_smoke_fits"] != 4:
            raise RuntimeError("Checkpoint/callback smoke checks are required")
        self.integrity("before")
        self.audit_inputs("before")
        for market in ("csi300", "csi800"):
            self.phase("states." + market, self.cli("prepare-state", market, "baseline", self.output / "preparation" / market),
                       self.output / f"states_{market}.log", [self.output / "context/states" / f"{market}.json"])
        for variant in REFERENCES:
            for market in ("csi300", "csi800"):
                self.run_model(market, variant)
        for direction, variants in PHASES:
            self.state["current_direction"] = direction
            self.state["direction_status"][direction] = "running"; self.save()
            for variant in variants:
                for market in ("csi300", "csi800"):
                    self.run_model(market, variant)
            self.state["direction_status"][direction] = "complete"; self.save()
        self.integrity("after")
        self.audit_inputs("after")
        if identity() != self.source:
            raise RuntimeError("Frozen source changed before finalization")
        self.state.update(status="complete", finished=now(), failed_runs=[])
        self.save(); self.summarize(); self.event("All fourteen new fits and six reference replays complete")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts/research/mechanism_screen_20261028")
    parser.add_argument("--previous-dir", type=Path, default=ROOT / "artifacts/research/next_directions_20261027")
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    screen = Screen(args)
    try:
        screen.execute()
    except BaseException as error:
        screen.state.update(status="failed", error=str(error), traceback=traceback.format_exc())
        screen.save(); screen.event("screen failed: " + str(error)); raise


if __name__ == "__main__":
    main()
