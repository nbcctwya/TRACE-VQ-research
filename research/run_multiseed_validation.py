"""Serial seed1-4 validation; reuse all historical baseline and seed0 models."""
import argparse
import copy
import csv
import fcntl
import json
from pathlib import Path
import shutil
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
from omegaconf import OmegaConf
from research.run_combination_screen import Screen as PreviousScreen, identity, now
from research.run_innovation_screen import PROTOCOL
from trace_vq.cli.combination_screen import cache_directory
from trace_vq.cli.multiseed_validation import VARIANTS, SEEDS, tasks
from trace_vq.data.frozen import atomic_json, sha256
from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics

SINGLE = ROOT / "artifacts/research/combination_screen_20261029"
METHODS = ("baseline", *VARIANTS)


def local(path):
    path = Path(path)
    old = ROOT.parent / "TRACE-VQ"
    if path.is_absolute() and path.is_relative_to(old):
        path = ROOT / path.relative_to(old)
    elif not path.is_absolute():
        path = ROOT / path
    path = path.resolve()
    if not path.is_relative_to(ROOT) or not path.exists():
        raise ValueError("A complete reference inside this research copy is required: " + str(path))
    return path


def reference_spec(market, variant, seed):
    if seed == 0:
        folder = SINGLE / market / variant
        record = json.loads((folder / "completion.json").read_text())
        return dict(universe=market, variant=variant, seed=seed, record=record,
            checkpoint=local(record["checkpoint"]), checkpoint_sha256=record["checkpoint_sha256"],
            config=local(record["config"]), prediction=folder / "evaluation/test_predictions.pkl",
            valid_prediction=folder / "evaluation/valid_predictions.pkl", backtest=folder / "backtest",
            needs_backtest=False, evidence=[folder / "completion.json", folder / "evaluation/completion.json"])
    if variant != "baseline":
        raise ValueError("Only historical baseline models have preexisting seeds1-4")
    if market == "csi300":
        folder = ROOT / f"artifacts/034/phase3/baseline-010-019-025-034-r2/seed{seed}"
        evidence = json.loads((folder / "training_evidence.json").read_text())
        assert evidence["actual_training_seed"] == seed and evidence["stage1_verified"]
        with (ROOT / "results/metrics/seed_metrics.csv").open() as stream:
            rows = [x for x in csv.DictReader(stream) if x["model"] == "034" and int(x["seed"]) == seed]
        assert len(rows) == 1
        row = rows[0]
        record = dict(stage1_seed=42, stage2_seed=seed, stage1_retrained=False,
            validation_metrics={}, best_val_loss=None, reference_stage1_sha256=evidence["stage1_sha256"],
            legacy_test_metrics={k: float(row[k]) for k in ("IC", "ICIR", "RankIC", "RankICIR", "AR", "Sharpe", "MDD")})
        return dict(universe=market, variant=variant, seed=seed, record=record,
            checkpoint=local(folder / evidence["best_checkpoint"]), checkpoint_sha256=evidence["checkpoint_sha256"],
            config=folder / f"seed{seed}.source_config.json", prediction=local(row["pred_path_or_ckpt_path"]),
            valid_prediction=None, backtest=None, needs_backtest=True,
            evidence=[folder / "training_evidence.json", folder / "result_manifest.json", ROOT / "results/metrics/seed_metrics.csv"])
    folder = ROOT / f"artifacts/csi800/s1seed42_s2seed{seed}"
    receipt = json.loads((folder / "completion_report.json").read_text())
    assert receipt["status"] == "complete" and receipt["seed"] == seed
    assert receipt["standalone_predictions_exact"] and receipt["seed0_configuration_equal"]
    record = dict(stage1_seed=42, stage2_seed=seed, stage1_retrained=False, validation_metrics={},
        reference_stage1_sha256=receipt["stage1_sha256"], legacy_test_metrics=receipt["protocol_metrics"])
    return dict(universe=market, variant=variant, seed=seed, record=record,
        checkpoint=local(receipt["stage2_checkpoint"]), checkpoint_sha256=sha256(local(receipt["stage2_checkpoint"])),
        config=local(receipt["effective_config"]), prediction=local(receipt["test_prediction"]),
        valid_prediction=None, backtest=folder / "backtest", needs_backtest=False,
        evidence=[folder / "completion_report.json"])


def references():
    return [reference_spec(market, variant, seed) for market in ("csi300", "csi800")
            for variant in METHODS for seed in range(5) if variant == "baseline" or seed == 0]


def frozen_inputs():
    result = {}
    native_stage1 = {}
    def add(path, digest=None):
        path = local(path)
        value = digest or sha256(path)
        if str(path) in result and result[str(path)] != value:
            raise RuntimeError("Conflicting input identity")
        result[str(path)] = value
    for market in ("csi300", "csi800"):
        for variant in ("ranknet", "fsq"):
            folder = cache_directory(market, variant)
            for split in ("train", "valid", "test"):
                path = folder / (split + ".npy")
                metadata_path = path.with_suffix(".json")
                metadata = json.loads(metadata_path.read_text())
                if variant == 'ranknet':
                    native_stage1[market] = metadata['identity']['stage1_sha256']
                add(path, metadata["cache_sha256"]); add(metadata_path)
                add(metadata["identity"]["source"], metadata["source_sha256"])
                add(metadata["identity"]["stage1_checkpoint"], metadata["identity"]["stage1_sha256"])
    for spec in references():
        if spec['record'].get('reference_stage1_sha256'):
            assert spec['record']['reference_stage1_sha256'] == native_stage1[spec['universe']]
        add(spec["checkpoint"], spec["checkpoint_sha256"]); add(spec["config"]); add(spec["prediction"])
        if spec["valid_prediction"]: add(spec["valid_prediction"])
        for path in spec["evidence"]: add(path)
        if spec["backtest"]:
            for name in ["portfolio_return.csv", "backtest_config.json", "qlib_report.csv"]:
                add(spec["backtest"] / name)
    return result


def check_backtest(config, market):
    s, e = config["strategy"], config["exchange"]
    assert config["universe"] == market and config["quote_universe"] == "all"
    assert config["account"] == 1e8 and s["topk"] == 30 and s["n_drop"] == 5 and s["risk_degree"] == .95
    assert e["open_cost"] == .0005 and e["close_cost"] == .0015 and e["min_cost"] == 0
    assert e["deal_price"] == "close" and e["limit_threshold"] is None
    assert (config["start_time"], config["end_time"]) == ("2023-01-01", "2025-12-31")


def link(source, target):
    source = local(source); target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() != source: raise RuntimeError("Reference link changed")
    elif target.exists():
        raise FileExistsError("Refusing to overwrite reference link: " + str(target))
    else:
        target.symlink_to(source, target_is_directory=source.is_dir())


class Screen(PreviousScreen):
    def __init__(self, args):
        self.args, self.output = args, args.output_dir.resolve()
        if not self.output.is_relative_to(ROOT / "artifacts/research"):
            raise ValueError("Explicit research output required")
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "screen.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.source = identity()
        protocol = dict(PROTOCOL, stage1_retrained=False, stage2_seeds=[0, 1, 2, 3, 4],
            ordering="Finish FSQ+RankNet seeds1-4 on both datasets, then FSQ, then RankNet",
            selection="All24 prespecified fits completed regardless of intermediate test outcomes")
        protocol.pop('stage2_seed', None)
        self.plan = dict(protocol=protocol, tasks=tasks(), methods=METHODS,
            source_sha256=self.source, input_sha256=frozen_inputs(), new_stage2_runs=24,
            new_stage1_runs=0, baseline_training_runs=0, reused_reference_results=16,
            baseline_backtests_recomputed=4, total_market_seed_results=40,
            num_workers=args.num_workers, parallel_fits=1, new_large_cache_copies=0,
            limitations=["Stage1seed42 fixed; repetitions measure Stage2 randomness only.",
                         "Previously inspected shared test; all seeds reported without selection.",
                         "Historical baseline predictions preserved; CSI300 seeds1-4 rebacktested with all quotes."])
        self.plan = json.loads(json.dumps(self.plan))
        path = self.output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != self.plan:
            raise RuntimeError("Frozen multiseed plan changed")
        atomic_json(path, self.plan)
        for name in self.source:
            target = self.output / "source_snapshot" / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / name, target)
        self.path = self.output / "progress.json"
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(started=now(), phases={}, runs={}, direction_status={})
        self.state.update(status="running", pid=__import__('os').getpid(), python=sys.executable)
        self.save()

    def directory(self, market, variant, seed):
        return self.output / market / variant / f"seed{seed}"

    def cli(self, command, market, variant, seed, output):
        return [sys.executable, "-m", "trace_vq.cli.multiseed_validation", command,
                "--universe", market, "--variant", variant, "--seed", str(seed),
                "--output-dir", str(output), "--num-workers", str(self.args.num_workers), "--device", "cuda"]

    def record_metrics(self, record, output):
        market, seed = record["universe"], record["stage2_seed"]
        prediction = pd.read_pickle(output / "evaluation/test_predictions.pkl")
        baseline = self.state["runs"].get(f"{market}.baseline.seed{seed}")
        if baseline and record["variant"] != "baseline":
            other = pd.read_pickle(self.directory(market, "baseline", seed) / "evaluation/test_predictions.pkl")
            assert prediction.index.equals(other.index)
            pd.testing.assert_series_equal(prediction.label, other.label, check_exact=True)
        metrics = ranking_metrics(prediction.score.astype('float64'), prediction.label.astype('float64'))
        config = json.loads((output / "backtest/backtest_config.json").read_text()); check_backtest(config, market)
        returns = pd.read_csv(output / "backtest/portfolio_return.csv", index_col=0, parse_dates=True)["return"]
        metrics.update(portfolio_metrics(returns)); assert metrics["num_test_days"] == 727
        record.update(status="complete", test_metrics=metrics, backtest_config=config, completed=now())
        atomic_json(output / "completion.json", record)
        key = f"{market}.{record['variant']}.seed{seed}"
        self.state["runs"][key] = record; self.save(); self.summarize()
        self.event(f"RESULT {key}: RankIC={metrics['RankIC']:.6f}, AR={metrics['AR']:.6f}, Sharpe={metrics['Sharpe']:.6f}, MDD={metrics['MDD']:.6f}")

    def adopt_reference(self, spec):
        market, variant, seed = spec["universe"], spec["variant"], spec["seed"]
        key = f"{market}.{variant}.seed{seed}"
        if self.state["runs"].get(key, {}).get("status") == "complete": return
        output = self.directory(market, variant, seed); output.mkdir(parents=True, exist_ok=True)
        assert sha256(spec["checkpoint"]) == spec["checkpoint_sha256"]
        link(spec["prediction"], output / "evaluation/test_predictions.pkl")
        if spec["valid_prediction"]:
            link(spec["valid_prediction"], output / "evaluation/valid_predictions.pkl")
        if spec["needs_backtest"]:
            self.phase(key + ".backtest", self.backtest(market, output), output / "backtest.log",
                       [output / "backtest/backtest_config.json", output / "backtest/portfolio_return.csv"])
        else:
            link(spec["backtest"], output / "backtest")
        record = copy.deepcopy(spec["record"])
        record.update(universe=market, variant=variant, stage1_seed=42, stage2_seed=seed,
            checkpoint=str(spec["checkpoint"]), checkpoint_sha256=spec["checkpoint_sha256"], config=str(spec["config"]),
            reused_reference=True, new_training=False, stage1_retrained=False,
            prediction_source=str(spec["prediction"]), prediction_sha256=sha256(spec["prediction"]),
            backtest_recomputed=spec["needs_backtest"], historical_result_preserved=True)
        self.record_metrics(record, output)

    def run_model(self, market, variant, seed):
        key = f"{market}.{variant}.seed{seed}"
        if self.state["runs"].get(key, {}).get("status") == "complete": return
        output = self.directory(market, variant, seed)
        self.phase(key + ".train", self.cli("train", market, variant, seed, output), output / "train.log",
                   [output / "training_result.json"])
        self.phase(key + ".evaluate", self.cli("evaluate", market, variant, seed, output), output / "evaluate.log",
                   [output / "evaluation/completion.json"])
        self.phase(key + ".backtest", self.backtest(market, output), output / "backtest.log",
                   [output / "backtest/portfolio_return.csv", output / "backtest/backtest_config.json"])
        record = json.loads((output / "training_result.json").read_text())
        assert record["stage2_seed"] == seed and record["stage1_unchanged"]
        assert json.loads((output / "evaluation/completion.json").read_text())["standalone_exact"]
        record.update(new_training=True, reused_reference=False, standalone_exact=True)
        self.record_metrics(record, output)

    def summarize(self):
        rows = []
        for record in self.state["runs"].values():
            row = dict(universe=record["universe"], variant=record["variant"], seed=record["stage2_seed"],
                status=record["status"], reused_reference=record["reused_reference"],
                elapsed_seconds=record.get("elapsed_seconds"), completed_epochs=record.get("completed_epochs"),
                run_dir=str(self.directory(record["universe"], record["variant"], record["stage2_seed"])),
                **record["test_metrics"], **{"valid_" + k: v for k, v in record.get("validation_metrics", {}).items()})
            for parent in ["baseline", "fsq", "ranknet"]:
                other = self.state["runs"].get(f"{row['universe']}.{parent}.seed{row['seed']}")
                if other:
                    for metric in ["IC", "RankIC", "AR", "Sharpe", "MDD"]:
                        row[f"delta_{parent}_{metric}"] = row[metric] - other["test_metrics"][metric]
            rows.append(row)
        if not rows: return
        frame = pd.DataFrame(rows).sort_values(["universe", "variant", "seed"])
        target = self.output / "seed_metrics.csv"; temporary = target.with_suffix('.partial.csv')
        frame.to_csv(temporary, index=False); temporary.replace(target)
        metrics = ["IC", "ICIR", "RankIC", "RankICIR", "AR", "STD", "Sharpe", "MDD", "Sortino", "Calmar"]
        aggregate = frame.groupby(["universe", "variant"])[metrics].agg(['count', 'mean', 'std'])
        aggregate.columns = ['_'.join(x) for x in aggregate.columns]
        aggregate.reset_index().to_csv(self.output / "mean_std.csv", index=False)

    def execute(self):
        gpu = json.loads((self.output / "gpu_validation.json").read_text())
        smoke = json.loads((self.output / "smoke_validation.json").read_text())
        assert gpu["status"] == 'passed' and len(gpu["checks"]) == 24
        assert smoke["status"] == 'passed' and smoke["technical_smoke_fits"] == 3
        self.integrity("before")
        self.state['current_direction'] = 'existing_references'; self.save()
        for spec in references(): self.adopt_reference(spec)
        assert len(self.state['runs']) >= 16
        for variant in VARIANTS:
            self.state['current_direction'] = variant; self.state['direction_status'][variant] = 'running'; self.save()
            for selected, market, seed in tasks():
                if selected == variant: self.run_model(market, variant, seed)
            self.state['direction_status'][variant] = 'complete'; self.save()
        self.integrity("after")
        assert identity() == self.source and len(self.state['runs']) == 40
        assert sum(not r['reused_reference'] for r in self.state['runs'].values()) == 24
        self.state.update(status='complete', finished=now()); self.save(); self.summarize()
        self.event('All24 new fits and16 reused reference results complete; no baseline or Stage1 retraining')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'artifacts/research/multiseed_validation_20261030')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args(); screen = Screen(args)
    try:
        screen.execute()
    except BaseException as error:
        screen.state.update(status='failed', error=str(error), traceback=traceback.format_exc()); screen.save()
        screen.event('Multiseed validation failed: ' + str(error)); raise


if __name__ == '__main__':
    main()
