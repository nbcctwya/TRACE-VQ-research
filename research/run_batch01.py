"""Run checklist batch 01 only; preserve all prior artifacts and Stage 1 weights."""
import argparse
import csv
import fcntl
import gc
import inspect
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
from omegaconf import OmegaConf

from trace_vq.cli.batch01 import make_config, reference_directory, cache_directory
from trace_vq.cli.screening import now
from trace_vq.data.frozen import atomic_json, sha256
from trace_vq.evaluation.ensemble import build_ensemble
from trace_vq.evaluation.protocol_metrics import ranking_metrics, portfolio_metrics
from trace_vq.models.batch01 import VARIANTS

METRICS = ("IC", "RankIC", "AR", "Sharpe", "MDD", "turnover")
SOURCES = ["trace_vq/models/batch01.py", "trace_vq/cli/batch01.py",
           "trace_vq/models/prediction/temporal_transformer.py", "research/run_batch01.py"]


def tasks():
    return [("csi800", name) for name in VARIANTS if name.startswith("tune800_")] + [
        (market, name) for name in VARIANTS if not name.startswith("tune800_")
        for market in ("csi300", "csi800")]


def result_metrics(folder):
    frame = pd.read_pickle(folder / "evaluation/test_predictions.pkl")
    result = ranking_metrics(frame.score.astype("float64"), frame.label.astype("float64"))
    returns = pd.read_csv(folder / "backtest/portfolio_return.csv", index_col=0)["return"]
    result.update(portfolio_metrics(returns))
    report = pd.read_csv(folder / "backtest/qlib_report.csv", index_col=0)
    result["turnover"] = float(report["turnover"].mean())
    return result


def validate_backtest(folder, market):
    record = json.loads((folder / "backtest/backtest_config.json").read_text())
    assert record["universe"] == market and record["quote_universe"] == "all"
    strategy, exchange = record["strategy"], record["exchange"]
    assert (strategy["topk"], strategy["n_drop"], strategy["risk_degree"]) == (30, 5, 0.95)
    assert (exchange["open_cost"], exchange["close_cost"], exchange["min_cost"]) == (0.0005, 0.0015, 0)
    assert exchange["deal_price"] == "close" and exchange["limit_threshold"] is None
    assert record["start_time"] == "2023-01-01" and record["end_time"] == "2025-12-31"
    return record


class Batch:
    def __init__(self, args):
        self.args, self.output = args, args.output_dir.resolve()
        if not self.output.is_relative_to(ROOT / "artifacts/research"):
            raise ValueError("Explicit research artifact root required")
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "batch.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.state_path = self.output / "progress.json"
        self.source = {name: sha256(ROOT / name) for name in SOURCES}
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "status": "running", "started": now(), "runs": {}, "phases": {}, "checks": {},
            "planned_new_stage2": 12, "planned_new_stage1": 0, "source_sha256": self.source}
        if self.state.get("source_sha256") != self.source:
            raise RuntimeError("Batch implementation changed after launch; inspect before resuming")
        self.state.update(pid=os.getpid(), status="running", current_phase=None)
        self.save()

    def save(self):
        self.state["updated"] = now()
        atomic_json(self.state_path, self.state)

    def event(self, message):
        line = f"{now()} {message}"
        print(line, flush=True)
        with (self.output / "events.log").open("a") as stream:
            stream.write(line + "\n")

    def phase(self, key, command, log, required=()):
        previous = self.state["phases"].get(key, {})
        if previous.get("status") == "complete" and all(p.exists() for p in required):
            return previous["seconds"]
        if any(p.name == "training_result.json" and p.exists() for p in required):
            record = json.loads(next(p for p in required if p.name == "training_result.json").read_text())
            seconds = record.get("train_phase_seconds", record.get("training_seconds"))
            self.state["phases"][key] = dict(status="complete", seconds=seconds, adopted_completed_phase=True)
            self.save()
            return seconds
        log.parent.mkdir(parents=True, exist_ok=True)
        self.event("START " + key)
        self.state["current_phase"] = key
        start = time.perf_counter()
        with log.open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            self.state["phases"][key] = dict(status="running", started=now(), pid=process.pid,
                                                    command=command, log=str(log))
            self.save()
            code = process.wait()
        elapsed = time.perf_counter() - start
        self.state["phases"][key].update(status="complete" if code == 0 else "failed", seconds=elapsed,
                                            returncode=code, finished=now())
        self.save()
        if code != 0:
            raise RuntimeError(f"{key} exited {code}; see {log}")
        if not all(p.exists() for p in required):
            raise RuntimeError("Missing completion artifacts: " + key)
        self.event(f"DONE {key}: {elapsed:.1f}s")
        return elapsed

    def check_items(self):
        from trace_vq.models.prediction.heads import LatentValueHead
        start = time.perf_counter()
        layers = [type(layer).__name__ for layer in LatentValueHead(128, 128).head]
        assert layers == ["Linear", "LayerNorm", "GELU", "Linear"]
        self.state["checks"]["G2"] = dict(status="already_implemented", training_skipped=True,
                                           layers=layers, seconds=time.perf_counter() - start,
                                           source="trace_vq/models/prediction/heads.py")
        checks = self.output / "checks"
        checks.mkdir(exist_ok=True)
        atomic_json(checks / "G2.json", self.state["checks"]["G2"])
        start = time.perf_counter()
        import qlib
        from qlib.config import C
        from qlib.constant import REG_CN
        from qlib.backtest.exchange import Exchange
        from trace_vq.evaluation.backtest import _resolve_universe
        provider, _, _ = _resolve_universe("csi800", None)
        qlib.init(provider_uri=provider, region=REG_CN,
                  exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                               "kwargs": {"uri": str(checks / "qlib_runs"), "default_exp_name": "audit"}})
        self.state["checks"]["H4"] = dict(
            status="audited", training_skipped=True, seconds=time.perf_counter() - start,
            explicit_st_prefilter_found=False,
            st_note="No dedicated ST exclusion found in current data-generation YAML/code. Historical membership is reused unchanged.",
            label_processing="DropnaLabel/CSRankNorm on learning view; test uses inference view. No test filtering by future holding-period events added.",
            suspension="Qlib Exchange treats absent/all-NaN close quotes as suspended and rejects non-tradable orders.",
            strategy_only_tradable=False, forbid_all_trade_at_limit=True,
            configured_limit_threshold=None, effective_qlib_cn_limit_threshold=C.limit_threshold,
            price_limit_note="None in backtest kwargs falls back to Qlib regional configuration. No new ST/board-specific rules are introduced.",
            deal_price="close", topk=30, drop=5, risk_degree=0.95,
            open_cost=0.0005, close_cost=0.0015, quote_universe="all", rules_changed=False,
            exchange_source_sha256=sha256(inspect.getfile(Exchange)))
        atomic_json(checks / "H4.json", self.state["checks"]["H4"])
        (checks / "README.md").write_text(
            "# G2 / H4 检查\n\nG2：当前因子值头已是 Linear → LayerNorm → GELU → Linear，跳过新增训练。\n\n"
            "H4：未发现独立 ST 预过滤；保留原有历史股票池。Qlib 对缺失/全 NaN close 报价视为停牌并拒绝不可成交订单。"
            "策略 only_tradable=False，forbid_all_trade_at_limit=True；传入 limit_threshold=None 会使用 Qlib 的区域默认值，"
            f"本次初始化后的中国市场默认值为 {C.limit_threshold}。这不等于按各股票 ST/板块实际涨跌停制度逐一建模。"
            "学习视图含 DropnaLabel/CSRankNorm，测试使用推理视图；没有按未来持有期事件筛测试股票。"
            "本批只记录现状，不修改数据或 TopkDropout 规则。\n")
        self.save()
        self.event("G2 already implemented; H4 audited without changing rules")

    def baseline(self, market, family):
        key = f"{market}.baseline_{family}"
        if key in self.state["runs"]:
            return self.state["runs"][key]
        folder = reference_directory(market, family)
        validate_backtest(folder, market)
        record = dict(status="complete", universe=market, variant="baseline_" + family, item="reference",
                      role="baseline", stage1_seed=42, stage2_seed=0, output_dir=str(folder),
                      reused_reference=True, new_training=False, training_seconds=0,
                      evaluation_backtest_seconds=0, metrics=result_metrics(folder), baseline_variant="")
        self.state["runs"][key] = record
        self.save()
        return record

    def backtest(self, market, folder):
        return [sys.executable, "-m", "trace_vq.evaluation.backtest", "--pred_path", str(folder / "evaluation/test_predictions.pkl"),
                "--universe", market, "--quote-universe", "all", "--start_time", "2023-01-01", "--end_time", "2025-12-31",
                "--topk", "30", "--drop", "5", "--open_cost", "0.0005", "--close_cost", "0.0015",
                "--min_cost", "0", "--account", "100000000", "--output_dir", str(folder / "backtest")]

    def ensembles(self):
        for market in ("csi300", "csi800"):
            self.baseline(market, "vq")
            self.baseline(market, "fsq")
            for name, families in [("ens_vq5", ["vq"]), ("ens_fsq5", ["fsq"]), ("ens_vq_fsq", ["vq", "fsq"])]:
                key = market + "." + name
                if self.state["runs"].get(key, {}).get("status") == "complete":
                    continue
                folder = self.output / market / name
                (folder / "evaluation").mkdir(parents=True, exist_ok=True)
                paths = [ROOT / "artifacts/research/multiseed_validation_20261030" / market /
                         ("baseline" if family == "vq" else "fsq") / f"seed{seed}" / "evaluation/test_predictions.pkl"
                         for family in families for seed in range(5)]
                try:
                    missing = [str(p) for p in paths if not p.exists()]
                    if missing:
                        raise FileNotFoundError("Missing ensemble predictions: " + ", ".join(missing))
                    start = time.perf_counter()
                    destination = folder / "evaluation/test_predictions.pkl"
                    frame = build_ensemble(paths, destination, "score", "label", False)
                    assert frame.index.is_unique
                    prediction_seconds = time.perf_counter() - start
                    atomic_json(folder / "input_manifest.json", dict(
                        status="complete", training_runs=0, alignment="exact sorted date/stock index and labels",
                        group_weights="equal; VQ/FSQ each 0.5 in mixed ensemble", input_sha256={str(p): sha256(p) for p in paths}))
                    backtest_seconds = self.phase(key + ".backtest", self.backtest(market, folder), folder / "backtest.log",
                                                  [folder / "backtest/backtest_config.json", folder / "backtest/portfolio_return.csv"])
                    validate_backtest(folder, market)
                    start = time.perf_counter()
                    metrics = result_metrics(folder)
                    metric_seconds = time.perf_counter() - start
                    baseline = "ens_vq5" if name == "ens_vq_fsq" else "baseline_" + families[0]
                    record = dict(status="complete", universe=market, variant=name, item="I1", role="ensemble",
                                  stage1_seed=42, stage2_seed=None, stage2_seeds=[0, 1, 2, 3, 4],
                                  output_dir=str(folder), training_seconds=0, new_training=False,
                                  prediction_seconds=prediction_seconds, backtest_seconds=backtest_seconds,
                                  evaluation_backtest_seconds=prediction_seconds + backtest_seconds + metric_seconds,
                                  gpu_peak_reserved_mib=0, gpu_peak_allocated_mib=0, metrics=metrics,
                                  baseline_variant=baseline, completed=now())
                    self.state["runs"][key] = record
                    atomic_json(folder / "completion.json", record)
                    self.event(f"RESULT {key}: {metrics}")
                except Exception as error:
                    self.fail(key, market, name, "I1", error)
                self.save()
                self.summarize()
                gc.collect()

    def fail(self, key, market, variant, item, error):
        self.state["runs"][key] = dict(status="failed", universe=market, variant=variant, item=item, error=str(error))
        self.event(f"FAILED {key}: {error}")
        with (self.output / "failures.log").open("a") as stream:
            traceback.print_exc(file=stream)

    def run_model(self, market, variant):
        key = market + "." + variant
        if self.state["runs"].get(key, {}).get("status") == "complete":
            return
        folder = self.output / market / variant / "seed0"
        folder.mkdir(parents=True, exist_ok=True)
        config = make_config(market, variant, folder, self.args.num_workers)
        OmegaConf.save(OmegaConf.create(config), folder / "planned.config.yaml")
        base = [sys.executable, "-u", "-m", "trace_vq.cli.batch01"]
        common = ["--universe", market, "--variant", variant, "--output-dir", str(folder),
                  "--device", "cuda", "--num-workers", str(self.args.num_workers)]
        try:
            self.phase(key + ".train", base + ["train"] + common, folder / "train.log", [folder / "training_result.json"])
            eval_seconds = self.phase(key + ".evaluate", base + ["evaluate"] + common, folder / "evaluate.log",
                                      [folder / "evaluation/completion.json"])
            backtest_seconds = self.phase(key + ".backtest", self.backtest(market, folder), folder / "backtest.log",
                                          [folder / "backtest/backtest_config.json", folder / "backtest/portfolio_return.csv"])
            validate_backtest(folder, market)
            record = json.loads((folder / "training_result.json").read_text())
            evaluation = json.loads((folder / "evaluation/completion.json").read_text())
            record.update(item=VARIANTS[variant]["item"], role="new_model", output_dir=str(folder), new_training=True,
                          standalone_exact=evaluation["standalone_exact"], metrics=result_metrics(folder),
                          baseline_variant="baseline_" + VARIANTS[variant]["family"],
                          evaluation_seconds=eval_seconds, backtest_seconds=backtest_seconds,
                          evaluation_backtest_seconds=record["training_inference_seconds"] + eval_seconds + backtest_seconds,
                          completed=now())
            for metric in ("gpu_peak_allocated_mib", "gpu_peak_reserved_mib"):
                record[metric] = max(record[metric], evaluation[metric])
            self.state["runs"][key] = record
            atomic_json(folder / "completion.json", record)
            self.event(f"RESULT {key}: {record['metrics']}")
        except Exception as error:
            self.fail(key, market, variant, VARIANTS[variant]["item"], error)
        self.save()
        self.summarize()

    def summarize(self):
        rows = []
        for key, record in self.state["runs"].items():
            row = {k: record.get(k) for k in ("universe", "variant", "item", "role", "status", "stage1_seed", "stage2_seed",
                    "new_training", "reused_reference", "baseline_variant", "training_seconds", "evaluation_backtest_seconds",
                    "evaluation_seconds", "backtest_seconds", "prediction_seconds", "trainable_parameters", "total_parameters",
                    "gpu_peak_allocated_mib", "gpu_peak_reserved_mib", "output_dir", "error")}
            row.update(record.get("metrics", {}))
            control = self.state["runs"].get(record.get("universe", "") + "." + record.get("baseline_variant", ""))
            if control and control.get("status") == "complete" and record.get("metrics"):
                for metric in METRICS:
                    row["delta_" + metric] = record["metrics"][metric] - control["metrics"][metric]
            if record.get("variant") == "ens_vq_fsq" and record.get("metrics"):
                control = self.state["runs"].get(record["universe"] + ".ens_fsq5")
                if control and control.get("metrics"):
                    row.update({"delta_vs_fsq5_" + metric: record["metrics"][metric] - control["metrics"][metric]
                                for metric in METRICS})
            rows.append(row)
        if rows:
            frame = pd.DataFrame(rows).sort_values(["universe", "role", "variant"])
            target = self.output / "results.csv"
            temp = target.with_suffix(".partial.csv")
            frame.to_csv(temp, index=False)
            temp.replace(target)
            frame[frame["role"] == "baseline"].to_csv(self.output / "baselines.csv", index=False)
            frame[frame["role"] != "baseline"].to_csv(self.output / "variant_results.csv", index=False)

    def execute(self):
        gpu = json.loads((self.output / "preflight/gpu_validation.json").read_text())
        assert gpu["status"] == "passed" and len(gpu["checks"]) == 12
        planned = []
        config_dir = self.output / "configs"
        for market, variant in tasks():
            folder = self.output / market / variant / "seed0"
            config = make_config(market, variant, folder, self.args.num_workers)
            destination = config_dir / market / (variant + ".yaml")
            destination.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(OmegaConf.create(config), destination)
            planned.append(dict(universe=market, variant=variant, stage1_seed=42, stage2_seed=0,
                                family=VARIANTS[variant]["family"], config=str(destination), output=str(folder)))
        atomic_json(self.output / "plan.json", dict(batch=1, items=list(range(1, 9)),
            new_stage1_runs=0, new_stage2_runs=12, ensemble_outputs=6, tasks=planned,
            no_later_batches=True, protocol="Baseline Results Protocol v1.0; existing all-quotes TopkDropout",
            validation_selection="original validation MSE plus auxiliary", source_sha256=self.source,
            grid_note="3 latent factors; CA uses normalized-grid MSE; unchanged Stage 1 and matching downstream tensors retained"))
        self.check_items()
        self.ensembles()
        for market, variant in tasks():
            self.run_model(market, variant)
        self.phase("original_integrity.after", [sys.executable, "research/verify_baseline.py"], self.output / "original_integrity_after.log")
        assert {name: sha256(ROOT / name) for name in SOURCES} == self.source
        failures = [key for key, value in self.state["runs"].items() if value.get("status") != "complete"]
        self.state.update(status="complete" if not failures else "complete_with_failures", finished=now(),
                          current_phase=None, failed_runs=failures)
        self.save()
        self.summarize()
        self.event("BATCH01 FINISHED: " + self.state["status"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    Batch(args).execute()


if __name__ == "__main__":
    main()
