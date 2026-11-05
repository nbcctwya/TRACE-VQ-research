"""Run only checklist 09–10, with persistent monitoring and paired comparisons."""
import argparse
import fcntl
import gc
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

from research.run_batch01 import result_metrics, validate_backtest, METRICS
from trace_vq.cli.batch02 import VARIANTS, PREVIOUS, family, make_config, reference_directory
from trace_vq.cli.screening import now
from trace_vq.data.frozen import atomic_json, sha256

SOURCES = ["research/run_batch02.py", "trace_vq/cli/batch02.py", "trace_vq/training/research_run.py",
           "trace_vq/models/next_screen.py", "trace_vq/models/screening.py", "trace_vq/training/prediction.py",
           "trace_vq/data/research_context.py", "trace_vq/data/frozen.py",
           "trace_vq/models/prediction/temporal_transformer.py", "trace_vq/cli/next_screen.py",
           "trace_vq/cli/batch01.py", "research/run_batch01.py"]


def tasks():
    return [(m, v, s) for s in range(1, 5) for m in ("csi300", "csi800")
            for v in ("turnover_control", "turnover")] + [
            (m, v, 0) for m in ("csi300", "csi800") for v in ("fsq_turnover_control", "fsq_turnover")]


def key(market, variant, seed):
    return f"{market}.{variant}.seed{seed}"


def sampled_gpu(pid):
    try:
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=5, check=True)
        for line in result.stdout.splitlines():
            parts = line.split(",")
            if int(parts[0]) == pid:
                return float(parts[1].strip())
    except (ValueError, subprocess.SubprocessError, FileNotFoundError):
        pass
    return None


class Batch02:
    def __init__(self, args):
        self.args, self.output = args, args.output_dir.resolve()
        if not self.output.is_relative_to(ROOT / "artifacts/research"):
            raise ValueError("Explicit research artifact root required")
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / "batch.lock").open("w")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.path = self.output / "progress.json"
        self.source = {name: sha256(ROOT / name) for name in SOURCES}
        self.state = json.loads(self.path.read_text()) if self.path.exists() else dict(
            status="preparing", started=now(), runs={}, phases={}, benchmarks={}, source_sha256=self.source,
            planned_new_stage2=20, planned_new_stage1=0, planned_reused_seed0=4)
        if self.source != self.state["source_sha256"]:
            raise RuntimeError("Training implementation changed; inspect before resuming this batch")
        previous = self.state.get("phases", {}).get(self.state.get("current_phase"), {})
        pid = previous.get("pid")
        if previous.get("status") == "running" and pid and Path(f"/proc/{pid}/cmdline").exists():
            actual = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
            if actual[:len(previous["command"])] == [p.encode() for p in previous["command"]]:
                raise RuntimeError("The prior training process is still alive; refusing a duplicate fit")
        self.state.update(pid=os.getpid(), status="running", current_phase=None)
        self.save()

    def save(self):
        self.state["updated"] = now()
        atomic_json(self.path, self.state)

    def event(self, message):
        line = f"{now()} {message}"
        print(line, flush=True)
        with (self.output / "events.log").open("a") as stream:
            stream.write(line + "\n")

    def monitor(self, phase=None, pid=None):
        runs = list(self.state["runs"].values())
        snapshot = dict(updated=now(), batch_status=self.state["status"], batch_pid=os.getpid(),
                        current_phase=phase, child_pid=pid,
                        completed_new_training=sum(r.get("new_training", False) and r["status"] == "complete" for r in runs),
                        planned_new_training=20,
                        reused_seed0=sum(r.get("reused_reference", False) and r["status"] == "complete" for r in runs),
                        failures=[k for k,r in self.state["runs"].items() if r["status"] == "failed"])
        if phase:
            record = self.state["phases"][phase]
            progress = Path(record["log"]).parent / "training_progress.json"
            if progress.exists():
                epochs = json.loads(progress.read_text())
                latest = epochs["latest"]
                snapshot.update(epoch=latest["epoch"], completed_epochs=latest["epoch"]+1,
                                val_loss=latest.get("val_loss"),
                                best_val_loss=min(r["val_loss"] for r in epochs["epochs"]),
                                last_epoch_updated=latest["updated"])
                stale = time.time()-progress.stat().st_mtime
                if phase.endswith(".train") and stale > 1200:
                    snapshot["warning"] = f"No completed epoch for {stale/60:.1f} minutes; inspect phase log and GPU"
                    if not record.get("stall_warning_recorded"):
                        with (self.output/"alerts.jsonl").open("a") as stream:
                            stream.write(json.dumps(dict(updated=now(),phase=phase,warning=snapshot["warning"]))+"\n")
                        record["stall_warning_recorded"] = True
            memory = sampled_gpu(pid) if pid else None
            snapshot["gpu_process_sample_mib"] = memory
            if memory is not None:
                record["sampled_gpu_process_peak_mib"] = max(record.get("sampled_gpu_process_peak_mib", 0), memory)
        atomic_json(self.output / "monitoring.json", snapshot)
        text = ["# 第02批实时进度", "", f"更新时间（UTC）：{snapshot['updated']}", "",
                f"状态：{snapshot['batch_status']}；新增训练完成 {snapshot['completed_new_training']}/20；"
                f"复用 seed0 完成 {snapshot['reused_seed0']}/4；失败 {len(snapshot['failures'])}。", "",
                f"当前阶段：{phase or '无'}。"]
        if "completed_epochs" in snapshot:
            text += ["", f"已完成 {snapshot['completed_epochs']} epochs，最近验证损失 {snapshot['val_loss']}，"
                     f"最佳 {snapshot['best_val_loss']}。"]
        if snapshot.get("warning"):
            text += ["", "警示：" + snapshot["warning"]]
        if snapshot["failures"]:
            text += ["", "失败项：" + ", ".join(snapshot["failures"])]
        (self.output / "STATUS.md").write_text("\n".join(text)+"\n")

    def phase(self, name, command, log, required=()):
        old = self.state["phases"].get(name, {})
        if old.get("status") == "complete" and all(p.exists() for p in required):
            return old["seconds"]
        training = [p for p in required if p.name == "training_result.json" and p.exists()]
        if training:
            data = json.loads(training[0].read_text())
            seconds = data["train_phase_seconds"]
            self.state["phases"][name] = dict(status="complete", seconds=seconds, adopted_completed=True)
            self.save()
            return seconds
        assert self.source == {filename:sha256(ROOT/filename) for filename in SOURCES}
        if old:
            self.state.setdefault("attempt_history", {}).setdefault(name, []).append(old)
        log.parent.mkdir(parents=True, exist_ok=True)
        self.event("START " + name)
        start = time.perf_counter()
        with log.open("a") as stream:
            process = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stream, stderr=subprocess.STDOUT)
            self.state["current_phase"] = name
            record = dict(status="running", started=now(), pid=process.pid, command=command, log=str(log))
            self.state["phases"][name] = record
            self.save()
            while True:
                self.monitor(name, process.pid)
                self.save()
                try:
                    code = process.wait(timeout=20)
                    break
                except subprocess.TimeoutExpired:
                    pass
        elapsed = time.perf_counter()-start
        record.update(status="complete" if code == 0 else "failed", seconds=elapsed, returncode=code, finished=now())
        self.save()
        if code or not all(p.exists() for p in required):
            tail = log.read_text(errors="replace")[-3000:]
            atomic_json(log.parent / (name.rsplit(".",1)[-1]+"_failure.json"), dict(phase=name, returncode=code, log=str(log), tail=tail))
            raise RuntimeError(f"{name} exited {code} or missing completion artifact; see {log}")
        self.event(f"DONE {name}: {elapsed:.1f}s")
        return elapsed

    def benchmarks(self):
        for market in ("csi300", "csi800"):
            entries = [("vq", seed, ROOT / "artifacts/research/multiseed_validation_20261030" / market /
                        "baseline" / f"seed{seed}") for seed in range(5)]
            entries += [("fsq", 0, PREVIOUS / market / "fsq")]
            for kind, seed, folder in entries:
                name = key(market, "baseline_"+kind, seed)
                if name in self.state["benchmarks"]:
                    continue
                try:
                    validate_backtest(folder, market)
                    self.state["benchmarks"][name] = dict(status="complete", universe=market,
                        variant="baseline_"+kind, stage1_seed=42, stage2_seed=seed, reused_reference=True,
                        new_training=False, output_dir=str(folder), metrics=result_metrics(folder))
                except Exception as error:
                    self.state["benchmarks"][name] = dict(status="failed", error=str(error))
                    self.event(f"BENCHMARK UNAVAILABLE {name}: {error}")
                self.save()

    def adopt_seed0(self):
        for market in ("csi300", "csi800"):
            for variant in ("turnover_control", "turnover"):
                name = key(market, variant, 0)
                if self.state["runs"][name]["status"] == "complete":
                    continue
                output = self.output / market / variant / "seed0"
                source = reference_directory(market, variant)
                try:
                    validate_backtest(source, market)
                    seconds = self.phase(name+".audit", self.command("audit_reference", market, variant, 0, output),
                                         output/"audit.log", [output/"reference_audit.json"])
                    old = json.loads((source / "training_result.json").read_text())
                    audit = json.loads((output / "reference_audit.json").read_text())
                    self.state["runs"][name].update(status="complete", config=audit["config"],
                        reference_source=str(source), reference_predictions_exact=True, metrics=result_metrics(source),
                        training_seconds=0, reference_training_seconds=old.get("elapsed_seconds"),
                        evaluation_backtest_seconds=seconds, gpu_peak_allocated_mib=None, gpu_peak_reserved_mib=None,
                        replay_gpu_peak_reserved_mib=audit["gpu_peak_reserved_mib"], completed=now())
                    self.event("REUSED " + name)
                except Exception as error:
                    self.fail(name, error)
                self.save()
                self.summarize()
                gc.collect()

    def command(self, action, market, variant, seed, folder):
        command = [sys.executable, "-u", "-m", "trace_vq.cli.batch02", action,
                   "--universe", market, "--variant", variant, "--seed", str(seed), "--output-dir", str(folder),
                   "--device", "cuda", "--num-workers", str(self.args.num_workers)]
        if action == "train":
            command += ["--resume"]
        return command

    def fail(self, name, error):
        self.state["runs"][name].update(status="failed", error=str(error))
        self.event(f"FAILED {name}: {error}")
        with (self.output / "failures.log").open("a") as stream:
            traceback.print_exc(file=stream)

    def run_model(self, market, variant, seed):
        name = key(market, variant, seed)
        if self.state["runs"][name]["status"] == "complete":
            return
        folder = self.output / market / variant / f"seed{seed}"
        try:
            self.phase(name+".train", self.command("train", market, variant, seed, folder), folder/"train.log",
                       [folder/"training_result.json"])
            evaluate_seconds = self.phase(name+".evaluate", self.command("evaluate", market, variant, seed, folder),
                                           folder/"evaluate.log", [folder/"evaluation/completion.json"])
            backtest = [sys.executable, "-m", "trace_vq.evaluation.backtest", "--pred_path", str(folder/"evaluation/test_predictions.pkl"),
                        "--universe", market, "--quote-universe", "all", "--start_time", "2023-01-01", "--end_time", "2025-12-31",
                        "--topk", "30", "--drop", "5", "--open_cost", "0.0005", "--close_cost", "0.0015", "--min_cost", "0",
                        "--account", "100000000", "--output_dir", str(folder/"backtest")]
            backtest_seconds = self.phase(name+".backtest", backtest, folder/"backtest.log",
                                           [folder/"backtest/backtest_config.json", folder/"backtest/portfolio_return.csv"])
            validate_backtest(folder, market)
            train = json.loads((folder/"training_result.json").read_text())
            evaluation = json.loads((folder/"evaluation/completion.json").read_text())
            assert evaluation["standalone_exact"]
            record = self.state["runs"][name]
            record.update(train, status="complete", metrics=result_metrics(folder), standalone_exact=True,
                          evaluation_seconds=evaluate_seconds, backtest_seconds=backtest_seconds,
                          evaluation_backtest_seconds=train["training_inference_seconds"]+evaluate_seconds+backtest_seconds,
                          sampled_gpu_process_peak_mib=max(self.state["phases"][name+"."+stage].get("sampled_gpu_process_peak_mib",0)
                                                           for stage in ("train", "evaluate", "backtest")), completed=now())
            for field in ("gpu_peak_allocated_mib", "gpu_peak_reserved_mib"):
                record[field] = max(train[field], evaluation[field])
            atomic_json(folder/"completion.json", record)
            self.event(f"RESULT {name}: {record['metrics']}")
        except Exception as error:
            self.fail(name, error)
        self.save()
        self.summarize()
        self.monitor()

    def summarize(self):
        rows = []
        for name, record in self.state["runs"].items():
            row = {k:v for k,v in record.items() if k not in ("metrics", "validation_metrics", "test_metrics")}
            row.update(record.get("metrics", {}))
            market, variant, seed = record["universe"], record["variant"], record["stage2_seed"]
            baseline = self.state["benchmarks"].get(key(market, "baseline_"+family(variant), seed))
            if variant.endswith("control"):
                control, control_name = baseline, "baseline_"+family(variant)
            else:
                control_name = "fsq_turnover_control" if family(variant)=="fsq" else "turnover_control"
                control = self.state["runs"].get(key(market, control_name, seed))
            row["baseline_variant"] = control_name
            row["baseline_stage2_seed"] = seed
            for prefix, target in (("delta_", control), ("delta_vs_family_", baseline)):
                if record.get("metrics") and target and target.get("metrics"):
                    for metric in METRICS:
                        row[prefix+metric] = record["metrics"][metric]-target["metrics"][metric]
            if family(variant)=="fsq" and record.get("metrics"):
                vq = self.state["runs"].get(key(market, variant.removeprefix("fsq_"), seed))
                if vq and vq.get("metrics"):
                    row.update({"delta_vs_vq_paired_"+metric:record["metrics"][metric]-vq["metrics"][metric] for metric in METRICS})
            rows.append(row)
        frame = pd.DataFrame(rows).sort_values(["universe", "variant", "stage2_seed"])
        temporary = self.output/"results.partial.csv"
        frame.to_csv(temporary,index=False); temporary.replace(self.output/"results.csv")
        metrics = [m for m in METRICS if m in frame]
        complete = frame[frame.status=="complete"]
        aggregates = []
        for (market, variant), group in complete.groupby(["universe", "variant"]):
            row = dict(universe=market, variant=variant, completed_seeds=len(group),
                       expected_seeds=1 if variant.startswith("fsq") else 5,
                       stage2_seeds=",".join(str(int(s)) for s in sorted(group.stage2_seed)))
            for metric in metrics:
                row[metric+"_mean"]=group[metric].mean()
                row[metric+"_std"]=group[metric].std(ddof=1)
                delta="delta_"+metric
                if delta in group:
                    row[delta+"_mean"]=group[delta].mean()
                    row[delta+"_std"]=group[delta].std(ddof=1)
            aggregates.append(row)
        pd.DataFrame(aggregates).to_csv(self.output/"mean_std.csv",index=False)
        benchmarks=[]
        for r in self.state["benchmarks"].values():
            benchmarks.append({**{k:v for k,v in r.items() if k!="metrics"}, **r.get("metrics",{})})
        pd.DataFrame(benchmarks).to_csv(self.output/"benchmarks.csv",index=False)

    def execute(self):
        preflight=json.loads((self.output/"preflight/gpu_validation.json").read_text())
        assert preflight["status"]=="passed" and len(preflight["checks"])==8
        test_log=(self.output/"unit_tests.log").read_text()
        assert "\nOK\n" in test_log and "cuda" in test_log
        plan=[]
        all_tasks=[(m,v,0) for m in ("csi300","csi800") for v in ("turnover_control","turnover")]+tasks()
        for market,variant,seed in all_tasks:
            folder=self.output/market/variant/f"seed{seed}"
            cfg=make_config(market,variant,seed,folder,self.args.num_workers)
            path=self.output/"configs"/market/variant/f"seed{seed}.yaml"
            path.parent.mkdir(parents=True,exist_ok=True)
            OmegaConf.save(OmegaConf.create(cfg),path)
            reused=not variant.startswith("fsq") and seed==0
            record=dict(universe=market,variant=variant,stage1_seed=42,stage2_seed=seed,
                        item="E2" if variant.startswith("fsq") else "E1", status="pending",
                        new_training=not reused,reused_reference=reused,output_dir=str(folder),config=str(path))
            self.state["runs"].setdefault(key(market,variant,seed),record)
            plan.append(record)
        atomic_json(self.output/"plan.json",dict(batch=2,items=[9,10],new_stage1_runs=0,new_stage2_runs=20,
            reused_seed0=4,tasks=plan,source_sha256=self.source,no_later_batches=True,
            checkpoint_selection="original validation MSE plus auxiliary", paired_controls=True,
            monitoring_interval_seconds=20,stall_warning_after_seconds=1200,
            resume="epoch checkpoint; complete Python/NumPy/Torch CPU/CUDA RNG; validated uninterrupted equivalence"))
        self.save()
        self.monitor()
        self.phase("original_integrity.before",[sys.executable,"research/verify_baseline.py"],self.output/"original_integrity_before.log")
        self.benchmarks()
        self.summarize()
        self.adopt_seed0()
        for market,variant,seed in tasks():
            self.run_model(market,variant,seed)
        self.phase("original_integrity.after",[sys.executable,"research/verify_baseline.py"],self.output/"original_integrity_after.log")
        assert self.source=={filename:sha256(ROOT/filename) for filename in SOURCES}
        failures=[k for k,r in self.state["runs"].items() if r["status"]!="complete"]
        self.state.update(status="complete" if not failures else "complete_with_failures", finished=now(),
                          failed_runs=failures,current_phase=None)
        self.save(); self.summarize(); self.monitor()
        self.event("BATCH02 FINISHED: "+self.state["status"])
        self.phase("aggregate_report",[sys.executable,"research/report_batch02.py","--output-dir",str(self.output)],
                   self.output/"report.log",[self.output/"report_verification.json"])
        self.state["current_phase"]=None
        self.save(); self.monitor()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--num-workers",type=int,default=2)
    args=parser.parse_args()
    batch=Batch02(args)
    try:
        batch.execute()
    except BaseException as error:
        batch.state.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "controller_failed",
                           controller_error=str(error))
        batch.save()
        batch.monitor()
        raise


if __name__=="__main__":
    main()
