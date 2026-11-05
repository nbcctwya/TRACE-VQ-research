"""Serial monitored execution: batch03 first, then batch04, no later batches."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
from omegaconf import OmegaConf

from research.run_batch01 import result_metrics,validate_backtest,METRICS
from trace_vq.cli.batch34 import make_config,reference
from trace_vq.cli.screening import now
from trace_vq.data.frozen import atomic_json,sha256
from trace_vq.models.batch34 import BATCH03,BATCH04,VARIANTS

SOURCES=["research/run_batch34.py","trace_vq/cli/batch34.py","trace_vq/models/batch34.py",
         "trace_vq/models/experts/routing_research.py","trace_vq/data/fsq_history.py",
         "trace_vq/training/research_run.py","trace_vq/models/batch01.py","trace_vq/models/screening.py",
         "trace_vq/models/experts/shared_routed.py","trace_vq/models/experts/fusion.py",
         "trace_vq/models/prediction/loadings.py","trace_vq/models/prediction/temporal_transformer.py",
         "trace_vq/training/prediction.py","trace_vq/data/frozen.py","trace_vq/cli/next_screen.py"]


def tasks(batch):
    return [(market,name) for name in (BATCH03 if batch==3 else BATCH04) for market in ("csi300","csi800")]


class Queue:
    def __init__(self,args):
        self.args=args;self.output=args.output_dir.resolve();self.history=args.history_root.resolve()
        assert self.output.is_relative_to(ROOT/"artifacts/research") and self.history.is_relative_to(ROOT/"artifacts/research")
        self.output.mkdir(parents=True,exist_ok=True)
        self.lock=(self.output/"batch.lock").open("w");fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        self.path=self.output/"progress.json"
        self.sources={path:sha256(ROOT/path) for path in SOURCES}
        self.state=json.loads(self.path.read_text()) if self.path.exists() else dict(status="running",started=now(),
                    runs={},phases={},batches={},planned_new_stage2=30,planned_new_stage1=0,source_sha256=self.sources)
        assert self.sources==self.state["source_sha256"],"Implementation changed after launch; inspect before recovery"
        previous=self.state["phases"].get(self.state.get("current_phase"),{})
        if previous.get("status")=="running" and previous.get("pid"):
            command=Path(f"/proc/{previous['pid']}/cmdline")
            if command.exists() and command.read_bytes().split(b"\0")[:len(previous["command"])]==[s.encode() for s in previous["command"]]:
                raise RuntimeError("Previous phase is still alive; refusing duplicate GPU work")
        self.state.update(status="running",pid=os.getpid(),current_phase=None)
        self.save()

    def folder(self,batch):return self.args.batch03_root.resolve() if batch==3 else self.args.batch04_root.resolve()
    def name(self,batch,market,variant):return f"batch{batch:02d}.{market}.{variant}.seed0"

    def save(self):
        self.state["updated"]=now();atomic_json(self.path,self.state)

    def event(self,message):
        line=now()+" "+message;print(line,flush=True)
        with (self.output/"events.log").open("a") as f:f.write(line+"\n")

    def monitor(self,phase=None,pid=None):
        models=[r for r in self.state["runs"].values() if r.get("role")=="new_model"]
        snapshot=dict(updated=now(),batch_status=self.state["status"],batch_pid=os.getpid(),child_pid=pid,
                      current_phase=phase,current_batch=self.state.get("current_batch"),planned_new_training=30,
                      completed_new_training=sum(r["status"]=="complete" for r in models),
                      failures=[k for k,r in self.state["runs"].items() if r["status"]=="failed"],
                      batches=self.state["batches"])
        if phase:
            record=self.state["phases"][phase];path=Path(record["log"]).parent/"training_progress.json"
            if path.exists():
                epochs=json.loads(path.read_text());last=epochs["latest"]
                snapshot.update(completed_epochs=last["epoch"]+1,val_loss=last.get("val_loss"),
                                best_val_loss=min(r["val_loss"] for r in epochs["epochs"]),last_epoch_updated=last["updated"])
                if phase.endswith(".train") and time.time()-path.stat().st_mtime>1200:
                    snapshot["warning"]="No completed epoch for 20 minutes; inspect GPU and phase log"
                    if not record.get("stall_alert"):
                        self.alert(phase,snapshot["warning"]);record["stall_alert"]=True
        atomic_json(self.output/"monitoring.json",snapshot)
        lines=["# 第三、第四批串行实验进度","",f"更新时间（UTC）：{snapshot['updated']}","",
               f"状态：{snapshot['batch_status']}；新增训练完成 {snapshot['completed_new_training']}/30；失败 {len(snapshot['failures'])}。","",
               f"当前阶段：{phase or '无'}。"]
        for number,info in self.state["batches"].items():
            lines += ["",f"第{number}批：{info['status']}，完整结果 {info.get('completed',0)}/{info['planned']}。"]
        if "completed_epochs" in snapshot:
            lines += ["",f"当前配置已完成 {snapshot['completed_epochs']} epochs，最新验证损失 {snapshot['val_loss']}，最佳 {snapshot['best_val_loss']}。"]
        if snapshot.get("warning"):lines += ["",snapshot["warning"]]
        (self.output/"STATUS.md").write_text("\n".join(lines)+"\n")

    def alert(self,phase,message):
        with (self.output/"alerts.jsonl").open("a") as f:f.write(json.dumps(dict(updated=now(),phase=phase,warning=message))+"\n")

    def phase(self,name,command,log,required=()):
        previous=self.state["phases"].get(name,{})
        if previous.get("status")=="complete" and all(p.exists() for p in required):return previous["seconds"]
        completed=[p for p in required if p.name=="training_result.json" and p.exists()]
        if completed:
            record=json.loads(completed[0].read_text());seconds=record["train_phase_seconds"]
            self.state["phases"][name]=dict(status="complete",seconds=seconds,adopted_completed=True);self.save();return seconds
        assert self.sources=={path:sha256(ROOT/path) for path in SOURCES}
        if previous:self.state.setdefault("attempt_history",{}).setdefault(name,[]).append(previous)
        log.parent.mkdir(parents=True,exist_ok=True);self.event("START "+name);start=time.perf_counter()
        with log.open("a") as f:
            process=subprocess.Popen(command,cwd=ROOT,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT)
            self.state["current_phase"]=name
            record=dict(status="running",started=now(),pid=process.pid,command=command,log=str(log))
            self.state["phases"][name]=record;self.save()
            while True:
                self.monitor(name,process.pid);self.save()
                try:code=process.wait(timeout=20);break
                except subprocess.TimeoutExpired:pass
        elapsed=time.perf_counter()-start
        record.update(status="complete" if code==0 else "failed",seconds=elapsed,returncode=code,finished=now());self.save()
        if code or not all(p.exists() for p in required):
            self.alert(name,f"Phase failed, code={code}; {log}")
            raise RuntimeError(f"{name} failed; inspect {log}")
        self.event(f"DONE {name}: {elapsed:.1f}s");return elapsed

    def summarize(self,batch):
        folder=self.folder(batch);rows=[]
        for name,r in self.state["runs"].items():
            if r["batch"]!=batch:continue
            row={k:v for k,v in r.items() if k not in ("metrics","validation_metrics","test_metrics","routing_valid","routing_test")}
            row.update(r.get("metrics",{}))
            if r.get("metrics") and r.get("baseline_variant"):
                control=self.state["runs"].get(self.name(batch,r["universe"],r["baseline_variant"]))
                if control and control.get("metrics"):
                    for m in METRICS:row["delta_"+m]=r["metrics"][m]-control["metrics"][m]
            if r.get("routing_test"):row.update({"routing_"+k:v for k,v in r["routing_test"].items()})
            rows.append(row)
        frame=pd.DataFrame(rows).sort_values(["universe","role","variant"])
        temporary=folder/"results.partial.csv";frame.to_csv(temporary,index=False);temporary.replace(folder/"results.csv")
        frame[frame.role.eq("baseline")].to_csv(folder/"baselines.csv",index=False)
        frame[frame.role.eq("new_model")].to_csv(folder/"variant_results.csv",index=False)
        relevant={k:r for k,r in self.state["runs"].items() if r["batch"]==batch}
        local=dict(status=self.state["batches"][str(batch)]["status"],batch=batch,planned_new_stage2=len(tasks(batch)),
                   stage1_new_training=0,runs=relevant,source_sha256=self.sources,history_root=str(self.history),
                   queue_root=str(self.output),updated=now(),gpu_preflight=str(self.output/f"preflight_batch{batch:02d}/gpu_validation.json"))
        atomic_json(folder/"progress.json",local)

    def setup(self):
        for batch in (3,4):
            folder=self.folder(batch);assert folder.is_relative_to(ROOT/"artifacts/research");folder.mkdir(parents=True,exist_ok=True)
            self.state["batches"].setdefault(str(batch),dict(status="pending",planned=len(tasks(batch)),completed=0))
            plan=[]
            for market,variant in tasks(batch):
                output=folder/market/variant/"seed0"
                config=make_config(market,variant,output,self.history,self.args.num_workers)
                destination=folder/"configs"/market/(variant+".yaml");destination.parent.mkdir(parents=True,exist_ok=True)
                OmegaConf.save(OmegaConf.create(config),destination)
                record=dict(batch=batch,universe=market,variant=variant,stage1_seed=42,stage2_seed=0,item=VARIANTS[variant]["item"],
                            role="new_model",status="pending",new_training=True,reused_reference=False,config=str(destination),
                            output_dir=str(output),baseline_variant="baseline_"+VARIANTS[variant]["family"])
                self.state["runs"].setdefault(self.name(batch,market,variant),record);plan.append(record)
            atomic_json(folder/"plan.json",dict(batch=batch,items=list(range(11,18)) if batch==3 else list(range(18,24)),
                new_stage2_runs=len(plan),new_stage1_runs=0,stage1_seed=42,stage2_seed=0,tasks=plan,
                reused_top1_reference=batch==4,source_sha256=self.sources,history_root=str(self.history),serial_queue=str(self.output)))
        self.save()

    def baselines(self,batch):
        families=("fsq",) if batch==3 else ("vq","fsq")
        for market in ("csi300","csi800"):
            for family in families:
                name=self.name(batch,market,"baseline_"+family)
                if name in self.state["runs"]:continue
                folder=reference(market,family);validate_backtest(folder,market)
                self.state["runs"][name]=dict(batch=batch,universe=market,variant="baseline_"+family,stage1_seed=42,stage2_seed=0,
                    role="baseline",status="complete",new_training=False,reused_reference=True,output_dir=str(folder),
                    metrics=result_metrics(folder),training_seconds=0,evaluation_backtest_seconds=0,baseline_variant="",
                    reference_role="same family seed0; native VQ also supplies the existing top1 reference")
        self.save();self.summarize(batch)

    def model(self,batch,market,variant):
        name=self.name(batch,market,variant);r=self.state["runs"][name]
        if r["status"]=="complete":return
        folder=Path(r["output_dir"])
        common=["--batch",str(batch),"--universe",market,"--variant",variant,"--output-dir",str(folder),
                "--history-root",str(self.history),"--device","cuda","--num-workers",str(self.args.num_workers)]
        base=[sys.executable,"-u","-m","trace_vq.cli.batch34"]
        try:
            self.phase(name+".train",base+["train"]+common,folder/"train.log",[folder/"training_result.json"])
            eval_seconds=self.phase(name+".evaluate",base+["evaluate"]+common,folder/"evaluate.log",[folder/"evaluation/completion.json"])
            backtest=[sys.executable,"-m","trace_vq.evaluation.backtest","--pred_path",str(folder/"evaluation/test_predictions.pkl"),
                      "--universe",market,"--quote-universe","all","--start_time","2023-01-01","--end_time","2025-12-31",
                      "--topk","30","--drop","5","--open_cost","0.0005","--close_cost","0.0015","--min_cost","0",
                      "--account","100000000","--output_dir",str(folder/"backtest")]
            backtest_seconds=self.phase(name+".backtest",backtest,folder/"backtest.log",
                                       [folder/"backtest/backtest_config.json",folder/"backtest/portfolio_return.csv"])
            validate_backtest(folder,market)
            trained=json.loads((folder/"training_result.json").read_text());evaluation=json.loads((folder/"evaluation/completion.json").read_text())
            assert evaluation["standalone_exact"]
            r.update(trained,status="complete",metrics=result_metrics(folder),standalone_exact=True,
                     evaluation_seconds=eval_seconds,backtest_seconds=backtest_seconds,
                     evaluation_backtest_seconds=trained["training_inference_seconds"]+eval_seconds+backtest_seconds,completed=now())
            for field in ("gpu_peak_reserved_mib","gpu_peak_allocated_mib"):r[field]=max(trained[field],evaluation[field])
            atomic_json(folder/"completion.json",r);self.event(f"RESULT {name}: {r['metrics']}")
        except Exception as error:
            r.update(status="failed",error=str(error));self.event(f"FAILED {name}: {error}")
            with (self.output/"failures.log").open("a") as f:traceback.print_exc(file=f)
        relevant=[r for r in self.state["runs"].values() if r["batch"]==batch and r["role"]=="new_model"]
        self.state["batches"][str(batch)]["completed"]=sum(r["status"]=="complete" for r in relevant)
        self.save();self.summarize(batch);self.monitor()

    def execute(self):
        validation=json.loads((self.output/"validation_exit_codes.json").read_text());assert not any(validation.values())
        for batch,count in ((3,16),(4,14)):
            pre=json.loads((self.output/f"preflight_batch{batch:02d}/gpu_validation.json").read_text())
            assert pre["status"]=="passed" and len(pre["checks"])==count
        self.setup()
        self.phase("original_integrity.before",[sys.executable,"research/verify_baseline.py"],self.output/"original_integrity_before.log")
        for batch in (3,4):
            if self.state["batches"][str(batch)].get("report_verified"):continue
            self.state["current_batch"]=batch;self.state["batches"][str(batch)]["status"]="running"
            self.baselines(batch)
            for market,variant in tasks(batch):self.model(batch,market,variant)
            rows=[r for r in self.state["runs"].values() if r["batch"]==batch and r["role"]=="new_model"]
            self.state["batches"][str(batch)]["status"]="complete" if all(r["status"]=="complete" for r in rows) else "complete_with_failures"
            self.save();self.summarize(batch)
            self.phase(f"original_integrity.batch{batch:02d}",[sys.executable,"research/verify_baseline.py"],self.folder(batch)/"original_integrity_after.log")
            self.phase(f"batch{batch:02d}.aggregate_report",[sys.executable,"research/report_batch34.py","--output-dir",str(self.folder(batch))],
                       self.folder(batch)/"report.log",[self.folder(batch)/"report_verification.json"])
            self.state["batches"][str(batch)]["report_verified"]=True
            self.event(f"BATCH{batch:02d} FINISHED: "+self.state["batches"][str(batch)]["status"]);self.save()
        assert self.sources=={path:sha256(ROOT/path) for path in SOURCES}
        failures=[k for k,r in self.state["runs"].items() if r["status"]!="complete"]
        self.state.update(status="complete" if not failures else "complete_with_failures",finished=now(),current_phase=None,
                          current_batch=None,failed_runs=failures)
        self.save();self.monitor()
        atomic_json(self.output/"report_verification.json",dict(status="passed",batch_status=self.state["status"],
                    completed_new_training=sum(r.get("role")=="new_model" and r["status"]=="complete" for r in self.state["runs"].values()),
                    batches=self.state["batches"]))
        self.event("SERIAL QUEUE FINISHED: "+self.state["status"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    parser.add_argument("--batch03-root",type=Path,required=True)
    parser.add_argument("--batch04-root",type=Path,required=True)
    parser.add_argument("--history-root",type=Path,required=True)
    parser.add_argument("--num-workers",type=int,default=2)
    args=parser.parse_args();queue=Queue(args)
    try:queue.execute()
    except BaseException as error:
        queue.state.update(status="interrupted" if isinstance(error,KeyboardInterrupt) else "controller_failed",controller_error=str(error))
        queue.save();queue.monitor();raise


if __name__=="__main__":main()
