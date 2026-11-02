"""Serial, resumable TRACE-VQ SP500 Stage1seed42 + Stage2seeds0-4."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from trace_vq.runtime import configure_runtime
configure_runtime()
from trace_vq.data.frozen import atomic_json,sha256
from trace_vq.cli.screening import now


def source_identity():
    paths=[]
    for folder in ["trace_vq","configs"]:
        paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in [".py",".yaml"])
    paths += [Path(__file__),ROOT/"research/run.sh",ROOT/"AGENTS.md",ROOT/"tests/test_sp500_multiseed.py"]
    return {str(p.relative_to(ROOT)):sha256(p) for p in sorted(set(paths))}


def check_backtest(config):
    assert config["universe"]=="sp500" and config["quote_universe"]=="all"
    assert config["strategy"]["topk"]==30 and config["strategy"]["n_drop"]==5
    assert config["strategy"]["risk_degree"]==.95
    assert config["account"]==100000000 and config["benchmark"]=="^gspc"
    assert config["exchange"]["open_cost"]==.0005 and config["exchange"]["close_cost"]==.0015
    assert config["exchange"]["min_cost"]==0 and config["exchange"]["deal_price"]=="close"
    assert config["start_time"]=="2023-01-01" and config["end_time"]=="2025-12-31"
    assert config["exchange_instruments"]=="all"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT/"artifacts/research/sp500_multiseed_20261031")
    args=parser.parse_args();root=args.root.resolve()
    if not root.is_relative_to(ROOT/"artifacts/research"):raise ValueError("Research output required")
    root.mkdir(parents=True,exist_ok=True)
    lock=(root/"pipeline.lock").open("a");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=root/"progress.json"
    state=json.loads(path.read_text()) if path.exists() else dict(started=now(),phases={},runs={})
    source=source_identity()
    if "source_sha256" in state and source!=state["source_sha256"]:raise RuntimeError("Frozen source changed")
    state.update(pid=os.getpid(),status="running",source_sha256=source)
    plan=dict(universe="sp500",model="TRACE-VQ full (TA+CA+SR)",stage1_seed=42,stage2_seeds=list(range(5)),
              train_period=["2009-01-01","2020-12-31"],valid_period=["2021-01-01","2022-12-31"],
              test_period=["2023-01-01","2025-12-31"],n_expert=8,k=4,attention_heads=4,aux_weight=.001,
              epochs=70,learning_rate=.0001,early_stopping_patience=15,num_workers=2,
              stage1_validation_only=True,stage2_validation_only=True,ranking_test_used_for_selection=False,
              individual_portfolios=True,prediction_ensemble=False,source_sha256=source)
    if (root/"plan.json").exists():
        if json.loads((root/"plan.json").read_text())!=plan:raise RuntimeError("Prespecified plan changed")
    else:atomic_json(root/"plan.json",plan)
    for name in source:
        p=root/"source_snapshot"/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes((ROOT/name).read_bytes())
        assert sha256(p)==source[name]
    env=dict(os.environ,PYTHONUNBUFFERED="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",
             OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1")
    def save():
        state["updated"]=now();atomic_json(path,state)
    def phase(name,arguments,log=None):
        if source_identity()!=source:raise RuntimeError("Frozen implementation changed")
        if state["phases"].get(name,{}).get("status")=="complete":return
        log=log or root/(name+".log");log.parent.mkdir(parents=True,exist_ok=True)
        entry=dict(status="running",started=now(),command=[sys.executable,*arguments],log=str(log))
        state["phases"][name]=entry;state["current_phase"]=name;save()
        print(f"[{now()}] {name} starting",flush=True)
        with log.open("a") as stream:
            p=subprocess.Popen(entry["command"],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            entry["pid"]=p.pid;save();code=p.wait()
        entry.update(status="complete" if code==0 else "failed",returncode=code,finished=now());save()
        if code:raise RuntimeError(f"{name} failed with {code}; {log}")
    def cli(command,*extra):
        return ["-m","trace_vq.cli.sp500_multiseed",command,"--root",str(root),*extra]
    def aggregate(seed):
        import pandas as pd
        from trace_vq.evaluation.protocol_metrics import ranking_metrics,portfolio_metrics
        folder=root/f"seed{seed}"
        record=json.loads((folder/"training_result.json").read_text())
        evaluation=json.loads((folder/"evaluation/completion.json").read_text())
        assert evaluation["standalone_exact"] and evaluation["stage1_unchanged"]
        frame=pd.read_pickle(folder/"evaluation/test_predictions.pkl")
        config=json.loads((folder/"backtest/backtest_config.json").read_text());check_backtest(config)
        returns=pd.read_csv(folder/"backtest/portfolio_return.csv",index_col=0,parse_dates=True)
        assert len(returns)==frame.index.get_level_values("datetime").nunique()==752
        import numpy as np
        assert np.allclose(returns["return"],returns.gross_return-returns.cost,rtol=0,atol=1e-14)
        metrics=ranking_metrics(frame.score.astype("float64"),frame.label.astype("float64"))
        metrics.update(portfolio_metrics(returns["return"]))
        record.update(test_metrics=metrics,standalone_exact=True,backtest_config=config,seed=seed,
                      run_dir=str(folder),stage1_sha256=json.loads((root/"stage1/training_result.json").read_text())["checkpoint_sha256"])
        atomic_json(folder/"completion.json",record)
        state["runs"][str(seed)]=record
        table=pd.DataFrame([dict(seed=k,universe="sp500",**v["test_metrics"],
                                valid_RankIC=v["validation_metrics"]["RankIC"],run_dir=v["run_dir"])
                            for k,v in sorted(state["runs"].items(),key=lambda kv:int(kv[0]))])
        table.to_csv(root/"seed_metrics.csv",index=False)
        table.select_dtypes(include="number").drop(columns="seed").agg(["mean","std","count"]).to_csv(root/"mean_std.csv")
        save()
    try:
        save()
        before=root/"integrity_before.json"
        if not before.exists() or not json.loads(before.read_text()).get("source_unchanged"):
            phase("integrity_before",["research/verify_baseline.py","--full"],before)
        if (root/"data_audit.json").exists():
            assert json.loads((root/"data_audit.json").read_text())["status"]=="passed"
            state["phases"].setdefault("data",dict(status="complete",reused_preflight=True))
        else:phase("data",cli("audit"))
        audit=json.loads((root/"data_audit.json").read_text())
        state["input_sha256"]={x["source"]:x["sha256"] for x in audit["splits"].values()}
        state["input_sha256"][audit["original_handler_config"]]=audit["original_handler_sha256"]
        if not (root/"all_tests.log").exists():raise RuntimeError("Run and pass tests before launching")
        if "\nOK" not in (root/"all_tests.log").read_text():raise RuntimeError("Tests did not pass")
        phase("stage1_preflight",cli("stage1-check"))
        phase("stage1",cli("stage1"),root/"stage1/train.log")
        first=json.loads((root/"stage1/training_result.json").read_text())
        state["stage1_checkpoint"]=first["checkpoint"];state["stage1_sha256"]=first["checkpoint_sha256"];save()
        phase("cache",cli("cache"))
        phase("stage2_preflight",cli("validate-cache"))
        for seed in range(5):
            if sha256(first["checkpoint"])!=first["checkpoint_sha256"]:raise RuntimeError("Stage1 checkpoint changed")
            phase(f"seed{seed}.train",cli("train","--seed",str(seed)),root/f"seed{seed}/train.log")
            phase(f"seed{seed}.evaluate",cli("evaluate","--seed",str(seed)),root/f"seed{seed}/evaluate.log")
            phase(f"seed{seed}.backtest",["-m","trace_vq.evaluation.backtest","--universe","sp500",
                  "--quote-universe","all","--pred_path",str(root/f"seed{seed}/evaluation/test_predictions.pkl"),
                  "--output_dir",str(root/f"seed{seed}/backtest")],root/f"seed{seed}/backtest.log")
            aggregate(seed)
        if sorted(map(int,state["runs"]))!=list(range(5)):raise RuntimeError("Incomplete five-seed results")
        phase("integrity_after",["research/verify_baseline.py","--full"],root/"integrity_after.json")
        assert json.loads((root/"integrity_after.json").read_text())["source_unchanged"]
        unchanged={p:sha256(p)==h for p,h in state["input_sha256"].items()}
        assert all(unchanged.values()) and source_identity()==source
        atomic_json(root/"input_integrity_after.json",dict(status="passed",files=unchanged))
        atomic_json(root/"completion.json",dict(status="complete",universe="sp500",model="TRACE-VQ",
            stage1_runs=1,stage1_seed=42,stage1_checkpoint=first["checkpoint"],stage1_sha256=first["checkpoint_sha256"],
            stage2_runs=5,stage2_seeds=list(range(5)),standalone_predictions_exact=True,original_unchanged=True,
            input_files_unchanged=True,source_files_unchanged=True,prediction_ensemble=False,finished=now()))
        state.update(status="complete",current_phase=None,finished=now());save()
        print("TRACE-VQ SP500 Stage1seed42 + Stage2seeds0-4 complete",flush=True)
    except BaseException as error:
        state.update(status="failed",error=str(error),traceback=traceback.format_exc());save();raise


if __name__=="__main__":main()
