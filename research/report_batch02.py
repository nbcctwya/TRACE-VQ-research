"""Aggregate-only batch02 report; keep paired seeds separate from ensembles."""
import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("IC", "RankIC", "AR", "Sharpe", "MDD", "turnover")


def portable(text):
    return re.sub(r"/home/[^/\s\"'`]+", "USER_HOME", text.replace(str(ROOT), "."))


def fmt(value, metric):
    if pd.isna(value):
        return ""
    scale = 100 if metric in ("AR", "MDD", "turnover") else 1
    return f"{value*scale:.{5 if metric in ('IC','RankIC') else 3}f}"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args()
    folder=args.output_dir.resolve()
    assert folder.is_relative_to(ROOT/"artifacts/research")
    state=json.loads((folder/"progress.json").read_text())
    assert state["status"] in ("complete","complete_with_failures")
    results=pd.read_csv(folder/"results.csv")
    if "sampled_gpu_process_peak_mib" in results:
        results["sampled_gpu_process_peak_mib"]=results.sampled_gpu_process_peak_mib.mask(results.sampled_gpu_process_peak_mib.eq(0))
    resource_peaks=folder/"resource_peaks.json"
    if resource_peaks.exists():
        peaks=json.loads(resource_peaks.read_text())
        device_peaks=[]
        for row in results.to_dict("records"):
            prefix=f"{row['universe']}.{row['variant']}.seed{row['stage2_seed']}."
            values=[value["device_memory_peak_mib"] for name,value in peaks.items() if name.startswith(prefix)]
            device_peaks.append(max(values) if values else float("nan"))
        results["sampled_whole_gpu_peak_mib"]=device_peaks
    results.to_csv(folder/"results.csv",index=False)
    assert len(results)==24 and not results.duplicated(["universe","variant","stage2_seed"]).any()
    pairs=[]
    for market in ("csi300","csi800"):
        for prefix in ("","fsq_"):
            for seed in ([0] if prefix else range(5)):
                a=results[(results.universe==market)&(results.variant==prefix+"turnover")&(results.stage2_seed==seed)]
                b=results[(results.universe==market)&(results.variant==prefix+"turnover_control")&(results.stage2_seed==seed)]
                if len(a)!=1 or len(b)!=1 or a.iloc[0].status!="complete" or b.iloc[0].status!="complete":
                    continue
                a,b=a.iloc[0],b.iloc[0]
                row=dict(universe=market,family="fsq" if prefix else "vq",stage2_seed=seed,
                         treatment=prefix+"turnover",control=prefix+"turnover_control")
                for metric in METRICS:
                    delta=a[metric]-b[metric]
                    assert (pd.isna(a["delta_"+metric]) and pd.isna(delta)) or math.isclose(a["delta_"+metric],delta,abs_tol=1e-12)
                    row["delta_"+metric]=delta
                pairs.append(row)
    paired=pd.DataFrame(pairs)
    paired.to_csv(folder/"paired_deltas.csv",index=False)
    summary=[]
    for (market,family),group in paired.groupby(["universe","family"]):
        row=dict(universe=market,family=family,paired_seeds=len(group),expected_seeds=5 if family=="vq" else 1)
        for metric in METRICS:
            values=group["delta_"+metric]
            row["delta_"+metric+"_mean"]=values.mean()
            row["delta_"+metric+"_std"]=values.std(ddof=1)
            row[metric+"_better_seeds"]=int((values<0 if metric=="turnover" else values>0).sum())
        summary.append(row)
    pair_summary=pd.DataFrame(summary)
    pair_summary.to_csv(folder/"paired_summary.csv",index=False)
    timing_columns=["universe","variant","stage1_seed","stage2_seed","status","new_training","reused_reference",
                    "training_seconds","reference_training_seconds","evaluation_backtest_seconds",
                    "evaluation_seconds","backtest_seconds","gpu_peak_allocated_mib","gpu_peak_reserved_mib",
                    "sampled_gpu_process_peak_mib","sampled_whole_gpu_peak_mib","replay_gpu_peak_reserved_mib"]
    results.reindex(columns=timing_columns).to_csv(folder/"timing_memory.csv",index=False)
    complete=results[results.status=="complete"]
    new=complete[complete.new_training]
    reused=complete[complete.reused_reference]
    if state["status"]=="complete":
        assert len(new)==20 and len(reused)==4 and len(paired)==12
    checks=json.loads((folder/"preflight/gpu_validation.json").read_text())
    original=json.loads((folder/"original_integrity_after.log").read_text())
    test_log=(folder/"unit_tests.log").read_text()
    controller_log=(folder/"controller_tests.log").read_text() if (folder/"controller_tests.log").exists() else ""
    tests=[int(n) for n in re.findall(r"^Ran (\d+) tests",test_log+"\n"+controller_log,re.MULTILINE)]
    assert checks["status"]=="passed" and original["source_unchanged"]
    for record in state["runs"].values():
        if record["status"]=="complete":
            assert record.get("standalone_exact") if record["new_training"] else record.get("reference_predictions_exact")
            if record["new_training"]:
                assert record["stage1_unchanged"] and not record["stage1_retrained"]
    lines=["# 第02批：已有强信号复验", "", f"状态：`{state['status']}`；范围09–10（E1/E2）。", "",
           f"新增完成 {len(new)}/20 次 Stage2 训练，复用 {len(reused)}/4 份 VQ seed0，Stage1 新训练0次。"
           "E1对两个股票池的平滑组和控制组各补seeds1–4；E2为FSQ平滑组和控制组各跑seed0。", "",
           "固定 pairwise_weight=0.1；turnover_weight 为0.05和0。控制组也执行相同的相邻日第二次前向，"
           "相邻交易日按股票对齐；只有训练视图配对。验证检查点沿用原主标签MSE与辅助项，不使用测试集挑选模型。", "",
           "训练2009–2020，验证2021–2022，测试2023–2025。TopkDropout沿用 all 报价、top30/drop5、"
           "risk=0.95、close成交、买入0.0005/卖出0.0015、最低费0。", "",
           "各seed是独立组合指标；均值与样本标准差不是预测集成。以下AR/MDD/换手用百分比，原CSV用比例。", "",
           "| 股票池 | 变体 | seed | 复用 | IC | RankIC | 净年化 % | Sharpe | MDD % | 日均换手 % |",
           "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results.to_dict("records"):
        lines.append(f"| {row['universe']} | {row['variant']} | {int(row['stage2_seed'])} | "
                     f"{'是' if row['reused_reference'] else '否'} | "+" | ".join(fmt(row.get(m),m) for m in METRICS)+" |")
    lines += ["", "**基线与差值**。平滑组主对照为同股票池、同seed、同量化器的零惩罚控制组；"
              "控制组主对照为同seed的VQ/FSQ原基线。results.csv 的 delta_* 为主对照差值，"
              "delta_vs_family_* 为对应原基线差值，FSQ行另有 delta_vs_vq_paired_* 对应VQ配对版本差值。"
              "全部基线见 benchmarks.csv。", "",
              "**五seed配对结果**。paired_deltas.csv列逐seed差值，paired_summary.csv列差值均值/样本标准差和胜出次数。", "",
              "| 股票池 | 量化器 | 完成配对seed数 | ΔRankIC均值 | Δ净年化百分点 | ΔSharpe | Δ回撤百分点 | Δ换手百分点 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in pair_summary.to_dict("records"):
        lines.append(f"| {row['universe']} | {row['family']} | {row['paired_seeds']} | "+
                     " | ".join(fmt(row["delta_"+m+"_mean"],m) for m in ("RankIC","AR","Sharpe","MDD","turnover"))+" |")
    means=pd.read_csv(folder/"mean_std.csv")
    lines += ["", "**各seed指标的均值与样本标准差**。FSQ只有1个seed，不报告标准差。", "",
              "| 股票池 | 变体 | 完成/计划seeds | IC | RankIC | 净年化 % | Sharpe | MDD % | 日均换手 % |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in means.to_dict("records"):
        cells=[]
        for metric in METRICS:
            value=fmt(row[metric+"_mean"],metric)
            if row["completed_seeds"]>1:
                value+=" ± "+fmt(row[metric+"_std"],metric)
            cells.append(value)
        lines.append(f"| {row['universe']} | {row['variant']} | {row['completed_seeds']}/{row['expected_seeds']} | "+" | ".join(cells)+" |")
    lines += ["", "MDD差值为正表示回撤减少，换手差值为负表示换手降低。", "", "**简短判断**。", ""]
    for row in pair_summary.to_dict("records"):
        if row["delta_RankIC_mean"]>0 and row["delta_AR_mean"]>0 and row["delta_Sharpe_mean"]>0:
            judgement="RankIC、净年化和Sharpe同向改善"
        elif row["delta_AR_mean"]>0 or row["delta_RankIC_mean"]>0 or row["delta_Sharpe_mean"]>0:
            judgement="存在指标取舍，未呈现三项同向改善"
        else:
            judgement="未见平均RankIC、净年化或Sharpe改善"
        lines.append(f"- {row['universe']}/{row['family']}：{judgement}；"
                     f"收益改善 {row['AR_better_seeds']}/{row['paired_seeds']} seeds，"
                     f"RankIC改善 {row['RankIC_better_seeds']}/{row['paired_seeds']} seeds，"
                     f"换手降低 {row['turnover_better_seeds']}/{row['paired_seeds']} seeds。")
    failed=results[results.status!="complete"]
    lines += ["", "上述改善判断中的排序指标为RankIC；IC单独列示。FSQ平滑组相对同seed零惩罚控制组的IC均未提高，"
              "CSI800的IC下降约0.00514。相对控制组的改善也不等于相对原FSQ基线的全面改善："
              "CSI300 FSQ平滑组的回撤为-22.00%，原FSQ seed0基线为-11.68%；原基线差值见results.csv中的delta_vs_family_*。", "",
              "未完成项："+("无。" if failed.empty else "；".join(f"{r.universe}/{r.variant}/seed{r.stage2_seed}: {r.get('error','')}" for _,r in failed.iterrows())), "",
              "固定Stage1 seed42不覆盖Stage1随机性；五seed共享同一测试期，不能当作五次独立市场验证或宣称统计显著。"
              "FSQ仅seed0，不能据此判定稳定性。", "",
              f"**实际资源**。新训练fit累计 {new.training_seconds.sum()/3600:.2f} 小时；"
              f"本批预测、独立评估、参考重放和回测累计 {complete.evaluation_backtest_seconds.sum()/60:.2f} 分钟。"
              "逐项见 timing_memory.csv。fit含训练内验证，准备时间另在各training_result.json记录。"
              "复用参考的本批训练耗时为0，历史训练耗时保留为 reference_training_seconds，历史显存未测量则留空。", "",
              "显存同时记录PyTorch allocated/reserved峰值，以及每20秒采样的训练进程nvidia-smi峰值。"
              "采样峰值可能漏掉短时尖峰，不应与PyTorch reserved或整张卡占用混为一项。"
              "驱动未提供进程级采样时该列留空；sampled_whole_gpu_peak_mib为独立监控的整卡采样峰值，包含桌面和其他进程。"
              "参考重放显存另列 replay_gpu_peak_reserved_mib，不冒充历史训练峰值。", "",
              "运行期间monitoring.json和STATUS.md每约20秒更新；卡顿警示与失败信息保存在本地alerts.jsonl、failures.log和各阶段日志。"
              "检查点额外保存Python/NumPy/Torch CPU/CUDA RNG，经过含打乱日期、多worker的CPU/GPU连续/断点训练一致性测试。", "",
              "实现复用 trace_vq/models/next_screen.py 与 trace_vq/data/research_context.py；"
              "新入口 trace_vq/cli/batch02.py，续训记录 trace_vq/training/research_run.py，"
              "编排 research/run_batch02.py，汇总 research/report_batch02.py。独立配置见 configs/。", ""]
    report="\n".join(lines)
    (folder/"README.md").write_text(report)
    public=ROOT/"experiments"/folder.name
    public.mkdir(exist_ok=True)
    for name in ("results.csv","mean_std.csv","paired_deltas.csv","paired_summary.csv","benchmarks.csv","timing_memory.csv"):
        frame=(results.copy() if name=="results.csv" else pd.read_csv(folder/name)).drop(columns=["output_dir","checkpoint","config"],errors="ignore")
        (public/name).write_text(portable(frame.to_csv(index=False)))
    (public/"README.md").write_text(portable(report))
    (public/"plan.json").write_text(portable((folder/"plan.json").read_text()))
    validation=dict(gpu_preflight_checks=len(checks["checks"]),gpu_preflight_status=checks["status"],
                    unit_tests=sum(tests) if tests else None,
                    unit_tests_passed=bool(re.search(r"^OK$",test_log,re.MULTILINE)) and
                                      (not controller_log or bool(re.search(r"^OK$",controller_log,re.MULTILINE))),
                    completed_new_training=len(new),completed_reused_seed0=len(reused),original_source_unchanged=original["source_unchanged"],
                    original_full_hash_check=original["full_hash_check"],source_sha256=state["source_sha256"])
    (public/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    for record in state["runs"].values():
        source=Path(record["config"])
        target=public/"configs"/record["universe"]/record["variant"]/f"seed{record['stage2_seed']}.yaml"
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(portable(source.read_text()))
    for path in public.rglob("*"):
        if path.is_file():
            assert path.suffix in (".csv",".md",".json",".yaml")
            assert str(ROOT) not in path.read_text() and str(Path.home()) not in path.read_text()
    index_file=ROOT/"experiments/index.csv"
    index=pd.read_csv(index_file)
    if folder.name not in set(index.batch):
        entry=dict(batch=folder.name,description="第02批：已有强信号复验",universes="CSI300 / CSI800",
                   seeds="VQ Stage2 0–4; FSQ Stage2 0; Stage1 42",records="20次新训练、4份复用seed0",
                   finding="见配对差值和五seed均值/标准差；不等同预测集成",readme=f"{folder.name}/README.md")
        pd.concat([index,pd.DataFrame([entry])],ignore_index=True).to_csv(index_file,index=False)
    readme=ROOT/"experiments/README.md"
    text=readme.read_text()
    if f"]({folder.name}/README.md)" not in text:
        readme.write_text(text+f"\n第02批的配对平滑复验见 [{folder.name}]({folder.name}/README.md)。\n")
    verification=dict(status="passed",batch_status=state["status"],new_training_complete=len(new),
                       reused_seed0=len(reused),complete_pairs=len(paired),public_summary=str(public),
                       report=str(folder/"README.md"))
    (folder/"report_verification.json").write_text(json.dumps(verification,indent=2)+"\n")
    print(json.dumps(verification),flush=True)


if __name__=="__main__":
    main()
