"""Publish aggregate-only results for one completed batch03 or batch04."""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
METRICS=("IC","RankIC","AR","Sharpe","MDD","turnover")


def portable(text):return re.sub(r"/home/[^/\s\"'`]+","USER_HOME",text.replace(str(ROOT),"."))


def number(value,metric):
    if value is None or pd.isna(value):return ""
    return f"{value*(100 if metric in ('AR','MDD','turnover') else 1):.{5 if metric in ('IC','RankIC') else 3}f}"


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args();folder=args.output_dir.resolve();assert folder.is_relative_to(ROOT/"artifacts/research")
    state=json.loads((folder/"progress.json").read_text());batch=state["batch"]
    assert state["status"] in ("complete","complete_with_failures") and batch in (3,4)
    frame=pd.read_csv(folder/"results.csv");models=frame[frame.role.eq("new_model")];complete=models[models.status.eq("complete")]
    assert len(models)==(16 if batch==3 else 14)
    for record in state["runs"].values():
        if record["role"]=="new_model" and record["status"]=="complete":
            assert record["standalone_exact"] and record["stage1_unchanged"] and not record["stage1_retrained"]
    for row in complete.to_dict("records"):
        base=frame[(frame.universe==row["universe"])&(frame.variant==row["baseline_variant"])].iloc[0]
        for metric in METRICS:
            difference=row[metric]-base[metric]
            assert (pd.isna(row["delta_"+metric]) and pd.isna(difference)) or math.isclose(row["delta_"+metric],difference,abs_tol=1e-12)
    queue=Path(state["queue_root"]);preflight=json.loads(Path(state["gpu_preflight"]).read_text())
    original=json.loads((folder/"original_integrity_after.log").read_text())
    assert original["source_unchanged"] and preflight["status"]=="passed"
    log=(queue/"unit_tests.log").read_text();tests=re.search(r"^Ran (\d+) tests",log,re.MULTILINE)
    title="FSQ状态变化与记忆" if batch==3 else "专家路由"
    lines=[f"# 第{batch:02d}批：{title}","",f"状态：`{state['status']}`；新增Stage2完整结果 {len(complete)}/{len(models)}。"
           "Stage1复用对应VQ/FSQ seed42，Stage2 seed0；无新增Stage1训练。","",
           "训练2009–2020，验证2021–2022，测试2023–2025。验证检查点按主标签MSE加本变体可用的原辅助项选择。"
           "TopkDropout使用all报价、top30/drop5、risk0.95、close、买入0.0005/卖出0.0015、最低费0。","",
           "AR、MDD、换手在CSV中为比例，下表为百分比。换手为Qlib逐日报告均值；MDD越接近0越好。","",
           "| 股票池 | 方案 | 对照 | IC | RankIC | 净年化 % | Sharpe | MDD % | 日均换手 % |",
           "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in frame.to_dict("records"):
        lines.append(f"| {row['universe']} | {row['variant']} | {row.get('baseline_variant','')} | "+
                     " | ".join(number(row.get(m),m) for m in METRICS)+" |")
    lines += ["","**相对同股票池、同seed、对应VQ/FSQ基线的差值**。正MDD差值表示回撤减少。","",
              "| 股票池 | 变体 | ΔIC | ΔRankIC | Δ净年化百分点 | ΔSharpe | Δ回撤百分点 | Δ换手百分点 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in models.to_dict("records"):
        lines.append(f"| {row['universe']} | {row['variant']} | "+" | ".join(number(row.get('delta_'+m),m) for m in METRICS)+" |")
    timing=["universe","variant","status","stage1_seed","stage2_seed","new_training","reused_reference",
            "training_seconds","evaluation_backtest_seconds","evaluation_seconds","backtest_seconds",
            "gpu_peak_allocated_mib","gpu_peak_reserved_mib","total_parameters","trainable_parameters",
            "moe_trainable_parameters","moe_activated_parameters_estimate","routed_experts","activated_routed_experts"]
    frame.reindex(columns=timing).to_csv(folder/"timing_memory.csv",index=False)
    routing_cols=[column for column in frame if column.startswith("routing_")]+["universe","variant","status"]
    models.reindex(columns=routing_cols).to_csv(folder/"routing_diagnostics.csv",index=False)
    lines += ["",f"**实际资源**。完整新训练fit合计 {complete.training_seconds.sum()/3600:.2f} 小时，"
              f"预测、独立恢复评估和回测 {complete.evaluation_backtest_seconds.sum()/60:.2f} 分钟。"
              "fit包含训练内验证，数据加载等准备时间另存training_result.json。逐项见timing_memory.csv；"
              "显存为PyTorch allocated/reserved峰值，不含桌面和其他进程。整卡采样记录保留在本地队列目录。","",
              "**实现与比较边界**。"]
    if batch==3:
        lines += ["历史特征来自既有冻结FSQ h与坐标，不使用标签。delta为1/5/20市场交易日坐标差加缺失掩码；"
                  "dwell截断252日并log1p；trans为仅训练期拟合、Laplace=1的三维8×8转移矩阵查询；"
                  "vol用过去5/20日的总体标准差和有效观测比例，不含当日。","",
                  "2/3-gram只使用之前完整code，4096桶、32维嵌入、当前h的sigmoid门。prototype以冻结h为表示、"
                  "每日code均值作EMA(.99)，先读上一日期原型，再更新当天；跨split因果延续。"
                  "训练日期虽打乱，历史特征已按真实时间预计算，不随训练访问顺序改写。","",
                  "新增历史特征经零初始化残差投影注入128维Stage2表示；保留原FSQ的128维因子及CA。"
                  "code_hist_state使用三维边际8档占比24维与归一化熵3维，零初始化投影影响专家门控。"]
    else:
        lines += ["biasbal仅移除负载/重要性均衡辅助损失，保留外层beta范数项；负载EMA=.99、bias步长=.001，"
                  "bias只改变top-k索引，权重来自原概率。验证/推理不更新偏置。","",
                  "expert_choice容量为ceil(N*k/E)，保留全股票共享专家；每股被选专家数及零路由比例见routing_diagnostics.csv。"
                  "finegrained分别为8专家/宽16/k4与16专家/宽8/k8，输出宽度64、共享专家不变，"
                  "记录总参数和激活参数估计；并非第一批只增加专家数的配置。","",
                  "Soft MoE使用每专家4槽，dispatch沿股票归一化、combine沿所有槽归一化，保留共享分支。"
                  "固定槽容量不额外加原top-k均衡损失，beta范数项保留。"
                  "FSQ路由固定第0维、[0,1,2]/[3,4]/[5,6,7]三组，各一个专家；原学习路由冻结。"
                  "其路由专家数同时由2变3，容量和路由形式不可单独归因。","",
                  "expert_k_all保持2个路由专家全部激活；top1已由baseline_vq行直接复用，无重复训练。"
                  "激活参数是按设定k的估计；expert_choice的逐股票实际覆盖统计单独保存。"]
    lines += ["","**描述性判断**。所有新配置仅seed0，不据此宣称稳定性或统计显著。"," "]
    for row in models.to_dict("records"):
        if row["status"]!="complete":
            lines.append(f"- 未完成 {row['universe']}/{row['variant']}：{row.get('error','')}")
        else:
            d=[row.get('delta_'+m,float('nan')) for m in ('RankIC','AR','Sharpe','MDD')]
            judgement="四项主要指标均未下降" if all(pd.notna(v) and v>=0 for v in d) else "有指标取舍" if any(pd.notna(v) and v>0 for v in d) else "未见四项主要指标改善"
            lines.append(f"- {row['universe']}/{row['variant']}：{judgement}。")
    lines += ["","实现位于trace_vq/models/batch34.py、models/experts/routing_research.py、data/fsq_history.py；"
              "训练入口trace_vq/cli/batch34.py，串行编排research/run_batch34.py。独立有效配置见configs/，"
              "原始权重、特征缓存、预测明细和日志均在本地忽略的artifacts/research/中。",""]
    report="\n".join(lines).replace("\n \n","\n\n");(folder/"README.md").write_text(report)
    public=ROOT/"experiments"/folder.name;public.mkdir(exist_ok=True)
    for name in ("results.csv","variant_results.csv","baselines.csv","timing_memory.csv","routing_diagnostics.csv"):
        table=pd.read_csv(folder/name).drop(columns=["output_dir","checkpoint","config"],errors="ignore")
        (public/name).write_text(portable(table.to_csv(index=False)))
    (public/"README.md").write_text(portable(report));(public/"plan.json").write_text(portable((folder/"plan.json").read_text()))
    validation=dict(batch=batch,gpu_preflight_status=preflight["status"],gpu_preflight_checks=len(preflight["checks"]),
        unit_tests=int(tests.group(1)) if tests else None,unit_tests_passed=bool(re.search(r"^OK$",log,re.MULTILINE)),
        standalone_exact_runs=len(complete),original_source_unchanged=original["source_unchanged"],source_sha256=state["source_sha256"])
    (public/"validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    for r in state["runs"].values():
        if r["role"]!="new_model":continue
        source=Path(r["config"]);destination=public/"configs"/r["universe"]/r["variant"]/"seed0.yaml"
        destination.parent.mkdir(parents=True,exist_ok=True);destination.write_text(portable(source.read_text()))
    index_path=ROOT/"experiments/index.csv";index=pd.read_csv(index_path)
    if folder.name not in set(index.batch):
        entry=dict(batch=folder.name,description=f"第{batch:02d}批：{title}",universes="CSI300 / CSI800",seeds="Stage1 42; Stage2 0",
                   records=f"{len(models)}次新训练，原基线直接复用",finding="见本批全部差值与负结果；仅单seed",readme=f"{folder.name}/README.md")
        pd.concat([index,pd.DataFrame([entry])],ignore_index=True).to_csv(index_path,index=False)
    readme=ROOT/"experiments/README.md";text=readme.read_text()
    if f"]({folder.name}/README.md)" not in text:readme.write_text(text+f"\n第{batch:02d}批结果见 [{folder.name}]({folder.name}/README.md)。\n")
    subprocess.run([sys.executable,"research/organize_experiment_results.py"],cwd=ROOT,check=True)
    for path in public.rglob("*"):
        if path.is_file():
            assert path.suffix in (".md",".csv",".json",".yaml") and str(Path.home()) not in path.read_text()
    result=dict(status="passed",batch=batch,batch_status=state["status"],new_training_complete=len(complete),
                expected_new_training=len(models),public_summary=str(public),report=str(folder/"README.md"))
    atomic=folder/"report_verification.json";atomic.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":main()
