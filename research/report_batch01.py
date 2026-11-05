"""Summarize completed batch01 artifacts and copy aggregate-only public results."""
import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def number(value, digits=4, scale=1):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return ""
    return f"{value * scale:.{digits}f}" if math.isfinite(value) else ""


def portable(text):
    text = text.replace(str(ROOT), ".")
    return re.sub(r"/home/[^/\s\"'`]+", "USER_HOME", text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    assert folder.is_relative_to(ROOT / "artifacts/research")
    state = json.loads((folder / "progress.json").read_text())
    assert state["status"] in ("complete", "complete_with_failures")
    all_results = pd.read_csv(folder / "results.csv")
    variants = all_results[all_results.role != "baseline"].copy()
    readable = []
    for row in variants.to_dict("records"):
        readable.append({"股票池": row["universe"], "方案": row["variant"], "状态": row["status"],
            "对照": row.get("baseline_variant"), "IC": row.get("IC"), "RankIC": row.get("RankIC"),
            "净年化百分比": row.get("AR", float("nan")) * 100,
            "Sharpe": row.get("Sharpe"), "最大回撤百分比": row.get("MDD", float("nan")) * 100,
            "日均报告换手百分比": row.get("turnover", float("nan")) * 100,
            "IC差值": row.get("delta_IC"),
            "RankIC差值": row.get("delta_RankIC"), "净年化差值百分点": row.get("delta_AR", float("nan")) * 100,
            "Sharpe差值": row.get("delta_Sharpe"), "回撤差值百分点": row.get("delta_MDD", float("nan")) * 100,
            "换手差值百分点": row.get("delta_turnover", float("nan")) * 100,
            "训练分钟": row.get("training_seconds", float("nan")) / 60,
            "评估回测分钟": row.get("evaluation_backtest_seconds", float("nan")) / 60,
            "峰值显存reservedMiB": row.get("gpu_peak_reserved_mib")})
    pd.DataFrame(readable).to_csv(folder / "results_readable.csv", index=False, encoding="utf-8-sig")
    timing_columns = ["universe", "variant", "status", "training_seconds", "evaluation_backtest_seconds",
                      "evaluation_seconds", "backtest_seconds", "prediction_seconds", "gpu_peak_allocated_mib",
                      "gpu_peak_reserved_mib", "trainable_parameters", "total_parameters"]
    variants.reindex(columns=timing_columns).to_csv(folder / "timing_memory.csv", index=False)
    item_definitions = [("G2", "latent_mlp", 0), ("H4", "tradable_filter", 0), ("I1", "ens_", 6),
                        ("G1", "tune800_", 4), ("K1/J2", "gated_attn", 2), ("B3", "fsq_wo_ca", 2),
                        ("B2", "fsq_ca_dim", 2), ("B1", "fsq_grid", 2)]
    statuses = []
    for index, (item, prefix, expected) in enumerate(item_definitions, 1):
        if expected == 0:
            record = state["checks"][item]
            statuses.append(dict(number=index, item=item, name=prefix, status=record["status"],
                                 expected_new_training_runs=0, complete_results=0, failures=""))
        else:
            selected = variants[variants.variant.str.startswith(prefix)]
            failed = selected[selected.status != "complete"]
            complete = int((selected.status == "complete").sum())
            statuses.append(dict(number=index, item=item, name=prefix,
                status="complete" if complete == expected else "incomplete", expected_results=expected,
                expected_new_training_runs=0 if item == "I1" else expected, complete_results=complete,
                failures="; ".join(f"{r['universe']}: {r.get('error', '')}" for r in failed.to_dict("records"))))
    pd.DataFrame(statuses).to_csv(folder / "item_status.csv", index=False)

    lines = ["# 第01批：快速起步与公共接口", "",
             "范围：清单 01–08。Stage 1 沿用各股票池 VQ/FSQ seed 42；新增训练仅 Stage 2 seed 0。G1 只跑 CSI800 四个独立配置。", "",
             f"批次状态：`{state['status']}`。新增训练计划 12 次，集成结果计划 6 份，G2/H4 两项检查。", "",
             "测试区间为 2023–2025 年，沿用统一指标与 all 报价 TopkDropout 规则。"
             "集成为已有 seeds 0–4 的预测分数逐样本算术平均；混合组中 VQ/FSQ 各占 0.5。", "",
             "**结果**。AR/MDD/换手在原始 CSV 中是比例；下表换成百分比。换手为 Qlib 逐日报告 turnover 的均值。", "",
             "| 股票池 | 方案 | 对照 | IC | RankIC | 净年化 % | Sharpe | 最大回撤 % | 日均换手 % |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in variants.to_dict("records"):
        lines.append(f"| {row['universe']} | {row['variant']} | {row.get('baseline_variant', '')} | " +
                     " | ".join(number(row.get(k), 5 if k in {"IC", "RankIC"} else 3,
                                               100 if k in {"AR", "MDD", "turnover"} else 1)
                                for k in ("IC", "RankIC", "AR", "Sharpe", "MDD", "turnover")) + " |")
    lines += ["", "**相对对照的差值**。MDD 差值为正表示回撤减少。", "",
              "| 股票池 | 方案 | ΔIC | ΔRankIC | Δ净年化百分点 | ΔSharpe | Δ回撤百分点 | Δ换手百分点 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in variants.to_dict("records"):
        lines.append(f"| {row['universe']} | {row['variant']} | " +
                     " | ".join(number(row.get(k), 5 if k in {"delta_IC", "delta_RankIC"} else 3,
                                               100 if k in {"delta_AR", "delta_MDD", "delta_turnover"} else 1)
                                for k in ("delta_IC", "delta_RankIC", "delta_AR", "delta_Sharpe", "delta_MDD", "delta_turnover")) + " |")
    lines += ["", "VQ/FSQ 单模型变体分别与同股票池对应 seed 0 基线比较；五 seed 集成分别与对应 seed 0 基线比较。"
              "混合集成的主对照为 VQ 五 seed 集成，CSV 另列相对 FSQ 五 seed 集成的差值。", "",
              "**耗时与资源**。训练秒数仅 fit（含训练内验证）；评估回测秒数包含训练后预测、独立恢复评估和回测。"
              "显存为各阶段 PyTorch max_memory_reserved 的最大值，另保存 max_memory_allocated；不含其他进程和非 PyTorch 上下文。"
              "CPU 集成/回测无 GPU 分配记 0；未测量的字段留空。", "",
              "| 股票池 | 方案 | 训练分钟 | 评估回测分钟 | 峰值 reserved MiB |",
              "| --- | --- | ---: | ---: | ---: |"]
    for row in variants.to_dict("records"):
        lines.append(f"| {row['universe']} | {row['variant']} | {number(row.get('training_seconds'), 2, 1/60)} | "
                     f"{number(row.get('evaluation_backtest_seconds'), 2, 1/60)} | {number(row.get('gpu_peak_reserved_mib'), 1)} |")
    successful = variants[variants.status == "complete"]
    lines += ["", f"最终完整尝试累计 fit {number(successful.training_seconds.sum(), 2, 1/3600)} 小时，"
              f"预测、独立评估和回测 {number(successful.evaluation_backtest_seconds.sum(), 2, 1/60)} 分钟。"
              f"最高 PyTorch reserved 显存 {number(successful.gpu_peak_reserved_mib.max(), 1)} MiB。"
              "数据读取、缓存验证和进程初始化等准备时间另保存在本地 training_result.json，不计入 fit。"]
    recovery = state.get("recovery", {})
    if recovery:
        lost_seconds = recovery.get("known_interrupted_training_seconds_lower_bound")
        lines += ["", "本批执行期间电脑重启。已完成的六份集成与 G2/H4 检查直接复用；首个未完成的 "
                  "CSI800/tune800_experts4 训练保存至 epoch 4 后中断。由于检查点未保存完整随机数状态，"
                  "该次产物已在本地归档，并从相同 seed 0 重新训练。中断尝试不计作新增独立实验。"
                  f"其已记录训练耗时至少 {number(lost_seconds, 2, 1/60)} 分钟，未记录完整峰值显存；"
                  "上述资源表使用最终完整尝试的实测值，另不计机器关机与等待时间。"]
    lines += ["", "**检查与解释**。G2 的 MLP 已存在，跳过训练。H4 未新增 ST/停牌/涨跌停过滤，保持既有策略；"
              "Qlib 中国市场的有效默认涨跌停阈值为 0.095，具体见 checks/H4.json。", "",
              "gated_attn 为 SDPA 输出与 out_proj 之间的逐头 sigmoid 门，零权重/偏置初始化对应 0.5 门值。"
              "fsq_ca_dim 使用整数单位 abs(u-q) 三维输入、零初始化适配器。fsq_grid 的 Stage 1 仍为 128 维，"
              "下游潜在因子与相应投影变成 3 维；CA 使用归一化网格坐标量化误差，预测路径不调用 from_grid。"
              "该实验同时改变因子数和 CA 几何空间，不能将效果仅归因于坐标形式。", "",
              "以下为逐股票池、单 seed 的描述性判断，不能据此宣称统计显著或证明机制。", ""]
    for row in variants.to_dict("records"):
        if row["status"] != "complete":
            lines.append(f"- 未完成 {row['universe']}/{row['variant']}：{row.get('error', '')}")
            continue
        deltas = [row.get("delta_" + k, float("nan")) for k in ("RankIC", "AR", "Sharpe", "MDD")]
        if all(math.isfinite(float(x)) and x >= 0 for x in deltas):
            judgement = "相对主对照四项主要指标均未下降"
        elif any(math.isfinite(float(x)) and x > 0 for x in deltas):
            judgement = "指标有取舍，详见差值表"
        else:
            judgement = "未见主要指标改善"
        lines.append(f"- {row['universe']}/{row['variant']}：{judgement}。")
    models = successful[successful.role == "new_model"]
    for market in ("csi300", "csi800"):
        candidates = models[models.universe == market]
        if len(candidates):
            best = candidates.loc[candidates.Sharpe.idxmax()]
            lines += ["", f"{market} 已完成的新单模型中 Sharpe 最高为 `{best.variant}`："
                      f"净年化 {number(best.AR, 2, 100)}%，Sharpe {number(best.Sharpe, 3)}，"
                      f"MDD {number(best.MDD, 2, 100)}%；相对其 `{best.baseline_variant}`，"
                      f"净年化差值 {number(best.delta_AR, 2, 100)} 个百分点、"
                      f"Sharpe 差值 {number(best.delta_Sharpe, 3)}、"
                      f"回撤差值 {number(best.delta_MDD, 2, 100)} 个百分点、"
                      f"RankIC 差值 {number(best.delta_RankIC, 5)}。"]
    completion = "本批无未完成训练项。" if state["status"] == "complete" else "未完成的训练项见 item_status.csv。"
    lines += ["", completion + "G2/H4 为按要求跳过训练的检查项。"
              "四份复用基线的完整指标见 baselines.csv，所有六项指标差值见 results.csv。"]
    lines += ["", "代码：trace_vq/models/batch01.py、trace_vq/cli/batch01.py、research/run_batch01.py；"
              "注意力门接入 trace_vq/models/prediction/temporal_transformer.py。配置在本批 configs/ 及各运行的 effective.config.yaml。", ""]
    report = "\n".join(lines)
    (folder / "README.md").write_text(report)

    public = ROOT / "experiments" / folder.name
    public.mkdir(exist_ok=True)
    for name in ("results.csv", "variant_results.csv", "baselines.csv", "results_readable.csv", "timing_memory.csv", "item_status.csv"):
        frame = pd.read_csv(folder / name).drop(columns=["output_dir"], errors="ignore")
        (public / name).write_text(portable(frame.to_csv(index=False)))
    (public / "README.md").write_text(portable(report))
    (public / "plan.json").write_text(portable((folder / "plan.json").read_text()))
    preflight = json.loads((folder / "preflight/gpu_validation.json").read_text())
    invariance = json.loads((folder / "config_invariance.json").read_text())
    original = json.loads((folder / "original_integrity_after.log").read_text())
    test_log = (folder / "unit_tests.log").read_text()
    test_count = re.search(r"^Ran (\d+) tests", test_log, re.MULTILINE)
    validation = dict(
        gpu_preflight_status=preflight["status"], gpu_preflight_checks=len(preflight["checks"]),
        gpu_preflight_failures=len(preflight["failures"]),
        config_invariance_status=invariance["status"], config_invariance_checks=len(invariance["checks"]),
        unit_tests=int(test_count.group(1)) if test_count else None,
        unit_tests_passed=bool(re.search(r"^OK$", test_log, re.MULTILINE)),
        standalone_prediction_exact_runs=sum(r.get("role") == "new_model" and r.get("standalone_exact", False)
                                            for r in state["runs"].values()),
        original_source_unchanged=original["source_unchanged"],
        original_verification_full_hash_check=original["full_hash_check"],
        original_regular_files_hashed=original["regular_files_hashed"],
        source_sha256=state["source_sha256"])
    (public / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
    checks = public / "checks"
    checks.mkdir(exist_ok=True)
    for name in ("G2.json", "H4.json", "README.md"):
        (checks / name).write_text(portable((folder / "checks" / name).read_text()))
    for record in state["runs"].values():
        if record.get("role") == "new_model" and record.get("status") == "complete":
            source = Path(record["config"])
            destination = public / "configs" / record["universe"] / record["variant"] / "seed0.yaml"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(portable(source.read_text()))
        if record.get("role") == "ensemble" and record.get("status") == "complete":
            destination = public / "configs" / record["universe"] / record["variant"]
            destination.mkdir(parents=True, exist_ok=True)
            source = Path(record["output_dir"])
            for original, exported in (("input_manifest.json", "input_manifest.json"),
                                       ("backtest/backtest_config.json", "backtest.json")):
                (destination / exported).write_text(portable((source / original).read_text()))
    index_file = ROOT / "experiments/index.csv"
    index = pd.read_csv(index_file)
    if folder.name not in set(index.batch):
        entry = dict(batch=folder.name, description="第01批：快速起步与公共接口", universes="CSI300 / CSI800（G1 仅 CSI800）",
                     seeds="Stage 1: 42; Stage 2: 0; 集成使用既有0–4", records="12次Stage2、6份集成、2项检查",
                     finding="见本批 README 和差值表；仅执行01–08", readme=f"{folder.name}/README.md")
        pd.concat([index, pd.DataFrame([entry])], ignore_index=True).to_csv(index_file, index=False)
    main_readme = ROOT / "experiments/README.md"
    text = main_readme.read_text()
    if f"]({folder.name}/README.md)" not in text:
        text += f"\n后续第01批结果见 [{folder.name}]({folder.name}/README.md)。根目录的64条单 seed 与60行多 seed 表保留前期快照，本批结果单独保存。\n"
        main_readme.write_text(text)
    print(json.dumps({"status": state["status"], "report": str(folder / "README.md"),
                      "public_summary": str(public), "result_rows": len(variants),
                      "complete_rows": int((variants.status == "complete").sum())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
