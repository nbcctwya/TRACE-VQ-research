"""Build portable, deduplicated result documents from curated inputs only.

No data, weights, predictions, training logs or baseline evidence are read.
Historical quote/metric variants remain in their own documented archives.
"""
import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments"
METRICS = ("IC", "RankIC", "AR", "Sharpe", "MDD", "turnover")
ALL_METRICS = METRICS + ("ICIR", "RankICIR", "STD", "Sortino", "Calmar", "num_test_days")
ALIASES = {"baseline_vq": "baseline", "baseline_fsq": "fsq"}
DESCRIPTIONS = {
    "gated_attn": "Stage2注意力输出与投影之间的逐头sigmoid门。",
    "fsq_wo_ca": "FSQ关闭量化置信度适配器。",
    "fsq_ca_dim": "FSQ用整数网格单位的三维abs(u-q)作为适配器输入。",
    "fsq_grid": "FSQ下游直接使用归一化三维网格，同时将潜在因子数改为3。",
    "tune800_experts4": "CSI800单独将路由专家数改为4。",
    "tune800_experts8": "CSI800单独将路由专家数改为8。",
    "tune800_aux001": "CSI800单独将辅助损失权重改为0.001。",
    "tune800_heads4": "CSI800单独将Stage2注意力头数改为4。",
    "fsq_turnover": "原FSQ底座加相邻交易日标准化分数平滑，pairwise=0.1、平滑=0.05。",
    "fsq_turnover_control": "FSQ相同配对流程的零平滑惩罚控制组，保留pairwise=0.1。",
    "code_delta": "FSQ相对过去1/5/20市场交易日的坐标差和缺失掩码。",
    "code_dwell": "FSQ各维当前档位的持续时间，截断252日后log1p。",
    "code_trans": "仅训练期估计的三维8×8状态转移概率。",
    "code_vol": "FSQ过去5/20日坐标波动和有效观测比例。",
    "fsq_gram2": "过去2个完整FSQ code的哈希嵌入与门控记忆。",
    "fsq_gram3": "过去3个完整FSQ code的哈希嵌入与门控记忆。",
    "code_hist_state": "同日FSQ边际档位分布与熵影响专家门控。",
    "proto_memory": "读取历史code的冻结h EMA原型，按日期先读后更新。",
    "moe_biasbal": "历史负载偏置均衡替代MoE均衡辅助损失。",
    "expert_choice": "专家从完整当日截面选择固定容量股票。",
    "moe_finegrained8": "8个宽16的路由专家，激活4个，保留共享专家。",
    "moe_finegrained16": "16个宽8的路由专家，激活8个，保留共享专家。",
    "soft_moe": "每专家4槽的dispatch/专家处理/combine及共享分支。",
    "fsq_sort_route": "FSQ第0维固定低/中/高三组路由。",
    "expert_k_all": "保持2个路由专家全部激活，top1参考直接复用。",
    "kda_temporal": "Stage2时间块改为原生PyTorch逐通道门控Delta递推，结构读取token置末尾。",
    "mhc": "Stage2两条残差流，动态pre/post映射与20轮Sinkhorn双随机残差混合。",
    "raven_context_experts": "FSQ路由5/10/20日窗口专家，短窗口塔不读取更早输入，保留共享分支。",
    "s2_cross_attn": "Stage2载荷头前增加同日完整股票截面注意力，零初始化残差输出。",
}
PRIMARY_TABLES = {
    "innovation_screen_20261026": "final_comparison.csv",
    "next_directions_20261027": "final_comparison.csv",
    "mechanism_screen_20261028": "final_comparison.csv",
    "combination_screen_20261029": "final_comparison.csv",
    "multiseed_validation_20261030": "seed_metrics.csv",
    "sp500_multiseed_20261031": "seed_metrics.csv",
    "sp500_candidates_multiseed_20261031": "seed_metrics.csv",
    "historical_csi300_ablations": "metrics/seed_metrics.csv",
    "historical_csi800_baseline": "seed_metrics.csv",
    "historical_stage1_multiseed": "per_seed_metrics.csv",
    "historical_full_pipeline_multiseed": "full_pipeline_seed_metrics.csv",
    "sp500_baseline_protocol_20261031": "ensemble_metrics.json",
    "sp500_protocol_recheck_20261031_203648": "cross_baseline_metric_values.csv",
    "sp500_effect_diagnosis_20261101": "prediction_summary.csv",
    "batch01_20261103": "results.csv",
    "batch02_20261103": "results.csv",
    "batch03_20261105": "results.csv",
    "batch04_20261105": "results.csv",
    "batch05_20261106": "results.csv",
}


def present(value):
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def load_config(path):
    return OmegaConf.to_container(OmegaConf.load(ROOT / path), resolve=False)


def is_stage1(path):
    return any(part.startswith("stage1") for part in Path(path).parts)


def update_configs():
    index = pd.read_csv(EXP / "config_index.csv")
    for batch, variant in {"025": "wo_ta", "034": "full", "037": "wo_ca", "038": "wo_sr"}.items():
        index.loc[index.batch.eq("historical_csi300_ablations/"+batch), "variant"] = variant
    old_baseline = index.batch.eq("historical_csi800_baseline")
    index.loc[old_baseline, "variant"] = "baseline"
    known = set(index.config_file)
    additions = []
    for batch in ("batch01_20261103", "batch02_20261103", "batch03_20261105", "batch04_20261105", "batch05_20261106"):
        for path in sorted((EXP / batch / "configs").rglob("*.yaml")):
            relative = str(path.relative_to(ROOT))
            if relative in known:
                continue
            cfg = load_config(relative)
            additions.append(dict(batch=batch, config_file=relative, universe=cfg["data"]["universe"],
                                  variant=cfg["research"]["variant"], train_seed=cfg["train"]["seed"],
                                  source_file=str(Path(cfg["artifact_root"]) / "effective.config.yaml")))
    index = pd.concat([index, pd.DataFrame(additions)], ignore_index=True)
    assert index.config_file.is_unique
    assert set(index.config_file) == {str(p.relative_to(ROOT)) for p in EXP.rglob("*.yaml")}
    index.to_csv(EXP / "config_index.csv", index=False)
    flattened = []
    for row in index.to_dict("records"):
        cfg = load_config(row["config_file"])
        predictor, vq, train = cfg.get("predictor", {}), cfg.get("vqvae", {}), cfg.get("train", {})
        transformer, research = predictor.get("transformer", {}), cfg.get("research", {})
        quantizer = vq.get("quantizer", {})
        flattened.append(dict(
            batch=row["batch"], universe=row["universe"], variant=row["variant"],
            training_stage="stage1" if is_stage1(row["config_file"]) else "stage2", train_seed=row["train_seed"],
            config_file=row["config_file"], seq_len=vq.get("seq_len"), encoder_hidden=vq.get("hidden_size"),
            stage1_embed_dim=vq.get("vq_embed_dim"),
            stage2_latent_dim=predictor.get("stage2_latent_dim", vq.get("vq_embed_dim")),
            quantizer_kind=quantizer.get("research_kind", research.get("quantizer_kind", "legacy_vq")),
            fsq_levels=json.dumps(quantizer.get("research_levels", research.get("levels"))),
            num_embed=vq.get("num_embed"), n_expert=predictor.get("n_expert"), top_k=predictor.get("k"),
            moe_hidden=predictor.get("moe_hidden"), num_heads=transformer.get("num_heads"),
            d_model=transformer.get("d_model"), aux_weight=predictor.get("aux_weight"),
            predictor_rank=predictor.get("rank"), pairwise_weight=research.get("pairwise_weight", 0),
            turnover_weight=research.get("turnover_weight", 0), paired_days=research.get("paired_days", False),
            learning_rate=train.get("learning_rate"), max_epochs=train.get("num_epochs"),
            early_stopping_patience=train.get("early_stopping", {}).get("patience"), precision=train.get("precision"),
            research_settings=json.dumps(research, ensure_ascii=False, sort_keys=True)))
    pd.DataFrame(flattened).to_csv(EXP / "hyperparameters.csv", index=False)
    return index


def update_variants(configs):
    variants = pd.read_csv(EXP / "variants.csv")
    known = set(variants.variant)
    additions = []
    for row in configs.to_dict("records"):
        variant = row["variant"]
        if variant in known or variant not in DESCRIPTIONS:
            continue
        settings = load_config(row["config_file"]).get("research", {}).copy()
        settings.pop("variant", None)
        additions.append(dict(variant=variant, description=DESCRIPTIONS[variant],
                              settings=json.dumps(settings, ensure_ascii=False, sort_keys=True),
                              implementation="trace_vq/models/batch05.py" if row["batch"].startswith("batch05") else
                              "trace_vq/models/batch34.py" if row["batch"].startswith(("batch03","batch04"))
                              else "trace_vq/models/next_screen.py" if variant.startswith("fsq_turnover") else "trace_vq/models/batch01.py"))
        known.add(variant)
    for variant, weight in (("ens_vq5", 1), ("ens_fsq5", 0), ("ens_vq_fsq", .5)):
        if variant not in known:
            additions.append(dict(variant=variant, description="已有预测的算术平均集成；无新增训练。",
                                  settings=json.dumps(dict(seeds=list(range(5)), vq_group_weight=weight,
                                                           fsq_group_weight=1-weight)),
                                  implementation="trace_vq/evaluation/ensemble.py"))
    variants = pd.concat([variants, pd.DataFrame(additions)], ignore_index=True)
    assert variants.variant.is_unique
    variants.to_csv(EXP / "variants.csv", index=False)


def config_reference(configs, market, variant, seed):
    aliases = {variant, "full"} if variant == "baseline" and market == "csi300" else {variant}
    candidates = configs[(configs.universe == market) & configs.variant.isin(aliases) & (configs.train_seed == seed)]
    candidates = candidates[~candidates.config_file.map(is_stage1)]
    return candidates.iloc[-1].config_file if len(candidates) else ""


def refresh_sp500_cost_summary(configs):
    """Optional local-only extraction: publish 20 scalars, never daily rows."""
    metrics=pd.read_csv(EXP/"sp500_candidates_multiseed_20261031/seed_metrics.csv")
    baseline_source=EXP/"sp500_multiseed_20261031/cost_diagnostics.csv"
    baseline=pd.read_csv(baseline_source).set_index("seed")
    rows=[]
    for record in metrics.to_dict("records"):
        variant, seed=record["variant"],int(record["seed"])
        if variant=="baseline":
            source=baseline_source
            turnover=baseline.loc[seed,"mean_daily_turnover"]
            days=int(record["num_test_days"])
            method="existing curated baseline cost aggregate"
        else:
            selected=configs[(configs.batch=="sp500_candidates_multiseed_20261031") &
                             (configs.variant==variant) & (configs.train_seed==seed)]
            assert len(selected)==1
            source=ROOT/Path(selected.iloc[0].source_file).parent/"backtest/qlib_report.csv"
            assert source.resolve().is_relative_to(ROOT/"artifacts/research") and source.is_file()
            protocol=json.loads((source.parent/"backtest_config.json").read_text())
            assert protocol["quote_universe"]=="all"
            assert (protocol["strategy"]["topk"],protocol["strategy"]["n_drop"])==(30,5)
            assert (protocol["exchange"]["open_cost"],protocol["exchange"]["close_cost"],protocol["exchange"]["min_cost"])==(.0005,.0015,0)
            report=pd.read_csv(source,index_col=0)
            days=len(report)
            assert days==record["num_test_days"]
            turnover=float(report.turnover.mean())
            method="mean of local Qlib daily reported turnover; daily rows excluded"
        rows.append(dict(universe="sp500",variant=variant,seed=seed,net_AR=record["AR"],net_Sharpe=record["Sharpe"],
                         mean_daily_reported_turnover=turnover,days=days,source_report=str(source.relative_to(ROOT)),
                         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),aggregation=method))
    pd.DataFrame(rows).to_csv(EXP/"sp500_candidates_multiseed_20261031/cost_diagnostics.csv",index=False)


def collect_runs(configs):
    records, sources = {}, {}
    extra_inputs=[name+"/results.csv" for name in ("batch03_20261105","batch04_20261105","batch05_20261106") if (EXP/name/"results.csv").exists()]
    extra_rows=sum(int(pd.read_csv(EXP/name).new_training.sum()) for name in extra_inputs)
    inputs = ("single_seed_all.csv", "multiseed_per_seed.csv", "batch01_20261103/results.csv", "batch02_20261103/results.csv", *extra_inputs)
    for source in inputs:
        for incoming in pd.read_csv(EXP / source).to_dict("records"):
            seed = incoming.get("stage2_seed", incoming.get("seed"))
            if not present(seed):
                continue  # Prediction ensembles have their own table.
            market = incoming["universe"]
            variant = ALIASES.get(incoming["variant"], incoming["variant"])
            identity = (market, variant, int(seed))
            row = records.setdefault(identity, dict(universe=market, variant=variant, stage1_seed=42, stage2_seed=int(seed),
                                                    metric_protocol="protocol_v1", quote_universe="all",
                                                    **{f"new_training_in_batch{i:02d}":False for i in range(1,6)}))
            sources.setdefault(identity, []).append("experiments/"+source)
            for metric in ALL_METRICS:
                value = incoming.get(metric)
                if present(value):
                    if present(row.get(metric)):
                        assert math.isclose(row[metric], value, rel_tol=1e-10, abs_tol=1e-12), (identity, metric, source)
                    row[metric] = value
                    if metric=="turnover":
                        row["turnover_source"]="experiments/"+source
            row["status"] = incoming.get("status") if present(incoming.get("status")) else "complete"
            if not present(incoming.get("status")):
                assert all(present(incoming.get(metric)) and math.isfinite(incoming[metric])
                           for metric in ("IC", "RankIC", "AR", "Sharpe", "MDD"))
                row["status_evidence"] = "completed source report and finite aggregate metrics; source status column empty"
            else:
                row["status_evidence"] = "source status field"
            if present(incoming.get("elapsed_seconds")):
                row["historical_elapsed_seconds"] = incoming["elapsed_seconds"]
            new = incoming.get("role") == "new_model" if source.startswith("batch01") else incoming.get("new_training") is True
            if source.startswith("batch") and new:
                batch = source.split("/")[0].split("_")[0]
                row["new_training_in_"+batch] = True
                for field in ("training_seconds", "evaluation_backtest_seconds", "gpu_peak_allocated_mib", "gpu_peak_reserved_mib"):
                    if present(incoming.get(field)):
                        row[field] = incoming[field]
            row["source_csv"] = "experiments/"+source
    for path in sorted(EXP.glob("*/cost_diagnostics.csv")):
        for incoming in pd.read_csv(path).to_dict("records"):
            identity=(incoming.get("universe","sp500"),incoming.get("variant","baseline"),int(incoming.get("seed",0)))
            if identity not in records:
                continue
            row=records[identity]
            for cost_key,metric in (("net_AR","AR"),("net_Sharpe","Sharpe")):
                assert math.isclose(incoming[cost_key],row[metric],rel_tol=1e-10,abs_tol=1e-12),(identity,path)
            value=incoming.get("mean_daily_reported_turnover",incoming.get("mean_daily_turnover"))
            assert present(value) and math.isfinite(value)
            if present(row.get("turnover")):
                assert math.isclose(row["turnover"],value,rel_tol=1e-10,abs_tol=1e-12),(identity,path)
            row["turnover"]=value
            row["turnover_source"]=str(path.relative_to(ROOT))
            sources[identity].append(str(path.relative_to(ROOT)))
    rows = []
    for identity, row in records.items():
        row["source_views"] = ";".join(sources[identity])
        row["effective_config"] = config_reference(configs, *identity)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["universe", "variant", "stage2_seed"])
    assert len(frame) == 148+extra_rows
    assert int(frame.new_training_in_batch01.sum()) == 12
    assert int(frame.new_training_in_batch02.sum()) == 20
    frame.to_csv(EXP / "research_runs.csv", index=False)
    aggregates = []
    for (market, variant), group in frame.groupby(["universe", "variant"]):
        assert len(group) in (1, 5)
        row = dict(universe=market, variant=variant, stage1_seed=42, n_seeds=len(group),
                   stage2_seeds=",".join(str(seed) for seed in group.stage2_seed),
                   study_scope="five_seed_validation" if len(group)==5 else "single_seed_exploration",
                   metric_protocol="protocol_v1", quote_universe="all")
        for metric in ALL_METRICS:
            row[metric+"_n"] = int(group[metric].count())
            row[metric+"_mean"] = group[metric].mean()
            row[metric+"_std"] = group[metric].std(ddof=1)
        aggregates.append(row)
    summary = pd.DataFrame(aggregates)
    assert len(summary)==84+extra_rows and int(summary.n_seeds.eq(5).sum())==16
    summary.to_csv(EXP / "research_mean_std.csv", index=False)
    return frame, summary


def collect_ensembles():
    source = "experiments/batch01_20261103/results.csv"
    frame = pd.read_csv(ROOT / source)
    frame = frame[frame.role.eq("ensemble")].copy()
    keep = ["universe", "variant", *ALL_METRICS, "baseline_variant", *["delta_"+m for m in METRICS]]
    frame = frame.reindex(columns=keep)
    frame["source_result"] = source
    frame["input_seeds"] = "0,1,2,3,4"
    frame["stage1_seed"] = 42
    frame["new_training"] = False
    frame["metric_protocol"] = "protocol_v1"
    frame["quote_universe"] = "all"
    metrics = json.loads((EXP / "sp500_baseline_protocol_20261031/ensemble_metrics.json").read_text())
    sp500 = dict(universe="sp500", variant="ens_vq5", source_result="experiments/sp500_baseline_protocol_20261031/ensemble_metrics.json",
                 input_seeds="0,1,2,3,4", stage1_seed=42, new_training=False, metric_protocol="protocol_v1", quote_universe="all", **metrics)
    frame = pd.concat([frame, pd.DataFrame([sp500])], ignore_index=True)
    assert len(frame)==7
    frame.to_csv(EXP / "research_ensembles.csv", index=False)
    return frame


def update_catalog():
    index = pd.read_csv(EXP / "index.csv")
    for batch, finding in {
        "batch01_20261103": "CSI800逐维FSQ残差和混合集成改善组合指标；CSI300新单模型无收益/Sharpe优势。新模型仅seed0。",
        "batch02_20261103": "VQ平滑在CSI800未稳定复现；CSI300换手5/5降低但平均回撤加深。FSQ组合仅seed0，相对配对控制组较好。",
        "batch03_20261105": "CSI300状态分布门控增收但回撤加深，转移/波动特征小幅改善；CSI800驻留特征增收。原型记忆无综合优势，均仅seed0。",
        "batch04_20261105": "CSI800 Soft MoE对VQ六项指标改善，8专家增收但RankIC下降；两者收益仍低于FSQ。CSI300多数路由改动无收益优势，仅seed0。",
        "batch05_20261106": "CSI800 mHC对VQ六项指标改善；多窗口专家对FSQ增收、降低回撤/换手，但IC/RankIC下降。CSI300新方案无净年化优势，仅seed0。",
    }.items():
        index.loc[index.batch.eq(batch), "finding"] = finding
    for row_index, row in index.iterrows():
        table = EXP / row.batch / PRIMARY_TABLES[row.batch]
        assert table.exists() and (EXP / row.readme).exists()
        index.loc[row_index, "primary_results"] = str(table.relative_to(EXP))
        index.loc[row_index, "primary_table_rows"] = len(pd.read_csv(table)) if table.suffix==".csv" else 1
        boundary = "historical; see batch README" if row.batch.startswith("historical") else "see protocol.md and batch README"
        index.loc[row_index, "metric_boundary"] = boundary
    index.to_csv(EXP / "index.csv", index=False)
    return index


def overview(frame, summary, ensembles, catalog, configs):
    date=datetime.now(timezone.utc).strftime("%Y-%m-%d")
    lines = ["# 实验结果总览", "", f"整理日期：{date}。目录包含既往研究、历史消融与训练随机性实验，以及已发布的清单批次。", "",
             f"同一研究指标/报价口径下，合并并去重后有 **{len(frame)}条股票池×变体×Stage2 seed结果**，"
             f"覆盖{len(summary)}个股票池×变体设置，其中16组完成五seed、其余为单seed探索；另有{len(ensembles)}份预测集成。", "",
             f"{len(frame)}条是已保存命名结果记录的覆盖数量，包含复用历史模型，不能解释为{len(frame)}次新增训练。"
             "前两批新增Stage2训练分别为12次、20次，合计32次；Stage1均复用seed42。"
             "第01批有6份集成及G2/H4两项检查，第02批有4份复用seed0参考。"
             "第03、04批分别新增16、14次Stage2训练，第05批新增8次，均完成训练、独立预测核对和回测；前五批合计70次新增Stage2训练，均无新增Stage1训练。", "",
             "## 统一入口", "",
             "- [逐seed去重结果](research_runs.csv)：全部聚合指标、各指标来源和配置引用；空白表示原汇总未记录。",
             "- [按设置统计均值与样本标准差](research_mean_std.csv)：n_seeds=1的标准差留空，每项指标另有有效样本数。",
             "- [预测集成](research_ensembles.csv)：平均预测后单独回测，不能用各seed指标均值代替。",
             "- [超参数表](hyperparameters.csv) 与 [配置索引](config_index.csv)：实际YAML配置是超参数的完整出处。",
             "- [研究变体与实现](variants.csv)、[逐批目录](index.csv)、[指标口径](protocol.md)。", "",
             "统一表来自既有single_seed_all.csv、multiseed_per_seed.csv及各已发布批次results.csv；同一股票池、变体名和seed在不同表的重叠记录已合并，"
             "共同指标逐一核对一致，source_views保留全部出处。不同变体别名仍可能复用同一模型，详见各批复用说明，不能据此推算独立训练次数。"
             "换手依据已整理的cost_diagnostics.csv补齐，turnover_source注明出处；SP500候选新增费用汇总只导出本地逐日报告的均值及来源哈希。"
             "旧的64条/60行表保留为前期快照。"
             "旧SP500行缺少status字段，其完成状态依据已完成批次报告和有限聚合指标，status_evidence显式标注。", "",
             "## 全部研究与历史记录", "",
             "| 批次 | 内容 | 记录说明 |", "| --- | --- | --- |"]
    for row in catalog.to_dict("records"):
        lines.append(f"| [{row['batch']}]({row['readme']}) | {row['description']} | {row['records']} |")
    lines += ["", "历史CSI300消融的股票池报价、Stage1多seed和两阶段多seed的随机性设置，以及原native/protocol混合记录，"
              "保留在各自目录，未强行混入统一研究表。诊断与口径回放同样单独列示，不能按文件行数统计新增训练。", "",
              "## 已完成五seed的研究设置", "", "下表为各seed独立组合指标的均值±样本标准差；净年化、MDD为百分比。", "",
              "| 股票池 | 方案 | RankIC | 净年化 % | Sharpe | MDD % |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in summary[summary.n_seeds.eq(5)].to_dict("records"):
        cells=[]
        for metric, digits, scale in (("RankIC",5,1),("AR",2,100),("Sharpe",3,1),("MDD",2,100)):
            cells.append(f"{row[metric+'_mean']*scale:.{digits}f} ± {row[metric+'_std']*scale:.{digits}f}")
        lines.append(f"| {row['universe']} | {row['variant']} | "+" | ".join(cells)+" |")
    lines += ["", "## 分批实验结论", "",
              "- 第01批：CSI300新增单模型的收益和Sharpe均低于对应基线；VQ五seed集成有小幅改善。"
              "CSI800的fsq_ca_dim和混合集成改善组合指标，但前者RankIC下降，新增模型仅seed0。",
              "- 第02批：CSI300 VQ平滑相对配对控制组平均净年化提高1.02个百分点、日均换手减少3.49个百分点，"
              "平均回撤加深3.10个百分点。CSI800原seed0的强信号未稳定复现，新增seeds1–4平均净年化差值为-1.05个百分点。",
              "- FSQ平滑在两个股票池相对零惩罚配对控制组的RankIC、收益和Sharpe改善，IC未改善；"
              "只有seed0。CSI300相对原FSQ基线的回撤更深，不能把对控制组的改善解释为所有维度都优于原基线。",
              "- 第03批：CSI300 code_hist_state净年化21.660%、较FSQ提高3.633个百分点，但回撤加深4.214个百分点。"
              "code_trans/code_vol在CSI300改善收益、Sharpe和回撤，在CSI800未复现收益/Sharpe改善。"
              "CSI800 code_dwell净年化16.795%、提高2.513个百分点；proto_memory在CSI300虽提高相关系数，却显著损害组合指标。",
              "- 第04批：CSI800 soft_moe相对VQ六项指标均改善，净年化11.822%、提高4.939个百分点；"
              "moe_finegrained8净年化13.720%、提高6.837个百分点，但RankIC下降。两者收益仍低于原FSQ的14.281%。"
              "CSI300仅expert_k_all小幅增收，且相关系数下降、回撤略深。FSQ固定路由降低两个股票池的收益，且同时改变专家数量。",
              "- 第03/04批仅Stage2 seed0，Stage1复用对应seed42；没有新增预测集成。"
              "训练分别约12.76小时、至少10.12小时，后者另计至少14.15分钟中断重算。"
              "第四批整卡GPU采样有缺口，逐项PyTorch显存峰值、实际时间字段及恢复说明见各批timing_memory.csv和README。", "",
              "- 第05批：CSI800 mHC相对VQ六项指标改善，净年化10.614%、提高3.732个百分点，仍低于原FSQ的14.281%。"
              "多窗口专家相对FSQ净年化提高1.402个百分点、Sharpe提高0.083、回撤减少5.149个百分点、换手减少6.297个百分点，"
              "但IC/RankIC下降。CSI300新方案均未提高对应基线净年化。KDA和跨股注意力未显示综合优势。"
              "本批8/8项均完成，仅seed0；fit合计8.36小时，预测评估与回测11.28分钟，无中断或失败。", "",
              "所有弱结果、负结果和复用标签均保留。固定Stage1的五seed共享同一测试期；单seed结果和探索筛选不能支持统计显著或稳定性的结论。", "",
              "## Git内容与本地产物", "",
              f"当前收录{len(configs)}份有效YAML配置快照。Git只包含源码、超参数配置及Markdown/CSV/JSON聚合说明。"
              "artifacts/中的数据、权重、预测、逐日回测、日志、缓存、监控记录与历史Git备份由.gitignore排除。"
              "原始基线证据research/baseline/保留原字节。重建训练或回测需要另行提供本地数据和权重。", ""]
    (EXP / "overview.md").write_text("\n".join(lines))


def refresh_navigation(frame,summary,catalog,configs):
    updates={
        "README.md":[(r"\d+条去重后的逐seed研究结果",f"{len(frame)}条去重后的逐seed研究结果"),
                     (r"\d+个设置的均值与标准差",f"{len(summary)}个设置的均值与标准差"),
                     (r"\d+份配置快照索引",f"{len(configs)}份配置快照索引"),
                     (r"共\d+个研究/归档条目",f"共{len(catalog)}个研究/归档条目")],
        "experiments/README.md":[(r"(\]\(research_runs\.csv\)：)\d+条",rf"\g<1>{len(frame)}条"),
            (r"(\]\(research_mean_std\.csv\)：)\d+个设置",rf"\g<1>{len(summary)}个设置"),
            (r"\d+组为单seed探索",f"{int(summary.n_seeds.eq(1).sum())}组为单seed探索"),
            (r"(\]\(config_index\.csv\)：)\d+份",rf"\g<1>{len(configs)}份"),
            (r"\d+条结果包括复用模型",f"{len(frame)}条结果包括复用模型"),
            (r"不能当作\d+次新增训练",f"不能当作{len(frame)}次新增训练"),
            (r"包含此前所有已收录实验及清单第01、02批", "包含此前所有已收录实验及已发布的清单批次")],
        "docs/analysis-guide.md":[(r"\d+个研究/归档条目",f"{len(catalog)}个研究/归档条目"),
            (r"\d+条研究结果、\d+个设置",f"{len(frame)}条研究结果、{len(summary)}个设置"),
            (r"\d+份实际YAML配置",f"{len(configs)}份实际YAML配置")],
        "docs/github-sync.md":[(r"\d+条去重逐seed结果、\d+个设置",f"{len(frame)}条去重逐seed结果、{len(summary)}个设置"),
            (r"\d+份YAML配置",f"{len(configs)}份YAML配置")],
    }
    for name,patterns in updates.items():
        path=ROOT/name;text=path.read_text()
        for pattern,replacement in patterns:text=re.sub(pattern,replacement,text)
        path.write_text(text)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-local-turnover",action="store_true",
                        help="Extract aggregate SP500 turnover from existing local reports; default needs curated text only")
    args=parser.parse_args()
    configs=update_configs()
    if args.refresh_local_turnover:
        refresh_sp500_cost_summary(configs)
    update_variants(configs)
    frame, summary=collect_runs(configs)
    ensembles=collect_ensembles()
    catalog=update_catalog()
    overview(frame, summary, ensembles, catalog, configs)
    refresh_navigation(frame,summary,catalog,configs)
    print(json.dumps(dict(research_runs=len(frame), settings=len(summary), five_seed_settings=int(summary.n_seeds.eq(5).sum()),
                          prediction_ensembles=len(ensembles), studies=len(catalog), effective_configs=len(configs)),ensure_ascii=False))


if __name__=="__main__":
    main()
