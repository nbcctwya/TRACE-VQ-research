# 实验结果总览

整理日期：2026-11-06。目录包含既往研究、历史消融与训练随机性实验，以及已发布的清单批次。

同一研究指标/报价口径下，合并并去重后有 **178条股票池×变体×Stage2 seed结果**，覆盖114个股票池×变体设置，其中16组完成五seed、其余为单seed探索；另有7份预测集成。

178条是已保存命名结果记录的覆盖数量，包含复用历史模型，不能解释为178次新增训练。前两批新增Stage2训练分别为12次、20次，合计32次；Stage1均复用seed42。第01批有6份集成及G2/H4两项检查，第02批有4份复用seed0参考。第03、04批分别新增16、14次Stage2训练，全部完成训练、独立预测核对和回测；四批合计62次新增Stage2训练，均无新增Stage1训练。

## 统一入口

- [逐seed去重结果](research_runs.csv)：全部聚合指标、各指标来源和配置引用；空白表示原汇总未记录。
- [按设置统计均值与样本标准差](research_mean_std.csv)：n_seeds=1的标准差留空，每项指标另有有效样本数。
- [预测集成](research_ensembles.csv)：平均预测后单独回测，不能用各seed指标均值代替。
- [超参数表](hyperparameters.csv) 与 [配置索引](config_index.csv)：实际YAML配置是超参数的完整出处。
- [研究变体与实现](variants.csv)、[逐批目录](index.csv)、[指标口径](protocol.md)。

统一表来自既有single_seed_all.csv、multiseed_per_seed.csv及各已发布批次results.csv；同一股票池、变体名和seed在不同表的重叠记录已合并，共同指标逐一核对一致，source_views保留全部出处。不同变体别名仍可能复用同一模型，详见各批复用说明，不能据此推算独立训练次数。换手依据已整理的cost_diagnostics.csv补齐，turnover_source注明出处；SP500候选新增费用汇总只导出本地逐日报告的均值及来源哈希。旧的64条/60行表保留为前期快照。旧SP500行缺少status字段，其完成状态依据已完成批次报告和有限聚合指标，status_evidence显式标注。

## 全部研究与历史记录

| 批次 | 内容 | 记录说明 |
| --- | --- | --- |
| [innovation_screen_20261026](innovation_screen_20261026/README.md) | 首轮创新筛选 | 14 条（含 2 条基线） |
| [next_directions_20261027](next_directions_20261027/README.md) | 第二轮方向筛选 | 26 条（含 4 条复用对照） |
| [mechanism_screen_20261028](mechanism_screen_20261028/README.md) | 机制与空对照 | 20 条（含 6 条复用对照） |
| [combination_screen_20261029](combination_screen_20261029/README.md) | 七种组件组合 | 24 条（含 10 条复用对照） |
| [multiseed_validation_20261030](multiseed_validation_20261030/README.md) | A 股五 seed 验证 | 40 条（24 次新训练、16 条复用） |
| [sp500_multiseed_20261031](sp500_multiseed_20261031/README.md) | SP500 基线 | 5 条 |
| [sp500_candidates_multiseed_20261031](sp500_candidates_multiseed_20261031/README.md) | SP500 候选五 seed 验证 | 20 条（15 次新训练、5 条复用基线） |
| [historical_csi300_ablations](historical_csi300_ablations/README.md) | 历史 CSI300 全模型与三种消融 | 20 条 seed 结果；另有预测集成汇总 |
| [historical_csi800_baseline](historical_csi800_baseline/README.md) | 历史 CSI800 基线 | 5 条 |
| [historical_stage1_multiseed](historical_stage1_multiseed/README.md) | Stage 1 多 seed、Stage 2 固定 | 5 条 |
| [historical_full_pipeline_multiseed](historical_full_pipeline_multiseed/README.md) | 两阶段都重训的多 seed 实验 | 5 条；另有集成结果 |
| [sp500_baseline_protocol_20261031](sp500_baseline_protocol_20261031/README.md) | SP500 基线口径与集成复核 | 6 次回测；无新训练 |
| [sp500_protocol_recheck_20261031_203648](sp500_protocol_recheck_20261031_203648/README.md) | SP500 回测指标复核 | 6 个回放案例；无新训练 |
| [sp500_effect_diagnosis_20261101](sp500_effect_diagnosis_20261101/README.md) | SP500 预测、梯度与市场暴露诊断 | 无新训练 |
| [batch01_20261103](batch01_20261103/README.md) | 第01批：快速起步与公共接口 | 12次Stage2、6份集成、2项检查 |
| [batch02_20261103](batch02_20261103/README.md) | 第02批：已有强信号复验 | 20次新训练、4份复用seed0 |
| [batch03_20261105](batch03_20261105/README.md) | 第03批：FSQ状态变化与记忆 | 16次新训练，原基线直接复用 |
| [batch04_20261105](batch04_20261105/README.md) | 第04批：专家路由 | 14次新训练，原基线直接复用 |

历史CSI300消融的股票池报价、Stage1多seed和两阶段多seed的随机性设置，以及原native/protocol混合记录，保留在各自目录，未强行混入统一研究表。诊断与口径回放同样单独列示，不能按文件行数统计新增训练。

## 已完成五seed的研究设置

下表为各seed独立组合指标的均值±样本标准差；净年化、MDD为百分比。

| 股票池 | 方案 | RankIC | 净年化 % | Sharpe | MDD % |
| --- | --- | ---: | ---: | ---: | ---: |
| csi300 | baseline | 0.05634 ± 0.00395 | 15.54 ± 3.51 | 0.913 ± 0.244 | -18.02 ± 7.63 |
| csi300 | fsq | 0.05403 ± 0.00188 | 15.84 ± 2.88 | 0.917 ± 0.198 | -19.70 ± 7.71 |
| csi300 | fsq_ranknet | 0.05423 ± 0.00256 | 16.97 ± 4.05 | 0.975 ± 0.289 | -20.53 ± 7.11 |
| csi300 | ranknet | 0.05450 ± 0.00238 | 15.48 ± 3.39 | 0.875 ± 0.212 | -20.88 ± 6.44 |
| csi300 | turnover | 0.05939 ± 0.00401 | 16.05 ± 2.48 | 0.923 ± 0.161 | -23.93 ± 4.07 |
| csi300 | turnover_control | 0.05838 ± 0.00518 | 15.03 ± 3.40 | 0.860 ± 0.227 | -20.83 ± 7.63 |
| csi800 | baseline | 0.05319 ± 0.00408 | 8.93 ± 4.08 | 0.475 ± 0.266 | -31.35 ± 7.17 |
| csi800 | fsq | 0.05458 ± 0.00307 | 15.44 ± 2.32 | 0.815 ± 0.093 | -24.95 ± 6.04 |
| csi800 | fsq_ranknet | 0.05438 ± 0.00282 | 15.18 ± 2.00 | 0.819 ± 0.092 | -22.98 ± 6.67 |
| csi800 | ranknet | 0.05380 ± 0.00411 | 10.44 ± 3.36 | 0.545 ± 0.219 | -30.70 ± 7.11 |
| csi800 | turnover | 0.05189 ± 0.00226 | 13.43 ± 3.92 | 0.650 ± 0.256 | -33.79 ± 9.26 |
| csi800 | turnover_control | 0.05398 ± 0.00342 | 12.51 ± 2.47 | 0.626 ± 0.172 | -28.45 ± 11.64 |
| sp500 | baseline | 0.00077 ± 0.00381 | 7.72 ± 2.95 | 0.436 ± 0.183 | -21.92 ± 3.81 |
| sp500 | fsq | 0.00175 ± 0.00460 | 7.37 ± 4.17 | 0.416 ± 0.222 | -20.80 ± 2.11 |
| sp500 | fsq_ranknet | 0.00111 ± 0.00453 | 7.91 ± 4.80 | 0.430 ± 0.235 | -21.59 ± 1.67 |
| sp500 | ranknet | 0.00040 ± 0.00388 | 7.34 ± 3.27 | 0.426 ± 0.191 | -20.20 ± 3.36 |

## 分批实验结论

- 第01批：CSI300新增单模型的收益和Sharpe均低于对应基线；VQ五seed集成有小幅改善。CSI800的fsq_ca_dim和混合集成改善组合指标，但前者RankIC下降，新增模型仅seed0。
- 第02批：CSI300 VQ平滑相对配对控制组平均净年化提高1.02个百分点、日均换手减少3.49个百分点，平均回撤加深3.10个百分点。CSI800原seed0的强信号未稳定复现，新增seeds1–4平均净年化差值为-1.05个百分点。
- FSQ平滑在两个股票池相对零惩罚配对控制组的RankIC、收益和Sharpe改善，IC未改善；只有seed0。CSI300相对原FSQ基线的回撤更深，不能把对控制组的改善解释为所有维度都优于原基线。
- 第03批：CSI300 code_hist_state净年化21.660%、较FSQ提高3.633个百分点，但回撤加深4.214个百分点。code_trans/code_vol在CSI300改善收益、Sharpe和回撤，在CSI800未复现收益/Sharpe改善。CSI800 code_dwell净年化16.795%、提高2.513个百分点；proto_memory在CSI300虽提高相关系数，却显著损害组合指标。
- 第04批：CSI800 soft_moe相对VQ六项指标均改善，净年化11.822%、提高4.939个百分点；moe_finegrained8净年化13.720%、提高6.837个百分点，但RankIC下降。两者收益仍低于原FSQ的14.281%。CSI300仅expert_k_all小幅增收，且相关系数下降、回撤略深。FSQ固定路由降低两个股票池的收益，且同时改变专家数量。
- 第03/04批仅Stage2 seed0，Stage1复用对应seed42；没有新增预测集成。训练分别约12.76小时、至少10.12小时，后者另计至少14.15分钟中断重算。第四批整卡GPU采样有缺口，逐项PyTorch显存峰值、实际时间字段及恢复说明见各批timing_memory.csv和README。

所有弱结果、负结果和复用标签均保留。固定Stage1的五seed共享同一测试期；单seed结果和探索筛选不能支持统计显著或稳定性的结论。

## Git内容与本地产物

当前收录228份有效YAML配置快照。Git只包含源码、超参数配置及Markdown/CSV/JSON聚合说明。artifacts/中的数据、权重、预测、逐日回测、日志、缓存、监控记录与历史Git备份由.gitignore排除。原始基线证据research/baseline/保留原字节。重建训练或回测需要另行提供本地数据和权重。
