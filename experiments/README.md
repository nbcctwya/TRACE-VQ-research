# 实验结果索引

本目录提供实验设置、实际超参数与聚合结果文档，包含此前所有已收录实验及已发布的清单批次。

优先阅读[完整研究总览](overview.md)与[统一口径](protocol.md)：

- [research_runs.csv](research_runs.csv)：246条去重后的股票池×变体×Stage2 seed结果，保留全部来源视图。
- [research_mean_std.csv](research_mean_std.csv)：182个设置；16组完成五seed，166组为单seed探索。单seed标准差留空。
- [research_ensembles.csv](research_ensembles.csv)：7份平均预测后单独回测的集成结果。
- [hyperparameters.csv](hyperparameters.csv)、[config_index.csv](config_index.csv)：334份实际YAML配置及主要超参数。
- [variants.csv](variants.csv)：模型变体/预测集成与实现代码的对应关系。

246条结果包括复用模型，不能当作246次新增训练。历史报价、Stage1多seed、两阶段多seed及诊断回放保留在下方各自目录。

## 分批实验已完成结果

[第01批](batch01_20261103/README.md)新增12次Stage2训练、6份预测集成，G2/H4按要求检查完成。
CSI800的逐维FSQ残差与混合集成改善部分组合指标；CSI300新增单模型无收益/Sharpe优势。新增模型均仅seed0。

[第02批](batch02_20261103/README.md)新增20次Stage2训练、复用4份seed0，共24份结果。
CSI300的VQ平滑降低换手，但平均回撤加深；CSI800原seed0强信号未稳定复现。
FSQ平滑的单seed结果相对配对控制组较好，IC未改善，不能据此判定稳定性。

前两批均零失败，Stage1复用seed42，实际配置、全部指标差值、训练/回测耗时和显存记录均在各批目录。

[第03批](batch03_20261105/README.md)新增16次Stage2训练，另复用2份FSQ基线。CSI300状态分布门控净年化21.660%，
较FSQ提高3.633个百分点，但回撤加深4.214个百分点；转移/波动特征改善CSI300组合指标，CSI800未复现收益/Sharpe提升。
CSI800驻留特征净年化16.795%；原型记忆在CSI300虽提高相关系数，组合指标却明显变差。

[第04批](batch04_20261105/README.md)新增14次Stage2训练，另复用4份VQ/FSQ基线，原top1不重复训练。
CSI800 Soft MoE相对VQ六项指标改善，净年化11.822%；8专家细粒度方案净年化13.720%，但RankIC下降。
两者收益仍低于原FSQ的14.281%。CSI300多数路由改动没有收益优势；FSQ固定路由在两个股票池都降低收益，且同时改变专家数。

第三、第四批共30/30项于2026-11-06 19:08（UTC+8）完成，失败0项，全部完成独立预测核对与TopkDropout回测。
均仅Stage2 seed0，Stage1复用对应seed42；没有新增预测集成。训练约12.76小时和至少10.12小时，后者包含已记录的中断重算下界。
中断恢复、整卡GPU采样缺口及逐项PyTorch显存峰值均在报告中注明。四批合计62次新增Stage2训练，不能用含复用的结果行数替代。

[第05批](batch05_20261106/README.md)新增8次Stage2训练，另复用4份基线，8项均完成独立预测核对与TopkDropout回测。
CSI800的mHC相对VQ六项指标改善，净年化10.614%，仍低于原FSQ的14.281%；多窗口专家相对FSQ净年化提高1.402个百分点，
Sharpe、回撤及换手改善，但IC/RankIC下降。CSI300新方案均未提高对应基线的净年化。
本批仅Stage2 seed0；训练8.36小时，预测评估和回测11.28分钟，无中断/失败，全部差值、实际配置与显存记录见报告。
前五批合计70次新增Stage2训练，Stage1均复用。

[第06批](batch06_20261107/README.md)新增14次Stage2训练，另复用2份FSQ基线，全部完成。
CSI300逐维档位嵌入净年化19.394%，较FSQ提高1.367个百分点，Sharpe/回撤改善，IC/RankIC略降；
CSI800边界邻档增强净年化18.548%，提高4.267个百分点，Sharpe/回撤/换手改善，但IC/RankIC下降，且CSI300增强结果变差。

[第07批](batch07_20261107/README.md)新增8次Stage2训练，另复用2份FSQ基线，全部完成。
CSI300历史Hopfield检索六项指标改善，净年化19.082%、提高1.055个百分点；CSI800未复现，且本批CSI800无同时提高年化与Sharpe的方案。
两批合计22/22项于2026-11-08 07:40（UTC+8）完成，训练11.63/6.75小时，预测评估与回测17.69/9.86分钟，零中断、零失败。
均仅Stage2 seed0；前七批共92次新增Stage2训练，Stage1均复用，无新增预测集成。

[第08批](batch08_20261108/README.md)新增22组两阶段训练，比较Stage1收益监督权重与量化器。
CSI300监督权重0.001净年化21.508%、提高3.481个百分点，Sharpe提高0.080，但最大回撤加深8.041个百分点、RankIC下降；
CSI800的11种方案净年化均低于原FSQ。五维档位[4,4,4,4,2]提高CSI300的RankIC，但未改善组合收益。

[第09批](batch09_20261108/README.md)新增16组两阶段训练。CSI800可学习尺度FSQ净年化14.553%、提高0.271个百分点，
Sharpe提高0.079、回撤减少6.365个百分点、换手略降，但IC/RankIC下降。
CSI300对比学习净年化19.940%、提高1.913个百分点，但回撤加深；CSI800对比学习净年化下降5.851个百分点。

两批38/38组于2026-11-13 05:20（UTC+8）完成，新增38次Stage1 seed42与38次Stage2 seed0训练，原FSQ对照复用已有结果。
均完成独立预测核对和TopkDropout回测，无新增预测集成、训练失败或中断恢复。fit分别56.20/54.83小时，预测评估与回测30.41/21.90分钟。
全部38组和76份阶段配置均已收录；单seed结果不能说明稳定性或统计显著。

## 前期候选五seed快照

以下表格对应此前的[五seed汇总](multiseed_mean_std.csv)和[60行逐seed快照](multiseed_per_seed.csv)。
前四轮的[64条单seed快照](single_seed_all.csv)也保留原值；最新覆盖范围请使用上方统一表。

以下均为五个独立 Stage 2 seed 的指标均值和样本标准差（ddof=1），不是预测集成。AR 与 MDD 在 CSV 中为比例，下表换成百分数。

| 股票池 | 方案 | RankIC（均值 ± SD） | 净年化 %（均值 ± SD） | Sharpe（均值 ± SD） | 最大回撤 %（均值 ± SD） |
| --- | --- | ---: | ---: | ---: | ---: |
| csi300 | baseline | 0.05634 ± 0.00395 | 15.54 ± 3.51 | 0.913 ± 0.244 | -18.02 ± 7.63 |
| csi300 | fsq | 0.05403 ± 0.00188 | 15.84 ± 2.88 | 0.917 ± 0.198 | -19.70 ± 7.71 |
| csi300 | fsq_ranknet | 0.05423 ± 0.00256 | 16.97 ± 4.05 | 0.975 ± 0.289 | -20.53 ± 7.11 |
| csi300 | ranknet | 0.05450 ± 0.00238 | 15.48 ± 3.39 | 0.875 ± 0.212 | -20.88 ± 6.44 |
| csi800 | baseline | 0.05319 ± 0.00408 | 8.93 ± 4.08 | 0.475 ± 0.266 | -31.35 ± 7.17 |
| csi800 | fsq | 0.05458 ± 0.00307 | 15.44 ± 2.32 | 0.815 ± 0.093 | -24.95 ± 6.04 |
| csi800 | fsq_ranknet | 0.05438 ± 0.00282 | 15.18 ± 2.00 | 0.819 ± 0.092 | -22.98 ± 6.67 |
| csi800 | ranknet | 0.05380 ± 0.00411 | 10.44 ± 3.36 | 0.545 ± 0.219 | -30.70 ± 7.11 |
| sp500 | baseline | 0.00077 ± 0.00381 | 7.72 ± 2.95 | 0.436 ± 0.183 | -21.92 ± 3.81 |
| sp500 | fsq | 0.00175 ± 0.00460 | 7.37 ± 4.17 | 0.416 ± 0.222 | -20.80 ± 2.11 |
| sp500 | fsq_ranknet | 0.00111 ± 0.00453 | 7.91 ± 4.80 | 0.430 ± 0.235 | -21.59 ± 1.67 |
| sp500 | ranknet | 0.00040 ± 0.00388 | 7.34 ± 3.27 | 0.426 ± 0.191 | -20.20 ± 3.36 |

CSI800 的 FSQ 和 FSQ+RankNet 相对基线更有希望；CSI300 的组合提高平均收益，但 RankIC 与回撤存在取舍。SP500 三个候选均未显示稳定的综合优势。当前证据不足以替换默认模型。

## 批次

| 批次 | 内容 | 股票池 | 记录 |
| --- | --- | --- | --- |
| [innovation_screen_20261026](innovation_screen_20261026/README.md) | 首轮创新筛选 | CSI300 / CSI800 | 14 条（含 2 条基线） |
| [next_directions_20261027](next_directions_20261027/README.md) | 第二轮方向筛选 | CSI300 / CSI800 | 26 条（含 4 条复用对照） |
| [mechanism_screen_20261028](mechanism_screen_20261028/README.md) | 机制与空对照 | CSI300 / CSI800 | 20 条（含 6 条复用对照） |
| [combination_screen_20261029](combination_screen_20261029/README.md) | 七种组件组合 | CSI300 / CSI800 | 24 条（含 10 条复用对照） |
| [multiseed_validation_20261030](multiseed_validation_20261030/README.md) | A 股五 seed 验证 | CSI300 / CSI800 | 40 条（24 次新训练、16 条复用） |
| [sp500_multiseed_20261031](sp500_multiseed_20261031/README.md) | SP500 基线 | SP500 | 5 条 |
| [sp500_candidates_multiseed_20261031](sp500_candidates_multiseed_20261031/README.md) | SP500 候选五 seed 验证 | SP500 | 20 条（15 次新训练、5 条复用基线） |
| [historical_csi300_ablations](historical_csi300_ablations/README.md) | 历史 CSI300 全模型与三种消融 | CSI300 | 20 条 seed 结果；另有预测集成汇总 |
| [historical_csi800_baseline](historical_csi800_baseline/README.md) | 历史 CSI800 基线 | CSI800 | 5 条 |
| [historical_stage1_multiseed](historical_stage1_multiseed/README.md) | Stage 1 多 seed、Stage 2 固定 | CSI300 | 5 条 |
| [historical_full_pipeline_multiseed](historical_full_pipeline_multiseed/README.md) | 两阶段都重训的多 seed 实验 | CSI300 | 5 条；另有集成结果 |
| [sp500_baseline_protocol_20261031](sp500_baseline_protocol_20261031/README.md) | SP500 基线口径与集成复核 | SP500 | 6 次回测；无新训练 |
| [sp500_protocol_recheck_20261031_203648](sp500_protocol_recheck_20261031_203648/README.md) | SP500 回测指标复核 | SP500 | 6 个回放案例；无新训练 |
| [sp500_effect_diagnosis_20261101](sp500_effect_diagnosis_20261101/README.md) | SP500 预测、梯度与市场暴露诊断 | SP500 | 无新训练 |
| [batch01_20261103](batch01_20261103/README.md) | 清单01–08：检查、集成和结构筛选 | CSI300 / CSI800 | 12次新训练、6份集成、2项检查 |
| [batch02_20261103](batch02_20261103/README.md) | 清单09–10：平滑五seed复验与FSQ组合 | CSI300 / CSI800 | 20次新训练、4份复用，共24条结果 |
| [batch03_20261105](batch03_20261105/README.md) | 清单11–17：FSQ状态变化与记忆 | CSI300 / CSI800 | 16次新训练、2份复用，共18条结果 |
| [batch04_20261105](batch04_20261105/README.md) | 清单18–23：专家路由 | CSI300 / CSI800 | 14次新训练、4份复用，共18条结果 |
| [batch05_20261106](batch05_20261106/README.md) | 清单24–27：时间模块与截面交互 | CSI300 / CSI800 | 8次新训练、4份复用，共12条结果 |
| [batch06_20261107](batch06_20261107/README.md) | 清单28–34：FSQ网格表达与预测头 | CSI300 / CSI800 | 14次新训练、2份复用，共16条结果 |
| [batch07_20261107](batch07_20261107/README.md) | 清单35–38：码条件化与关系检索 | CSI300 / CSI800 | 8次新训练、2份复用，共10条结果 |
| [batch08_20261108](batch08_20261108/README.md) | 清单39–45：Stage1监督与量化器 | CSI300 / CSI800 | 22组两阶段新训练、2份复用，共24条结果 |
| [batch09_20261108](batch09_20261108/README.md) | 清单46–53：FSQ进一步改造 | CSI300 / CSI800 | 16组两阶段新训练、2份复用，共18条结果 |

前四轮的 14/26/20/24 行含复用对照，去重后为 64 条股票池 × 配置结果（32 个配置，每个用于两个股票池）。多 seed 表也含已跑的 seed 0 与历史基线；不能把各表行数相加当作新增训练次数。失败和较差的科学结果保留，技术 smoke 和资源中断重试不算新增方法。

代码中声明但没有结果行的可选变体不视为已完成实验。自动 promising 标签只是早期筛选标记，最终结论应结合多 seed 结果、验证集表现和每批 README。

更新统一表可运行 `research/run.sh research/organize_experiment_results.py`。
该命令只读取已整理的聚合结果与配置，不读取原始数据或预测明细。
