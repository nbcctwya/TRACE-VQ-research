# 实验结果索引

本目录仅提供实验设置与聚合结果，供分析代码、对照实验和研究结论。

优先阅读 [统一口径](protocol.md)、[五 seed 汇总](multiseed_mean_std.csv)、[每 seed 结果](multiseed_per_seed.csv)，再看 [64 条去重单 seed 结果](single_seed_all.csv)。变体代码映射见 [variants.csv](variants.csv)，实际保存配置见 [config_index.csv](config_index.csv)。

## 当前结果

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

前四轮的 14/26/20/24 行含复用对照，去重后为 64 条股票池 × 配置结果（32 个配置，每个用于两个股票池）。多 seed 表也含已跑的 seed 0 与历史基线；不能把各表行数相加当作新增训练次数。失败和较差的科学结果保留，技术 smoke 和资源中断重试不算新增方法。

代码中声明但没有结果行的可选变体不视为已完成实验。自动 promising 标签只是早期筛选标记，最终结论应结合多 seed 结果、验证集表现和每批 README。
