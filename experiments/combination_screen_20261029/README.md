# 七种组件组合

股票池：CSI300 / CSI800。种子：Stage 2: 0。记录数：24 条（含 10 条复用对照）。

FSQ+RankNet 在单 seed 双股票池比较中较好，后续已做多 seed 验证。新增组合共 14 次 Stage 2 训练。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [all_parent_comparisons.csv](all_parent_comparisons.csv)
- [combination_outcomes.json](combination_outcomes.json)
- [cost_diagnostics.csv](cost_diagnostics.csv)
- [final_comparison.csv](final_comparison.csv)
- [joint_market_summary.csv](joint_market_summary.csv)
- [leaderboard.csv](leaderboard.csv)
- [manual_review.csv](manual_review.csv)
- [paired_uncertainty.csv](paired_uncertainty.csv)
- [plan.json](plan.json)
- [prediction_change.csv](prediction_change.csv)
- [primary_uncertainty.csv](primary_uncertainty.csv)
- [report_summary.json](report_summary.json)
- [yearly_metrics.csv](yearly_metrics.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
