# 第二轮方向筛选

股票池：CSI300 / CSI800。种子：Stage 2: 0。记录数：26 条（含 4 条复用对照）。

检验 RankNet、换手约束、软路由、历史状态及 RVQ/FSQ。新增 Stage 2 训练 22 次、Stage 1 训练 6 次；一次资源中断已恢复。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [cost_diagnostics.csv](cost_diagnostics.csv)
- [decisions.json](decisions.json)
- [direction_review.json](direction_review.json)
- [final_comparison.csv](final_comparison.csv)
- [leaderboard.csv](leaderboard.csv)
- [paired_uncertainty.csv](paired_uncertainty.csv)
- [plan.json](plan.json)
- [prediction_change.csv](prediction_change.csv)
- [primary_uncertainty.csv](primary_uncertainty.csv)
- [quantizer_usage_summary.json](quantizer_usage_summary.json)
- [report_summary.json](report_summary.json)
- [shortlist.csv](shortlist.csv)
- [yearly_metrics.csv](yearly_metrics.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
