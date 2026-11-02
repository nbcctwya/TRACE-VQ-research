# A 股五 seed 验证

股票池：CSI300 / CSI800。种子：Stage 2: 0–4。记录数：40 条（24 次新训练、16 条复用）。

比较基线、FSQ、RankNet 和 FSQ+RankNet。CSI800 的 FSQ/组合较有希望；CSI300 没有同时改善所有主要指标的候选。Stage 1 固定 seed 42。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [conclusions_zh.json](conclusions_zh.json)
- [cost_diagnostics.csv](cost_diagnostics.csv)
- [interaction_seed_metrics.csv](interaction_seed_metrics.csv)
- [interaction_summary.csv](interaction_summary.csv)
- [matched_seed_deltas.csv](matched_seed_deltas.csv)
- [mean_std.csv](mean_std.csv)
- [paired_summary.csv](paired_summary.csv)
- [plan.json](plan.json)
- [seed_metrics.csv](seed_metrics.csv)
- [summary_readable.csv](summary_readable.csv)
- [yearly_metrics.csv](yearly_metrics.csv)
- [yearly_summary.csv](yearly_summary.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
