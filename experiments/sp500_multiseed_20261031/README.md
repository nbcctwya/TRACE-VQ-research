# SP500 基线

股票池：SP500。种子：Stage 2: 0–4。记录数：5 条。

新训练一个 Stage 1（seed 42），再训练五个 Stage 2。平均净年化约 7.72%，但 RankIC 接近零。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [benchmark_metrics.json](benchmark_metrics.json)
- [conclusions_zh.json](conclusions_zh.json)
- [cost_diagnostics.csv](cost_diagnostics.csv)
- [mean_std.csv](mean_std.csv)
- [plan.json](plan.json)
- [seed_metrics.csv](seed_metrics.csv)
- [summary_metrics.csv](summary_metrics.csv)
- [summary_readable.csv](summary_readable.csv)
- [yearly_mean_std.csv](yearly_mean_std.csv)
- [yearly_metrics.csv](yearly_metrics.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
