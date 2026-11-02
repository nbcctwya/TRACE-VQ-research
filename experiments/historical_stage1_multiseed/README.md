# Stage 1 多 seed、Stage 2 固定

股票池：CSI300。种子：Stage 1: 0–4; Stage 2: 42。记录数：5 条。

这是 Stage 1 随机性的历史补充实验，与固定 Stage 1 的候选多 seed 验证不同。

使用 protocol_v1 目录的汇总；超额收益列与绝对组合收益列分开保存。

结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [excess_table.csv](excess_table.csv)
- [per_seed_metrics.csv](per_seed_metrics.csv)
- [portfolio_table.csv](portfolio_table.csv)
- [ranking_table.csv](ranking_table.csv)
- [summary_statistics.csv](summary_statistics.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
