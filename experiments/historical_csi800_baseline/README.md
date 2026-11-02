# 历史 CSI800 基线

股票池：CSI800。种子：Stage 1: 42; Stage 2: 0–4。记录数：5 条。

后续 A 股多 seed 实验复用了这些基线；seed 1–4 的历史验证集指标没有完整保留。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [seed_mean_std.csv](seed_mean_std.csv)
- [seed_metrics.csv](seed_metrics.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
