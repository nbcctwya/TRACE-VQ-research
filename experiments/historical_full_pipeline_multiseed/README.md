# 两阶段都重训的多 seed 实验

股票池：CSI300。种子：Stage 1 = Stage 2: 0–4。记录数：5 条；另有集成结果。

每个 seed 都从头训练 Stage 1，再训练同 seed 的 Stage 2。

原报告同时包含 native ddof=0 与 protocol ddof=1 指标。protocol_ranking_metrics.csv 的 IC/RankIC 比率列为 protocol 口径；pf_ 组合列继承原报告，不自动宣称已统一重算。

结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [comparison_table.csv](comparison_table.csv)
- [ensemble_metrics.json](ensemble_metrics.json)
- [full_pipeline_seed_metrics.csv](full_pipeline_seed_metrics.csv)
- [full_pipeline_summary.md](full_pipeline_summary.md)
- [protocol_ranking_metrics.csv](protocol_ranking_metrics.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
