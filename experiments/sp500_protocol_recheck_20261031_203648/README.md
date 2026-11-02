# SP500 回测指标复核

股票池：SP500。种子：0–4 和平均预测。记录数：6 个回放案例；无新训练。

保留指标边界、报价策略敏感性及其他既有模型的指标口径检查汇总。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [cross_baseline_metric_values.csv](cross_baseline_metric_values.csv)
- [metric_boundary_checks.json](metric_boundary_checks.json)
- [quote_policy_sensitivity.json](quote_policy_sensitivity.json)
- [report.md](report.md)
- [report_zh.md](report_zh.md)
- [supplemental_summary.json](supplemental_summary.json)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
