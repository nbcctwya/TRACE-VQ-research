# SP500 基线口径与集成复核

股票池：SP500。种子：0–4 和平均预测。记录数：6 次回测；无新训练。

复核五个独立 seed 及其平均预测集成。集成净年化约 5.86%，不能用基线五 seed 均值 7.72% 代替。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [ensemble_metrics.json](ensemble_metrics.json)
- [plan.json](plan.json)
- [protocol_audit.json](protocol_audit.json)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
