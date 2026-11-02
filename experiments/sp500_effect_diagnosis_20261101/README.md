# SP500 预测、梯度与市场暴露诊断

股票池：SP500。种子：已有 20 个 seed 模型。记录数：无新训练。

RankNet 有非零梯度；抽样参数梯度范数约为 MSE 的 2.4%，不是损失未生效。FSQ 明显改变预测排序，但组合日收益仍高度相关。

这里的相关性、重合度、梯度和市场暴露是聚合诊断，不构成统计显著性或因果证明；不附逐日/逐股票原始记录。

结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [adding_ranknet_to_fsq.csv](adding_ranknet_to_fsq.csv)
- [adding_ranknet_to_fsq_summary.json](adding_ranknet_to_fsq_summary.json)
- [between_seed_prediction_comparison.csv](between_seed_prediction_comparison.csv)
- [gradient_direction_change.csv](gradient_direction_change.csv)
- [gradient_summary.json](gradient_summary.json)
- [market_exposure_summary.csv](market_exposure_summary.csv)
- [prediction_summary.csv](prediction_summary.csv)
- [prediction_summary.json](prediction_summary.json)
- [ranknet_gradient_summary.csv](ranknet_gradient_summary.csv)
- [same_seed_prediction_comparison.csv](same_seed_prediction_comparison.csv)
- [validation.json](validation.json)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
