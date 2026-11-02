# 历史 CSI300 全模型与三种消融

股票池：CSI300。种子：Stage 1: 42; Stage 2: 0–4。记录数：20 条 seed 结果；另有预测集成汇总。

025=wo_ta，034=full，037=wo_ca，038=wo_sr。保留归档指标。

归档组合指标使用 historical 股票池报价口径；不要与后续 quote_universe=all 的研究回测直接混比。集成指标来自平均预测后单独回测，不是各 seed 指标均值。

结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的结果表：

- [每 seed 指标](metrics/seed_metrics.csv)
- [均值与标准差](tables/seed_mean_std.csv)
- [预测集成指标](metrics/ensemble_metrics.csv)



若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
