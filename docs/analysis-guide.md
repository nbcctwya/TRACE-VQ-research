# 代码与结果阅读顺序

1. [实验索引](../experiments/README.md) 和 [指标口径](../experiments/protocol.md)：确认实验组、复用关系与比较边界。
2. [五 seed 均值/标准差](../experiments/multiseed_mean_std.csv) 与 [每 seed 结果](../experiments/multiseed_per_seed.csv)：比较稳定性与同 seed 的差值。
3. [单 seed 全部配置](../experiments/single_seed_all.csv) 和 [变体说明](../experiments/variants.csv)：查看探索、控制组和负结果。
4. [配置索引](../experiments/config_index.csv)：定位实际模型尺寸、损失权重、训练 seed 和所用 Stage 1。路径已做移植处理，数值参数保留。
5. [架构](architecture.md)，然后读 `trace_vq/models/representation.py`、`trace_vq/training/prediction.py`、`trace_vq/models/screening.py`、`next_screen.py`、`mechanism_screen.py`、`combination_screen.py`。
6. 查看对应 `trace_vq/cli/` 与 `research/run_*.py`，理解检查点选择、缓存和实验编排。

引用结论时给出股票池、方案、seed 范围、测试区间和表格路径；区分实测结果、已有报告判断与新的推断。不要把代码中的所有可选分支都当成已经跑过，也不要把重复引用的基线算作新训练。

推荐分析问题：FSQ 与 RankNet 的单独和组合效应是什么？收益改进能否覆盖 RankIC/回撤的取舍？同 seed 的胜出次数与均值是否一致？历史状态的错配控制能否支持机制解释？SP500 的弱排序能力与较高组合收益是否主要关联市场暴露？哪些结论还需要原始明细验证？

Git 跟踪的源码为当前研究版本，包含各轮之后的累计扩展。历史源码快照、权重和数据只在本地忽略目录保留；公开视图不能单凭当前代码对所有旧训练作逐字节重演保证。来源与署名见根目录 NOTICE.md。
