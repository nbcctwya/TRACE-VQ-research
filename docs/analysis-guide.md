# 代码与结果阅读顺序

1. [完整研究总览](../experiments/overview.md)、[实验索引](../experiments/README.md)和[指标口径](../experiments/protocol.md)：确认18个研究/归档条目的复用关系与比较边界。
2. [逐seed去重结果](../experiments/research_runs.csv)与[按设置统计](../experiments/research_mean_std.csv)：178条研究结果、114个设置，区分五seed复验和单seed探索；缺失指标留空。
3. [预测集成](../experiments/research_ensembles.csv)：7份集成单独分析。各seed的组合指标均值与平均预测后的回测结果分别报告。
4. [第一批](../experiments/batch01_20261103/README.md)、[第二批](../experiments/batch02_20261103/README.md)、[第三批](../experiments/batch03_20261105/README.md)、[第四批](../experiments/batch04_20261105/README.md)及对应差值CSV：前两批核对平滑组控制；第三批对照FSQ，第四批除FSQ固定路由外对照VQ。对VQ的提升不代表超过FSQ。
5. [超参数表](../experiments/hyperparameters.csv)、[配置索引](../experiments/config_index.csv)和[变体说明](../experiments/variants.csv)：228份实际YAML配置及代码对应关系。路径已做移植处理，数值参数保留。
6. [架构](architecture.md)，然后读 `trace_vq/models/representation.py`、`trace_vq/training/prediction.py`、`trace_vq/models/screening.py`、`next_screen.py`、`mechanism_screen.py`、`combination_screen.py`、`batch01.py`、`batch34.py`。第三批历史特征位于 `trace_vq/data/fsq_history.py`，第四批路由位于 `trace_vq/models/experts/routing_research.py`。
7. 查看对应 `trace_vq/cli/` 与 `research/run_*.py`，理解检查点选择、缓存和实验编排。第二批复用NextPredictor的配对实现；第三、第四批由 `research/run_batch34.py` 串行编排，报告生成位于 `research/report_batch34.py`。恢复记录位于 `trace_vq/training/research_run.py`。

第三、第四批仅Stage2 seed0，没有新增集成。读完整结果与主要取舍，不能只依赖自动的四指标描述性标签。
资源分析优先读取各批timing_memory.csv：training_seconds沿用保存/恢复路径的计时；
training_seconds_including_replay_lower_bound另计中断前未保存、后续需重算的已记录时间下界，停机等待不计训练。
第四批整卡采样存在缺口，不能把缺失填为零；逐项PyTorch allocated/reserved峰值与整卡占用不是同一指标。

前期64条单seed与60行多seed快照保留原值，已纳入统一表的重叠结果通过source_views链接，不能再次相加。
历史消融、Stage1随机性、两阶段随机性及其他指标/报价口径在独立目录保留，未混入统一研究表。

引用结论时给出股票池、方案、seed 范围、测试区间和表格路径；区分实测结果、已有报告判断与新的推断。不要把代码中的所有可选分支都当成已经跑过，也不要把重复引用的基线算作新训练。

推荐分析问题：FSQ 与 RankNet 的单独和组合效应是什么？收益改进能否覆盖 RankIC/回撤的取舍？同 seed 的胜出次数与均值是否一致？历史状态的错配控制能否支持机制解释？SP500 的弱排序能力与较高组合收益是否主要关联市场暴露？哪些结论还需要原始明细验证？

Git 跟踪的源码为当前研究版本，包含各轮之后的累计扩展。历史源码快照、权重和数据只在本地忽略目录保留；公开视图不能单凭当前代码对所有旧训练作逐字节重演保证。来源与署名见根目录 NOTICE.md。
