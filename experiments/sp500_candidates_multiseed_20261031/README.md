# SP500 候选五 seed 验证

本次整理补充了[逐seed费用/换手汇总](cost_diagnostics.csv)：仅保存各seed的日均报告换手、
原净收益/Sharpe及输入来源哈希，共20行。候选换手从本地Qlib逐日报告聚合，基线复用既有费用汇总；
逐日报告与预测明细不进入Git。统一结果表的换手列引用此文件。
维护命令为 `research/run.sh research/organize_experiment_results.py --refresh-local-turnover`，
需要本地既有报告；普通整理命令只读取已纳入Git的聚合文档。

股票池：SP500。种子：Stage 2: 0–4。记录数：20 条（15 次新训练、5 条复用基线）。

FSQ、RankNet、FSQ+RankNet 各五 seeds，另训练一个共享的 FSQ Stage 1（seed 42）。未观察到稳定的综合优势。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [matched_seed_deltas.csv](matched_seed_deltas.csv)
- [mean_std.csv](mean_std.csv)
- [paired_summary.csv](paired_summary.csv)
- [plan.json](plan.json)
- [seed_metrics.csv](seed_metrics.csv)
- [summary_readable.csv](summary_readable.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
