# 机制与空对照

股票池：CSI300 / CSI800。种子：Stage 2: 0。记录数：20 条（含 6 条复用对照）。

检验连续历史、错配历史、载荷平滑、peer 关系与状态均衡。最终审查没有从本轮新方向中晋级候选；脚本的 promising 标签不能代替最终审查。



结果表保留所有已收录方案，包括较差结果；每个 seed 一行不属于逐股票或逐日明细。复用基线不等于新增训练，批次行数不能直接相加。

此目录的现有文件：

- [cost_diagnostics.csv](cost_diagnostics.csv)
- [decisions.json](decisions.json)
- [drawdown_events.csv](drawdown_events.csv)
- [final_comparison.csv](final_comparison.csv)
- [horizon_decay.csv](horizon_decay.csv)
- [leaderboard.csv](leaderboard.csv)
- [loading_stability_summary.csv](loading_stability_summary.csv)
- [paired_uncertainty.csv](paired_uncertainty.csv)
- [peer_self_uncertainty.csv](peer_self_uncertainty.csv)
- [plan.json](plan.json)
- [prediction_change.csv](prediction_change.csv)
- [previous_candidate_uncertainty.csv](previous_candidate_uncertainty.csv)
- [primary_uncertainty.csv](primary_uncertainty.csv)
- [regime_metrics.csv](regime_metrics.csv)
- [report_summary.json](report_summary.json)
- [shortlist.csv](shortlist.csv)
- [yearly_metrics.csv](yearly_metrics.csv)

若有 `configs/`，其中为已保存的实际配置；本机路径已改为相对引用，数值设置保留。完整配置索引见 [config_index.csv](../config_index.csv)。指标约定见 [protocol.md](../protocol.md)。
