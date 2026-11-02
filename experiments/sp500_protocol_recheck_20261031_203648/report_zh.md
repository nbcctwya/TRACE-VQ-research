复核结论：TRACE-VQ SP500 已入库指标复算正确，主要指标公式和交易参数与其他 baseline 一致；全部历史实验条件尚未完全统一。

本次独立复核于 2026-10-31T12:44:40.744677+00:00 完成，主检查 424 项通过、0 项失败。

使用 MASTER、MATCC、PRISM-VQ、FactorVAE、FactorVQVAE、Qlib baselines 六套实际指标实现，对 TRACE-VQ 五个 seed 和 avg_none 的十项指标交叉复算，共 360 个数值，全部一致；最大绝对误差 1.20348e-13。

直接调用 FactorVQVAE 的标准 Qlib 适配器，对五个 seed 和 avg_none 分别重新回测。逐日毛收益、费用和净收益均一致，最大绝对差 1.18829e-15。净收益还与账户净值变动核对一致，交易成本只扣除一次。

另从当前 baseline-results 中 10 个 SP500 模型/组合的公开曲线重新计算全部六项组合指标，60 项均与各自的已发布数值一致。30 项指标边界检查也全部通过。

共同计算约定：净简单收益等于 Qlib report.return − report.cost；g=log1p(净收益)；年化 252 天；标准差 ddof=1；无风险利率和 MAR 均为 0；最大回撤包含初始净值 1；Sortino 的下行均方根覆盖全部交易日；ICIR/RankICIR 不年化；跨 seed 标准差使用 ddof=1。avg_none 为原始预测按键 inner join 后取均值，独立回测。

共同交易参数：Top30/Drop5、risk_degree=0.95、初始资金 1e8、日频收盘价、买入费率 0.0005、卖出费率 0.0015、最低费用 0；Qlib 内部将信号移后一日，无手工重复移位；美股交易单位 1 股、无 A 股涨跌停阈值；测试 2023–2025 共 752 个交易日。

已确认的历史差异：

- MASTER 的历史 SP500 导出器显式传入 codes=sp500；TRACE-VQ 和其余主要适配器使用默认的全部行情。选股预测仍来自 SP500；区别会影响已退出成分股的估值、卖出行情等。
- 保持同一 TRACE-VQ 集成预测和当前 Qlib 0.9.7 不变，仅切换行情范围，得到以下对照。该对照保存在独立审计目录，没有替换正式结果。

| 行情范围 | 净年化收益 | 年化波动率 | 最大回撤 | Sharpe | Sortino | Calmar |
|---|---:|---:|---:|---:|---:|---:|
| all（正式 TRACE-VQ） | 5.8556% | 17.8787% | -22.7825% | 0.318288 | 0.449183 | 0.257022 |
| sp500（MASTER 行情范围） | 5.2477% | 16.8542% | -23.8911% | 0.303467 | 0.439800 | 0.219653 |

- TRACE-VQ、MASTER、MATCC、PRISM-VQ 的标签为 close(t+5)/close(t+1)−1；FactorVAE、FactorVQVAE 和 Qlib baselines 使用 close(t+2)/close(t+1)−1，FactorVAE 还声明 CSRankNorm。因此 IC/RankIC 的预测目标并非全部相同。
- MASTER SP500 元数据记录 Qlib 0.9.3.99，TRACE-VQ 和多数包记录 0.9.7，不能声称所有历史引擎版本完全相同。当前入库 MASTER 数值与 master_corrected 的修正结果吻合，本次没有将其误判为尚未修正的日期索引故障。

保护检查：本次复核前后的全部 353 个已入库文件 SHA256 完全相同。原增量导入前的 335 个文件再次通过保护检查：319 个字节不变，其余只增加新记录/元数据，原 CSV 行、精度及顺序和旧元数据均保留。原 TRACE-VQ 工程完整性检查通过。

机器可读主报告：report.json；360 项逐值明细：cross_baseline_metric_values.csv；逐日重放：seed*_factorvqvae_replay.csv 和 avg_none_factorvqvae_replay.csv；行情对照：quote_policy_sensitivity.json；其余检查见 supplemental_summary.json。
