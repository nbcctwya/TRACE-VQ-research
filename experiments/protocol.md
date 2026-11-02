# 指标口径与比较边界

## 研究筛选与候选多 seed 实验

- 训练：2009-01-01 至 2020-12-31；验证：2021-01-01 至 2022-12-31；测试：2023-01-01 至 2025-12-31。
- 输入窗口 20 日，158 个股票特征、13 个先验因子、63 个市场特征、10 个未来收益标签。主标签为第 5 列：`Ref($close,-5)/Ref($close,-1)-1`。学习期标签经过处理器，预测分数不是原始收益率。
- Stage 1 基线量化器为 512 × 128 单码本；RVQ 与 FSQ 另行训练各自 Stage 1。A 股多 seed 和 SP500 候选重复实验固定对应 Stage 1 seed 42，只变化 Stage 2 seeds 0–4。
- 单 seed 探索使用 Stage 2 seed 0。最多 70 epochs，early stopping patience=15、min_delta=1e-5。新增排序等损失用于训练；检查点沿用原验证目标（主标签 MSE + 原辅助项），不能以测试集最好结果挑选检查点。
- 研究回测为 TopkDropout，topk=30、drop=5、risk_degree=0.95、close 成交价、买入成本 0.0005、卖出成本 0.0015、min_cost=0；报价为 quote_universe=all。SP500 与 A 股专家数量等预设不同，以实际配置为准。
- 测试交易日通常为 A 股 727、SP500 752。对同一模型比较时应核对 num_test_days，不能用股票池之间的原始指标差作为模型效应。

## 指标定义

实现位于 [protocol_metrics.py](../trace_vq/evaluation/protocol_metrics.py)。

- IC：逐交易日股票截面的 Pearson 相关，再对有效日期取均值。RankIC 用 Spearman 相关。无法定义的日期跳过。
- ICIR、RankICIR：对应逐日相关序列的均值 / 样本标准差（ddof=1），不乘 sqrt(252)。
- 输入组合指标的是扣费后的每日简单收益 r，令 g=log(1+r)。AR=exp(mean(g)×252)-1；STD=std(g,ddof=1)×sqrt(252)。
- Sharpe=sqrt(252)×mean(g)/std(g,ddof=1)，无风险利率为零。
- NAV 从 1 开始累积，MDD=min(NAV/累计峰值-1)。MDD 为负数，越接近零表示回撤越小。
- Sortino 使用所有交易日的下行偏差，MAR=0；Calmar=AR/abs(MDD)。未定义的值为 NaN/空白，不用零补充。
- `_mean` / `_std` 是各 seed 指标的均值与样本标准差。CSV 中 AR/MDD 为比例，例如 0.15 是 15%，-0.20 是 -20%。

## 历史归档

历史 CSI300 全模型/消融组合指标保留历史股票池报价；后续研究使用 all 报价，不应直接混比两者。Stage 1 多 seed 与两阶段多 seed 又对应不同随机性设置。部分历史全流程报告含 native ddof=0 的组合指标；这些列按原报告保留，相关目录 README 会注明，不能自动视为全量 protocol 重算。

## 解释限制

五个 seeds 共享同一测试期；候选此前已通过测试表现参与探索筛选，未做多重检验校正。五 seeds 不等于五次独立样本外验证。CSI300 与 CSI800 股票池有重叠，不是两个独立市场。提高均值不等于统计显著；样本标准差不等于均值的标准误。固定 Stage 1 的重复实验不覆盖 Stage 1 训练随机性。

汇总指标足以检查相对表现、seed 稳定性和代码/配置的对应关系。逐日统计检验、重建持仓或重新计算全部回测需另行提供原始明细；本仓库没有这些输入。候选结果均值与平均预测后的集成回测必须分别讨论。
