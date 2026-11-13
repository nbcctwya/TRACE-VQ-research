# 第八、九批运行说明

本次只执行清单39–53：第八批11种配置、第九批8种配置，各用于CSI300/CSI800，共38组两阶段实验。
每组从头训练Stage1 seed42，再冻结对应权重训练标准Stage2 seed0；共计划38次Stage1与38次Stage2训练。
第八批全部训练、评估、回测和汇总完成后进入第九批。原FSQ seed42/seed0结果仅作为复用对照，不重复训练。

两批已于2026-11-13 05:20（UTC+8）全部完成：第八批22/22组、第九批16/16组，零训练失败、零中断恢复。
完整指标、相对基线差值和资源统计见[第八批结果](../experiments/batch08_20261108/README.md)与[第九批结果](../experiments/batch09_20261108/README.md)。
两批fit分别56.20、54.83小时，预测评估与回测分别30.41、21.90分钟；单seed探索不构成稳定提升的证据。

模型实现：[batch89.py](../trace_vq/models/batch89.py)、[batch89_quantizers.py](../trace_vq/models/batch89_quantizers.py)。
相邻日数据：[batch89.py](../trace_vq/data/batch89.py)；训练入口：[CLI](../trace_vq/cli/batch89.py)。
每项独立设置与比较边界见[报告生成器](../research/report_batch89.py)中的METHODS。
`s1_predw_1e3/1e2/1e1`分别对应预测监督权重0.001、0.01、0.1；FSQ档位形状只跑清单首轮三组512码。

本地目录为`artifacts/research/batch0809_20261108/`：

- `STATUS.md`、`monitoring.json`提供完整结果数、Stage1/Stage2完成数、当前阶段、epoch与心跳。
- `batch08_20261108/configs/`、`batch09_20261108/configs/`保存76份阶段配置；每个模型目录另保存实际生效配置。
- 各模型输出分为`stage1/seed42/`及`stage2/seed0/`；后者保存该新Stage1对应的独立冻结缓存和回测。
- Stage2只能使用通过Stage1验证选择的真实权重；预检查权重不得进入正式训练。
- 残差/分组FSQ分量码以两个整数单独保存；分组表示保留在z_q的两个64维半区，不生成联合one-hot。
- 每批结束自动整理到`experiments/batch08_20261108/`、`experiments/batch09_20261108/`，不把计划或预检查视为完成结果。

Stage1沿用原重建与十期限收益监督，验证目标包含对应量化器内在目标及指定收益监督权重；
额外风格/维度相关和时间对比惩罚仅用于训练。不同Stage1目标的绝对验证损失不直接跨方案比较。
Stage2沿用[统一协议](../experiments/protocol.md)的主标签MSE、原辅助项、数据划分与TopkDropout回测。
新批次`training_seconds`为两阶段已记录fit之和，`stage1_training_seconds`与`stage2_training_seconds`分别保留；
fit含训练内验证，停机等待不计训练，中断未保存工作另计下界。失败项保留原因与已知资源时间，指标不补零。

先验13列JKP因子与个股风格区分处理：风格去相关使用实际存在的ROC20/STD20/BETA20/VSTD20股票特征。
时间对比按完整交易日配对同股票，缺前一天时跳过；两天均保留完整截面，不读取未来标签。
BSQ包含球面归一化、符号量化和联合熵目标，与九维二档FSQ分开。
Sinkhorn分别记录软质量与硬占用；CSI300股票数小于512时不要求硬码等量。

队列服务为`trace-vq-batch89-20261108.service`，独立监控为`trace-vq-batch89-watchdog-20261108.service`。
用户systemd启动后可自动续跑；每个阶段每轮原子保存latest检查点及上一份备份，含优化器、调度器与完整随机数状态。
已完成阶段直接复用；不完整缓存仅在本次目录内归档后重建。单阶段最多3次尝试，独立修复累计最多16次、30分钟最多3次。
数据加载进程异常可降为workers=0，不自动改变模型或数据/回测协议；持续失败记录原因后继续其他项目。
`FINISHED`防止重启后重复训练；`PAUSED`阻止继续启动；训练来源哈希变化会阻止混用实现。

本次不修改历史量化器源码及其缓存指纹目录，不安装或升级现有环境包。所有新输出都在忽略目录，Git仅保留代码、配置和聚合文档。
机制参考：[BSQ](https://arxiv.org/abs/2406.07548)、[SwAV软分配](https://github.com/facebookresearch/swav/blob/main/main_swav.py)。
