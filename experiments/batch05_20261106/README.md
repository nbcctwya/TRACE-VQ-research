# 第05批：新时间模块与截面交互

状态：complete；完整新Stage2结果 8/8；无新增Stage1训练。

CSI300/CSI800各Stage2 seed0，Stage1复用对应seed42。训练2009–2020，验证2021–2022，测试2023–2025；验证检查点按原主MSE与原辅助项选择。
TopkDropout沿用all报价、top30/drop5、risk0.95、close、买入0.0005/卖出0.0015、最低费0。
raven_context_experts对照FSQ，其余对照VQ。以下AR/MDD/换手为百分比，CSV保留比例。

| 股票池 | 方案 | 对照 | IC | RankIC | 净年化 % | Sharpe | 最大回撤 % | 日均换手 % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300 | baseline_fsq |  | 0.03821 | 0.05702 | 18.027 | 1.117 | -11.684 | 32.200 |
| csi300 | baseline_vq |  | 0.04034 | 0.05679 | 18.496 | 1.078 | -18.118 | 32.780 |
| csi300 | kda_temporal | baseline_vq | 0.03086 | 0.04951 | 13.502 | 0.787 | -13.782 | 33.016 |
| csi300 | mhc | baseline_vq | 0.04019 | 0.05654 | 18.428 | 1.082 | -17.936 | 32.920 |
| csi300 | raven_context_experts | baseline_fsq | 0.03608 | 0.04951 | 16.925 | 0.998 | -18.901 | 32.537 |
| csi300 | s2_cross_attn | baseline_vq | 0.03695 | 0.05389 | 15.281 | 0.794 | -25.197 | 32.867 |
| csi800 | baseline_fsq |  | 0.03585 | 0.05701 | 14.281 | 0.781 | -27.883 | 32.968 |
| csi800 | baseline_vq |  | 0.03224 | 0.05468 | 6.883 | 0.411 | -27.217 | 33.010 |
| csi800 | kda_temporal | baseline_vq | 0.03026 | 0.04774 | 9.035 | 0.477 | -30.437 | 32.633 |
| csi800 | mhc | baseline_vq | 0.03235 | 0.05481 | 10.614 | 0.620 | -26.484 | 32.831 |
| csi800 | raven_context_experts | baseline_fsq | 0.02821 | 0.05466 | 15.683 | 0.864 | -22.734 | 26.671 |
| csi800 | s2_cross_attn | baseline_vq | 0.03220 | 0.05236 | 9.089 | 0.495 | -29.811 | 32.804 |

**相对对应基线的差值**。AR/MDD/换手为百分点，正MDD差值表示回撤减少。

| 股票池 | 方案 | ΔIC | ΔRankIC | Δ年化 | ΔSharpe | Δ回撤 | Δ换手 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300 | kda_temporal | -0.00948 | -0.00727 | -4.994 | -0.291 | 4.336 | 0.236 |
| csi300 | mhc | -0.00015 | -0.00025 | -0.068 | 0.004 | 0.182 | 0.140 |
| csi300 | raven_context_experts | -0.00213 | -0.00751 | -1.102 | -0.119 | -7.217 | 0.337 |
| csi300 | s2_cross_attn | -0.00340 | -0.00290 | -3.215 | -0.285 | -7.080 | 0.087 |
| csi800 | kda_temporal | -0.00198 | -0.00695 | 2.152 | 0.066 | -3.220 | -0.376 |
| csi800 | mhc | 0.00011 | 0.00013 | 3.732 | 0.208 | 0.732 | -0.178 |
| csi800 | raven_context_experts | -0.00764 | -0.00235 | 1.402 | 0.083 | 5.149 | -6.297 |
| csi800 | s2_cross_attn | -0.00004 | -0.00232 | 2.207 | 0.084 | -2.594 | -0.205 |

**实际资源**。保存/恢复路径fit合计8.36小时，另计中断前未保存训练至少0.00分钟；已记录训练合计至少8.36小时。预测、独立评估与回测11.28分钟。fit含训练内验证，不含停机等待。逐项见timing_memory.csv。PyTorch allocated/reserved峰值与整卡占用分别记录；整卡明细仅留本地。

**实现与比较边界**。

- KDA使用已有PyTorch的参考递推，逐key通道遗忘门、delta更新、q/k L2归一化、4阶因果短卷积及sigmoid输出门。环境没有FLA，未安装或升级包；不据此宣称FLA融合内核性能或完整复现Kimi Linear大模型。保留Stage2前后的MLP/归一化，结构读取token改放最后且不使用RoPE，以便因果递推读取全部历史。
- mHC使用2条残差流，动态正pre/post映射和20轮log-space Sinkhorn残差映射，分别作用于原注意力/FFN两条残差连接；输出取流均值。保留原Stage2 post-norm约定，未采用论文的大模型混合精度融合内核。
- 多窗口专家使用FSQ路由，分别读取5/10/20日；5/10日塔先裁剪原始输入再做DLinear，采用学习CLS而不注入完整20日FSQ信息。共享分支保留完整20日输入。路由专家数由2变3，新增两个时间塔；所有时间塔均计算，top1不等于跳过其他时间塔计算。需同时考察容量和耗时差异。
- 跨股票注意力在同一交易日的完整股票截面上、原融合/载荷头之前计算；输出投影零初始化。没有跨日期读取，也不读取标签。

**恢复与监控**。每轮原子保存latest检查点及上一份备份，完整保存优化器、调度器、Python/NumPy/Torch/CUDA RNG。训练进程和独立GPU监控使用用户级系统服务，服务异常自动重启；单阶段最多3次尝试，独立修复最多8次，30分钟内最多3次。数据加载进程异常时仅降到workers=0；不自动改变模型、样本范围或回测规则。反复失败保留原因并继续其他项目。

**逐项描述**。仅seed0，不能据此断言稳定性或统计显著。

- csi300/kda_temporal：ΔRankIC -0.00727，Δ净年化 -4.994个百分点，ΔSharpe -0.291，ΔMDD +4.336个百分点。
- csi300/mhc：ΔRankIC -0.00025，Δ净年化 -0.068个百分点，ΔSharpe +0.004，ΔMDD +0.182个百分点。
- csi300/raven_context_experts：ΔRankIC -0.00751，Δ净年化 -1.102个百分点，ΔSharpe -0.119，ΔMDD -7.217个百分点。
- csi300/s2_cross_attn：ΔRankIC -0.00290，Δ净年化 -3.215个百分点，ΔSharpe -0.285，ΔMDD -7.080个百分点。
- csi800/kda_temporal：ΔRankIC -0.00695，Δ净年化 +2.152个百分点，ΔSharpe +0.066，ΔMDD -3.220个百分点。
- csi800/mhc：ΔRankIC +0.00013，Δ净年化 +3.732个百分点，ΔSharpe +0.208，ΔMDD +0.732个百分点。
- csi800/raven_context_experts：ΔRankIC -0.00235，Δ净年化 +1.402个百分点，ΔSharpe +0.083，ΔMDD +5.149个百分点。
- csi800/s2_cross_attn：ΔRankIC -0.00232，Δ净年化 +2.207个百分点，ΔSharpe +0.084，ΔMDD -2.594个百分点。

机制核对来源：[Kimi Linear](https://arxiv.org/abs/2510.26692)、[FLA KDA层](https://github.com/fla-org/flash-linear-attention/blob/main/fla/layers/kda.py)、[FLA参考算子](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py)、[mHC](https://arxiv.org/abs/2512.24880)。

实现：trace_vq/models/batch05.py、trace_vq/models/prediction/batch05_temporal.py；训练入口trace_vq/cli/batch05.py，恢复回调trace_vq/training/resilient.py；编排/监控/报告research/run_batch05.py、monitor_batch05.py、report_batch05.py。配置见configs/。
