# 第04批：专家路由

状态：`complete`；新增Stage2完整结果 14/14。Stage1复用对应VQ/FSQ seed42，Stage2 seed0；无新增Stage1训练。

训练2009–2020，验证2021–2022，测试2023–2025。验证检查点按主标签MSE加本变体可用的原辅助项选择。TopkDropout使用all报价、top30/drop5、risk0.95、close、买入0.0005/卖出0.0015、最低费0。

AR、MDD、换手在CSV中为比例，下表为百分比。换手为Qlib逐日报告均值；MDD越接近0越好。

**主要结果与取舍**。以下解读对应本批固定seed0的实测结果。

- CSI800的soft_moe相对VQ基线六项指标均改善：RankIC提高0.00815、净年化提高4.939个百分点、Sharpe提高0.325、回撤减少8.135个百分点，换手下降。其净年化11.822%仍低于原FSQ的14.281%，不能将对VQ的改善解释为超过FSQ收益。
- CSI800的moe_finegrained8净年化13.720%，较VQ提高6.837个百分点，Sharpe和回撤改善；RankIC下降0.00326，净年化仍比FSQ低0.561个百分点。16专家方案在CSI800虽小幅增收，回撤却加深至-31.669%；增加专家数没有呈现一致优势。
- CSI300仅expert_k_all提高净年化：18.968%，较VQ提高0.472个百分点，Sharpe提高0.054；IC和RankIC下降，回撤略深。其余六个新配置的净年化均低于对应VQ/FSQ基线。
- expert_choice在CSI800的IC/RankIC略升，但净年化、Sharpe和回撤均变差；在CSI300也未提高净年化。
- fsq_sort_route使CSI300/CSI800净年化分别下降5.592/2.639个百分点；CSI800回撤改善，CSI300回撤更深。该配置同时将路由专家数从2改为3，容量与路由方式的作用不可单独归因。

单seed结果不支持稳定性或统计显著性的结论；较差结果和复用基线均保留。

fsq_sort_route的差值对照FSQ，其余新配置对照VQ。FSQ另作为跨模型参照；原top1直接复用VQ基线，无新增训练。

| 股票池 | 方案 | 对照 | IC | RankIC | 净年化 % | Sharpe | MDD % | 日均换手 % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300 | baseline_fsq |  | 0.03821 | 0.05702 | 18.027 | 1.117 | -11.684 | 32.200 |
| csi300 | baseline_vq |  | 0.04034 | 0.05679 | 18.496 | 1.078 | -18.118 | 32.780 |
| csi300 | expert_choice | baseline_vq | 0.03857 | 0.05469 | 18.175 | 1.084 | -16.565 | 32.783 |
| csi300 | expert_k_all | baseline_vq | 0.03898 | 0.05518 | 18.968 | 1.132 | -18.177 | 32.794 |
| csi300 | fsq_sort_route | baseline_fsq | 0.03459 | 0.04780 | 12.435 | 0.764 | -14.334 | 32.557 |
| csi300 | moe_biasbal | baseline_vq | 0.03960 | 0.05642 | 18.117 | 1.096 | -16.848 | 32.914 |
| csi300 | moe_finegrained16 | baseline_vq | 0.03115 | 0.04393 | 13.905 | 0.830 | -15.948 | 32.935 |
| csi300 | moe_finegrained8 | baseline_vq | 0.03890 | 0.05642 | 17.094 | 1.013 | -16.775 | 32.795 |
| csi300 | soft_moe | baseline_vq | 0.03120 | 0.04696 | 15.378 | 0.905 | -19.476 | 33.017 |
| csi800 | baseline_fsq |  | 0.03585 | 0.05701 | 14.281 | 0.781 | -27.883 | 32.968 |
| csi800 | baseline_vq |  | 0.03224 | 0.05468 | 6.883 | 0.411 | -27.217 | 33.010 |
| csi800 | expert_choice | baseline_vq | 0.03239 | 0.05481 | 6.601 | 0.392 | -30.270 | 32.830 |
| csi800 | expert_k_all | baseline_vq | 0.03197 | 0.05323 | 8.697 | 0.515 | -27.274 | 32.804 |
| csi800 | fsq_sort_route | baseline_fsq | 0.03371 | 0.05602 | 11.642 | 0.708 | -20.585 | 32.857 |
| csi800 | moe_biasbal | baseline_vq | 0.03202 | 0.05467 | 9.805 | 0.585 | -25.767 | 32.971 |
| csi800 | moe_finegrained16 | baseline_vq | 0.03196 | 0.05234 | 7.548 | 0.436 | -31.669 | 32.946 |
| csi800 | moe_finegrained8 | baseline_vq | 0.03314 | 0.05142 | 13.720 | 0.789 | -21.530 | 32.870 |
| csi800 | soft_moe | baseline_vq | 0.03626 | 0.06283 | 11.822 | 0.736 | -19.082 | 32.611 |

**相对同股票池、同seed、对应VQ/FSQ基线的差值**。正MDD差值表示回撤减少。

| 股票池 | 变体 | ΔIC | ΔRankIC | Δ净年化百分点 | ΔSharpe | Δ回撤百分点 | Δ换手百分点 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300 | expert_choice | -0.00177 | -0.00209 | -0.321 | 0.006 | 1.553 | 0.003 |
| csi300 | expert_k_all | -0.00137 | -0.00160 | 0.472 | 0.054 | -0.060 | 0.014 |
| csi300 | fsq_sort_route | -0.00362 | -0.00922 | -5.592 | -0.352 | -2.650 | 0.357 |
| csi300 | moe_biasbal | -0.00075 | -0.00036 | -0.379 | 0.018 | 1.269 | 0.134 |
| csi300 | moe_finegrained16 | -0.00919 | -0.01286 | -4.591 | -0.249 | 2.170 | 0.155 |
| csi300 | moe_finegrained8 | -0.00145 | -0.00037 | -1.402 | -0.066 | 1.342 | 0.015 |
| csi300 | soft_moe | -0.00915 | -0.00983 | -3.118 | -0.174 | -1.358 | 0.237 |
| csi800 | expert_choice | 0.00015 | 0.00012 | -0.282 | -0.020 | -3.054 | -0.180 |
| csi800 | expert_k_all | -0.00027 | -0.00146 | 1.814 | 0.104 | -0.058 | -0.206 |
| csi800 | fsq_sort_route | -0.00214 | -0.00098 | -2.639 | -0.072 | 7.298 | -0.111 |
| csi800 | moe_biasbal | -0.00022 | -0.00001 | 2.922 | 0.174 | 1.449 | -0.039 |
| csi800 | moe_finegrained16 | -0.00028 | -0.00234 | 0.665 | 0.025 | -4.453 | -0.064 |
| csi800 | moe_finegrained8 | 0.00090 | -0.00326 | 6.837 | 0.378 | 5.687 | -0.140 |
| csi800 | soft_moe | 0.00402 | 0.00815 | 4.939 | 0.325 | 8.135 | -0.399 |

**实际资源**。保存与恢复路径上的fit计时合计 9.88 小时，预测、独立恢复评估和回测 17.42 分钟。fit包含训练内验证，数据加载等准备时间另存training_result.json。逐项见timing_memory.csv；显存为PyTorch allocated/reserved峰值，不含桌面和其他进程。整卡采样记录保留在本地队列目录。

training_seconds保留原训练记录；unsaved_training_seconds_lower_bound另计中断前已执行、未保存而后需重算的时间下界。training_seconds_including_replay_lower_bound为两者之和，不计无训练进程期间的等待。

**中断与恢复**。计入中断前未保存的fit至少 14.15 分钟后，本批实际已记录fit耗时至少 10.12 小时。
batch04.csi300.soft_moe.seed0.train日志记录到第15轮，从第5轮检查点恢复，需重算10轮；恢复时间（UTC）2026-11-06T07:12:38.630846+00:00。实现和Stage1权重未改变，已有完整结果直接复用。首个重跑轮次的验证损失与原记录完全一致。

整卡GPU采样不完整：5/14项缺少训练期整卡采样。恢复后监控服务缺少WSL GPU命令路径；路径修复已写入源码，服务重启的自动审批超时，因此不能补出缺失整卡采样。逐项PyTorch峰值仍按训练与独立评估记录报告。

**实现与比较边界**。
biasbal仅移除负载/重要性均衡辅助损失，保留外层beta范数项；负载EMA=.99、bias步长=.001，bias只改变top-k索引，权重来自原概率。验证/推理不更新偏置。

expert_choice容量为ceil(N*k/E)，保留全股票共享专家；每股被选专家数及零路由比例见routing_diagnostics.csv。finegrained分别为8专家/宽16/k4与16专家/宽8/k8，输出宽度64、共享专家不变，记录总参数和激活参数估计；并非第一批只增加专家数的配置。

Soft MoE使用每专家4槽，dispatch沿股票归一化、combine沿所有槽归一化，保留共享分支。固定槽容量不额外加原top-k均衡损失，beta范数项保留。FSQ路由固定第0维、[0,1,2]/[3,4]/[5,6,7]三组，各一个专家；原学习路由冻结。其路由专家数同时由2变3，容量和路由形式不可单独归因。

expert_k_all保持2个路由专家全部激活；top1已由baseline_vq行直接复用，无重复训练。激活参数是按设定k的估计；expert_choice的逐股票实际覆盖统计单独保存。

**描述性判断**。以下自动判断只看RankIC、净年化、Sharpe和MDD四项，不含IC与换手。所有新配置仅seed0，不据此宣称稳定性或统计显著。

- csi300/expert_choice：有指标取舍。
- csi300/expert_k_all：有指标取舍。
- csi300/fsq_sort_route：未见四项主要指标改善。
- csi300/moe_biasbal：有指标取舍。
- csi300/moe_finegrained16：有指标取舍。
- csi300/moe_finegrained8：有指标取舍。
- csi300/soft_moe：未见四项主要指标改善。
- csi800/expert_choice：有指标取舍。
- csi800/expert_k_all：有指标取舍。
- csi800/fsq_sort_route：有指标取舍。
- csi800/moe_biasbal：有指标取舍。
- csi800/moe_finegrained16：有指标取舍。
- csi800/moe_finegrained8：有指标取舍。
- csi800/soft_moe：四项主要指标均未下降。

实现位于trace_vq/models/batch34.py、models/experts/routing_research.py、data/fsq_history.py；训练入口trace_vq/cli/batch34.py，串行编排research/run_batch34.py。独立有效配置见configs/，原始权重、特征缓存、预测明细和日志均在本地忽略的artifacts/research/中。
