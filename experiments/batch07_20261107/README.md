# 第07批：码条件化与关系检索

状态：complete；完整新Stage2结果 8/8；无新增Stage1训练。

CSI300/CSI800各Stage2 seed0，Stage1复用对应seed42。训练2009–2020，验证2021–2022，测试2023–2025；验证检查点按原主MSE与原辅助项选择。
TopkDropout沿用all报价、top30/drop5、risk0.95、close、买入0.0005/卖出0.0015、最低费0。
本批全部以原FSQ为对照，保持128个潜在因子；以下AR/MDD/换手为百分比，CSV保留比例。

| 股票池 | 方案 | 对照 | IC | RankIC | 净年化 % | Sharpe | 最大回撤 % | 日均换手 % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300 | baseline_fsq |  | 0.03821 | 0.05702 | 18.027 | 1.117 | -11.684 | 32.200 |
| csi300 | code_hypergraph | baseline_fsq | 0.03706 | 0.05548 | 14.750 | 0.937 | -15.165 | 32.811 |
| csi300 | film_code | baseline_fsq | 0.03860 | 0.05516 | 16.466 | 0.920 | -19.995 | 32.707 |
| csi300 | hopfield_retrieval | baseline_fsq | 0.03897 | 0.05794 | 19.082 | 1.159 | -11.482 | 32.157 |
| csi300 | hyper_loading | baseline_fsq | 0.03651 | 0.05142 | 18.762 | 1.118 | -16.140 | 32.860 |
| csi800 | baseline_fsq |  | 0.03585 | 0.05701 | 14.281 | 0.781 | -27.883 | 32.968 |
| csi800 | code_hypergraph | baseline_fsq | 0.03661 | 0.05744 | 10.918 | 0.637 | -23.879 | 32.749 |
| csi800 | film_code | baseline_fsq | 0.03118 | 0.05231 | 16.377 | 0.739 | -28.294 | 32.689 |
| csi800 | hopfield_retrieval | baseline_fsq | 0.03373 | 0.05467 | 13.056 | 0.777 | -22.660 | 32.792 |
| csi800 | hyper_loading | baseline_fsq | 0.03232 | 0.04839 | 12.594 | 0.675 | -26.807 | 32.913 |

**相对对应基线的差值**。AR/MDD/换手为百分点，正MDD差值表示回撤减少。

| 股票池 | 方案 | ΔIC | ΔRankIC | Δ年化 | ΔSharpe | Δ回撤 | Δ换手 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300 | code_hypergraph | -0.00115 | -0.00154 | -3.277 | -0.180 | -3.481 | 0.612 |
| csi300 | film_code | 0.00038 | -0.00186 | -1.561 | -0.197 | -8.310 | 0.507 |
| csi300 | hopfield_retrieval | 0.00076 | 0.00092 | 1.055 | 0.042 | 0.202 | -0.043 |
| csi300 | hyper_loading | -0.00170 | -0.00560 | 0.735 | 0.001 | -4.455 | 0.660 |
| csi800 | code_hypergraph | 0.00075 | 0.00043 | -3.363 | -0.143 | 4.003 | -0.219 |
| csi800 | film_code | -0.00467 | -0.00470 | 2.096 | -0.042 | -0.412 | -0.279 |
| csi800 | hopfield_retrieval | -0.00213 | -0.00234 | -1.225 | -0.004 | 5.223 | -0.176 |
| csi800 | hyper_loading | -0.00354 | -0.00862 | -1.687 | -0.105 | 1.076 | -0.055 |

**实际资源**。保存/恢复路径fit合计6.75小时，另计中断前未保存训练至少0.00分钟；已记录训练合计至少6.75小时。预测、独立评估与回测9.86分钟。fit含训练内验证，不含停机等待。逐项见timing_memory.csv。PyTorch allocated/reserved峰值与整卡占用分别记录；整卡明细仅留本地。

**实现与比较边界**。

- code_hypergraph在Stage2以512个格点为超边中心，同码或曼哈顿距离1的股票参与同一超边；按超边/顶点度归一化做一次同日完整截面消息传递，残差输出零初始化，无标签输入。
- hyper_loading以16维code embedding生成rank4载荷增量的输出矩阵，输入低秩投影共享；分别加到先验/潜在载荷，按更新后载荷计算原范数辅助项。不生成整个大网络。
- film_code取消原前置结构token，改用学习CLS；由16维code embedding为每个时间块生成gamma/beta，在块输入做(1+gamma)*x+beta，零初始化。FSQ仍用于原下游融合和因子头。
- hopfield_retrieval复用第三批坐标/交易日缓存和原冻结h；每码最多保留8项，从同码或六向邻码选择最近16项。32维归一化键/查询、逆温度4、一次softmax关联更新；值为历史h与已到期的处理后主标签，残差输出零初始化。
- 标签成熟规则为source_day+5严格早于query_day，最小间隔6个完整市场交易日，排除自身和同日样本。按日期先解封过去标签，再生成候选；允许验证/测试期间已知的历史结果进入记忆，不用未到期标签，也不对验证/测试标签做梯度更新。
- 候选按固定日期/索引和码选择，不按标签大小选择；空记忆输出为零。使用未来标签扰动、完整交易日间隔和严格恢复测试；检索是受限候选池原型，不代表无限记忆或论文完整应用效果。

**恢复与监控**。每轮原子保存latest检查点及上一份备份，完整保存优化器、调度器、Python/NumPy/Torch/CUDA RNG。训练进程和独立GPU监控使用用户级系统服务，服务异常自动重启；单阶段最多3次尝试，独立修复最多8次，30分钟内最多3次。数据加载进程异常时仅降到workers=0；不自动改变模型、样本范围或回测规则。反复失败保留原因并继续其他项目。

**逐项描述**。仅seed0，不能据此断言稳定性或统计显著。

- csi300/code_hypergraph：ΔRankIC -0.00154，Δ净年化 -3.277个百分点，ΔSharpe -0.180，ΔMDD -3.481个百分点。
- csi300/film_code：ΔRankIC -0.00186，Δ净年化 -1.561个百分点，ΔSharpe -0.197，ΔMDD -8.310个百分点。
- csi300/hopfield_retrieval：ΔRankIC +0.00092，Δ净年化 +1.055个百分点，ΔSharpe +0.042，ΔMDD +0.202个百分点。
- csi300/hyper_loading：ΔRankIC -0.00560，Δ净年化 +0.735个百分点，ΔSharpe +0.001，ΔMDD -4.455个百分点。
- csi800/code_hypergraph：ΔRankIC +0.00043，Δ净年化 -3.363个百分点，ΔSharpe -0.143，ΔMDD +4.003个百分点。
- csi800/film_code：ΔRankIC -0.00470，Δ净年化 +2.096个百分点，ΔSharpe -0.042，ΔMDD -0.412个百分点。
- csi800/hopfield_retrieval：ΔRankIC -0.00234，Δ净年化 -1.225个百分点，ΔSharpe -0.004，ΔMDD +5.223个百分点。
- csi800/hyper_loading：ΔRankIC -0.00862，Δ净年化 -1.687个百分点，ΔSharpe -0.105，ΔMDD +1.076个百分点。

机制定义参考：[KAN](https://arxiv.org/abs/2404.19756)、[现代Hopfield检索](https://arxiv.org/abs/2008.02217)。

实现：trace_vq/models/batch67.py；因果检索缓存trace_vq/data/hopfield.py；训练入口trace_vq/cli/batch67.py，恢复回调trace_vq/training/resilient.py；串行编排/监控/报告research/run_batch67.py、monitor_batch67.py、report_batch67.py。配置见configs/。
