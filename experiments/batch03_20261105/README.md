# 第03批：FSQ状态变化与记忆

状态：`complete`；新增Stage2完整结果 16/16。Stage1复用对应VQ/FSQ seed42，Stage2 seed0；无新增Stage1训练。

训练2009–2020，验证2021–2022，测试2023–2025。验证检查点按主标签MSE加本变体可用的原辅助项选择。TopkDropout使用all报价、top30/drop5、risk0.95、close、买入0.0005/卖出0.0015、最低费0。

AR、MDD、换手在CSV中为比例，下表为百分比。换手为Qlib逐日报告均值；MDD越接近0越好。

**主要结果与取舍**。以下解读对应本批固定seed0的实测结果。

- CSI300的code_hist_state相对FSQ将净年化从18.027%提高到21.660%、Sharpe从1.117提高到1.331，IC和RankIC也提高；最大回撤从-11.684%变为-15.898%，加深4.214个百分点，换手上升。CSI800同一方案净年化提高0.945个百分点、回撤减少10.060个百分点，但IC和RankIC下降。
- CSI300的code_trans/code_vol同时改善净年化、Sharpe和回撤；code_trans的IC略降、换手略升。这两种特征在CSI800的净年化和Sharpe均下降，不能据此宣称跨股票池一致改善。
- CSI800的code_dwell净年化为16.795%，较FSQ提高2.513个百分点，Sharpe为0.918；IC略升、RankIC基本不变但略降，回撤略有改善。
- CSI800的fsq_gram2净年化15.593%、Sharpe0.928、最大回撤-18.009%，组合指标改善但IC和RankIC下降；CSI300同一方案的收益、Sharpe和回撤均变差。fsq_gram3未显示两个股票池一致的收益/Sharpe提升。
- proto_memory在CSI300提高IC和RankIC，却使净年化下降7.134个百分点、Sharpe下降0.505，回撤加深16.259个百分点；CSI800净年化基本持平而Sharpe下降。相关系数改善不能直接视为组合收益改善。

单seed结果不支持稳定性或统计显著性的结论；较差结果和复用基线均保留。

| 股票池 | 方案 | 对照 | IC | RankIC | 净年化 % | Sharpe | MDD % | 日均换手 % |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300 | baseline_fsq |  | 0.03821 | 0.05702 | 18.027 | 1.117 | -11.684 | 32.200 |
| csi300 | code_delta | baseline_fsq | 0.03857 | 0.05718 | 18.129 | 1.102 | -11.831 | 32.163 |
| csi300 | code_dwell | baseline_fsq | 0.03843 | 0.05702 | 18.035 | 1.119 | -11.355 | 32.033 |
| csi300 | code_hist_state | baseline_fsq | 0.04059 | 0.05850 | 21.660 | 1.331 | -15.898 | 32.762 |
| csi300 | code_trans | baseline_fsq | 0.03820 | 0.05707 | 19.129 | 1.170 | -10.645 | 32.246 |
| csi300 | code_vol | baseline_fsq | 0.03877 | 0.05744 | 18.758 | 1.148 | -11.084 | 32.199 |
| csi300 | fsq_gram2 | baseline_fsq | 0.03622 | 0.05128 | 15.953 | 0.947 | -16.485 | 32.886 |
| csi300 | fsq_gram3 | baseline_fsq | 0.03686 | 0.05230 | 18.440 | 1.110 | -16.268 | 32.839 |
| csi300 | proto_memory | baseline_fsq | 0.04103 | 0.05873 | 10.893 | 0.612 | -27.944 | 32.973 |
| csi800 | baseline_fsq |  | 0.03585 | 0.05701 | 14.281 | 0.781 | -27.883 | 32.968 |
| csi800 | code_delta | baseline_fsq | 0.03474 | 0.05587 | 14.553 | 0.873 | -18.632 | 32.872 |
| csi800 | code_dwell | baseline_fsq | 0.03597 | 0.05696 | 16.795 | 0.918 | -27.036 | 32.897 |
| csi800 | code_hist_state | baseline_fsq | 0.03428 | 0.05594 | 15.226 | 0.907 | -17.823 | 32.858 |
| csi800 | code_trans | baseline_fsq | 0.03273 | 0.05424 | 13.315 | 0.649 | -23.936 | 32.829 |
| csi800 | code_vol | baseline_fsq | 0.03299 | 0.05482 | 12.922 | 0.641 | -22.904 | 32.805 |
| csi800 | fsq_gram2 | baseline_fsq | 0.03451 | 0.05547 | 15.593 | 0.928 | -18.009 | 32.794 |
| csi800 | fsq_gram3 | baseline_fsq | 0.03454 | 0.05541 | 11.795 | 0.721 | -19.849 | 32.867 |
| csi800 | proto_memory | baseline_fsq | 0.03216 | 0.05427 | 14.241 | 0.724 | -21.407 | 32.652 |

**相对同股票池、同seed、对应VQ/FSQ基线的差值**。正MDD差值表示回撤减少。

| 股票池 | 变体 | ΔIC | ΔRankIC | Δ净年化百分点 | ΔSharpe | Δ回撤百分点 | Δ换手百分点 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| csi300 | code_delta | 0.00036 | 0.00016 | 0.102 | -0.015 | -0.147 | -0.037 |
| csi300 | code_dwell | 0.00022 | -0.00000 | 0.008 | 0.002 | 0.330 | -0.167 |
| csi300 | code_hist_state | 0.00237 | 0.00147 | 3.633 | 0.214 | -4.214 | 0.562 |
| csi300 | code_trans | -0.00001 | 0.00005 | 1.102 | 0.053 | 1.039 | 0.046 |
| csi300 | code_vol | 0.00056 | 0.00041 | 0.731 | 0.032 | 0.600 | -0.001 |
| csi300 | fsq_gram2 | -0.00199 | -0.00574 | -2.074 | -0.170 | -4.800 | 0.687 |
| csi300 | fsq_gram3 | -0.00135 | -0.00472 | 0.413 | -0.006 | -4.583 | 0.639 |
| csi300 | proto_memory | 0.00282 | 0.00171 | -7.134 | -0.505 | -16.259 | 0.773 |
| csi800 | code_delta | -0.00111 | -0.00114 | 0.272 | 0.093 | 9.251 | -0.097 |
| csi800 | code_dwell | 0.00012 | -0.00005 | 2.513 | 0.137 | 0.847 | -0.071 |
| csi800 | code_hist_state | -0.00157 | -0.00106 | 0.945 | 0.126 | 10.060 | -0.110 |
| csi800 | code_trans | -0.00313 | -0.00276 | -0.966 | -0.132 | 3.946 | -0.139 |
| csi800 | code_vol | -0.00286 | -0.00219 | -1.360 | -0.139 | 4.979 | -0.163 |
| csi800 | fsq_gram2 | -0.00134 | -0.00154 | 1.312 | 0.147 | 9.874 | -0.174 |
| csi800 | fsq_gram3 | -0.00132 | -0.00159 | -2.486 | -0.060 | 8.034 | -0.101 |
| csi800 | proto_memory | -0.00369 | -0.00274 | -0.040 | -0.057 | 6.476 | -0.317 |

**实际资源**。保存与恢复路径上的fit计时合计 12.76 小时，预测、独立恢复评估和回测 20.70 分钟。fit包含训练内验证，数据加载等准备时间另存training_result.json。逐项见timing_memory.csv；显存为PyTorch allocated/reserved峰值，不含桌面和其他进程。整卡采样记录保留在本地队列目录。

training_seconds保留原训练记录；unsaved_training_seconds_lower_bound另计中断前已执行、未保存而后需重算的时间下界。training_seconds_including_replay_lower_bound为两者之和，不计无训练进程期间的等待。

**实现与比较边界**。
历史特征来自既有冻结FSQ h与坐标，不使用标签。delta为1/5/20市场交易日坐标差加缺失掩码；dwell截断252日并log1p；trans为仅训练期拟合、Laplace=1的三维8×8转移矩阵查询；vol用过去5/20日的总体标准差和有效观测比例，不含当日。

2/3-gram只使用之前完整code，4096桶、32维嵌入、当前h的sigmoid门。prototype以冻结h为表示、每日code均值作EMA(.99)，先读上一日期原型，再更新当天；跨split因果延续。训练日期虽打乱，历史特征已按真实时间预计算，不随训练访问顺序改写。

新增历史特征经零初始化残差投影注入128维Stage2表示；保留原FSQ的128维因子及CA。code_hist_state使用三维边际8档占比24维与归一化熵3维，零初始化投影影响专家门控。

**描述性判断**。以下自动判断只看RankIC、净年化、Sharpe和MDD四项，不含IC与换手。所有新配置仅seed0，不据此宣称稳定性或统计显著。

- csi300/code_delta：有指标取舍。
- csi300/code_dwell：有指标取舍。
- csi300/code_hist_state：有指标取舍。
- csi300/code_trans：四项主要指标均未下降。
- csi300/code_vol：四项主要指标均未下降。
- csi300/fsq_gram2：未见四项主要指标改善。
- csi300/fsq_gram3：有指标取舍。
- csi300/proto_memory：有指标取舍。
- csi800/code_delta：有指标取舍。
- csi800/code_dwell：有指标取舍。
- csi800/code_hist_state：有指标取舍。
- csi800/code_trans：有指标取舍。
- csi800/code_vol：有指标取舍。
- csi800/fsq_gram2：有指标取舍。
- csi800/fsq_gram3：有指标取舍。
- csi800/proto_memory：有指标取舍。

实现位于trace_vq/models/batch34.py、models/experts/routing_research.py、data/fsq_history.py；训练入口trace_vq/cli/batch34.py，串行编排research/run_batch34.py。独立有效配置见configs/，原始权重、特征缓存、预测明细和日志均在本地忽略的artifacts/research/中。
