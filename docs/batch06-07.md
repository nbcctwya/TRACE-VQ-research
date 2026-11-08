# 第六、七批运行说明

本次只执行清单28–38：第六批7个变体×CSI300/CSI800，第七批4个变体×两个股票池，共22次Stage2 seed0训练。
Stage1均复用对应FSQ seed42。各变体独立配置、独立输出；第六批训练、评估、回测及汇总完成后进入第七批。

22/22项已于2026-11-08 07:40（UTC+8）完成，全程无训练中断或失败。结果、基线差值、实际配置及资源记录见
[第六批报告](../experiments/batch06_20261107/README.md)与[第七批报告](../experiments/batch07_20261107/README.md)。
训练及监控服务已正常结束，FINISHED标记防止重复启动。

实现位于[batch67.py](../trace_vq/models/batch67.py)，历史检索位于[hopfield.py](../trace_vq/data/hopfield.py)。
逐项方法与参数见[报告生成器](../research/report_batch67.py)中的METHODS，以及每项effective.config.yaml。
首轮单seed结果用于探索，参数量不同；结果提升不等于稳定性或统计显著。

本地队列目录为`artifacts/research/batch0607_20261107/`：

- `STATUS.md`、`monitoring.json`提供当前阶段与进度，`watchdog.json`记录健康状态和GPU采样是否可用。
- `batch06_20261107/configs/`及`batch07_20261107/configs/`保存22份计划配置，各模型目录保留实际生效配置。
- `progress.json`保留分阶段状态、尝试次数、恢复记录；`results.csv`保留基线复用标记、缺失/失败与指标差值。
- 每批结束，自动整理指标、时间、显存与可移植配置到`experiments/batch06_20261107/`、`experiments/batch07_20261107/`。
  这些文档目录在对应批次结束时才生成，不把计划视为完成结果。

训练与回测沿用[统一协议](../experiments/protocol.md)。未到期的历史标签不能用于检索：source_day+5必须严格早于query_day，
其中day是完整市场交易日索引，最小间隔6日。候选只按码/日期选择，最多16项；训练、验证、测试按时间解封已知历史标签，
验证/测试标签不用于更新模型权重。第三批原始历史缓存保持不变，新增带标签检索缓存仅存于本次忽略目录。

队列由`trace-vq-batch67-20261107.service`运行，独立监控为`trace-vq-batch67-watchdog-20261107.service`。
两者已配置为用户级持久服务；用户systemd启动后可继续，电脑关机期间不执行计算。
每轮原子保存latest检查点，并保留上一份备份；包含优化器、调度器和完整随机数状态。
服务重启后从最近完整轮恢复，已完成训练或评估阶段直接复用。
单阶段最多3次尝试；监控修复累计最多8次，30分钟内最多3次，反复失败会记录原因并继续其他配置。
数据加载进程异常可降到workers=0，模型、数据划分与回测协议保持原设置。

查询运行状态：

```bash
systemctl --user status trace-vq-batch67-20261107.service trace-vq-batch67-watchdog-20261107.service
cat artifacts/research/batch0607_20261107/STATUS.md
```

`FINISHED`防止已完成队列在登录/重启时重复训练；`PAUSED`阻止继续启动工作。
训练实现的来源哈希变化会阻止自动续跑，需先核实，避免在同一实验中混用实现。
日志、检查点、缓存、预测和逐日回测均由现有.gitignore排除；公开文档仅保留聚合结果。
