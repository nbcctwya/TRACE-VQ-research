# 当前研究目录的 Git 同步范围

直接在 `TRACE-VQ-research` 这个仓库中维护源码、配置与实验汇总。
`artifacts/` 下的原始实验与本机资源继续保留，Git 只同步整理后的公开视图。

## 纳入 Git 的内容

- `trace_vq/`：当前模型、训练、数据处理和评估源码。
- `configs/`、`research/run_*.py`、`scripts/`、`tests/` 及兼容入口。
- `experiments/`：实验描述、计划、变体定义、聚合结果和实际配置快照。
- 项目 README、架构与分析说明、`AGENTS.md` 和 `CLAUDE.md`。

实验索引见 [experiments/README.md](../experiments/README.md)。主要表为
64 条去重后的单 seed 股票池/配置结果、12 组五 seed 均值/标准差和 60 行每 seed
指标，另保留历史消融、训练随机性与诊断汇总。原始表中的数值保留，汇总中的
本机绝对路径已改为相对引用。每 seed 一行属于稳定性分析所需的汇总，
没有逐日收益、逐股票预测或持仓记录。

## 只在本地保留的内容

`.gitignore` 屏蔽：

- 数据集、数组、模型权重、序列化预测与持仓。
- `artifacts/`、原 `res/` 和 `results/`、运行输出、日志、缓存、W&B、压缩包和生成图片。
- `research/baseline/`、`research/validation.json` 等本机复制、校验和部署记录。
- `.env`、本地配置、凭据目录、私钥、服务账户文件和编辑器状态。

`dataset/*.py` 兼容代码仍进入 Git；`experiments/` 显式允许 Markdown、CSV、JSON
与 YAML。汇总目录中的常见逐日结果和训练曲线文件名仍会被忽略。
忽略规则按文件路径和类型工作，不能识别任意文件内嵌的秘密；新增源码或汇总时
仍需确保没有写入真实凭据或本机私有配置。

本机基线证据保持原字节，停止跟踪不会删除磁盘文件。原研究基线 tag 保留在本地
忽略的旧 Git 元数据备份中，当前仓库不再引用它。
`research/verify_baseline.py` 继续使用本地证据校验原 TRACE-VQ。新的 Git 克隆
不会自动带上这些本机记录，归档验证和训练也需要外部数据/权重。

## 维护汇总

新实验仍输出到独立的 `artifacts/research/<run_name>/`。完成实验后，将设置、
每配置/seed 指标和说明整理进 `experiments/`，保留较差结果、负结果和复用基线标记。
实际配置只移植路径，不修改数值参数。新增批次时更新 `experiments/index.csv`
与实验 README，区分独立 seed 的均值和预测集成后的回测。

用以下只读命令检查当前同步范围：

```bash
git status --short
git ls-files --cached --others --exclude-standard
git check-ignore -v artifacts/research/example/train.log
git check-ignore -v .env
```

`.gitignore` 只约束后续文件跟踪，不会删除旧提交或 tag 中已经出现的内容。
已按用户请求单独清除本仓库的旧历史：当前仓库从 `main` 的一个初始提交开始，
不包含旧仓库的分支、tag、远程跟踪 refs 或 reflog。旧 Git 元数据仅保留在
`artifacts/research/history_reset_*/git-before/`，供本机恢复与基线核查，整个
`artifacts/` 继续被忽略。原始 `../TRACE-VQ` 的 Git 元数据和数据不变。

用户已创建独立科研仓库，`origin` 的拉取和推送地址均指向：

```text
git@github.com:nbcctwya/TRACE-VQ-research.git
```

默认分支为 `main`。首次同步的范围为源码、配置和 `experiments/` 中的汇总，
不推送旧历史备份或本机产物。后续同步继续遵循相同忽略规则，并按用户请求推送。
原始 `TRACE-VQ` 仓库不作为本研究目录的推送目标。
