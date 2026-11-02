# TRACE-VQ 消融实验

每个消融有一个独立文件，直接展示结构差异，并提供模型构建、训练和评估入口。
公共模型组件继续由 `trace_vq` 提供。

| 文件 | 历史实验 | 时间编码器 | CA 置信度适配器 | SR 共享专家 |
| --- | --- | --- | --- | --- |
| [wo_ta.py](wo_ta.py) | 025 | GRU | 保留 | 保留 |
| [wo_ca.py](wo_ca.py) | 037 | 时间注意力 | 移除 | 保留 |
| [wo_sr.py](wo_sr.py) | 038 | 时间注意力 | 保留 | 移除 |

`wo_ta` 继续保留跨资产注意力；`wo_sr` 继续保留稀疏路由专家。
`_common.py` 复用现有模型与 CLI。训练配置来自 `configs/experiment/`，
新训练输出仍写入 `artifacts/new_runs/<消融名>/`。

在 TRACE-VQ 目录下使用 `prism-vq` 环境：

```bash
conda activate prism-vq

# 使用已迁移的 Stage 1 权重，分别训练三个 Stage 2 消融。
python ablation/wo_ta.py stage2 train.seed=0
python ablation/wo_ca.py stage2 train.seed=0
python ablation/wo_sr.py stage2 train.seed=0

# 评估各消融的历史 checkpoint。
python ablation/wo_ta.py evaluate --seed 0 --split test
python ablation/wo_ca.py evaluate --seed 0 --split test
python ablation/wo_sr.py evaluate --seed 0 --split test
```

训练参数和评估参数直接传给原入口，例如 `train.num_epochs=2`、
`data.data_path=/path/to/processed`、`--device cuda`、`--output-dir /path/to/output`。
运行 `python ablation/wo_ca.py --help` 查看入口说明，
`python ablation/wo_ca.py stage2 --help` 查看训练配置。
也支持模块形式，如 `python -m ablation.wo_ca stage2 train.seed=0`。
使用文件的绝对路径时，可以从其他工作目录启动。

w/o TA 使用 baseline 的 GRU Stage 1；w/o CA 和 w/o SR 复用 028 的
时间注意力 Stage 1。重新训练 GRU Stage 1 可以运行：

```bash
python ablation/wo_ta.py stage1
```

之后通过 `predictor.saved_model` 将新 Stage 1 权重传给 Stage 2。
Checkpoint 路径包含 `=` 时保留 Hydra 需要的内部引号：

```bash
python ablation/wo_ta.py stage2 "predictor.saved_model='/path/to/epoch=5.ckpt'"
```

评估自己新训练的模型时，显式提供 checkpoint 和当次训练的有效配置：

```bash
python ablation/wo_ca.py evaluate \
  --checkpoint /path/to/stage2.ckpt \
  --stage1-checkpoint /path/to/stage1.ckpt \
  --config /path/to/run.config.yaml \
  --data-path /path/to/processed
```

每个文件也提供 `build_model(config, T_max)`，可以在 Python 中直接构建该消融。
传入配置中的 `predictor.saved_model` 必须指向对应类型的 Stage 1 权重。
函数复制配置并应用文件内的 `MODEL_OVERRIDES`，不会修改调用者的配置；
模型参数名保持兼容，可严格加载历史 Stage 2 权重。

```python
from ablation.wo_ca import build_model

model = build_model(config, T_max=100)
```

数据继续复用 HVQ-Stock；迁移记录和一致性验证见
[实验说明](../docs/experiments.md)。
