# TRACE-VQ Research

这个仓库同步当前研究目录的源码、配置、实验说明和汇总指标，便于阅读模型实现、
检查实验设置，以及用 ChatGPT/Claude 分析结果。

从 [实验索引](experiments/README.md)、[指标口径](experiments/protocol.md) 和
[分析指南](docs/analysis-guide.md) 开始。主要结果在：

- [64 条去重单 seed 实验结果](experiments/single_seed_all.csv)。
- [12 组五 seed 均值与标准差](experiments/multiseed_mean_std.csv)。
- [60 行每 seed 指标](experiments/multiseed_per_seed.csv)。
- [变体与代码对应](experiments/variants.csv)、[实际实验配置索引](experiments/config_index.csv)。

本地 `artifacts/`、原 `results/`、数据、权重、逐日/逐股票明细、日志及缓存保留在磁盘，
由 `.gitignore` 排除。可公开的汇总统一放在 `experiments/`，包含较差结果和负结果。
配置中的本机绝对路径已转换为相对引用，数值设置与结果保留。
完整跟踪范围见 [Git 同步说明](docs/github-sync.md)。
新克隆可以阅读代码与汇总；训练、推理、归档一致性测试和重建回测还需要本地数据及权重。

## Original implementation and usage

Official implementation of TRACE-VQ, a vector-quantized framework for cross-sectional stock prediction with temporal-aware latent modeling and shared-routed experts.

This standalone project retains the temporal-attention Stage 1 encoder, one
512-entry / 128-dimensional codebook, and the shared-routed MoE with a
quantization-confidence adapter. Both stages use `train.seed` (default 0);
the imported historical Stage 1 checkpoints use seed 42.
Architecture, loss functions, and train/validation/test periods follow the
source revision recorded in [migration record](docs/migration/MIGRATION.md).

The local workspace also contains the historical full-model and ablation
artifacts imported from HVQ-Stock. See [experiment usage and verification](docs/experiments.md).

| Configuration | HVQ experiment | Stage 1 | Confidence adapter | Shared expert |
| --- | --- | --- | --- | --- |
| `experiment=full` | 034, TRACE-VQ | Temporal attention | Yes | Yes |
| `experiment=wo_ta` | 025, w/o TA | Original GRU | Yes | Yes |
| `experiment=wo_ca` | 037, w/o CA | Temporal attention | No | Yes |
| `experiment=wo_sr` | 038, w/o SR | Temporal attention | Yes | No |

Each ablation also has an independent file under [ablation/](ablation/README.md):
`wo_ta.py`, `wo_ca.py` and `wo_sr.py`. These files expose the architecture
differences and reuse the shared model components and training/evaluation CLIs:

```bash
conda run -n prism-vq python ablation/wo_ca.py stage2 train.seed=0
conda run -n prism-vq python ablation/wo_ca.py evaluate --seed 0
```

Historical CSI300 runs include seeds 0–4, both required Stage 1 checkpoints,
predictions, backtest artifacts, training logs and accepted Phase 3 receipts.
Evaluate a migrated checkpoint without training or spelling out its filenames:

```bash
conda run -n prism-vq python -m trace_vq.cli.evaluate --experiment full --seed 0
conda run -n prism-vq python -m trace_vq.cli.evaluate --experiment wo_ta --seed 0
conda run -n prism-vq python -m trace_vq.cli.evaluate --experiment wo_ca --seed 0
conda run -n prism-vq python -m trace_vq.cli.evaluate --experiment wo_sr --seed 0
```

The experiment configurations reuse `../HVQ-Stock/dataset/processed`; large
datasets are not copied. New training with `experiment=...` writes under
`artifacts/new_runs/<variant>/`, separate from the imported historical runs.
Binary artifacts remain gitignored; a fresh clone needs an explicit import
from a populated HVQ-Stock workspace:

```bash
conda run -n prism-vq python scripts/import_hvq_experiments.py --source ../HVQ-Stock
```

For a code-only server deployment reusing existing data and Stage 1 weights,
see [remote usage](docs/remote.md).

Use `--device cuda` to evaluate with each run's recorded effective matrix
precision. Changing the GPU or using CPU can change scores near discrete
VQ/routing boundaries. Verification compares the original and migrated code
on the same device exactly, and reports historical score differences separately.

## Project organization

The implementation lives in the `trace_vq` package and is organized around the
model mechanisms. See [architecture and data flow](docs/architecture.md).

```text
trace_vq/
  models/
    temporal/       # Temporal-attention encoding and reconstruction auxiliaries
    quantization/   # Discrete codebook and quantization losses
    confidence/     # Zero-initialized correction from frozen VQ distortion
    experts/        # Always-on shared expert, sparse routed experts and fusion
    prediction/     # Temporal loading generator and factor return heads
    representation.py  # Stage 1 VQ representation model
    components/     # Reusable normalization, decomposition and position layers
  training/         # Two-stage Lightning lifecycle, losses/optimizers orchestration
  data/             # Canonical schema, market features, loaders and preparation
  evaluation/       # Inference, metrics, backtest and ensemble
  losses/           # Ranking objectives
  cli/              # Training/evaluation applications
  utils/            # Configuration and logging helpers
configs/
  config.yaml       # Model and training settings (compatible keys)
  data/             # CSI300/SP500 generation settings
  experiment/       # Full model and three trainable ablations
  hvq/              # Original configurations from the four frozen commits
```

The original `stage1.py`, `stage2.py`, `evaluate.py` and data/backtest commands
remain supported as thin compatibility entrypoints. Likewise, `module/`,
`trainer/`, `dataset/*.py` and `utils/` only forward old imports; add new code
under `trace_vq/`. This also preserves old dataset pickle module paths. Raw and
processed data continue to use the ignored `dataset/` storage directories.

## Environment

Use Python 3.11 and an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The dependency versions match the existing environment used for migration
validation (PyTorch 2.8.0+cu129, Lightning 2.6.4, Qlib 0.9.7). The remote
`quantbase` environment uses PyTorch 2.8.0+cu128. Both Stage 1 encoders and all
four Stage 2 variants passed short real-data GPU training and standalone
evaluation checks on its RTX 2080 Ti; see [remote usage](docs/remote.md).
A fresh installation has not been validated. The tests use CPU and unittest;
pytest is not required. Training defaults to GPU; for CPU add
`+train.accelerator=cpu train.device=cpu train.gpu_counts=1`.

Run the following commands from the project directory. Relative training data
paths resolve against this project. Checkpoints, predictions, Hydra logs and
default caches remain here. W&B defaults to **offline**; explicitly set
`train.wandb_mode=online` if you want remote logging.

## Data preparation

Raw data and processed datasets are not bundled. The migrated experiment
configurations reuse HVQ-Stock's processed data and imported checkpoints;
the generic commands below prepare new standalone datasets when needed.
Provide your own Qlib binary data and JKP factor CSV for generation.
Data-generation settings are in `configs/data/csi300.yaml` and
`configs/data/sp500.yaml`. To generate CSI300 splits:

```bash
python -m trace_vq.data.prepare --universe csi300 \
  --qlib-provider-uri /absolute/path/to/qlib/cn_data \
  --jkp-path /absolute/path/to/jkp.csv
```

Use `--universe sp500` with US data for SP500. Use `--data-handler-config` for a
custom YAML or `--output-dir` for a custom output root. Existing split files are
never overwritten by the generator. Raw factor CSV must contain `date`, `name`,
`ret`, `location`, `weighting` and `freq`, with 13 selected daily `vw_cap` factors
for `chn` or `usa`.

Default generated files are
`dataset/processed/CN/csi300_20_h10_{train,valid,test}.pkl` (or
`US/sp500_20_h10_{train,valid,test}.pkl`). The canonical layout is `[N,20,244]`:
158 stock features, 13 prior factors, 63 market features, and 10 future returns.
Market features remain in the data schema but are unused by this model.
Periods are 2009–2020 for training, 2021–2022 for validation, and 2023–2025 for
testing. If changing periods, update both data-generation YAML and model config.

CSI800 is constructed from the daily historical union of Qlib's CSI300 and
CSI500 membership, with duplicate stocks removed and original membership gaps
retained. `configs/data/csi800.yaml` keeps the CSI300 features, labels,
processors and periods. Generation caches annual raw features, fits one set
of normalization parameters on all training rows, and stores canonical FP32
samplers under `dataset/processed/CN/csi800_20_h10_{train,valid,test}.pkl`.
It leaves the Qlib provider and existing CSI300 datasets untouched.

The following command builds CSI800, trains Stage 1 with seed 42, trains
Stage 2 with seed 0 using the selected Stage 1 checkpoint, verifies standalone
test predictions, and runs the CSI800 portfolio backtest:

```bash
conda run -n prism-vq python scripts/run_csi800.py
```

The runner defaults to the sibling HVQ-Stock Chinese JKP CSV; `--jkp-path`
overrides it. Model and training settings match the CSI300 full model.
Outputs and phase logs are isolated under `artifacts/csi800/s1seed42_s2seed0/`,
with live status in `progress.json` and native/protocol metrics kept separately.
Its completed phases can be reused by running the same command again.

Backtests retain all instrument quotes so holdings remain priced after index
removals. Archived CSI300 portfolio metrics used historical pool-filtered
quotes; `--quote-universe historical` reproduces that convention, which can
leave removed holdings without quotes. Compare portfolios under the same
quote policy; the CSI800 runner explicitly uses `--quote-universe all` and
saves the backtest settings in `backtest/backtest_config.json`.

## Training

Stage 1 learns the codebook and temporal encoder:

```bash
python -m trace_vq.cli.stage1 data.universe=csi300 data.data_path=dataset/processed \
  artifact_root=artifacts/my_run
```

Stage 2 requires an explicitly chosen compatible Stage 1 checkpoint. Use the
best validation-loss checkpoint produced above (a `.ckpt` file other than
`last.ckpt`):

```bash
python -m trace_vq.cli.stage2 data.universe=csi300 data.data_path=dataset/processed \
  artifact_root=artifacts/my_run \
  "predictor.saved_model='/absolute/path/to/stage1-best.ckpt'"
```

A bare checkpoint filename resolves under the effective `train.save_dir`.
Keep the inner quotes when the actual checkpoint name contains `=` (Lightning
uses names such as `epoch=5-val_loss=0.5865.ckpt`); Hydra requires them.
Stage 2 keeps the encoder, codebook and RevIN frozen, trains the predictor, then
evaluates the best validation checkpoint on validation and test data. Predictions
and IC/RankIC metrics are written under `artifacts/my_run/res/`. A
`<run_name>.config.yaml` beside the Stage 2 checkpoint records the effective
configuration, including universe presets and explicit overrides. Keep this
file and both checkpoints for standalone evaluation. Old checkpoint filenames
have intentionally been removed from defaults.

## Standalone evaluation and backtest

Evaluation loads an existing Stage 2 checkpoint without training:

```bash
python -m trace_vq.cli.evaluate --checkpoint /absolute/path/to/stage2-best.ckpt \
  --stage1-checkpoint /absolute/path/to/stage1-best.ckpt \
  --config /absolute/path/to/run_name.config.yaml \
  --data-path /absolute/path/to/processed --universe csi300 --split test
```

Use `--split valid` for validation. Evaluation defaults to CPU and writes
`res/evaluation/{split}_predictions.pkl` and `{split}_metrics.csv`; `--device`
and `--output-dir` override those defaults. The Stage 1 checkpoint is needed
because the predictor constructor loads it before restoring Stage 2 state.
Always pass the saved effective configuration for runs with custom settings.

Optional portfolio backtesting requires the corresponding Qlib market data:

```bash
python -m trace_vq.evaluation.backtest --pred_path res/evaluation/test_predictions.pkl \
  --universe csi300 --qlib_provider_uri /absolute/path/to/qlib/cn_data
```

See `python -m trace_vq.evaluation.backtest --help` and
`python -m trace_vq.evaluation.ensemble --help` for portfolio and ensemble
options. Default backtest and ensemble outputs are under this project's `res/`.

## Verification

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The tests use synthetic data and temporary generated checkpoints; they require
no sibling repository, source experiment artifacts or network access. They
cover architecture, codebook freezing, adapter gradients, data schema,
checkpoint restoration and standalone validation/test evaluation. The ablation
tests cover both encoder variants, absent CA/SR modules, gradients and strict
checkpoint loading, including the independent `ablation/` model factories.
The CLI smoke exercises those files for training and evaluation.
For the imported historical artifacts, run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n prism-vq python scripts/verify_hvq_migration.py
conda run -n prism-vq python scripts/smoke_ablation_clis.py
```

The historical audit compares frozen HVQ source code, complete test predictions,
file checksums and formal result metrics. Reports are saved in
`docs/migration/`. It performs no new full training or portfolio backtest.
The completed import, exact source parity and cross-hardware limits are recorded
locally in `docs/migration/verification_summary.json`. Generated reports,
manifests, datasets, checkpoints and experiment results are excluded from Git;
the repository retains source code, configuration and project documentation.
