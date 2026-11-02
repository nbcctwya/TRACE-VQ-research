# TRACE-VQ and historical ablations

当前研究实验（含各轮筛选、多 seed、SP500 与效果诊断）的公开汇总见
[experiments/README.md](../experiments/README.md)，指标约定见
[experiments/protocol.md](../experiments/protocol.md)。下文保留历史模型的使用与溯源说明；
其中引用的检查点、预测和完整归档只在本地保留，不随 Git 同步。

The four imported experiments use the frozen commits below. Each has CSI300
seed 0 from Phase 2 and seeds 1–4 from accepted Phase 3 runs. Historical runs
are under `artifacts/<HVQ ID>/`; their original predictions, checkpoints,
logs, receipts and backtests are preserved. The importer reads HVQ-Stock
without switching branches or modifying its files.

| Variant | HVQ ID | Frozen commit | Encoder | CA | SR |
| --- | --- | --- | --- | --- | --- |
| `full` / TRACE-VQ | 034 | `2b1f576e4a7d46b098b6278b2fe76672f5881a23` | Temporal attention | Yes | Yes |
| `wo_ta` / w/o TA | 025 | `5650e9fc9f3cf0ba89ac5e08b0965cc041ef4a2f` | Original GRU | Yes | Yes |
| `wo_ca` / w/o CA | 037 | `086d98fff392567c5a24751933a1fcd51926b401` | Temporal attention | No | Yes |
| `wo_sr` / w/o SR | 038 | `0a968f77e717c52cf5c48d997c77c104f7c1563d` | Temporal attention | Yes | No |

w/o SR removes the always-on shared expert; sparse routed experts remain.
w/o TA restores the original GRU temporal summary and retains cross-asset
attention. The two Stage 1 checkpoints are copied once:

- 025 reuses the corrected baseline GRU checkpoint, seed 42,
  `artifacts/baseline/run/checkpoints/*epoch=7-val_loss=0.5712.ckpt`.
- 034/037/038 reuse experiment 028's temporal-attention checkpoint, seed 42,
  `artifacts/028/run/checkpoints/*epoch=5-val_loss=0.5865.ckpt`.

The temporal-attention Stage 1 was trained on an RTX 5060 Ti. Its original
W&B run (`xbhdtgsh`) started on 2026-09-10 at 02:38:07 and finished at
03:33:25, UTC+8, about 55 minutes 19 seconds including training lifecycle
overhead. The selected checkpoint records epoch 5 and global step 17,508.
`migration/stage1_provenance.json` links these facts to the original logs,
W&B environment record, checkpoint hash and source experiment 028.

## Configurations and files

- `configs/hvq/{025,034,037,038}.yaml`: original bytes of each frozen
  experiment configuration, including its historical checkpoint placeholders.
  These files document the source; use the experiment groups for new training.
- `configs/experiment/{full,wo_ta,wo_ca,wo_sr}.yaml`: runnable Hydra groups
  selecting the proper encoder, adapter/shared-expert switches and Stage 1.
- `ablation/{wo_ta,wo_ca,wo_sr}.py`: independent ablation definitions, model
  factories and CLI entrypoints; public model components are shared.
- A `.config.yaml` beside each imported Stage 2 checkpoint: effective per-seed
  configuration with current workspace paths. Seeds 1–4 come from their
  `training_evidence.json`; seed 0 is reconstructed from the frozen config,
  market preset, Phase 2 marker and recorded checkpoint provenance.
- `docs/migration/source_records/`: original four experiment records.
- `docs/migration/experiments_manifest.json`: pinned source revisions, all
  imported artifact hashes, per-seed file paths and source/dataset integrity.
- `artifacts/_reference/<ID>/`: exact frozen implementation files used by an
  isolated process for comparison; the active implementations live in `trace_vq`.
- `artifacts/_hvq_results/`: immutable snapshot of source formal results.
- `results/`: source metrics/tables filtered to the four experiment IDs, plus
  their original ensemble curves. Model IDs remain unchanged; the names are
  listed in `results/metadata/experiment_names.json`.

The original archive manifests and log paths still describe the original
execution. The migration manifest and generated `.config.yaml` files identify
the current copies; historical JSON/log files are not rewritten.
The importer resolves checkpoint and Phase 3 archive paths within the selected
`--source` checkout, including when the original absolute path describes a
different machine or repository location.

## Reuse data

No dataset is copied or regenerated. Training groups point at
`../HVQ-Stock/dataset/processed`, resolved against TRACE-VQ's root. Override
`data.data_path=/absolute/path/to/processed` for training or `--data-path` for
evaluation. Imported evaluation configs also accept `TRACE_VQ_DATA_ROOT`.
Training groups also accept `TRACE_VQ_DATA_ROOT`; `TRACE_VQ_STAGE1_TA_CHECKPOINT`
and `TRACE_VQ_STAGE1_GRU_CHECKPOINT` override their corresponding Stage 1 paths
without changing model hyperparameters. See [remote usage](remote.md).
Legacy `dataset.market.CanonicalSampler` pickle globals resolve to TRACE-VQ's
canonical implementation.

The historical artifacts are CSI300 only. SP500's original configuration is
retained in the source presets, but no SP500 results or compatible Stage 1
weights are claimed for these four imported runs.

## Evaluate an imported checkpoint

From TRACE-VQ, using the `prism-vq` environment:

```bash
conda run -n prism-vq python -m trace_vq.cli.evaluate --experiment full --seed 0 --split test
conda run -n prism-vq python -m trace_vq.cli.evaluate --experiment wo_ta --seed 1 --split valid
conda run -n prism-vq python -m trace_vq.cli.evaluate --experiment wo_ca --seed 2 --split test
conda run -n prism-vq python -m trace_vq.cli.evaluate --experiment wo_sr --seed 4 --split test
```

The selector finds the matching Stage 1, Stage 2 and effective config.
Evaluation defaults to CPU and writes
`res/evaluation/<variant>/seed<seed>/<split>_{predictions.pkl,metrics.csv}`.
Explicit checkpoint/config/data/output arguments remain supported.
The explicit-config evaluation path also supports `predictor.use_prior=false`;
the predictor still accepts the canonical prior input and omits its return term,
matching the training forward call.

For CUDA evaluation, add `--device cuda`.
The saved configuration selects each historical run's effective matrix
precision: seed 0 uses `high`; seeds 1–4 use full FP32 (`highest`), matching
the accepted RTX 2080 Ti runs. The precision diagnostic in
`migration/gpu_precision_diagnostic.json` confirms the choice against original
predictions. `--matmul-precision` explicitly overrides it when needed.
CPU evaluation is supported. Changing the device or GPU generation can change
VQ selection and top-k routing at numerical boundaries. In particular, the
accepted Phase 3 artifacts used an RTX 2080 Ti with CUDA 12.8; validation here
uses an RTX 5060 Ti with CUDA 12.9. Every current source/migrated prediction is
compared exactly; historical score differences are recorded separately at
the original `atol=rtol=1e-5` tolerance. The archived predictions remain unchanged.

## Train the full model or an ablation

The independent files [wo_ta.py](../ablation/wo_ta.py),
[wo_ca.py](../ablation/wo_ca.py) and [wo_sr.py](../ablation/wo_sr.py) make each
ablation's structural switches explicit. Each exposes `build_model(config,
T_max)` and `stage1`, `stage2`, `evaluate` commands using the established
configurations and shared implementation. See [ablation usage](../ablation/README.md).

The group selects the historical Stage 1 weights, so Stage 2 can run directly:

```bash
conda run -n prism-vq python -m trace_vq.cli.stage2 experiment=full train.seed=0
conda run -n prism-vq python -m trace_vq.cli.stage2 experiment=wo_ta train.seed=0
conda run -n prism-vq python -m trace_vq.cli.stage2 experiment=wo_ca train.seed=0
conda run -n prism-vq python -m trace_vq.cli.stage2 experiment=wo_sr train.seed=0
```

New outputs go to `artifacts/new_runs/<variant>/checkpoints` and `res`.
An `artifact_root` override can select another output root. The full model and
w/o TA also support Stage 1 training through the corresponding group:

```bash
conda run -n prism-vq python -m trace_vq.cli.stage1 experiment=full
conda run -n prism-vq python -m trace_vq.cli.stage1 experiment=wo_ta
```

After training a new Stage 1, explicitly pass its checkpoint to Stage 2 using
`predictor.saved_model`; the group otherwise keeps selecting the imported
historical Stage 1. Stage 1 follows `train.seed`, as does Stage 2.

## Verify migration

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n prism-vq python -m unittest discover -s tests -v
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n prism-vq python scripts/verify_hvq_migration.py
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 conda run -n prism-vq python scripts/verify_shared_hvq_data.py
conda run -n prism-vq python scripts/smoke_ablation_clis.py
```

The historical audit checks every copied artifact hash, source/config
provenance, both complete Stage 1 checkpoints, all 20 Stage 2 checkpoints,
fixed-seed initialization, six full daily validation/test cross-sections per
run, and seed 0 loss/gradients/one optimizer step for every variant. Source
and migrated computations use the same device/environment and require exact
tensor equality, including all 217,909 test scores for each of the 20 models.
The audit uses CUDA when available and also supports `--sample-only --device cpu`
for exact CPU source parity. Copied historical predictions and artifacts must
match their source hashes; test labels and indexes must match exactly.

Cross-hardware historical recomputation is a separate report field:
`historical_reprediction_strict_passed`. It compares scores at
`atol=rtol=1e-5` and records the affected row count, maximum, mean, median and
99th-percentile error for every run. A migration pass requires exact current
source parity, intact archives and consistent metrics; it does not imply all
cross-hardware scores meet that tolerance. Add `--require-historical-close`
to make those historical differences fail the command as well. The initial
CPU and GPU attempts are retained in `migration/cpu_historical_comparison.json`
and `migration/gpu_historical_attempt.json`.

The data audit hashes every actual sampled window of train/valid/test, including
the independent market channel, and compares against the fingerprints of all
16 accepted Phase 3 training runs. It does not merely inspect a few rows or
hash the sampler's raw array.

The CLI smoke trains both encoders and all four predictors for one epoch on
temporary synthetic data. Standalone evaluation must reproduce the training
entrypoint predictions exactly. It also exercises the imported-run selector
for all four variants. The three ablations use their independent files in
`ablation/` for Stage 2 training, explicit-checkpoint evaluation and historical
run selection; w/o TA also uses its file for Stage 1 training.

Machine-readable reports:

- `docs/migration/consistency_report.json`
- `docs/migration/data_consistency_report.json`
- `docs/migration/cli_smoke_report.json`
- `docs/migration/training_config_report.json`
- `docs/migration/results_consistency_report.json`
- `docs/migration/gpu_cli_report.json`
- `docs/migration/integrity_report.json`
- `docs/migration/verification_summary.json`
- `docs/migration/recheck_report.json`
- `docs/migration/stage1_provenance.json`
- `docs/migration/ablation_entrypoint_report.json`
- `docs/migration/ablation_gpu_cli_report.json`
- `results/diagnostics/migration_consistency.json`

Completed validation in `prism-vq`: 101 tests pass; both encoders and all four
Stage 2 CLIs train and evaluate on synthetic data. The frozen original and
migrated implementations give exactly equal full-test predictions for every
run. Each has 217,909 stock/date rows covering 727 trading days; the 20 runs
give 4,358,180 scores in total. Initialization,
sampled Stage 1/Stage 2 outputs, and each variant's seed 0 loss, gradients and
one optimizer step also match exactly. All copied artifacts, original configs,
formal metrics and shared-data fingerprints pass verification. Real GPU
evaluation CLIs for `full/seed0` and `wo_ta/seed1` reproduce the corresponding
current source outputs exactly.

The independent ablation entrypoints retain the complete effective training
configuration in both direct-file and module execution, including when the
file is started from another working directory. Real GPU evaluations through
all three files (seed 0) match the previously verified source output exactly:
653,727 stock/date predictions in total. Their model factories preserve the
caller's config and pass gradient, freezing and strict restoration checks
against the shared predictor implementation.

The code-only deployment on the remote RTX 2080 Ti server also passed 101 tests. It reuses
the server's existing data and both Stage 1 checkpoints via environment paths.
CUDA smoke checks trained both Stage 1 encoders and all four Stage 2 variants
for one epoch on two complete real trading days per split. Each variant's 600
test predictions matched its standalone evaluation exactly. See
[remote usage and validation](remote.md); no historical artifacts were uploaded.

The subsequent review corrected historical absolute-path resolution in the
importer and an inherited evaluation call that failed when prior factors were
disabled. Tests cover relocation even when another copy remains at the old
path, and exact CLI predictions with prior factors enabled and disabled on
both validation and test splits. The complete GPU audit was rerun from scratch
and passed in one command; `migration/recheck_report.json` records that run.

Historical cross-GPU recomputation is not fully identical. All four seed 0
runs are exact. At the original `atol=rtol=1e-5` tolerance, w/o TA seeds 1–4
have 170 affected rows in total, and w/o SR seeds 1 and 3 have one each;
the full model and w/o CA have none. Across all 4,358,180 scores, 172 exceed
the tolerance; the maximum absolute difference is 0.0842234. The maximum
per-run change in formal IC is 0.00001646 and in RankIC is 0.00002077.
The exact same deviations occur with the original code on the current GPU.
`results/diagnostics/cross_hardware_score_differences.csv` identifies each row,
and `cross_hardware_metrics.csv` records the metric impact. No archived score,
metric or tolerance was changed to make these comparisons pass.

Native training/evaluation ICIR retains the inherited population standard
deviation (`ddof=0`). The imported formal `results/` use Baseline Results
Protocol v1.0 (`ddof=1`, log-return portfolio metrics, 252 trading days).
`trace_vq/evaluation/protocol_metrics.py` preserves that source protocol;
the audit recomputes seed ranking metrics, seed aggregates, ensemble ranking
metrics, and portfolio metrics from the copied ensemble curves. Historical
seed portfolio numbers are imported unchanged; no new portfolio backtest or
full training run is performed.
Formal prediction loading casts scores and labels to float64 before averaging
seeds, matching HVQ's loader and avoiding float32 ensemble-rounding differences.

Other historical numerical conventions are retained, including the source
label horizon definitions and the quantizer's effective fixed `decay=0.99`
despite the YAML `decay=0.95` field. Changing these would alter the experiment
and is outside this migration.
