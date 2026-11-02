# TRACE-VQ remote usage

The code-only deployment uses a server-local project directory. It uses the
existing `quantbase` environment, the server's CSI300 processed data, and its
existing temporal-attention and GRU Stage 1 checkpoints. No local datasets,
historical experiment artifacts, caches or Git database are uploaded.

The deployment passed 101 unit tests and the real-data CUDA smoke on an RTX
2080 Ti with PyTorch 2.8.0+cu128. The smoke used 551 train, 600 validation and
600 test stock/date rows from two complete trading days per split. Both
Stage 1 encoders and all four Stage 2 variants completed one epoch; all 2,400
test scores matched standalone evaluation exactly. Original data, Stage 1
weights and the old repository were preserved. Server reports are in
`.deployment/verification.json` and `artifacts/verification/remote_smoke_report.json`.

Log in and load the server-specific paths:

```bash
ssh YOUR_SERVER
cd "$REMOTE_PROJECT_ROOT"
source .deployment/environment.sh
```

The environment file activates `quantbase` and sets `TRACE_VQ_DATA_ROOT`,
`TRACE_VQ_STAGE1_TA_CHECKPOINT` and `TRACE_VQ_STAGE1_GRU_CHECKPOINT` to existing
server files. Experiment groups accept these environment variables; their
local defaults retain the original migration paths.

Train Stage 2 with the existing Stage 1 weights:

```bash
python stage2.py experiment=full train.seed=0
python ablation/wo_ta.py stage2 train.seed=0
python ablation/wo_ca.py stage2 train.seed=0
python ablation/wo_sr.py stage2 train.seed=0
```

Each command is an independent training job. New results go under
`artifacts/new_runs/<variant>/`. Stage 1 training is also available:

```bash
python stage1.py experiment=full
python ablation/wo_ta.py stage1
```

To use a newly trained Stage 1 in Stage 2, explicitly override
`predictor.saved_model`; the environment otherwise selects the existing
historical Stage 1. Keep Hydra's inner quotes when a checkpoint name contains
`=`:

```bash
python stage2.py experiment=full "predictor.saved_model='/path/to/epoch=5.ckpt'"
```

Evaluate a newly trained Stage 2 with its saved effective configuration:

```bash
python evaluate.py \
  --checkpoint /path/to/stage2.ckpt \
  --stage1-checkpoint /path/to/stage1.ckpt \
  --config /path/to/run.config.yaml \
  --data-path "$TRACE_VQ_DATA_ROOT" --device cuda
```

The historical `evaluate --experiment ...` selector requires the historical
Stage 2 artifacts and manifest; use explicit paths on this code-only deployment.

The short GPU check samples two complete trading days from each existing
train/validation/test pickle. Stock, prior, market and return windows are
retained, with the same FP32 conversion as the production loader. It trains
both Stage 1 encoders and all four Stage 2 variants for one epoch, then checks
exact standalone evaluation predictions. Small sampled inputs and newly
generated checkpoints are temporary and removed afterwards. This checks the
pipeline; it is not a full training run or historical result reproduction.

```bash
python scripts/smoke_ablation_clis.py \
  --device cuda --data-path "$TRACE_VQ_DATA_ROOT" --days 2 --skip-imported \
  --report artifacts/verification/remote_smoke_report.json
```

Unit tests run without historical artifacts:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -v
```
