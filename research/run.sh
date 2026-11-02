#!/usr/bin/env bash
set -euo pipefail

research_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
research_python=${TRACE_VQ_RESEARCH_PYTHON:-${HOME}/anaconda3/envs/prism-vq/bin/python}
if [[ ! -x "$research_python" ]]; then
    printf 'Python interpreter unavailable: %s\n' "$research_python" >&2
    exit 1
fi
if [[ ! -d "$research_root/.git" || -L "$research_root/.git" ]]; then
    printf 'An independent Git directory is required.\n' >&2
    exit 1
fi

cd -- "$research_root"
export TRACE_VQ_ROOT="$research_root"
export TRACE_VQ_DATA_ROOT="$research_root/dataset/processed"
export TRACE_VQ_STAGE1_TA_CHECKPOINT="$research_root/artifacts/028/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=5-val_loss=0.5865.ckpt"
export TRACE_VQ_STAGE1_GRU_CHECKPOINT="$research_root/artifacts/baseline/run/checkpoints/infucsi300_h128_VQK512_C128_emb128_dl2p10_s42-epoch=7-val_loss=0.5712.ckpt"
export PYTHONPATH="$research_root"
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="$research_root/.cache/matplotlib"
export XDG_CACHE_HOME="$research_root/.cache"
export CUDA_CACHE_PATH="$research_root/.cache/cuda"
export TRITON_CACHE_DIR="$research_root/.cache/triton"
export TORCH_HOME="$research_root/.cache/torch"
export WANDB_MODE=offline
export WANDB_DIR="$research_root/artifacts/research/wandb"
export WANDB_CACHE_DIR="$research_root/.cache/wandb"
export WANDB_CONFIG_DIR="$research_root/.cache/wandb-config"
export TMPDIR="$research_root/.cache/tmp"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
unset STAGE2_CONFIG_SNAPSHOT_DIR
mkdir -p -- "$TMPDIR" "$WANDB_DIR" "$research_root/research/logs"
exec "$research_python" -B "$@"
