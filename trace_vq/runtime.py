"""Project-local defaults for entrypoint logs and caches."""

import os
from pathlib import Path


def configure_runtime():
    root = Path(__file__).resolve().parents[1]
    os.environ["TRACE_VQ_ROOT"] = str(root)
    for key, relative in {
        "MPLCONFIGDIR": ".cache/matplotlib",
        "XDG_CACHE_HOME": ".cache",
        "WANDB_CACHE_DIR": ".cache/wandb",
        "WANDB_CONFIG_DIR": ".cache/wandb-config",
        "CUDA_CACHE_PATH": ".cache/cuda",
        "TRITON_CACHE_DIR": ".cache/triton",
    }.items():
        os.environ.setdefault(key, str(root / relative))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
