"""Locate the imported full model and ablation runs without long filenames."""

import json
from pathlib import Path

EXPERIMENT_IDS = {"full": "034", "wo_ta": "025", "wo_ca": "037", "wo_sr": "038"}
ROOT = Path(__file__).resolve().parents[1]


def migrated_run(experiment, seed=0):
    if experiment not in EXPERIMENT_IDS or seed not in range(5):
        raise ValueError(f"Unsupported historical run: {experiment}/seed{seed}")
    path = ROOT / "docs/migration/experiments_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Import the HVQ artifacts first: {path}")
    manifest = json.loads(path.read_text())
    info = manifest["experiments"][EXPERIMENT_IDS[experiment]]
    run = info["runs"][str(seed)]
    return {
        "checkpoint": ROOT / run["checkpoint"],
        "stage1_checkpoint": ROOT / info["stage1_checkpoint"],
        "config": ROOT / run["config"],
        "prediction": ROOT / run["prediction"],
        "universe": "csi300",
    }
