"""w/o SR (HVQ 038): omit the shared expert; retain TA, CA and routed experts.

Stage 1 reuses experiment 028. Sparse top-k routing remains active in Stage 2;
only the always-on shared expert is removed.
Run: python ablation/wo_sr.py stage2 train.seed=0
     python ablation/wo_sr.py evaluate --seed 0
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation._common import build_predictor, run_experiment

EXPERIMENT = "wo_sr"
HVQ_EXPERIMENT = "038"
MODEL_OVERRIDES = {
    "encoder_type": "temporal-attention",
    "quantization_confidence_adapter": True,
    "shared_expert": False,
}


def build_model(config, T_max):
    """Build w/o SR without creating shared-expert parameters."""
    return build_predictor(config, T_max, **MODEL_OVERRIDES)


def main(argv=None):
    return run_experiment(EXPERIMENT, __doc__, argv)


if __name__ == "__main__":
    main()
