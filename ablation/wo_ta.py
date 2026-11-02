"""w/o TA (HVQ 025): use the original GRU; retain CA and shared experts.

Stage 1 uses the imported baseline GRU checkpoint, rather than experiment 028.
Run: python ablation/wo_ta.py stage2 train.seed=0
     python ablation/wo_ta.py evaluate --seed 0
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation._common import build_predictor, run_experiment

EXPERIMENT = "wo_ta"
HVQ_EXPERIMENT = "025"
MODEL_OVERRIDES = {
    "encoder_type": "gru",
    "quantization_confidence_adapter": True,
    "shared_expert": True,
}


def build_model(config, T_max):
    """Build w/o TA using a config with a compatible GRU Stage 1 checkpoint."""
    return build_predictor(config, T_max, **MODEL_OVERRIDES)


def main(argv=None):
    return run_experiment(EXPERIMENT, __doc__, argv)


if __name__ == "__main__":
    main()
