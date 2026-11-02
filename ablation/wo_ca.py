"""w/o CA (HVQ 037): omit the confidence adapter; retain TA and shared experts.

Stage 1 reuses experiment 028. Stage 2 consumes the detached quantized latent
without the quantization-confidence residual.
Run: python ablation/wo_ca.py stage2 train.seed=0
     python ablation/wo_ca.py evaluate --seed 0
"""

if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ablation._common import build_predictor, run_experiment

EXPERIMENT = "wo_ca"
HVQ_EXPERIMENT = "037"
MODEL_OVERRIDES = {
    "encoder_type": "temporal-attention",
    "quantization_confidence_adapter": False,
    "shared_expert": True,
}


def build_model(config, T_max):
    """Build w/o CA without adding adapter parameters to the state dict."""
    return build_predictor(config, T_max, **MODEL_OVERRIDES)


def main(argv=None):
    return run_experiment(EXPERIMENT, __doc__, argv)


if __name__ == "__main__":
    main()
