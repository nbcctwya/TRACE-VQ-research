"""Shared model construction and CLI dispatch for the ablation files."""

import argparse
import copy
import importlib
import sys


def build_predictor(config, T_max, *, encoder_type,
                    quantization_confidence_adapter, shared_expert):
    """Apply the ablation to a private config while retaining checkpoint keys."""
    from omegaconf import OmegaConf
    from trace_vq.training.prediction import GenerateReturn

    cfg = (OmegaConf.to_container(config, resolve=True)
           if OmegaConf.is_config(config) else copy.deepcopy(config))
    cfg["vqvae"]["encoder"]["type"] = encoder_type
    cfg["predictor"]["quantization_confidence_adapter"] = quantization_confidence_adapter
    cfg["predictor"]["shared_expert"] = shared_expert
    return GenerateReturn(cfg, T_max=T_max)


def run_experiment(experiment, description, argv=None):
    """Reuse the established CLIs with this file's experiment selected."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("command", choices=["stage1", "stage2", "evaluate"])
    parser.add_argument("arguments", nargs=argparse.REMAINDER,
                        help="Arguments forwarded to the training/evaluation CLI")
    args = parser.parse_args(argv)
    forwarded = args.arguments
    if args.command == "evaluate":
        if any(arg == "--experiment" or arg.startswith("--experiment=")
               for arg in forwarded):
            parser.error(f"This entrypoint already selects {experiment}")
        # Explicit checkpoints use their saved effective config; otherwise
        # select the imported historical run for this ablation.
        if not any(arg == "--checkpoint" or arg.startswith("--checkpoint=")
                   for arg in forwarded):
            forwarded = ["--experiment", experiment, *forwarded]
    else:
        if any(arg.lstrip("+").split("=", 1)[0] in {"experiment", "~experiment"}
               for arg in forwarded):
            parser.error(f"This entrypoint already selects experiment={experiment}")
        forwarded = [f"experiment={experiment}", *forwarded]

    previous_argv = sys.argv
    try:
        sys.argv = [previous_argv[0], *forwarded]
        implementation = importlib.import_module(f"trace_vq.cli.{args.command}")
        return implementation.main()
    finally:
        sys.argv = previous_argv
