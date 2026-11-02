"""Evaluate a trained TRACE-VQ predictor on one canonical data split."""

import argparse
import pickle
from pathlib import Path

from trace_vq.runtime import configure_runtime
configure_runtime()

import pandas as pd
import torch
from omegaconf import OmegaConf

from trace_vq.data.loaders import init_data_loader
from trace_vq.data.schema import dataset_basename
from trace_vq.training.prediction import GenerateReturn
from trace_vq.utils import get_root_dir, run_inference, seed_everything
from trace_vq.experiments import EXPERIMENT_IDS, migrated_run


def main():
    root = Path(get_root_dir())
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(EXPERIMENT_IDS), help="Imported HVQ model/ablation")
    parser.add_argument("--seed", type=int, choices=range(5), default=0, help="Imported seed (0-4)")
    parser.add_argument("--checkpoint", type=Path, help="Stage 2 checkpoint")
    parser.add_argument("--stage1-checkpoint", type=Path)
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--universe", choices=["csi300", "csi800", "sp500"])
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--matmul-precision", choices=["high", "highest"],
                        help="Override the checkpoint's recorded matrix execution precision")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.experiment:
        run = migrated_run(args.experiment, args.seed)
        args.checkpoint = args.checkpoint or run["checkpoint"]
        args.stage1_checkpoint = args.stage1_checkpoint or run["stage1_checkpoint"]
        args.config = args.config or run["config"]
        if args.universe and args.universe != run["universe"]:
            parser.error("Imported historical experiments use CSI300; select explicit checkpoints for another market")
        args.universe = run["universe"]
        args.output_dir = args.output_dir or root / "res/evaluation" / args.experiment / f"seed{args.seed}"
    else:
        for name in ("checkpoint", "stage1_checkpoint", "data_path"):
            if getattr(args, name) is None:
                parser.error(f"--{name.replace('_', '-')} is required without --experiment")
        args.config = args.config or root / "configs/config.yaml"
        args.output_dir = args.output_dir or root / "res/evaluation"
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)
    cfg = OmegaConf.load(args.config)
    args.universe = args.universe or cfg.data.universe
    cfg.data.universe = args.universe
    # Training's saved, resolved .hydra/config.yaml is authoritative. Only the
    # bundled default needs the per-universe preset applied here.
    if args.config.resolve() == (root / "configs/config.yaml").resolve():
        preset = cfg.get("stage2_presets", {}).get(args.universe, {})
        cfg = OmegaConf.merge(cfg, preset)
    cfg.predictor.saved_model = str(args.stage1_checkpoint.expanduser().resolve())
    config = OmegaConf.to_container(cfg, resolve=True)
    torch.set_float32_matmul_precision(
        args.matmul_precision or cfg.get("inference", {}).get("matmul_precision", "high")
    )
    region = {"csi300": "CN", "csi800": "CN", "sp500": "US"}[args.universe]
    base = dataset_basename(args.universe, cfg.data.window_size, cfg.vqvae.predictor.pred_len)
    data_path = args.data_path or root / Path(cfg.data.data_path).expanduser()
    data_file = data_path.expanduser() / region / f"{base}_{args.split}.pkl"
    for path in (args.checkpoint.expanduser(), Path(cfg.predictor.saved_model), data_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    with data_file.open("rb") as stream:
        dataset = pickle.load(stream)
    loader, _ = init_data_loader(dataset, shuffle=False, num_workers=0)
    seed_everything(cfg.train.seed)
    model = GenerateReturn.load_from_checkpoint(
        args.checkpoint.expanduser(), config=config, T_max=1, map_location="cpu", strict=True
    )
    prediction, _, metrics = run_inference(model, loader, config, device=args.device)
    output = args.output_dir.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    prediction.to_pickle(output / f"{args.split}_predictions.pkl")
    pd.Series(metrics, name="value").to_csv(output / f"{args.split}_metrics.csv")
    print(f"Saved {len(prediction)} predictions and metrics to {output}")


if __name__ == "__main__":
    main()
