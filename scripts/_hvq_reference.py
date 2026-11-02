"""Isolated reference process: import only the frozen HVQ model packages."""

import argparse
import copy
import sys
from pathlib import Path


def clone_state(module):
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.reference))
    import torch
    from trainer.train_vqvae import FactorVQVAE
    from trainer.train_ypred import GenerateReturn
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("high")
    torch.use_deterministic_algorithms(True)
    request = torch.load(args.input, map_location="cpu", weights_only=False)
    device = request.get("device", "cpu")
    result = {"runs": {}}
    dataset = None
    if request.get("full_data_path"):
        import pickle
        import numpy as np
        from dataset.dataset import DailyBatchSamplerRandom
        with open(request["full_data_path"], "rb") as stream:
            dataset = pickle.load(stream)
        days = list(DailyBatchSamplerRandom(dataset))
        result["full_index"] = dataset.get_index()[np.concatenate(days)]
    first_config = copy.deepcopy(request["configs"][0])
    torch.manual_seed(42)
    first = FactorVQVAE(first_config, T_max=1)
    result["stage1_initial"] = clone_state(first)
    state = torch.load(first_config["predictor"]["saved_model"], map_location="cpu", weights_only=False)["state_dict"]
    first.load_state_dict(state, strict=True)
    first.to(device).eval()
    torch.set_float32_matmul_precision(first_config.get("inference", {}).get("matmul_precision", "high"))
    sample = request["samples"][0].to(device)
    with torch.no_grad():
        result["stage1_outputs"] = tuple(x.detach().cpu().clone() for x in first(
            sample[:, :, :158], sample[:, -1, 158:171], sample[:, -1, 234:244]
        )[:5])
    for config, checkpoint in zip(request["configs"], request["checkpoints"]):
        config = copy.deepcopy(config)
        seed = config["train"]["seed"]
        torch.set_float32_matmul_precision(config.get("inference", {}).get("matmul_precision", "high"))
        torch.manual_seed(seed)
        model = GenerateReturn(config, T_max=1)
        run = {"initial": clone_state(model)}
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        with torch.no_grad():
            run["outputs"] = [tuple(x.detach().cpu().clone() for x in model(
                batch[:, :, :158], batch[:, -1, 158:171]
            )) for batch in [sample.to(device) for sample in request["samples"]]]
        if seed == 0:
            torch.manual_seed(123)
            model.train()
            batch = request["samples"][0].to(device)
            output = model(batch[:, :, :158], batch[:, -1, 158:171])
            target = batch[:, -1, 234 + config["predictor"]["target_day"] - 1]
            loss = model.rank_loss(output[0], target) + model.aux_weight * output[-1]
            loss.backward()
            run["training_loss"] = loss.detach().cpu().clone()
            run["gradients"] = {name: p.grad.detach().cpu().clone() if p.grad is not None else None
                                for name, p in model.named_parameters()}
            optimizers, _ = model.configure_optimizers()
            optimizers[0].step()
            run["after_step"] = clone_state(model)
            model.load_state_dict(state, strict=True)
            model.zero_grad(set_to_none=True)
        if dataset is not None:
            model.eval()
            predictions = []
            with torch.no_grad():
                for positions in days:
                    batch = torch.as_tensor(dataset[positions]).float().to(device)
                    predictions.append(model(batch[:, :, :158], batch[:, -1, 158:171])[0].cpu().numpy())
            run["full_scores"] = np.concatenate(predictions)
            print(f"Reference seed {seed}: all {len(dataset):,} predictions complete", flush=True)
        result["runs"][str(seed)] = run
    torch.save(result, args.output)


if __name__ == "__main__":
    main()
