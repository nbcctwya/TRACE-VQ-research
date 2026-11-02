"""Trace both stages on one synthetic daily cross-section, using default dimensions."""

import argparse
import json
import tempfile
from datetime import datetime
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from trace_vq.data.schema import GROUP_SLICES, TOTAL_DIM, unpack_batch
from trace_vq.training.prediction import GenerateReturn
from trace_vq.training.representation import FactorVQVAE
from trace_vq.utils import seed_everything


def shapes(value):
    if isinstance(value, torch.Tensor):
        return list(value.shape)
    if isinstance(value, (tuple, list)):
        return [shapes(item) for item in value]
    return type(value).__name__


def trace_modules(model, names, events):
    handles = []
    for name in names:
        def capture(module, arguments, output, name=name):
            events.append({'module': name, 'input_shapes': shapes(arguments), 'output_shapes': shapes(output)})
        handles.append(model.get_submodule(name).register_forward_hook(capture))
    return handles


def parameter_counts(module):
    return {
        'total': sum(p.numel() for p in module.parameters()),
        'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stocks', type=int, default=8)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.stocks < 2:
        parser.error('--stocks must be at least 2')
    root = Path(__file__).resolve().parents[1]
    seed_everything(args.seed)
    if not OmegaConf.has_resolver('half'):
        OmegaConf.register_new_resolver('half', lambda count: int(count) // 2)
    with initialize_config_dir(version_base='1.3', config_dir=str(root / 'configs')):
        cfg = compose(config_name='config', overrides=['experiment=full', f'train.seed={args.seed}'])
    stage1_trace, stage2_trace = [], []
    with tempfile.TemporaryDirectory(prefix='architecture_', dir=root / '.cache/tmp') as temporary:
        cfg.predictor.saved_model = str(Path(temporary) / 'synthetic-stage1.ckpt')
        config = OmegaConf.to_container(cfg, resolve=True)
        batch = torch.randn(args.stocks, cfg.vqvae.seq_len, TOTAL_DIM)
        batch[:, :, GROUP_SLICES['prior']] = torch.randn(1, 1, cfg.vqvae.num_prior_factors)
        parts = unpack_batch(batch)
        first = FactorVQVAE(config, T_max=2)
        first.train()
        handles = trace_modules(first, [
            'vqvae.spatial_encoder.feature_transform',
            'vqvae.spatial_encoder.temporal_encoder',
            'vqvae.spatial_encoder.cross_asset_transformer',
            'vqvae.quantizer', 'vqvae.decoder', 'vqvae.predictor',
        ], stage1_trace)
        reconstruction, vq, auxiliary, total, latent, details = first(
            parts.stock_feature, parts.prior_factor, parts.future_returns
        )
        total.backward()
        for handle in handles:
            handle.remove()
        torch.save({'state_dict': first.state_dict()}, cfg.predictor.saved_model)

        second = GenerateReturn(config, T_max=2)
        second.train()
        codebook_before = second.quantizer.embedding.weight.detach().clone()
        handles = trace_modules(second, [
            'encoder', 'quantizer', 'quantization_confidence_adapter',
            'loadings.dliner', 'loadings.dim_projection',
            'loadings.temporal_transformer', 'loadings.fusion.moe',
            'loadings.fusion', 'latent_value_head', 'return_predictor',
        ], stage2_trace)
        prediction, beta_prior, beta_latent, corrected_latent, auxiliary2 = second(
            parts.stock_feature, parts.prior_factor
        )
        main_loss = second.rank_loss(prediction, parts.target(cfg.predictor.target_day))
        total2 = main_loss + second.aux_weight * auxiliary2
        total2.backward()
        for handle in handles:
            handle.remove()
        frozen = {
            name: {'training': module.training, **parameter_counts(module),
                   'parameters_with_gradient': sum(p.grad is not None for p in module.parameters())}
            for name, module in (('encoder', second.encoder), ('quantizer', second.quantizer), ('revin', second.revin))
        }
        invariants = {
            'stage1_encoder_receives_gradients': any(p.grad is not None for p in first.vqvae.spatial_encoder.parameters()),
            'stage2_frozen_modules_remain_eval': all(not item['training'] for item in frozen.values()),
            'stage2_frozen_modules_have_no_gradients': all(item['parameters_with_gradient'] == 0 for item in frozen.values()),
            'stage2_codebook_unchanged': torch.equal(codebook_before, second.quantizer.embedding.weight),
            'confidence_adapter_starts_at_zero': bool(torch.count_nonzero(second.quantization_confidence_adapter.weight) == 0 and torch.count_nonzero(second.quantization_confidence_adapter.bias) == 0),
            'confidence_adapter_receives_gradients': bool(second.quantization_confidence_adapter.weight.grad.abs().sum() > 0),
            'shared_expert_receives_gradients': bool(second.loadings.fusion.moe.shared_expert.net[-1].weight.grad.abs().sum() > 0),
        }
        if not all(invariants.values()):
            raise RuntimeError(f'Unexpected model behavior: {invariants}')
        report = {
            'data': 'synthetic; one daily cross-section; random initialization; CPU; no optimizer step',
            'seed': args.seed,
            'inputs': {name: shapes(value) for name, value in zip(parts._fields, parts)},
            'codebook_shape': shapes(second.quantizer.embedding.weight),
            'stage1': {
                'parameters': parameter_counts(first), 'trace': stage1_trace,
                'losses': {'reconstruction': reconstruction.item(), 'vq_including_training_contrastive': vq.item(),
                           'auxiliary_return': auxiliary.item(), 'auxiliary_weight': first.vqvae.pred_weight, 'total': total.item()},
                'latent_shape': shapes(latent), 'batch_code_perplexity': details[0].item(),
                'configured_decay': config['vqvae']['quantizer']['decay'],
                'effective_decay': first.vqvae.quantizer.decay,
            },
            'stage2': {
                'parameters': parameter_counts(second), 'trace': stage2_trace, 'frozen_modules': frozen,
                'outputs': {'prediction': shapes(prediction), 'beta_prior': shapes(beta_prior),
                            'beta_latent': shapes(beta_latent), 'corrected_latent': shapes(corrected_latent)},
                'losses': {'main': main_loss.item(), 'rank_weight': second.rank, 'softcapped_auxiliary': auxiliary2.item(),
                           'auxiliary_weight': second.aux_weight, 'total': total2.item()},
                'routed_experts': second.loadings.fusion.moe.num_experts,
                'top_k': second.loadings.fusion.moe.k,
                'shared_expert': second.loadings.fusion.moe.use_shared_expert,
            },
            'invariants': invariants,
        }
    output = root / 'artifacts/research/walkthrough' / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(f'Report: {output / "report.json"}')


if __name__ == '__main__':
    main()
