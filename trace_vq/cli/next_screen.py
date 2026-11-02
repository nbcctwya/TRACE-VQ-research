"""Train/evaluate the second screen with explicit, isolated output paths."""
import argparse
import copy
import gc
import json
import time
from pathlib import Path

from trace_vq.runtime import configure_runtime
configure_runtime()

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from trace_vq.cli.screening import (ROOT, EpochRecord, initialize, make_config as base_config,
                                    now, read_dataset, scalar, write_prediction)
from trace_vq.data.frozen import atomic_json, daily_loader, prepare_frozen_cache, sha256
from trace_vq.data.research_context import PairedDays, CodeHistory, build_history_positions
from trace_vq.models.next_screen import NextPredictor, NEXT_VARIANTS
from trace_vq.training.representation import FactorVQVAE


def make_config(args):
    config = base_config(args.universe, 'baseline', args.output_dir)
    config['research'] = dict(variant=args.variant, **copy.deepcopy(NEXT_VARIANTS[args.variant]))
    config['train']['num_workers'] = args.num_workers
    if args.stage1_checkpoint:
        path = Path(args.stage1_checkpoint).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('Stage 1 checkpoint must be inside the research copy')
        config['predictor']['saved_model'] = str(path)
    research = config['research']
    if research.get('quantizer_kind'):
        quant = config['vqvae']['quantizer']
        quant['research_kind'] = research['quantizer_kind']
        if research['quantizer_kind'] == 'rvq':
            quant['research_sizes'] = research['sizes']
        else:
            quant['research_levels'] = research['levels']
    return config


def cached(model, config, split, args, context=True):
    source, path = read_dataset(config, split)
    dataset, metadata = prepare_frozen_cache(model, source, path,
                                            args.cache_root / args.universe / f'{split}.npy',
                                            config, args.device, args.num_workers)
    if context and config['research'].get('history'):
        dataset = CodeHistory(dataset, args.history_root, args.cache_root, args.universe, split)
    elif context and split == 'train' and config['research'].get('paired_days'):
        dataset = PairedDays(dataset)
    return dataset, metadata


def build_cache(args):
    initialize()
    config = make_config(args)
    model = NextPredictor(config, T_max=1)
    records = {}
    for split in ('train', 'valid', 'test'):
        dataset, records[split] = cached(model, config, split, args, context=False)
        del dataset
        gc.collect()
    atomic_json(args.cache_root / args.universe / 'completion.json', records)


def history_cache(args):
    config = make_config(args)
    indices, identities = [], []
    for split in ('train', 'valid', 'test'):
        source, path = read_dataset(config, split)
        indices.append(source.get_index().copy())
        record = json.loads((args.cache_root / args.universe / f'{split}.json').read_text())
        identities.append({'source_sha256': sha256(path),
                           'stage1_sha256': record['identity']['stage1_sha256']})
        if record['source_sha256'] != identities[-1]['source_sha256']:
            raise RuntimeError('Dataset changed before history construction')
        del source
        gc.collect()
    build_history_positions(indices, args.history_root / args.universe, identities, days=20)


class QuietFactorVQVAE(FactorVQVAE):
    def on_train_epoch_end(self):
        if self.epoch_vq_codes:
            codes = np.concatenate(self.epoch_vq_codes)
            unique, counts = np.unique(codes, return_counts=True)
            p = counts / counts.sum()
            record = {'epoch': self.current_epoch, 'used_codes': len(unique),
                      'empirical_perplexity': float(np.exp(-(p*np.log(p)).sum()))}
            quantizer = self.vqvae.quantizer
            if hasattr(quantizer, 'sizes'):
                record.update(coarse_used=int(len(np.unique(codes // quantizer.sizes[1]))),
                              fine_used=int(len(np.unique(codes % quantizer.sizes[1]))),
                              combined_vocab_size=quantizer.combined_vocab_size)
            output = Path(self.config['artifact_root'])
            atomic_json(output / f'code_usage_epoch{self.current_epoch:03d}.json', record)
            self.epoch_vq_codes.clear()

    def on_validation_epoch_end(self):
        # The native hook only logs/plots. Suppress WandB and enormous RVQ plots.
        pass


def callbacks(config, output, seed):
    best = ModelCheckpoint(dirpath=output / 'checkpoints', monitor='val_loss', mode='min',
                           save_top_k=1, save_last=True, filename=f'seed{seed}-{{epoch}}-{{val_loss:.6f}}')
    early = config['train']['early_stopping']
    stop = EarlyStopping(monitor='val_loss', mode='min', min_delta=early['min_delta'],
                         patience=early['patience'], verbose=True)
    return best, stop


def trainer(config, output, callbacks_, args, seed):
    return pl.Trainer(logger=CSVLogger(save_dir=str(output), name='training', version=f'seed{seed}'),
                      callbacks=callbacks_, max_epochs=config['train']['num_epochs'],
                      accelerator='gpu' if args.device.startswith('cuda') else 'cpu', devices=1,
                      precision=config['train']['precision'], deterministic=True,
                      gradient_clip_val=config['train']['gradient_clip_val'], enable_progress_bar=False,
                      log_every_n_steps=50, default_root_dir=str(output))


def train_stage1(args):
    initialize(42)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training_result.json').exists():
        raise FileExistsError('Refusing to overwrite a completed Stage 1')
    config = make_config(args)
    config['train']['seed'] = 42
    train_data, train_path = read_dataset(config, 'train')
    valid_data, valid_path = read_dataset(config, 'valid')
    train_loader = daily_loader(train_data, shuffle=True, num_workers=args.num_workers)
    valid_loader = daily_loader(valid_data, num_workers=args.num_workers)
    model = QuietFactorVQVAE(config, len(train_loader) * config['train']['num_epochs'])
    OmegaConf.save(OmegaConf.create(config), output / 'effective.config.yaml')
    atomic_json(output / 'input_manifest.json', {'seed': 42,
                'train_sha256': sha256(train_path), 'valid_sha256': sha256(valid_path),
                'test_used_for_training_or_selection': False})
    best, stop = callbacks(config, output, 42)
    fit = trainer(config, output, [best, stop, EpochRecord(output)], args, 42)
    start = time.monotonic()
    fit.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if not best.best_model_path:
        raise RuntimeError('No Stage 1 validation checkpoint')
    atomic_json(output / 'training_result.json', {'status': 'complete', 'universe': args.universe,
                'variant': args.variant, 'stage1_seed': 42, 'checkpoint': best.best_model_path,
                'checkpoint_sha256': sha256(best.best_model_path),
                'best_val_loss': scalar(best.best_model_score), 'completed_epochs': fit.current_epoch,
                'elapsed_seconds': time.monotonic()-start, 'finished': now()})


def train(args):
    initialize()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'training_result.json').exists():
        raise FileExistsError('Refusing to overwrite completed Stage 2')
    config = make_config(args)
    model = NextPredictor(config, T_max=1)
    train_data, train_metadata = cached(model, config, 'train', args)
    valid_data, valid_metadata = cached(model, config, 'valid', args)
    train_loader = daily_loader(train_data, shuffle=True, num_workers=args.num_workers)
    valid_loader = daily_loader(valid_data, num_workers=args.num_workers)
    model.T_max = len(train_loader) * config['train']['num_epochs']
    best, stop = callbacks(config, output, 0)
    snapshot = copy.deepcopy(model.config)
    OmegaConf.save(OmegaConf.create(snapshot), output / 'effective.config.yaml')
    atomic_json(output / 'input_manifest.json', {'stage1_seed': 42, 'stage2_seed': 0,
                'stage1_sha256': sha256(config['predictor']['saved_model']),
                'train': train_metadata, 'valid': valid_metadata,
                'selection': 'original validation MSE plus original auxiliary loss'})
    fit = trainer(config, output, [best, stop, EpochRecord(output)], args, 0)
    start = time.monotonic()
    fit.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)
    if not best.best_model_path:
        raise RuntimeError('No Stage 2 validation checkpoint')
    first = torch.load(config['predictor']['saved_model'], map_location='cpu', weights_only=False)['state_dict']
    for name, tensor in model.state_dict().items():
        for prefix, source in [('encoder.', 'vqvae.spatial_encoder.'), ('quantizer.', 'vqvae.quantizer.'),
                               ('revin.', 'vqvae.revin.')]:
            if name.startswith(prefix) and not torch.equal(tensor.cpu(), first[source+name[len(prefix):]]):
                raise RuntimeError(f'Frozen Stage 1 changed: {name}')
    del first
    result = {'status': 'complete', 'universe': args.universe, 'variant': args.variant,
              'stage1_seed': 42, 'stage2_seed': 0, 'checkpoint': best.best_model_path,
              'checkpoint_sha256': sha256(best.best_model_path), 'best_val_loss': scalar(best.best_model_score),
              'completed_epochs': fit.current_epoch, 'elapsed_seconds': time.monotonic()-start,
              'stage1_unchanged': True, 'config': str(output / 'effective.config.yaml'), 'finished': now()}
    restored = NextPredictor.load_from_checkpoint(best.best_model_path, config=snapshot, T_max=1,
                                                  map_location='cpu', strict=True)
    del model, fit, train_loader, train_data
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    result['validation_metrics'] = write_prediction(restored, valid_data, snapshot, args.device,
                                                    output / 'training_evaluation', 'valid', args.num_workers)
    del valid_loader, valid_data
    gc.collect()
    test_data, _ = cached(restored, snapshot, 'test', args)
    result['test_metrics'] = write_prediction(restored, test_data, snapshot, args.device,
                                              output / 'training_evaluation', 'test', args.num_workers)
    atomic_json(output / 'training_result.json', result)


def evaluate(args):
    initialize()
    output = args.output_dir
    config = OmegaConf.to_container(OmegaConf.load(output / 'effective.config.yaml'), resolve=True)
    record = json.loads((output / 'training_result.json').read_text())
    model = NextPredictor.load_from_checkpoint(record['checkpoint'], config=config, T_max=1,
                                               map_location='cpu', strict=True)
    metrics = {}
    for split in ('valid', 'test'):
        data, _ = cached(model, config, split, args)
        metrics[split] = write_prediction(model, data, config, args.device, output/'evaluation', split, args.num_workers)
        actual = pd.read_pickle(output/'evaluation'/f'{split}_predictions.pkl')
        expected = pd.read_pickle(output/'training_evaluation'/f'{split}_predictions.pkl')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        del data
        gc.collect()
    atomic_json(output/'evaluation/completion.json', {'status': 'complete', 'standalone_exact': True, 'metrics': metrics})


def adopt(args):
    initialize()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    config = make_config(args)
    source = args.reference_dir.resolve()
    if not source.is_relative_to(ROOT):
        raise ValueError('Reference must be inside this copy')
    record = json.loads((source/'completion.json').read_text())
    model = NextPredictor.load_from_checkpoint(record['checkpoint'], config=config, T_max=1,
                                               map_location='cpu', strict=True)
    metrics = {}
    for split in ('valid', 'test'):
        data, _ = cached(model, config, split, args)
        metrics[split] = write_prediction(model, data, config, args.device, output/'training_evaluation', split, args.num_workers)
        actual = pd.read_pickle(output/'training_evaluation'/f'{split}_predictions.pkl')
        expected = pd.read_pickle(source/'evaluation'/f'{split}_predictions.pkl')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        del data
        gc.collect()
    OmegaConf.save(OmegaConf.create(config), output/'effective.config.yaml')
    record.update(config=str(output/'effective.config.yaml'), universe=args.universe, variant=args.variant,
                  reference_source=str(source), reference_predictions_exact=True,
                  validation_metrics=metrics['valid'], test_metrics=metrics['test'])
    atomic_json(output/'training_result.json', record)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['cache', 'history-cache', 'train-stage1', 'train', 'evaluate', 'adopt'])
    parser.add_argument('--universe', choices=['csi300', 'csi800'], required=True)
    parser.add_argument('--variant', choices=NEXT_VARIANTS, default='baseline')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-root', type=Path, required=True)
    parser.add_argument('--history-root', type=Path)
    parser.add_argument('--stage1-checkpoint', type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=2)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    if not args.output_dir.is_relative_to(ROOT/'artifacts/research'):
        raise ValueError('All new outputs must be under artifacts/research')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    functions = {'cache': build_cache, 'history-cache': history_cache, 'train-stage1': train_stage1,
                 'train': train, 'evaluate': evaluate, 'adopt': adopt}
    functions[args.command](args)


if __name__ == '__main__':
    main()
