"""Contracts for real seed propagation, frozen inputs and historical reuse."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from omegaconf import OmegaConf

from trace_vq.cli.multiseed_validation import VARIANTS, tasks, make_config
from trace_vq.cli.screening import initialize
from trace_vq.models.next_screen import NextPredictor
from research.run_multiseed_validation import ROOT, SINGLE, check_backtest, references


class MultiseedTests(unittest.TestCase):
    def args(self, variant='fsq_ranknet', market='csi300', seed=1):
        return SimpleNamespace(variant=variant, universe=market, seed=seed, num_workers=2,
            device='cpu', output_dir=Path('/tmp/trace_vq_multiseed_test') / market / variant / f'seed{seed}')

    def test_queue_has_exactly_24_unique_fits_in_authorized_order(self):
        queue = tasks()
        self.assertEqual(len(queue), 24)
        self.assertEqual(len(set(queue)), 24)
        self.assertEqual([row[0] for row in queue], ['fsq_ranknet'] * 8 + ['fsq'] * 8 + ['ranknet'] * 8)
        self.assertEqual({row[2] for row in queue}, {1, 2, 3, 4})
        self.assertEqual({row[1] for row in queue}, {'csi300', 'csi800'})

    def test_real_configs_only_change_seed_and_output_locations(self):
        for market in ['csi300', 'csi800']:
            for variant in VARIANTS:
                old = OmegaConf.to_container(OmegaConf.load(SINGLE / market / variant / 'effective.config.yaml'), resolve=True)
                for seed in range(5):
                    cfg = make_config(self.args(variant, market, seed))
                    for key in ['vqvae', 'predictor', 'data', 'research']:
                        self.assertEqual(cfg[key], old[key], (market, variant, seed, key))
                    self.assertEqual(cfg['train']['seed'], seed)
                    for key in ['num_epochs', 'learning_rate', 'precision', 'early_stopping', 'gradient_clip_val']:
                        self.assertEqual(cfg['train'][key], old['train'][key])
                    self.assertTrue(cfg['train']['save_dir'].startswith(str(self.args(variant, market, seed).output_dir)))
        with self.assertRaises(ValueError): make_config(self.args(seed=5))

    def test_seeds_change_stage2_initialization_but_reproduce_and_preserve_stage1(self):
        torch.set_num_threads(1)
        for variant in ['fsq_ranknet', 'ranknet']:
            snapshots = []
            for seed in [1, 2, 1]:
                initialize(seed)
                model = NextPredictor(make_config(self.args(variant=variant, seed=seed)), T_max=2)
                trainable = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
                frozen = {n: p.detach().clone() for n, p in model.named_parameters()
                          if n.startswith(('encoder.', 'quantizer.', 'revin.'))}
                self.assertTrue(all(not p.requires_grad for n, p in model.named_parameters() if n in frozen))
                snapshots.append((trainable, frozen))
            self.assertTrue(torch.equal(snapshots[0][0], snapshots[2][0]))
            self.assertFalse(torch.equal(snapshots[0][0], snapshots[1][0]))
            for name in snapshots[0][1]:
                self.assertTrue(torch.equal(snapshots[0][1][name], snapshots[1][1][name]), name)

    def test_all_sixteen_references_are_present_and_only_four_need_rebacktesting(self):
        specs = references()
        self.assertEqual(len(specs), 16)
        self.assertEqual(sum(x['variant'] == 'baseline' for x in specs), 10)
        self.assertEqual(sum(x['needs_backtest'] for x in specs), 4)
        for spec in specs:
            self.assertTrue(spec['checkpoint'].is_file())
            self.assertTrue(spec['prediction'].is_file())
            self.assertTrue(spec['config'].is_file())
            if spec['needs_backtest']:
                self.assertEqual(spec['universe'], 'csi300')
                self.assertNotEqual(spec['seed'], 0)

    def test_protocol_rejects_legacy_quote_range_and_changed_exposure(self):
        for market in ['csi300', 'csi800']:
            config = json.loads((SINGLE / market / 'baseline/backtest/backtest_config.json').read_text())
            check_backtest(config, market)
            changed = copy.deepcopy(config); changed['quote_universe'] = market
            with self.assertRaises(AssertionError): check_backtest(changed, market)
            changed = copy.deepcopy(config); changed['strategy']['risk_degree'] = .9
            with self.assertRaises(AssertionError): check_backtest(changed, market)


if __name__ == '__main__':
    unittest.main()
