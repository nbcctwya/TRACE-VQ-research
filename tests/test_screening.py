"""Research screen invariants: unchanged baseline, exact caches, trainable additions."""

import copy
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from trace_vq.data.frozen import DailyPositions, daily_loader, prepare_frozen_cache
from trace_vq.data.loaders import DailyBatchSamplerRandom, init_data_loader
from trace_vq.data.schema import TOTAL_DIM, unpack_batch
from trace_vq.models.screening import ScreeningPredictor, VARIANTS, variant_config
from trace_vq.training.prediction import GenerateReturn
from trace_vq.training.representation import FactorVQVAE

ROOT = Path(__file__).resolve().parents[1]


class ToyDays:
    def __init__(self):
        rng = np.random.default_rng(8)
        self.index = pd.MultiIndex.from_tuples([
            (pd.Timestamp("2023-01-03"), "B"), (pd.Timestamp("2023-01-02"), "A"),
            (pd.Timestamp("2023-01-03"), "A"), (pd.Timestamp("2023-01-02"), "B"),
            (pd.Timestamp("2023-01-03"), "C"),
        ], names=["datetime", "instrument"])
        self.values = rng.normal(size=(len(self.index), 20, TOTAL_DIM))
        # Common market window per endpoint date, matching the real sampler.
        for date in self.index.get_level_values("datetime").unique():
            self.values[self.index.get_level_values("datetime") == date, :, 171:234] = rng.normal(size=(20, 63))

    def __len__(self):
        return len(self.index)

    def get_index(self):
        return self.index

    def __getitem__(self, position):
        return self.values[position]


class ScreeningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        if not OmegaConf.has_resolver("half"):
            OmegaConf.register_new_resolver("half", lambda value: int(value) // 2)
        cfg = OmegaConf.load(ROOT / "configs/config.yaml")
        cfg.predictor.saved_model = str(cls.root / "stage1.ckpt")
        cfg.vqvae.hidden_size = 8
        cfg.vqvae.vq_embed_dim = 8
        cfg.vqvae.num_embed = 16
        cfg.vqvae.decoder.hidden_channels = 8
        cfg.predictor.moe_hidden = 8
        cfg.predictor.transformer.d_model = 8
        cfg.predictor.transformer.dim_feedforward = 16
        cls.config = OmegaConf.to_container(cfg, resolve=True)
        torch.manual_seed(42)
        first = FactorVQVAE(copy.deepcopy(cls.config), T_max=2)
        torch.save({"state_dict": first.state_dict()}, cfg.predictor.saved_model)
        torch.manual_seed(9)
        cls.batch = torch.randn(7, 20, TOTAL_DIM)
        cls.parts = unpack_batch(cls.batch)
        cls.market = cls.parts.market_feature[:1].expand(7, -1, -1).clone()

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self, variant="baseline"):
        torch.manual_seed(0)
        return ScreeningPredictor(variant_config(self.config, variant), T_max=20)

    def test_baseline_state_and_prediction_are_exactly_historical(self):
        torch.manual_seed(0)
        historical = GenerateReturn(copy.deepcopy(self.config), T_max=20).eval()
        model = self.model().eval()
        self.assertEqual(historical.state_dict().keys(), model.state_dict().keys())
        for key, value in historical.state_dict().items():
            self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
        with torch.no_grad():
            expected = historical(self.parts.stock_feature, self.parts.prior_factor)
            actual = model(self.parts.stock_feature, self.parts.prior_factor, self.market)
        for left, right in zip(expected, actual):
            self.assertTrue(torch.equal(left, right))

    def test_residual_branches_start_at_the_same_prediction(self):
        with torch.no_grad():
            expected = self.model().eval()(self.parts.stock_feature, self.parts.prior_factor)[0]
            for variant in VARIANTS:
                with self.subTest(variant=variant):
                    model = self.model(variant).eval()
                    actual = model(self.parts.stock_feature, self.parts.prior_factor, self.market)[0]
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertIsNone(getattr(model.loadings.fusion.moe.gate, "condition", None))

    def test_optional_branches_preserve_the_shared_rng_stream(self):
        self.model()
        expected = torch.get_rng_state()
        for variant in VARIANTS:
            self.model(variant)
            self.assertTrue(torch.equal(expected, torch.get_rng_state()), variant)

    def test_all_variants_train_and_restore_strictly(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant).train()
                before = model.quantizer.embedding.weight.detach().clone()
                result = model(self.parts.stock_feature, self.parts.prior_factor, self.market)
                loss, _ = model.training_loss(self.parts, result)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for name, parameter in model.named_parameters():
                    if name.startswith(("encoder.", "quantizer.", "revin.")):
                        self.assertIsNone(parameter.grad, name)
                    elif parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                if model.market_kind == "route":
                    self.assertGreater(model.loadings.fusion.moe.gate.market[-1].weight.grad.abs().sum(), 0)
                if model.market_kind == "direct":
                    self.assertGreater(model.market_modulation[-1].weight.grad.abs().sum(), 0)
                if hasattr(model, "prior_gate"):
                    self.assertGreater(model.prior_gate[-1].weight.grad.abs().sum(), 0)
                if hasattr(model, "horizon_head"):
                    self.assertGreater(model.horizon_head.weight.grad.abs().sum(), 0)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                optimizer.step()
                self.assertTrue(torch.equal(before, model.quantizer.embedding.weight))
                restored = self.model(variant)
                restored.load_state_dict(model.state_dict(), strict=True)
                with torch.no_grad():
                    left = model.eval()(self.parts.stock_feature, self.parts.prior_factor, self.market)[0]
                    right = restored.eval()(self.parts.stock_feature, self.parts.prior_factor, self.market)[0]
                self.assertTrue(torch.equal(left, right))

    def test_cached_forward_and_stage2_gradients_are_exact(self):
        for variant in ("baseline", "market_route", "multihorizon", "sparse_prior"):
            with self.subTest(variant=variant):
                model = self.model(variant).train()
                with torch.no_grad():
                    h = model.encoder(model.revin(self.parts.stock_feature, "norm"))
                    z_q = model.quantizer(h)[0]
                torch.manual_seed(123)
                reference = model(self.parts.stock_feature, self.parts.prior_factor, self.market)
                reference_loss = model.training_loss(self.parts, reference)[0]
                reference_loss.backward()
                gradients = {name: parameter.grad.clone() for name, parameter in model.named_parameters()
                             if parameter.grad is not None}
                model.zero_grad(set_to_none=True)
                torch.manual_seed(123)
                actual = model(self.parts.stock_feature, self.parts.prior_factor, self.market, h.clone(), z_q.clone())
                actual_loss = model.training_loss(self.parts, actual)[0]
                actual_loss.backward()
                self.assertTrue(torch.equal(reference_loss, actual_loss))
                for left, right in zip(reference, actual):
                    self.assertTrue(torch.equal(left, right))
                for name, parameter in model.named_parameters():
                    if name in gradients:
                        self.assertTrue(torch.equal(gradients[name], parameter.grad), name)

    def test_daily_sampler_and_fp32_windows_match_original_loader(self):
        source = ToyDays()
        for shuffle in (False, True):
            np.random.seed(42)
            expected = list(DailyBatchSamplerRandom(source, shuffle))
            np.random.seed(42)
            actual = list(DailyPositions(source, shuffle))
            self.assertEqual(len(expected), len(actual))
            for left, right in zip(expected, actual):
                np.testing.assert_array_equal(left, right)
        original, _ = init_data_loader(source, shuffle=False)
        optimized = daily_loader(source)
        for left, right in zip(original, optimized):
            self.assertTrue(torch.equal(left, right))

    def test_frozen_cache_is_aligned_reusable_and_rejects_stale_input(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = ToyDays()
            source_path = directory / "dataset.pkl"
            with source_path.open("wb") as stream:
                pickle.dump(source, stream)
            path = directory / "cache.npy"
            model = self.model().eval()
            cached, metadata = prepare_frozen_cache(model, source, source_path, path, model.config, "cpu", 0)
            self.assertEqual(metadata["rows"], len(source))
            self.assertTrue(metadata["quantizer_unchanged"])
            for positions, batch in zip(DailyPositions(source), daily_loader(cached)):
                with torch.no_grad():
                    parts = unpack_batch(torch.tensor(source[positions]).float())
                    h = model.encoder(model.revin(parts.stock_feature, "norm"))
                    z_q = model.quantizer(h)[0]
                self.assertTrue(torch.equal(batch["frozen_h"], h))
                self.assertTrue(torch.equal(batch["frozen_zq"], z_q))
            reused, _ = prepare_frozen_cache(model, source, source_path, path, model.config, "cpu", 0)
            self.assertTrue(np.array_equal(cached.values, reused.values))
            with source_path.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaises(RuntimeError):
                prepare_frozen_cache(model, source, source_path, path, model.config, "cpu", 0)

    def test_market_variants_fail_if_market_input_is_missing(self):
        for variant in ("market_route", "market_direct"):
            model = self.model(variant)
            with self.assertRaises(ValueError):
                model(self.parts.stock_feature, self.parts.prior_factor)

    def test_invalid_multihorizon_does_not_supervise_constant_ret1(self):
        config = variant_config(self.config, "multihorizon")
        config["research"]["auxiliary_horizons"] = [1, 2, 10]
        with self.assertRaises(ValueError):
            ScreeningPredictor(config, T_max=1)


if __name__ == "__main__":
    unittest.main()
