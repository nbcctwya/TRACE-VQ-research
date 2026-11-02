"""Mechanism controls, meaningful relation gradients and training-only states."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from trace_vq.data.frozen import DailyPositions
from trace_vq.data.mechanism_context import (RepresentationHistory, StateBalancedDays,
                                            date_derangement, fit_state_balance)
from trace_vq.data.research_context import build_history_positions
from trace_vq.data.schema import TOTAL_DIM, unpack_batch
from trace_vq.models.mechanism_screen import (MECHANISM_VARIANTS, MechanismPredictor,
                                              loading_stability, peer_average)
from trace_vq.models.next_screen import NextPredictor
from trace_vq.training.representation import FactorVQVAE

ROOT = Path(__file__).resolve().parents[1]


class ToySource:
    def __init__(self, index, market_frame=None):
        self.index = index
        self.market_frame = market_frame

    def __len__(self):
        return len(self.index)

    def get_index(self):
        return self.index

    def __getitem__(self, rows):
        shape = np.asarray(rows).shape
        canonical = np.zeros((*shape, 20, TOTAL_DIM), dtype=np.float32)
        if self.market_frame is not None:
            dates = self.index.get_level_values("datetime")[rows]
            market = self.market_frame.loc[np.atleast_1d(dates)].to_numpy(dtype=np.float32)
            canonical[..., :, 171:234] = market.reshape((*shape, 1, 63))
        return {"canonical": canonical, "frozen_h": np.zeros((*shape, 4), dtype=np.float32),
                "frozen_zq": np.zeros((*shape, 4), dtype=np.float32)}


class MechanismContextTests(unittest.TestCase):
    def test_date_derangement_has_no_fixed_stocks_and_preserves_global_rng(self):
        np.random.seed(13)
        before = np.random.get_state()
        for n in [0, 1, 2, 7, 800]:
            order = date_derangement(n, "2020-01-02")
            np.testing.assert_array_equal(np.sort(order), np.arange(n))
            np.testing.assert_array_equal(order, date_derangement(n, "2020-01-02"))
            if n > 1:
                self.assertFalse(np.any(order == np.arange(n)))
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_history_controls_switch_only_representation_or_stock_identity(self):
        indices = [pd.MultiIndex.from_product([pd.to_datetime(dates), ["A", "B"]],
                    names=["datetime", "instrument"]) for dates in
                   [["2020-01-01", "2020-01-02"], ["2020-01-03"], ["2020-01-06"]]]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identities = [dict(source_sha256=str(i), stage1_sha256="fixed") for i in range(3)]
            meta = build_history_positions(indices, root / "history/csi300", identities, days=2)
            folder = root / "cache/csi300"; folder.mkdir(parents=True)
            for i, name in enumerate(["train", "valid", "test"]):
                values = np.zeros((len(indices[i]), 2, 4), dtype=np.float32)
                values[:, 1] = np.arange(meta["offsets"][i], meta["offsets"][i + 1])[:, None]
                values[:, 0] = values[:, 1] + 100
                np.save(folder / f"{name}.npy", values)
                (folder / f"{name}.json").write_text(json.dumps({"source_sha256": str(i), "identity": {"stage1_sha256": "fixed"}}))
            def wrapped(**options):
                return RepresentationHistory(ToySource(indices[1]), root / "history", root / "cache", "csi300", "valid", **options)
            ordinary = wrapped()[np.arange(2)]
            continuous = wrapped(feature="continuous")[np.arange(2)]
            mismatched = wrapped(mismatch=True)[np.arange(2)]
            np.testing.assert_array_equal(continuous["code_history"], ordinary["code_history"] + 100)
            np.testing.assert_array_equal(mismatched["code_history"], ordinary["code_history"][[1, 0]])
            np.testing.assert_array_equal(mismatched["history_mask"], ordinary["history_mask"][[1, 0]])
            self.assertTrue((ordinary["code_history"][:, :, 0] < meta["offsets"][1]).all())

    def test_state_weights_are_bounded_and_preserve_mean_training_loss_scale(self):
        trend = np.arange(100, dtype=float)
        volatility = trend[::-1] ** 2
        result = fit_state_balance(trend, volatility)
        weights, counts = np.array(result["weights"]), np.array(result["counts"])
        self.assertTrue(((weights >= .5) & (weights <= 2)).all())
        self.assertAlmostEqual(float(np.dot(counts / counts.sum(), weights)), 1, places=12)
        self.assertEqual(sum(counts), 100)

    def test_cli_state_fit_ignores_heldout_feature_changes(self):
        from trace_vq.cli.mechanism_screen import prepare_state
        training_dates = pd.date_range("2020-01-01", periods=40)
        dates = training_dates.append(pd.to_datetime(["2021-06-01", "2025-12-01"]))
        frame = pd.DataFrame(np.arange(len(dates) * 63).reshape(-1, 63).astype(float), index=dates)
        index = pd.MultiIndex.from_product([training_dates, ["A", "B"]], names=["datetime", "instrument"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "unchanged_source.pkl"; source_path.write_bytes(b"training source identity")
            results = []
            for attempt in [0, 1]:
                context = root / str(attempt)
                source = ToySource(index, frame.copy())
                if attempt:
                    source.market_frame.loc[source.market_frame.index.year > 2020] *= -10000
                args = SimpleNamespace(universe="csi300", output_dir=context, context_root=context)
                with patch("trace_vq.cli.mechanism_screen.base_config", return_value={}), \
                     patch("trace_vq.cli.mechanism_screen.read_dataset", return_value=(source, source_path)):
                    prepare_state(args)
                    prepare_state(args)  # Exact metadata and tolerant numeric CSV round trip.
                results.append(json.loads((context / "states/csi300.json").read_text()))
            self.assertEqual(results[0], results[1])
            self.assertEqual(results[0]["fit_splits"], ["train"])
            self.assertFalse(results[0]["test_used_for_fitting"])


class MechanismMathTests(unittest.TestCase):
    def test_peer_means_exclude_self_and_mask_singletons(self):
        x = torch.tensor([[2., 0.], [4., 1.], [9., -2.], [8., 4.]], requires_grad=True)
        codes = torch.tensor([0, 0, 1, 2])
        peer, valid = peer_average(x, codes, "code")
        self.assertTrue(torch.equal(peer[:2], x[[1, 0]]))
        self.assertTrue(torch.equal(peer[2:], torch.zeros(2, 2)))
        self.assertEqual(valid.tolist(), [True, True, False, False])
        peer[0].sum().backward()
        self.assertTrue(torch.equal(x.grad[0], torch.zeros(2)))
        self.assertTrue(torch.equal(x.grad[1], torch.ones(2)))
        global_mean, _ = peer_average(x.detach(), codes, "market")
        self.assertTrue(torch.allclose(global_mean[0], x.detach()[1:].mean(0)))
        own, _ = peer_average(x.detach(), codes, "self")
        self.assertTrue(torch.equal(own, x.detach()))

    def test_loading_penalty_detects_direction_changes_and_backpropagates_both_days(self):
        current = [torch.tensor([[1., -2., 3.]], requires_grad=True)] * 2
        previous = [torch.tensor([[-1., 2., -3.]], requires_grad=True)] * 2
        self.assertAlmostEqual(float(loading_stability(current, current)), 0)
        loss = loading_stability(current, previous)
        self.assertGreater(float(loss), 0)
        loss.backward()
        self.assertGreater(float(current[0].grad.abs().sum()), 0)
        self.assertGreater(float(previous[0].grad.abs().sum()), 0)


class MechanismModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        if not OmegaConf.has_resolver("half"):
            OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)
        cfg = OmegaConf.load(ROOT / "configs/config.yaml")
        cfg.vqvae.hidden_size = 8; cfg.vqvae.vq_embed_dim = 8; cfg.vqvae.num_embed = 16
        cfg.vqvae.decoder.hidden_channels = 8
        cfg.predictor.moe_hidden = 8; cfg.predictor.transformer.d_model = 8
        cfg.predictor.transformer.dim_feedforward = 16
        cls.config_base = OmegaConf.to_container(cfg, resolve=True)
        torch.manual_seed(42)
        first = FactorVQVAE(cls.config_base, T_max=2)
        checkpoint = root / "seed42.ckpt"
        torch.save({"state_dict": first.state_dict()}, checkpoint)
        cls.config_base["predictor"]["saved_model"] = str(checkpoint)
        torch.manual_seed(9)
        cls.canonical = torch.randn(7, 20, TOTAL_DIM)
        cls.canonical[:, :, 171:234] = cls.canonical[:1, :, 171:234]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def model(self, variant, **changes):
        cfg = copy.deepcopy(self.config_base)
        cfg["research"] = dict(variant=variant, **copy.deepcopy(MECHANISM_VARIANTS[variant]))
        cfg["research"].update(changes)
        torch.manual_seed(0)
        result = MechanismPredictor(cfg, T_max=2)
        result.log = lambda *a, **kw: None
        result.log_dict = lambda *a, **kw: None
        return result

    def batch(self, model):
        parts = unpack_batch(self.canonical)
        with torch.no_grad():
            h = model.encoder(model.revin(parts.stock_feature, mode="norm")); z = model.quantizer(h)[0]
        return {"canonical": self.canonical, "frozen_h": h, "frozen_zq": z,
                "prev_canonical": self.canonical, "prev_frozen_h": h, "prev_frozen_zq": z,
                "align_current": torch.arange(7).float(), "align_previous": torch.arange(7).float(),
                "code_history": z[:, None].expand(-1, 20, -1).clone(), "history_mask": torch.ones(7, 20),
                "state_weight": torch.tensor(1.3)}

    def test_reference_models_preserve_exact_state_and_constructor_rng(self):
        for variant in ["baseline", "history_gru", "turnover_control"]:
            new = self.model(variant)
            rng = torch.get_rng_state()
            torch.manual_seed(0)
            old = NextPredictor(copy.deepcopy(new.config), T_max=2)
            self.assertTrue(torch.equal(rng, torch.get_rng_state()))
            self.assertEqual(set(new.state_dict()), set(old.state_dict()))
            for name, value in old.state_dict().items():
                self.assertTrue(torch.equal(value, new.state_dict()[name]), name)

    def test_peer_arms_have_equal_capacity_initialization_and_zero_initial_correction(self):
        baseline = self.model("baseline").eval()
        batch = self.batch(baseline)
        expected = baseline.batch_prediction(batch)[1][0]
        peer_states = []
        for variant in ["peer_self", "peer_market", "peer_code"]:
            peer = self.model(variant).eval()
            self.assertTrue(torch.equal(peer.batch_prediction(batch)[1][0], expected))
            peer_states.append(peer.state_dict())
        for other in peer_states[1:]:
            self.assertEqual(set(other), set(peer_states[0]))
            for name, value in other.items():
                self.assertTrue(torch.equal(value, peer_states[0][name]), name)

    def test_all_variants_have_finite_gradients_and_restore_exactly(self):
        for variant in MECHANISM_VARIANTS:
            with self.subTest(variant=variant):
                model = self.model(variant).train()
                batch = self.batch(model)
                loss = model.training_step(batch, 0)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for name, parameter in model.named_parameters():
                    if name.startswith(("encoder.", "quantizer.", "revin.")):
                        self.assertIsNone(parameter.grad, name)
                    elif parameter.grad is not None:
                        self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                restored = self.model(variant).eval()
                restored.load_state_dict(model.state_dict(), strict=True)
                self.assertTrue(torch.equal(model.eval().batch_prediction(batch)[1][0],
                                            restored.batch_prediction(batch)[1][0]))

    def test_zero_beta_penalty_and_unit_state_weight_reproduce_control_loss_and_gradients(self):
        for current, reference, extra in [("beta_smooth", "turnover_control", {"beta_smooth_weight": 0}),
                                           ("state_balanced", "baseline", {})]:
            candidate, control = self.model(current, **extra).train(), self.model(reference).train()
            batch = self.batch(candidate); batch["state_weight"] = torch.tensor(1.)
            rng = torch.get_rng_state()
            value = candidate.training_step(batch, 0); value.backward()
            torch.set_rng_state(rng)
            expected = control.training_step(batch, 0); expected.backward()
            self.assertTrue(torch.equal(value, expected))
            for (name, a), (_, b) in zip(candidate.named_parameters(), control.named_parameters()):
                if a.grad is not None:
                    self.assertTrue(torch.equal(a.grad, b.grad), name)


if __name__ == "__main__":
    unittest.main()
