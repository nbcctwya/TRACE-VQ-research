"""Batch01 changes must preserve frozen Stage 1 and the native default path."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from omegaconf import OmegaConf

from trace_vq.models.batch01 import Batch01Predictor, CoordinateFSQ, VARIANTS
from trace_vq.models.prediction.temporal_transformer import RotaryTransformerEncoderLayer
from trace_vq.models.quantization.research import FiniteScalarQuantiser
from trace_vq.models.screening import ScreeningPredictor
from trace_vq.training.representation import FactorVQVAE
from trace_vq.data.schema import TOTAL_DIM, unpack_batch

ROOT = Path(__file__).resolve().parents[1]


class Batch01Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = Path(cls.temp.name)
        if not OmegaConf.has_resolver("half"):
            OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)
        base = OmegaConf.to_container(OmegaConf.load(ROOT / "configs/config.yaml"), resolve=True)
        base["vqvae"].update(hidden_size=8, vq_embed_dim=8, num_embed=16)
        base["vqvae"]["decoder"]["hidden_channels"] = 8
        base["predictor"]["moe_hidden"] = 8
        base["predictor"]["transformer"].update(d_model=8, dim_feedforward=16)
        base["research"] = {"variant": "baseline"}
        cls.configs = {}
        for family in ("vq", "fsq"):
            cfg = copy.deepcopy(base)
            if family == "fsq":
                cfg["vqvae"]["num_embed"] = 512
                cfg["vqvae"]["quantizer"].update(research_kind="fsq", research_levels=[8, 8, 8])
            checkpoint = cls.directory / (family + ".ckpt")
            cfg["predictor"]["saved_model"] = str(checkpoint)
            torch.manual_seed(42)
            model = FactorVQVAE(copy.deepcopy(cfg), T_max=2)
            torch.save({"state_dict": model.state_dict()}, checkpoint)
            cls.configs[family] = cfg
        torch.manual_seed(9)
        cls.parts = unpack_batch(torch.randn(7, 20, TOTAL_DIM))

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def model(self, variant, family=None):
        family = family or VARIANTS.get(variant, {}).get("family", "vq")
        cfg = copy.deepcopy(self.configs[family])
        cfg["research"] = {"variant": variant, **VARIANTS.get(variant, {})}
        if variant == "fsq_wo_ca":
            cfg["predictor"]["quantization_confidence_adapter"] = False
        torch.manual_seed(0)
        return Batch01Predictor(cfg, T_max=10)

    def test_native_default_state_and_prediction_remain_exact(self):
        torch.manual_seed(0)
        expected = ScreeningPredictor(copy.deepcopy(self.configs["vq"]), T_max=10).eval()
        actual = self.model("baseline").eval()
        self.assertEqual(expected.state_dict().keys(), actual.state_dict().keys())
        with torch.no_grad():
            left = expected(self.parts.stock_feature, self.parts.prior_factor)
            right = actual(self.parts.stock_feature, self.parts.prior_factor)
        for a, b in zip(left, right):
            self.assertTrue(torch.equal(a, b))

    def test_coordinate_api_preserves_native_quantization_exactly(self):
        torch.manual_seed(3)
        original = FiniteScalarQuantiser([8, 8, 8], 8).eval()
        extended = CoordinateFSQ.from_frozen(original)
        h = torch.randn(17, 8)
        a, b = original(h), extended(h)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[2][2], b[2][2]))
        u, q = extended.coordinates(h)
        self.assertTrue(((u - q).abs() <= 0.5).all())
        self.assertTrue(torch.equal((q.long() * extended.basis).sum(-1), a[2][2]))
        self.assertFalse(any(p.requires_grad for p in extended.parameters()))

    def test_gate_operates_before_output_projection(self):
        torch.manual_seed(0)
        native = RotaryTransformerEncoderLayer(8, 2, dim_feedforward=16, dropout=0).eval()
        gated = copy.deepcopy(native)
        gated.head_gate = nn.Linear(8, 2)
        nn.init.zeros_(gated.head_gate.weight)
        nn.init.zeros_(gated.head_gate.bias)
        seen = []
        handles = [layer.out_proj.register_forward_pre_hook(lambda _, inputs: seen.append(inputs[0].detach().clone()))
                   for layer in (native, gated)]
        x = torch.randn(3, 6, 8)
        native(x)
        loss = gated(x).square().mean()
        loss.backward()
        for handle in handles:
            handle.remove()
        self.assertTrue(torch.equal(seen[1], seen[0] * 0.5))
        self.assertGreater(float(gated.head_gate.weight.grad.abs().sum()), 0)

    def test_dimensional_adapter_starts_at_native_fsq_prediction(self):
        torch.manual_seed(0)
        expected = ScreeningPredictor(copy.deepcopy(self.configs["fsq"]), T_max=10).eval()
        actual = self.model("fsq_ca_dim").eval()
        self.assertEqual(actual.quantization_confidence_adapter.in_features, 3)
        self.assertEqual(actual.quantization_confidence_adapter.out_features, 8)
        with torch.no_grad():
            a = expected(self.parts.stock_feature, self.parts.prior_factor)[0]
            b = actual(self.parts.stock_feature, self.parts.prior_factor)[0]
        self.assertTrue(torch.equal(a, b))

    def test_direct_grid_has_three_factors_and_bypasses_from_grid(self):
        model = self.model("fsq_grid").eval()
        with mock.patch.object(model.quantizer.from_grid, "forward", side_effect=AssertionError("from_grid used")):
            result = model(self.parts.stock_feature, self.parts.prior_factor)
        self.assertEqual(result[2].shape, (7, 3))
        self.assertEqual(result[3].shape, (7, 3))
        self.assertEqual(model.encoder.final_embed_dim_d if hasattr(model.encoder, "final_embed_dim_d") else model.vq_embed_dim, 8)
        self.assertEqual(model.loadings.temporal_transformer.z_q_proj.in_features, 3)
        self.assertEqual(model.latent_value_head.head[0].in_features, 3)
        self.assertEqual(model.loadings.fusion.moe.gate_input_size, 3)

    def test_every_new_model_backpropagates_without_updating_stage1(self):
        for variant in ("gated_attn", "fsq_wo_ca", "fsq_ca_dim", "fsq_grid"):
            with self.subTest(variant=variant):
                model = self.model(variant).train()
                frozen = {k: t.clone() for k, t in model.state_dict().items()
                          if k.startswith(("encoder.", "quantizer.", "revin."))}
                result = model(self.parts.stock_feature, self.parts.prior_factor)
                loss, _ = model.training_loss(self.parts, result)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                for name in ("encoder", "quantizer", "revin"):
                    module = getattr(model, name)
                    self.assertFalse(module.training)
                    self.assertFalse(any(p.requires_grad for p in module.parameters()))
                    self.assertTrue(all(p.grad is None for p in module.parameters()))
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
                optimizer.step()
                self.assertTrue(all(torch.equal(t, model.state_dict()[k]) for k, t in frozen.items()))

    def test_strict_state_restoration_for_changed_shapes(self):
        for variant in ("gated_attn", "fsq_ca_dim", "fsq_grid"):
            with self.subTest(variant=variant):
                a = self.model(variant).eval()
                b = self.model(variant).eval()
                b.load_state_dict(a.state_dict(), strict=True)
                with torch.no_grad():
                    expected = a(self.parts.stock_feature, self.parts.prior_factor)[0]
                    actual = b(self.parts.stock_feature, self.parts.prior_factor)[0]
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
