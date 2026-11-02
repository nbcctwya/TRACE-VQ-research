"""Trainable ablations, historical parameter paths, and strict restoration."""

import copy
import importlib
import os
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import torch
from torch import nn

from trace_vq.models.temporal.gru import GRUSpatialEncoder
from trace_vq.training.prediction import GenerateReturn
from trace_vq.training.representation import FactorVQVAE

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "full": ("temporal-attention", True, True),
    "wo_ta": ("gru", True, True),
    "wo_ca": ("temporal-attention", False, True),
    "wo_sr": ("temporal-attention", True, False),
}


def configuration(variant):
    if not OmegaConf.has_resolver("half"):
        OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "configs")):
        cfg = compose(config_name="config", overrides=[f"experiment={variant}"])
    config = OmegaConf.to_container(cfg, resolve=True)
    config["vqvae"].update(num_features=8, hidden_size=8, vq_embed_dim=8, num_embed=16, num_prior_factors=3)
    config["vqvae"]["decoder"]["hidden_channels"] = 8
    config["predictor"]["num_features"] = 8
    config["predictor"]["moe_hidden"] = 8
    config["predictor"]["transformer"].update(d_model=8, dim_feedforward=16)
    return config


class AblationTests(unittest.TestCase):
    def test_hydra_variants_select_the_historical_mechanisms(self):
        for variant, (kind, ca, sr) in VARIANTS.items():
            with self.subTest(variant=variant):
                cfg = configuration(variant)
                self.assertEqual(cfg["vqvae"]["encoder"]["type"], kind)
                self.assertEqual(cfg["predictor"]["quantization_confidence_adapter"], ca)
                self.assertEqual(cfg["predictor"]["shared_expert"], sr)
                self.assertEqual(cfg["data"]["data_path"], os.environ.get(
                    "TRACE_VQ_DATA_ROOT", "../HVQ-Stock/dataset/processed"))
                self.assertIn("artifacts/new_runs/", cfg["artifact_root"])

    def test_environment_paths_preserve_all_model_hyperparameters(self):
        paths = {"TRACE_VQ_DATA_ROOT": "/shared/processed",
                 "TRACE_VQ_STAGE1_TA_CHECKPOINT": "/weights/temporal-attention.ckpt",
                 "TRACE_VQ_STAGE1_GRU_CHECKPOINT": "/weights/gru.ckpt"}
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                expected = configuration(variant)
                expected["data"]["data_path"] = paths["TRACE_VQ_DATA_ROOT"]
                kind = "GRU" if variant == "wo_ta" else "TA"
                expected["predictor"]["saved_model"] = paths[f"TRACE_VQ_STAGE1_{kind}_CHECKPOINT"]
                with mock.patch.dict(os.environ, paths):
                    self.assertEqual(configuration(variant), expected)

    def test_every_variant_trains_freezes_and_strictly_restores(self):
        for variant, (kind, ca, sr) in VARIANTS.items():
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                torch.manual_seed(42)
                cfg = configuration(variant)
                cfg["predictor"]["saved_model"] = str(Path(tmp) / "stage1.ckpt")
                first = FactorVQVAE(copy.deepcopy(cfg), T_max=2)
                feature = torch.randn(6, 20, 8)
                prior = torch.randn(6, 3)
                returns = torch.randn(6, 10)
                loss = first(feature, prior, returns)[3]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                self.assertTrue(any(p.grad is not None for p in first.vqvae.spatial_encoder.parameters()))
                torch.save({"state_dict": first.state_dict()}, cfg["predictor"]["saved_model"])
                original_config = copy.deepcopy(cfg)
                factory = (GenerateReturn if variant == "full" else
                           importlib.import_module(f"ablation.{variant}").build_model)
                model = factory(copy.deepcopy(cfg) if variant == "full" else cfg, T_max=2)
                self.assertEqual(cfg, original_config)
                keys = set(model.state_dict())
                self.assertEqual(any(k.startswith("quantization_confidence_adapter.") for k in keys), ca)
                self.assertEqual(any("shared_expert.net." in k for k in keys), sr)
                self.assertEqual(isinstance(model.encoder, GRUSpatialEncoder), kind == "gru")
                if kind == "gru":
                    self.assertIsInstance(model.encoder.temporal_extractor.get_h, nn.GRU)
                    self.assertIn("encoder.temporal_extractor.get_h.weight_ih_l0", keys)
                    self.assertFalse(any(k.startswith("encoder.temporal_encoder.") for k in keys))
                frozen = {name: value.clone() for name, value in model.encoder.state_dict().items()}
                codebook = model.quantizer.embedding.weight.detach().clone()
                model.train()
                prediction, _, _, _, aux = model(feature, prior)
                total = model.rank_loss(prediction, returns[:, 4]) + model.aux_weight * aux
                total.backward()
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                optimizer.step()
                self.assertTrue(torch.isfinite(total))
                self.assertTrue(torch.equal(codebook, model.quantizer.embedding.weight))
                self.assertFalse(model.encoder.training)
                self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
                for name, value in model.encoder.state_dict().items():
                    self.assertTrue(torch.equal(value, frozen[name]))
                if ca:
                    self.assertGreater(model.quantization_confidence_adapter.weight.grad.abs().sum().item(), 0)
                if sr:
                    self.assertGreater(model.loadings.fusion.moe.shared_expert.net[-1].weight.grad.abs().sum().item(), 0)
                restored = GenerateReturn(copy.deepcopy(cfg), T_max=2)
                restored.load_state_dict(model.state_dict(), strict=True)
                model.eval()
                restored.eval()
                with torch.no_grad():
                    self.assertTrue(torch.equal(model(feature, prior)[0], restored(feature, prior)[0]))

    def test_gru_weights_cannot_silently_load_into_the_attention_encoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = configuration("wo_ta")
            checkpoint = Path(tmp) / "gru.ckpt"
            first = FactorVQVAE(copy.deepcopy(cfg), T_max=1)
            torch.save({"state_dict": first.state_dict()}, checkpoint)
            attention = configuration("full")
            attention["predictor"]["saved_model"] = str(checkpoint)
            with self.assertRaises(RuntimeError):
                GenerateReturn(attention, T_max=1)
