"""Check SP500 parity and paired initialization before the expensive fits."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

from trace_vq.cli.screening import initialize
from trace_vq.cli.sp500_candidates import BASELINE, VARIANTS, cache_path, common_config, make_config, tasks
from trace_vq.models.combination_screen import settings
from trace_vq.models.representation import VQVAE


class SP500CandidateTests(unittest.TestCase):
    def test_actual_baseline_shared_hyperparameters_for_every_fit(self):
        baseline = OmegaConf.to_container(OmegaConf.load(BASELINE / "seed0/effective.config.yaml"), resolve=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for variant, seed in tasks():
                cfg = make_config(root, variant, seed)
                self.assertEqual(common_config(cfg), common_config(baseline))
                self.assertEqual(cfg["research"], settings(variant))
                self.assertEqual(cfg["train"]["seed"], seed)
                self.assertTrue(cfg["train"]["save_dir"].startswith(str(root / variant / f"seed{seed}")))
                self.assertEqual(cfg["predictor"]["n_expert"], 8)
                self.assertEqual(cfg["predictor"]["k"], 4)
                self.assertEqual(cfg["predictor"]["transformer"]["num_heads"], 4)
                self.assertEqual(cfg["predictor"]["aux_weight"], .001)
                self.assertEqual(cfg["inference"]["matmul_precision"], "high")

    def test_fsq_and_combination_share_stage1_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a, b = make_config(root, "fsq", 0), make_config(root, "fsq_ranknet", 4)
            self.assertEqual(a["predictor"]["saved_model"], b["predictor"]["saved_model"])
            self.assertEqual(a["vqvae"], b["vqvae"])
            self.assertEqual(a["vqvae"]["quantizer"]["research_levels"], [8, 8, 8])
            for split in ("train", "valid", "test"):
                self.assertEqual(cache_path(root, "fsq", split), cache_path(root, "fsq_ranknet", split))
                self.assertEqual(cache_path(root, "ranknet", split), BASELINE / "cache" / f"{split}.npy")
            native = make_config(root, "ranknet", 0)
            self.assertTrue(Path(native["predictor"]["saved_model"]).is_file())
            self.assertNotIn("research_kind", native["vqvae"]["quantizer"])

    def test_stage1_preserves_actual_baseline_config_and_common_initialization(self):
        torch.set_num_threads(1)
        baseline = OmegaConf.to_container(OmegaConf.load(BASELINE / "stage1/effective.config.yaml"), resolve=True)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = make_config(Path(tmp), "fsq", 42, stage1=True)
            self.assertEqual(common_config(cfg), common_config(baseline))
            self.assertEqual(cfg["train"]["seed"], 42)
            initialize(42)
            native = VQVAE(copy.deepcopy(baseline))
            initialize(42)
            fsq = VQVAE(copy.deepcopy(cfg))
            for name, tensor in native.state_dict().items():
                if not name.startswith("quantizer."):
                    self.assertTrue(torch.equal(tensor, fsq.state_dict()[name]), name)
            x = torch.randn(11, 128, requires_grad=True)
            quantized, q_loss, (_, _, indices) = fsq.quantizer(x)
            self.assertEqual(tuple(quantized.shape), (11, 128))
            self.assertTrue(bool(((indices >= 0) & (indices < 512)).all()))
            (quantized.square().mean() + q_loss).backward()
            self.assertGreater(float(x.grad.abs().sum()), 0)

    def test_exact_queue_and_reject_unplanned_seeds(self):
        self.assertEqual(len(tasks()), 15)
        self.assertEqual(len(set(tasks())), 15)
        self.assertEqual(set(tasks()), {(v, s) for v in VARIANTS for s in range(5)})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for seed in (-1, 5, 42):
                with self.assertRaises(ValueError):
                    make_config(root, "fsq", seed)
            with self.assertRaises(ValueError):
                make_config(root, "ranknet", 42, stage1=True)
            with self.assertRaises(ValueError):
                make_config(root, "unplanned", 0)


if __name__ == "__main__":
    unittest.main()
