"""Migration regressions for explicit checkpoints and project-local paths."""

import os
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

import stage1
import stage2

ROOT = Path(__file__).resolve().parents[1]


class EntrypointTests(unittest.TestCase):
    def test_universe_presets_preserve_explicit_checkpoint_and_overrides(self):
        for universe in ("csi300", "sp500"):
            cfg = OmegaConf.load(ROOT / "configs/config.yaml")
            cfg.data.universe = universe
            cfg.predictor.saved_model = "/tmp/custom-epoch=1.ckpt"
            cfg.predictor.n_expert = 4
            stage2._apply_stage2_universe_preset(
                cfg, ["predictor.saved_model='/tmp/custom-epoch=1.ckpt'", "predictor.n_expert=4"]
            )
            self.assertEqual(cfg.predictor.saved_model, "/tmp/custom-epoch=1.ckpt")
            self.assertEqual(cfg.predictor.n_expert, 4)
            expected_heads = 2 if universe == "csi300" else 4
            self.assertEqual(cfg.predictor.transformer.num_heads, expected_heads)

    def test_relative_data_paths_are_independent_of_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                self.assertEqual(stage1._resolve_data_path("dataset/processed"), ROOT / "dataset/processed")
                self.assertEqual(Path(stage2._to_absolute_path("dataset/processed")), ROOT / "dataset/processed")
            finally:
                os.chdir(previous)

    def test_stage2_requires_explicit_stage1_checkpoint(self):
        cfg = OmegaConf.load(ROOT / "configs/config.yaml")
        stage2._apply_stage2_universe_preset(cfg, [])
        self.assertTrue(OmegaConf.is_missing(cfg.predictor, "saved_model"))
