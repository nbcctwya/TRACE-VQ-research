"""Paired protocol configuration and uninterrupted-versus-resumed RNG checks."""
import copy
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset

from trace_vq.cli.batch02 import make_config, reference_directory, settings, VARIANTS
from trace_vq.cli.screening import initialize
from trace_vq.data.frozen import DailyPositions
from trace_vq.training.research_run import RecoverableEpochRecord, capture_rng, restore_rng


class TinyDays(Dataset):
    def __init__(self):
        self.index = pd.MultiIndex.from_product([pd.date_range("2020-01-01", periods=4), list("ABC")],
                                               names=["datetime", "instrument"])
        self.values = torch.arange(len(self.index)*4).reshape(-1, 4).float()/17

    def __len__(self):
        return len(self.index)

    def get_index(self):
        return self.index

    def __getitem__(self, index):
        return self.values[index]


class TinyPredictor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = torch.nn.Sequential(torch.nn.Linear(3, 5), torch.nn.Dropout(.2), torch.nn.Linear(5, 1))

    def training_step(self, batch, batch_idx):
        loss = (self.network(batch[:, :3]).squeeze(-1)-batch[:, 3]).square().mean()
        return loss + self.network[0].weight.mean()*(torch.rand((), device=self.device)+np.random.rand()+random.random())*.001

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", (self.network(batch[:, :3]).squeeze(-1)-batch[:, 3]).square().mean(),
                 on_epoch=True, batch_size=len(batch))

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.01)


class StopFirstEpoch(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = trainer.current_epoch == 0


class Batch02Tests(unittest.TestCase):
    def test_exact_existing_research_settings_and_stage1_configs(self):
        for market in ("csi300", "csi800"):
            for variant in VARIANTS:
                source = "fsq" if variant.startswith("fsq") else variant
                old = OmegaConf.to_container(OmegaConf.load(reference_directory(market, source)/"effective.config.yaml"),
                                             resolve=True)
                for seed in range(5):
                    cfg = make_config(market, variant, seed, Path(tempfile.gettempdir())/"batch02_config_test")
                    self.assertEqual(cfg["vqvae"], old["vqvae"])
                    self.assertEqual(cfg["predictor"], old["predictor"])
                    self.assertEqual(cfg["data"], old["data"])
                    self.assertEqual(cfg["train"]["seed"], seed)
                    self.assertEqual(cfg["research"]["pairwise_weight"], .1)
                    if not variant.startswith("fsq"):
                        self.assertEqual(cfg["research"], old["research"])

    def test_matched_control_differs_only_in_variant_name_and_penalty_weight(self):
        for prefix in ("", "fsq_"):
            a, b = settings(prefix+"turnover"), settings(prefix+"turnover_control")
            for key in ("variant", "turnover_weight"):
                a.pop(key); b.pop(key)
            self.assertEqual(a, b)

    def test_rng_roundtrip(self):
        initialize(4)
        state = capture_rng()
        expected = (random.random(), np.random.rand(), torch.rand(5))
        restore_rng(state)
        self.assertEqual(random.random(), expected[0])
        self.assertEqual(np.random.rand(), expected[1])
        self.assertTrue(torch.equal(torch.rand(5), expected[2]))

    def test_epoch_checkpoint_resume_is_exact_with_shuffled_days_and_workers(self):
        torch.set_num_threads(1)
        def run(folder, device, stop=False, resume=False):
            initialize(17)
            model = TinyPredictor()
            data = TinyDays()
            train = DataLoader(data, sampler=DailyPositions(data, True), batch_size=None, num_workers=2)
            valid = DataLoader(data, sampler=DailyPositions(data), batch_size=None, num_workers=2)
            record = RecoverableEpochRecord(folder, device)
            best = ModelCheckpoint(dirpath=folder/"checkpoints", monitor="val_loss", save_last=True, save_top_k=1)
            callbacks = [record] + ([StopFirstEpoch()] if stop else []) + [best]
            fit = pl.Trainer(max_epochs=3, accelerator="gpu" if device == "cuda" else "cpu", devices=1,
                             logger=False, callbacks=callbacks,
                             enable_progress_bar=False, enable_model_summary=False, deterministic=True,
                             default_root_dir=str(folder))
            fit.fit(model, train_dataloaders=train, val_dataloaders=valid,
                    ckpt_path=str(folder/"checkpoints/last.ckpt") if resume else None)
            return copy.deepcopy(model.state_dict()), capture_rng(device), record.rows
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
                with self.subTest(device=device):
                    full, split = root/device/"full", root/device/"split"
                    full.mkdir(parents=True); split.mkdir(parents=True)
                    expected, rng_expected, rows_expected = run(full, device)
                    run(split, device, stop=True)
                    actual, rng_actual, rows_actual = run(split, device, resume=True)
                    for key in expected:
                        self.assertTrue(torch.equal(actual[key], expected[key]), key)
                    self.assertTrue(torch.equal(rng_actual["torch"], rng_expected["torch"]))
                    self.assertEqual(rng_actual["python"], rng_expected["python"])
                    self.assertTrue(np.array_equal(rng_actual["numpy"][1], rng_expected["numpy"][1]))
                    if device == "cuda":
                        self.assertTrue(all(torch.equal(a,b) for a,b in zip(rng_actual["cuda"], rng_expected["cuda"])))
                    self.assertEqual([r["val_loss"] for r in rows_actual], [r["val_loss"] for r in rows_expected])


if __name__ == "__main__":
    unittest.main()
