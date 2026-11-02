"""Synthetic two-stage checkpoint and standalone evaluation integration test."""

import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from trace_vq.data.market import CanonicalSampler, market_feature_config
from trace_vq.data.loaders import init_data_loader
from trace_vq.data.schema import GROUP_DIMS, MARKET_INDICES, TOTAL_DIM, dataset_basename, unpack_batch
from trace_vq.training.representation import FactorVQVAE
from trace_vq.training.prediction import GenerateReturn

ROOT = Path(__file__).resolve().parents[1]


class StandaloneTest(unittest.TestCase):
    def test_two_stage_training_checkpoint_and_evaluation_cli(self):
        torch.manual_seed(42)
        if not OmegaConf.has_resolver("half"):
            OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)
        cfg = OmegaConf.load(ROOT / "configs/config.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg.predictor.saved_model = str(root / "stage1.ckpt")
            cfg.train.save_dir = str(root)
            config = OmegaConf.to_container(cfg, resolve=True)
            batch = torch.randn(8, 20, TOTAL_DIM)
            parts = unpack_batch(batch)
            first = FactorVQVAE(config, T_max=2)
            optimizer = torch.optim.AdamW(first.parameters(), lr=1e-4)
            loss = first(parts.stock_feature, parts.prior_factor, parts.future_returns)[3]
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertTrue(any(p.grad is not None for p in first.vqvae.spatial_encoder.parameters()))
            optimizer.step()
            torch.save({"state_dict": first.state_dict()}, cfg.predictor.saved_model)

            second = GenerateReturn(config, T_max=2)
            frozen = second.quantizer.embedding.weight.detach().clone()
            optimizer = torch.optim.AdamW([p for p in second.parameters() if p.requires_grad], lr=1e-4)
            second.train()
            prediction, _, _, _, aux = second(parts.stock_feature, parts.prior_factor)
            loss = second.rank_loss(prediction, parts.target(5)) + second.aux_weight * aux
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertGreater(second.quantization_confidence_adapter.weight.grad.abs().sum().item(), 0)
            optimizer.step()
            self.assertTrue(torch.equal(frozen, second.quantizer.embedding.weight))
            checkpoint = root / "stage2.ckpt"
            torch.save({"state_dict": second.state_dict(), "pytorch-lightning_version": pl.__version__}, checkpoint)
            restored = GenerateReturn.load_from_checkpoint(checkpoint, config=config, T_max=2, strict=True)
            second.eval()
            restored.eval()
            with torch.no_grad():
                self.assertTrue(torch.equal(second(parts.stock_feature, parts.prior_factor)[0],
                                            restored(parts.stock_feature, parts.prior_factor)[0]))

            dates = pd.bdate_range("2023-01-02", periods=25, name="datetime")
            index = pd.MultiIndex.from_product([dates, [f"S{i}" for i in range(8)]],
                                               names=["datetime", "instrument"])
            columns = pd.MultiIndex.from_tuples([(g, f"{g}_{i}") for g, n in GROUP_DIMS.items() for i in range(n)])
            rng = np.random.default_rng(7)
            frame = pd.DataFrame(rng.normal(size=(len(index), TOTAL_DIM)), index=index, columns=columns)
            market = pd.DataFrame(rng.normal(size=(len(dates), 63)), index=dates,
                                  columns=pd.MultiIndex.from_product([["market"], market_feature_config(MARKET_INDICES["csi300"])[1]]))
            sampler = CanonicalSampler(frame, market, "csi300", dates[20], dates[-1])
            (root / "CN").mkdir()
            config_file = root / "effective.yaml"
            loader, _ = init_data_loader(sampler, shuffle=False, num_workers=0)
            for use_prior in (True, False):
                with self.subTest(use_prior=use_prior):
                    cfg.predictor.use_prior = use_prior
                    second.return_predictor.use_prior = use_prior
                    OmegaConf.save(cfg, config_file)
                    scores, labels = [], []
                    with torch.no_grad():
                        for daily_batch in loader:
                            daily = unpack_batch(daily_batch.float())
                            scores.append(second(daily.stock_feature, daily.prior_factor)[0].numpy())
                            labels.append(daily.target(cfg.predictor.target_day).numpy())
                    expected = pd.DataFrame({"score": np.concatenate(scores), "label": np.concatenate(labels)},
                                            index=sampler.get_index().sortlevel(0)[0])
                    for split in ("valid", "test"):
                        with (root / "CN" / f"{dataset_basename('csi300')}_{split}.pkl").open("wb") as stream:
                            pickle.dump(sampler, stream)
                        result = subprocess.run([
                            sys.executable, str(ROOT / "evaluate.py"), "--checkpoint", str(checkpoint),
                            "--stage1-checkpoint", cfg.predictor.saved_model, "--data-path", str(root),
                            "--config", str(config_file), "--split", split, "--output-dir", str(root / "evaluation"),
                        ], cwd=tmp, env=dict(os.environ, OMP_NUM_THREADS="1"), capture_output=True, text=True)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        predictions = pd.read_pickle(root / "evaluation" / f"{split}_predictions.pkl")
                        pd.testing.assert_frame_equal(predictions, expected, check_exact=True)
                        self.assertTrue(np.isfinite(predictions.to_numpy()).all())
                        metrics = pd.read_csv(root / "evaluation" / f"{split}_metrics.csv", index_col=0)
                        self.assertEqual(set(metrics.index), {"IC", "ICIR", "RankIC", "RankICIR"})
