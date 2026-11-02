"""Check combination loss semantics, initialization controls and Stage1 reuse."""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from omegaconf import OmegaConf

from trace_vq.data.schema import TOTAL_DIM
from trace_vq.models.combination_screen import COMBINATIONS, family, settings
from trace_vq.models.next_screen import NextPredictor, ranknet_objective
from trace_vq.models.screening import masked_mse, pairwise_objective
from trace_vq.training.representation import FactorVQVAE

ROOT = Path(__file__).resolve().parents[1]


class CombinationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        if not OmegaConf.has_resolver("half"):
            OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)
        cfg = OmegaConf.load(ROOT / "configs/config.yaml")
        cfg.vqvae.hidden_size = 8; cfg.vqvae.vq_embed_dim = 8; cfg.vqvae.num_embed = 16
        cfg.vqvae.decoder.hidden_channels = 8
        cfg.predictor.moe_hidden = 8; cfg.predictor.transformer.d_model = 8
        cfg.predictor.transformer.dim_feedforward = 16
        cls.base = OmegaConf.to_container(cfg, resolve=True)
        cls.paths = {}
        for kind in ["native", "fsq"]:
            c = copy.deepcopy(cls.base)
            if kind == "fsq":
                c["vqvae"]["quantizer"].update(research_kind="fsq", research_levels=[4, 4])
            torch.manual_seed(42)
            model = FactorVQVAE(c, T_max=2)
            path = Path(cls.temporary.name) / (kind + ".ckpt")
            torch.save({"state_dict": model.state_dict()}, path)
            cls.paths[kind] = str(path)
        torch.manual_seed(9)
        cls.batch = torch.randn(7, 20, TOTAL_DIM)
        cls.batch[:, :, 171:234] = cls.batch[:1, :, 171:234]

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self, variant):
        cfg = copy.deepcopy(self.base)
        cfg["research"] = settings(variant)
        kind = family(variant)
        if kind == "fsq":
            cfg["vqvae"]["quantizer"].update(research_kind="fsq", research_levels=[4, 4])
        cfg["predictor"]["saved_model"] = self.paths[kind]
        return cfg

    def model(self, variant):
        torch.manual_seed(0)
        return NextPredictor(self.config(variant), T_max=2)

    def test_real_config_reuses_stage1_and_preserves_split_and_training_protocol(self):
        from trace_vq.cli.combination_screen import make_config, reference_directory
        for market in ["csi300", "csi800"]:
            for variant in COMBINATIONS:
                args = SimpleNamespace(universe=market, variant=variant, num_workers=2,
                                       output_dir=Path(self.temporary.name) / market / variant)
                cfg = make_config(args)
                base = "fsq" if family(variant) == "fsq" else "baseline"
                old = OmegaConf.to_container(OmegaConf.load(reference_directory(market, base) / "effective.config.yaml"), resolve=True)
                self.assertEqual(cfg["vqvae"], old["vqvae"])
                self.assertEqual(cfg["predictor"], old["predictor"])
                self.assertEqual(cfg["data"], old["data"])
                self.assertEqual(cfg["train"]["seed"], 0)
                self.assertEqual(cfg["train"]["num_epochs"], 70)
                self.assertEqual(cfg["train"]["early_stopping"], old["train"]["early_stopping"])
                self.assertTrue(cfg["train"]["save_dir"].startswith(str(args.output_dir)))

    def test_every_archived_reference_has_matching_completed_artifacts(self):
        import json
        from trace_vq.cli.combination_screen import reference_directory
        from trace_vq.models.combination_screen import REFERENCES
        for market in ["csi300", "csi800"]:
            for variant in REFERENCES:
                folder = reference_directory(market, variant)
                record = json.loads((folder / "completion.json").read_text())
                self.assertEqual(record["status"], "complete")
                self.assertEqual(record["variant"], variant)
                self.assertTrue(Path(record["checkpoint"]).is_file())
                for name in ["effective.config.yaml", "evaluation/valid_predictions.pkl",
                             "evaluation/test_predictions.pkl", "backtest/portfolio_return.csv"]:
                    self.assertTrue((folder / name).is_file(), (market, variant, name))

    def test_combined_loss_keeps_original_mse_auxiliary_and_exactly_one_rank_loss(self):
        for variant in COMBINATIONS:
            model = self.model(variant).train()
            parts, result = model.batch_prediction(self.batch)
            loss, _ = model.training_loss(parts, result)
            expected = masked_mse(result[0], parts.target(5)) + model.aux_weight * result[4]
            if "ranknet" in variant:
                expected = expected + .1 * ranknet_objective(result[0], parts.target(5))
            elif "pairwise" in variant:
                expected = expected + .1 * pairwise_objective(result[0], parts.target(5))
            self.assertTrue(torch.equal(loss, expected), variant)
            loss.backward()
            self.assertTrue(all(p.grad is None for n, p in model.named_parameters()
                                if n.startswith(("encoder.", "quantizer.", "revin."))))
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_zero_initialized_market_branch_keeps_reference_scores_and_rng(self):
        for baseline, variant in [("baseline", "route_ranknet"), ("fsq", "fsq_route_ranknet")]:
            reference = self.model(baseline).eval(); before = torch.get_rng_state()
            combined = self.model(variant).eval(); after = torch.get_rng_state()
            self.assertTrue(torch.equal(before, after))
            with torch.no_grad():
                a = reference.batch_prediction(self.batch)[1]
                b = combined.batch_prediction(self.batch)[1]
            self.assertTrue(torch.equal(a[0], b[0]), variant)
            self.assertTrue(torch.equal(a[4], b[4]), variant)

    def test_zero_rank_weight_restores_matching_route_loss_and_all_gradients(self):
        for variant, weight in [("route_ranknet", "ranknet_weight"), ("route_pairwise", "pairwise_weight")]:
            reference = self.model("market_route").train()
            cfg = self.config(variant); cfg["research"][weight] = 0
            torch.manual_seed(0); zero = NextPredictor(cfg, T_max=2).train()
            rng = torch.get_rng_state()
            a = reference.training_loss(*reference.batch_prediction(self.batch))[0]; a.backward()
            torch.set_rng_state(rng)
            b = zero.training_loss(*zero.batch_prediction(self.batch))[0]; b.backward()
            self.assertTrue(torch.equal(a, b), variant)
            for (name, x), (other, y) in zip(reference.named_parameters(), zero.named_parameters()):
                self.assertEqual(name, other)
                if x.grad is None:
                    self.assertIsNone(y.grad)
                else:
                    self.assertTrue(torch.equal(x.grad, y.grad), (variant, name))


if __name__ == "__main__":
    unittest.main()
