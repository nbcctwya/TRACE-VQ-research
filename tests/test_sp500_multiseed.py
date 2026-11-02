"""Contracts for the fixed SP500 baseline and explicit per-stage seeds."""
from pathlib import Path
import copy
import json
import tempfile
import unittest
from trace_vq.cli.sp500_multiseed import make_config
from research.run_sp500_multiseed import check_backtest,ROOT


class SP500MultiseedTests(unittest.TestCase):
    def test_market_preset_preserves_trace_mechanisms(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=make_config(Path(tmp)/"run",Path(tmp)/"data",0,Path(tmp)/"stage1.ckpt")
            self.assertEqual(cfg["data"]["universe"],"sp500")
            self.assertEqual(cfg["vqvae"]["encoder"]["type"],"temporal-attention")
            self.assertTrue(cfg["predictor"]["shared_expert"])
            self.assertTrue(cfg["predictor"]["quantization_confidence_adapter"])
            self.assertEqual(cfg["predictor"]["n_expert"],8)
            self.assertEqual(cfg["predictor"]["k"],4)
            self.assertEqual(cfg["predictor"]["transformer"]["num_heads"],4)
            self.assertEqual(cfg["predictor"]["aux_weight"],.001)
            self.assertEqual(cfg["research"],{"variant":"baseline"})
            self.assertNotIn("research_kind",cfg["vqvae"]["quantizer"])

    def test_five_configs_only_change_stage2_seed_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            configs=[make_config(Path(tmp)/f"seed{s}",Path(tmp)/"data",s,Path(tmp)/"stage1.ckpt") for s in range(5)]
            for seed,cfg in enumerate(configs):
                self.assertEqual(cfg["train"]["seed"],seed)
                for key in ["vqvae","predictor","data","research"]:
                    self.assertEqual(cfg[key],configs[0][key])
                self.assertEqual(cfg["train"]["num_epochs"],70)
                self.assertEqual(cfg["train"]["early_stopping"]["patience"],15)
                self.assertTrue(cfg["train"]["save_dir"].startswith(str(Path(tmp)/f"seed{seed}")))

    def test_stage1_seed42_and_fixed_periods(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=make_config(Path(tmp),Path(tmp)/"data",42)
            self.assertEqual(cfg["train"]["seed"],42)
            self.assertEqual(cfg["predictor"]["saved_model"],"NOT_USED_DURING_STAGE1")
            self.assertEqual(cfg["data"]["train_period"],["2009-01-01","2020-12-31"])
            self.assertEqual(cfg["data"]["valid_period"],["2021-01-01","2022-12-31"])
            self.assertEqual(cfg["data"]["test_period"],["2023-01-01","2025-12-31"])
            self.assertEqual(cfg["predictor"]["target_day"],5)

    def test_unspecified_seeds_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in [-1,5,43]:
                with self.assertRaises(ValueError):make_config(Path(tmp),Path(tmp),seed)

    def test_unified_protocol_schema_and_market(self):
        fixture=ROOT/"artifacts/research/multiseed_validation_20261030/csi300/baseline/seed1/backtest/backtest_config.json"
        cfg=json.loads(fixture.read_text());cfg.update(universe="sp500",benchmark="^gspc")
        check_backtest(cfg)
        for path,value in [("quote_universe","sp500"),("account",10000),("benchmark","SH000300")]:
            altered=copy.deepcopy(cfg);altered[path]=value
            with self.assertRaises(AssertionError):check_backtest(altered)
        altered=copy.deepcopy(cfg);altered["exchange"]["close_cost"]=.0005
        with self.assertRaises(AssertionError):check_backtest(altered)


if __name__=="__main__":unittest.main()
