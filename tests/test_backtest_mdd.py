"""Regression tests: backtest MDD must include the initial NAV=1 in the peak.

Previously ``_paper_metrics`` computed drawdown from ``wealth.cummax()``
alone, so a first-day loss (or any period where wealth never exceeded 1)
produced MDD = 0 instead of the real drawdown.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from trace_vq.evaluation.backtest import _paper_metrics


def paper_metrics(returns):
    return _paper_metrics(pd.Series(returns), None, "x")["x"]


def expected_mdd(returns):
    g = np.log1p(np.asarray(returns, dtype=float))
    wealth = np.exp(np.cumsum(g))
    peak = np.maximum(np.maximum.accumulate(wealth), 1.0)
    return float(np.min(wealth / peak - 1.0))


class BacktestMddTest(unittest.TestCase):
    def test_single_day_loss_is_drawdown(self):
        self.assertAlmostEqual(paper_metrics([-0.10])["MDD"], -0.10, places=4)

    def test_single_day_gain_has_zero_mdd(self):
        self.assertAlmostEqual(paper_metrics([0.10])["MDD"], 0.0, places=4)

    def test_gain_then_loss_uses_initial_nav_peak(self):
        # wealth path: 1 -> 1.1 -> 0.99; max drawdown is 0.99/1.1 - 1 = -0.1
        self.assertAlmostEqual(paper_metrics([0.10, -0.10])["MDD"], -0.10, places=4)

    def test_multi_day_series_matches_independent_recalculation(self):
        r = [0.01, -0.02, 0.03, -0.05, 0.02, -0.01]
        self.assertAlmostEqual(paper_metrics(r)["MDD"], expected_mdd(r), places=4)
        # sanity: this series dips below the initial NAV, so MDD must be < 0
        self.assertLess(paper_metrics(r)["MDD"], 0.0)

    def test_calmar_uses_corrected_mdd(self):
        m = paper_metrics([-0.10])
        self.assertAlmostEqual(
            m["Calmar Ratio"], m["Annualized Return"] / abs(m["MDD"]), places=3
        )

    def test_sortino_uses_downside_rms_over_all_trading_days(self):
        # One loss still defines downside deviation. Positive/zero days also
        # belong in its denominator; a std of loss days alone is undefined.
        log_returns = np.array([0.03, -0.02, 0.01, 0.0])
        downside_rms = np.sqrt((0.02**2) / 4)
        expected = np.sqrt(252) * log_returns.mean() / downside_rms
        observed = paper_metrics(np.expm1(log_returns))["Sortino Ratio"]
        self.assertAlmostEqual(observed, expected, places=4)

    def test_sortino_is_undefined_without_downside(self):
        self.assertTrue(np.isnan(paper_metrics([0.01, 0.02, 0.0])["Sortino Ratio"]))


if __name__ == "__main__":
    unittest.main()
