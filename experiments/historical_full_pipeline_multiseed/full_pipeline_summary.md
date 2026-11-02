# Reviewer 2 — Full-pipeline 5-seed experiment (TRACE-VQ / HVQ 034)

Protocol: for each seed s in {0..4}, Stage-I(seed=s) is retrained from
scratch, then Stage-II(seed=s) is trained on that same-seed Stage-I
checkpoint. Only `train.seed` differs from the formal experiment=full
configuration. Backtest: Qlib TopkDropout topk=30 drop=5, csi300,
2023-01-01..2025-12-31, open 0.0005 / close 0.0015.

## Per-seed results (native stage2 metrics, ddof=0)

| seed | IC | ICIR | RankIC | RankICIR | AR | MDD | Sharpe | Calmar | Sortino | Omega | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0.035368 | 0.218435 | 0.051479 | 0.310416 | 0.167646 | -0.168915 | 1.011861 | 0.992486 | 1.529218 | 1.223265 | ok |
| 1 | 0.038188 | 0.246233 | 0.052358 | 0.355741 | 0.138182 | -0.272794 | 0.707967 | 0.506544 | 1.114764 | 1.164296 | ok |
| 2 | 0.033451 | 0.169726 | 0.061792 | 0.338376 | 0.108328 | -0.271957 | 0.610062 | 0.398328 | 0.927177 | 1.138977 | ok |
| 3 | 0.042392 | 0.275725 | 0.053223 | 0.328992 | 0.160813 | -0.152029 | 0.921305 | 1.057778 | 1.343413 | 1.207734 | ok |
| 4 | 0.034756 | 0.193809 | 0.053665 | 0.304281 | 0.126723 | -0.224872 | 0.711934 | 0.563535 | 1.055772 | 1.165129 | ok |

## Predictive mean ± std (protocol v1.0, ddof=1, comparable to formal tables)

- IC:      0.0368 ± 0.0036
- ICIR:    0.2206 ± 0.0418
- RankIC:  0.0545 ± 0.0042
- RankICIR:0.3273 ± 0.0209

## Portfolio descriptive statistics (per-seed independent portfolios)

- AR: 0.1403 ± 0.0244
- MDD: -0.2181 ± 0.0564
- Sharpe: 0.7926 ± 0.1671
- Calmar: 0.7037 ± 0.3002
- Sortino: 1.1941 ± 0.2405
- Omega: 1.1799 ± 0.0346

## Five-full-pipeline-seed ensemble (mean of 5 seed predictions)

This is the portfolio of the *averaged prediction*, NOT the average of
per-seed portfolio metrics.

- IC: 0.040665
- ICIR: 0.229454
- RankIC: 0.060373
- RankICIR: 0.347432
- AR: 0.194370
- MDD: -0.188928
- Sharpe: 1.103152
- Calmar: 1.028802
- Sortino: 1.745276
- Omega: 1.250433
- num_test_days: 727

Reference — formal paper 034 ensemble: AR=0.2102, MDD=-0.1585, Sharpe=1.2301, Sortino=1.8992, Calmar=1.3266, IC=0.0429, RankIC=0.0608

## Protocol comparison (A vs B vs C)

| Protocol | IC mean±std | RankIC mean±std | ICIR mean±std | RankICIR mean±std | Ensemble AR | Ensemble Sharpe |
|---|---|---|---|---|---|---|
| A: Fixed Stage-I + Stage-II multi-seed (paper 034) | 0.0399 ± 0.0026 | 0.0563 ± 0.0039 | 0.2388 ± 0.0203 | 0.3355 ± 0.0257 | 0.2102 | 1.2301 |
| B: Stage-I multi-seed + Stage-II seed42 (Reviewer 1) | 0.0389 ± 0.0012 | 0.0585 ± 0.0016 | 0.2110 ± 0.0067 | 0.3124 ± 0.0077 | n/a (no ensemble) | n/a (no ensemble) |
| C: Full pipeline multi-seed (Reviewer 2, this run) | 0.0368 ± 0.0036 | 0.0545 ± 0.0042 | 0.2206 ± 0.0418 | 0.3273 ± 0.0209 | 0.1944 | 1.1032 |

Note: A ranking/portfolio values come from the formal `results/metrics/`
tables (protocol v1.0). B and C ranking metrics are recomputed from
prediction pickles with the same protocol v1.0 functions. B has no
ensemble row (Reviewer 1 did not build one).
