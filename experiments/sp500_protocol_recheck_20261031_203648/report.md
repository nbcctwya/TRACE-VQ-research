# TRACE-VQ SP500 independent recheck



Passed: True; finished: 2026-10-31T12:44:40.744677+00:00



All 360 metric values recomputed using six other baseline implementations agree; maximum absolute error 1.2e-13.

All five seeds and the avg_none ensemble were replayed through the actual FactorVQVAE standard adapter; daily gross returns, costs and net returns agree within 1e-12.



Shared formulas: annualization 252; log1p(net return); ddof=1; rf=MAR=0; initial NAV=1; Sortino downside over all days; non-annualized ICIR/RankICIR.

Shared trading parameters: Top30/Drop5, risk degree 0.95, initial cash 1e8, close execution, open/close costs 0.0005/0.0015, minimum cost 0, Qlib internal signal shift only.



Scope differences: MASTER historical exporter uses codes=sp500; TRACE-VQ and the majority of SP500 adapters use all exchange quotes. Prediction candidates still come from the SP500 score index. See quote_policy_sensitivity.json.

Label horizons differ across model families; this preserves their native targets. IC/RankIC therefore do not all refer to the same prediction target.

MASTER metadata records an older Qlib version; identical historical engine versions cannot be claimed.



All 353 published result files retained identical SHA256s during this recheck. Original pre-import rows and unrelated files also passed preservation checks.
