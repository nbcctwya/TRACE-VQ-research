import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

from trace_vq.runtime import configure_runtime
configure_runtime()

import numpy as np
import pandas as pd

from trace_vq.evaluation.protocol_metrics import TRADING_DAYS, portfolio_metrics


# 兼容不同预测文件里常见的列名/索引名，最终统一成 Qlib 需要的格式。
DATE_COLUMNS = ("datetime", "date", "time")
INSTRUMENT_COLUMNS = ("instrument", "stock_id", "symbol", "code")
SCORE_COLUMNS = ("score", "pred", "prediction", "pred_score", "Pred", "PRED")
LABEL_COLUMNS = ("label", "realized_return", "realized_ret", "return", "ret", "LABEL0")


# 每个 universe 对应的 Qlib 数据路径、市场区域和默认 benchmark。
UNIVERSE_SETTINGS = {
    "sp500": {
        "region": "US",
        "provider_uri": "~/.qlib/qlib_data/us_data",
        "benchmark": "^gspc",
    },
    "csi300": {
        "region": "CN",
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "benchmark": "SH000300",
    },
    "csi500": {
        "region": "CN",
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "benchmark": "SH000905",
    },
    "csi800": {
        "region": "CN",
        "provider_uri": "~/.qlib/qlib_data/cn_data",
        "benchmark": "SH000906",
    },
}


def _find_first(candidates, names) -> Optional[str]:
    name_set = set(names)
    lower_to_name = {str(name).lower(): name for name in names}
    for candidate in candidates:
        if candidate in name_set:
            return candidate
        lowered = candidate.lower()
        if lowered in lower_to_name:
            return lower_to_name[lowered]
    return None


def _normalize_prediction_frame(pred_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """读取 stage2 的 *_best.pkl，并整理成 Qlib signal 所需的 MultiIndex Series。"""
    pred = pd.read_pickle(pred_path)
    if isinstance(pred, pd.Series):
        pred = pred.to_frame(name=pred.name or "score")
    if not isinstance(pred, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame or Series from {pred_path}, got {type(pred)!r}.")

    pred = pred.copy()

    # stage2 结果通常是 (datetime, instrument) MultiIndex；这里也兼容普通列格式。
    if isinstance(pred.index, pd.MultiIndex):
        index_names = list(pred.index.names)
        date_level = _find_first(DATE_COLUMNS, index_names)
        instrument_level = _find_first(INSTRUMENT_COLUMNS, index_names)
        if date_level is None or instrument_level is None:
            if pred.index.nlevels == 2:
                date_level = index_names[0]
                instrument_level = index_names[1]
            else:
                raise ValueError(
                    "Prediction MultiIndex must include datetime/date and instrument levels. "
                    f"Found index names: {index_names}."
                )

        pred = pred.reset_index()
        pred = pred.rename(columns={date_level: "datetime", instrument_level: "instrument"})
    else:
        date_col = _find_first(DATE_COLUMNS, pred.columns)
        instrument_col = _find_first(INSTRUMENT_COLUMNS, pred.columns)
        if date_col is None or instrument_col is None:
            raise ValueError(
                "Prediction file must contain datetime/date and instrument columns, or a "
                "MultiIndex with those levels. "
                f"Found columns: {list(pred.columns)}; index names: {getattr(pred.index, 'names', None)}."
            )
        pred = pred.rename(columns={date_col: "datetime", instrument_col: "instrument"})

    score_col = _find_first(SCORE_COLUMNS, pred.columns)
    if score_col is None:
        raise ValueError(
            "Prediction file must contain a score column. Accepted names: "
            f"{list(SCORE_COLUMNS)}. Found columns: {list(pred.columns)}."
        )

    label_col = _find_first(LABEL_COLUMNS, pred.columns)
    if label_col is None:
        print(
            "Warning: no label/realized return column found. "
            "Continuing because Qlib portfolio backtest uses only prediction scores."
        )

    pred["datetime"] = pd.to_datetime(pred["datetime"])
    pred["instrument"] = pred["instrument"].astype(str)
    pred["score"] = pd.to_numeric(pred[score_col], errors="coerce")

    missing_score = pred["score"].isna().sum()
    if missing_score:
        raise ValueError(f"Prediction score contains {missing_score} NaN/non-numeric values.")

    normalized_cols = ["datetime", "instrument", "score"]
    if label_col is not None:
        # label 仅用于留档和人工检查；Qlib 组合回测只消费 score。
        pred["label"] = pd.to_numeric(pred[label_col], errors="coerce")
        normalized_cols.append("label")

    # Qlib 的 TopkDropoutStrategy 期望 signal 的索引是 (datetime, instrument)。
    normalized = (
        pred[normalized_cols]
        .drop_duplicates(subset=["datetime", "instrument"], keep="last")
        .set_index(["datetime", "instrument"])
        .sort_index()
    )
    signal = normalized["score"].rename("score")
    return normalized, signal


def _resolve_universe(universe: str, provider_uri: Optional[str]) -> Tuple[str, str, str]:
    if universe not in UNIVERSE_SETTINGS:
        raise ValueError(f"Unsupported universe: {universe}. Expected one of {sorted(UNIVERSE_SETTINGS)}.")
    settings = UNIVERSE_SETTINGS[universe]
    resolved_provider = provider_uri or settings["provider_uri"]
    return (
        str(Path(resolved_provider).expanduser()),
        settings["region"],
        settings["benchmark"],
    )


def _exchange_codes(universe: str, start_time: str, end_time: str, quote_universe: str = "all"):
    """Keep quotes for exits; historical quotes reproduce archived evaluations."""
    if quote_universe == "all":
        return "all"
    if universe == "csi800":
        from trace_vq.data.universe import load_union

        instruments = load_union(["csi300", "csi500"], start_time, end_time)
        if not instruments:
            raise ValueError("CSI300/CSI500 union is empty for the backtest period")
        return instruments
    return universe


def _default_output_dir(pred_path: Path, topk: int, drop: int) -> Path:
    seed = pred_path.stem.split("_", 1)[0]
    if not seed:
        seed = "unknown"
    return Path(__file__).resolve().parents[2] / "res" / "backtest" / pred_path.parent.name / f"seed{seed}_top{topk}_drop{drop}"


def _portfolio_returns_after_cost(report: pd.DataFrame) -> pd.Series:
    """Qlib report 中 return 是扣费前收益；论文口径需要先扣除交易成本。"""
    gross_return = report["return"].astype(float)
    cost = report["cost"].astype(float).fillna(0.0) if "cost" in report else 0.0
    return (gross_return - cost).rename("return")


def _paper_metrics(simple_return: pd.Series, turnover: Optional[pd.Series], name: str) -> pd.DataFrame:
    """Display the same portfolio formulas as the unified results protocol."""
    simple_return = simple_return.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if simple_return.empty:
        raise ValueError(f"No valid return observations for {name}.")
    if (simple_return <= -1).any():
        bad_count = int((simple_return <= -1).sum())
        raise ValueError(f"{name} contains {bad_count} return values <= -100%, cannot compute log returns.")

    metrics = portfolio_metrics(simple_return)
    wealth = np.exp(np.log1p(simple_return).cumsum())

    turnover_value = np.nan
    if turnover is not None:
        turnover_value = turnover.reindex(simple_return.index).astype(float).mean()

    result = {
        "Annualized Return": round(metrics["AR"], 4),
        "Annual Std": round(metrics["STD"], 4),
        "MDD": round(metrics["MDD"], 4),
        "Sharpe Ratio": round(metrics["Sharpe"], 4),
        "Sortino Ratio": round(metrics["Sortino"], 4),
        "Calmar Ratio": round(metrics["Calmar"], 4),
        "Cumulative Returns": round(wealth.iloc[-1] - 1.0, 4),
        "Turnover": round(turnover_value, 4) if pd.notna(turnover_value) else np.nan,
    }
    return pd.DataFrame.from_dict(result, orient="index", columns=[name])


def _make_metric_frame(report: pd.DataFrame) -> pd.DataFrame:
    from qlib.contrib.evaluate import risk_analysis

    # 所有 portfolio / excess 指标都用扣成本后的净收益，和论文回测口径一致。
    portfolio_return = _portfolio_returns_after_cost(report)
    benchmark_return = report["bench"].astype(float) if "bench" in report else None
    excess_return = portfolio_return - benchmark_return if benchmark_return is not None else None

    metric_sections: Dict[str, pd.Series] = {}

    # 保留一份 Qlib 原生 risk_analysis 输出，便于和 Qlib 默认口径交叉检查。
    qlib_port = risk_analysis(portfolio_return.dropna(), freq="day", mode="sum")["risk"]
    metric_sections["qlib_portfolio"] = qlib_port

    if benchmark_return is not None:
        metric_sections["qlib_benchmark"] = risk_analysis(benchmark_return.dropna(), freq="day", mode="sum")["risk"]
    if excess_return is not None:
        metric_sections["qlib_excess"] = risk_analysis(excess_return.dropna(), freq="day", mode="sum")["risk"]

    turnover = report["turnover"].astype(float) if "turnover" in report else None

    # project_* 是论文公式口径：净收益 -> log return -> 年化/MDD/Sharpe。
    table_metrics = _paper_metrics(portfolio_return, turnover, name="project_portfolio")
    metric_sections["project_portfolio"] = table_metrics["project_portfolio"]

    if benchmark_return is not None:
        bench_metrics = _paper_metrics(benchmark_return, None, name="project_benchmark")
        metric_sections["project_benchmark"] = bench_metrics["project_benchmark"]

    if excess_return is not None:
        excess_metrics = _paper_metrics(excess_return, turnover, name="project_excess")
        metric_sections["project_excess"] = excess_metrics["project_excess"]

    return pd.DataFrame(metric_sections)


def _save_outputs(
    output_dir: Path,
    report: pd.DataFrame,
    positions,
    metric_df: pd.DataFrame,
    signal: pd.Series,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 对外保存的 return 使用扣费后净收益；gross_return/cost 保留用于追溯。
    returns = pd.DataFrame(index=report.index)
    returns["return"] = _portfolio_returns_after_cost(report)
    returns["gross_return"] = report["return"]
    if "cost" in report:
        returns["cost"] = report["cost"]
    if "bench" in report:
        returns["benchmark"] = report["bench"]
        returns["excess_return"] = returns["return"] - returns["benchmark"]
    for optional_col in ["turnover", "total_turnover", "total_cost", "account", "value", "cash"]:
        if optional_col in report:
            returns[optional_col] = report[optional_col]

    returns.to_csv(output_dir / "portfolio_return.csv")
    report.to_csv(output_dir / "qlib_report.csv")
    metric_df.to_csv(output_dir / "portfolio_metric.csv")
    signal.to_frame("score").to_pickle(output_dir / "qlib_signal.pkl")

    if isinstance(positions, pd.DataFrame):
        positions.to_pickle(output_dir / "positions.pkl")
    else:
        pd.to_pickle(positions, output_dir / "positions.pkl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Qlib native portfolio backtest from TRACE-VQ stage2 prediction scores."
    )
    parser.add_argument("--pred_path", required=True, help="Path to stage2 *_best.pkl prediction file.")
    parser.add_argument("--universe", required=True, choices=sorted(UNIVERSE_SETTINGS), help="Market universe.")
    parser.add_argument("--qlib_provider_uri", default=None, help="Qlib data path. Defaults by universe.")
    parser.add_argument("--start_time", default="2023-01-01", help="Backtest start date.")
    parser.add_argument("--end_time", default="2025-12-31", help="Backtest end date.")
    parser.add_argument("--topk", type=int, default=30, help="Top-k holdings for Qlib TopkDropoutStrategy.")
    parser.add_argument("--drop", type=int, default=5, help="Number of holdings to replace each trading day.")
    parser.add_argument("--open_cost", type=float, default=0.0005, help="Open transaction cost.")
    parser.add_argument("--close_cost", type=float, default=0.0015, help="Close transaction cost.")
    parser.add_argument("--min_cost", type=float, default=0.0, help="Minimum transaction cost.")
    parser.add_argument("--account", type=float, default=100000000.0, help="Initial account value.")
    parser.add_argument("--benchmark", default=None, help="Override benchmark instrument.")
    parser.add_argument(
        "--quote-universe", choices=("all", "historical"), default="all",
        help="Quote coverage: all retains prices after index exits; historical reproduces archived pool-filtered backtests.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output directory. Defaults to project res/backtest/RUN/seedX_topK_dropD.",
    )
    return parser.parse_args()


def main() -> None:
    import qlib
    from qlib.constant import REG_CN, REG_US
    from qlib.contrib.evaluate import backtest_daily
    from qlib.contrib.strategy import TopkDropoutStrategy

    args = parse_args()
    pred_path = Path(args.pred_path).expanduser()
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    provider_uri, region, default_benchmark = _resolve_universe(args.universe, args.qlib_provider_uri)
    benchmark = args.benchmark or default_benchmark
    output_dir = Path(args.output_dir).expanduser() if args.output_dir else _default_output_dir(
        pred_path,
        topk=args.topk,
        drop=args.drop,
    )

    normalized_pred, signal = _normalize_prediction_frame(pred_path)
    print(f"Loaded prediction: {pred_path}")
    print(f"Signal rows: {len(signal):,}")
    print(f"Signal date range: {signal.index.get_level_values('datetime').min()} -> {signal.index.get_level_values('datetime').max()}")
    print(f"Signal instruments: {signal.index.get_level_values('instrument').nunique():,}")
    print(f"Output directory: {output_dir}")

    qlib_region = REG_US if region == "US" else REG_CN
    qlib.init(provider_uri=provider_uri, region=qlib_region,
              exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm",
                           "kwargs": {"uri": str(output_dir.resolve() / "qlib_runs"),
                                      "default_exp_name": "backtest"}})

    # 论文使用 Qlib TopK-DropN：每天根据 5 日收益预测分数调仓，默认 K=30、DropN=5。
    strategy_kwargs = {
        "topk": args.topk,
        "n_drop": args.drop,
        "method_sell": "bottom",
        "method_buy": "top",
        "hold_thresh": 1,
        "only_tradable": False,
        "forbid_all_trade_at_limit": True,
        "risk_degree": 0.95,
    }
    strategy = TopkDropoutStrategy(signal=signal, **strategy_kwargs)
    codes = _exchange_codes(args.universe, args.start_time, args.end_time, args.quote_universe)
    exchange_kwargs = {
        "freq": "day",
        "codes": codes,
        "limit_threshold": None,
        "deal_price": "close",
        "open_cost": args.open_cost,
        "close_cost": args.close_cost,
        "min_cost": args.min_cost,
    }

    # Qlib 原生回测负责成交、持仓更新和交易成本；指标阶段再用净收益计算论文口径。
    report, positions = backtest_daily(
        start_time=args.start_time,
        end_time=args.end_time,
        strategy=strategy,
        account=args.account,
        benchmark=benchmark,
        exchange_kwargs=exchange_kwargs,
    )

    metric_df = _make_metric_frame(report)
    _save_outputs(output_dir, report, positions, metric_df, signal)
    metadata = {
        "universe": args.universe,
        "provider_uri": provider_uri,
        "benchmark": benchmark,
        "start_time": args.start_time,
        "end_time": args.end_time,
        "account": args.account,
        "quote_universe": args.quote_universe,
        "strategy": strategy_kwargs,
        "exchange": {key: value for key, value in exchange_kwargs.items() if key != "codes"},
        "exchange_instruments": codes if isinstance(codes, str) else {
            "source_universes": ["csi300", "csi500"],
            "point_in_time_union": True,
            "unique_stocks": len(codes),
            "membership_file": "exchange_instruments.txt",
        },
        "qlib_version": qlib.__version__,
    }
    (output_dir / "backtest_config.json").write_text(json.dumps(metadata, indent=2))
    if isinstance(codes, dict):
        with (output_dir / "exchange_instruments.txt").open("w") as stream:
            for symbol, intervals in codes.items():
                for start, end in intervals:
                    stream.write(f"{symbol}\t{pd.Timestamp(start).date()}\t{pd.Timestamp(end).date()}\n")

    print("\nPortfolio metrics:")
    print(metric_df)
    print(f"\nSaved portfolio returns to {output_dir / 'portfolio_return.csv'}")
    print(f"Saved portfolio metrics to {output_dir / 'portfolio_metric.csv'}")
    print(f"Saved Qlib native report to {output_dir / 'qlib_report.csv'}")


if __name__ == "__main__":
    main()
