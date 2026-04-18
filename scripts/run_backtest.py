"""
Run a single Trend Pullback Pro backtest.

Usage:
python scripts/run_backtest.py --config configs/sample_backtest.yaml
python scripts/run_backtest.py --config configs/base.yaml --log-level DEBUG
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtrader as bt
import pandas as pd

from trend_pullback.analyzers import EquityCurveAnalyzer, TradeListAnalyzer
from trend_pullback.config import load_config
from trend_pullback.datafeed import load_ohlcv, make_bt_feed
from trend_pullback.reporting import (
    build_equity_df,
    build_trades_df,
    compute_summary,
    print_summary,
    save_results,
)
from trend_pullback.signal_engine import compute_signals
from trend_pullback.strategy import TrendPullbackStrategy
from trend_pullback.utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trend Pullback Pro backtest")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML config file (e.g. configs/sample_backtest.yaml)",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="Directory to save results (default: output/)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def normalize_bt_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True)
        df["datetime"] = dt.dt.tz_convert("UTC").dt.tz_localize(None)

    if isinstance(df.index, pd.DatetimeIndex):
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
    elif "datetime" in df.columns:
        df = df.set_index("datetime")

    return df


def run_backtest_df(
    df: pd.DataFrame,
    strategy_params,
    backtest_params,
    output_dir: str | None = None,
    save: bool = False,
    print_report: bool = False,
) -> dict:
    df = normalize_bt_datetime_index(df)

    signals_df = compute_signals(df, strategy_params)
    signals_df = normalize_bt_datetime_index(signals_df)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(backtest_params.initial_capital)
    cerebro.broker.setcommission(commission=backtest_params.commission_pct / 100.0)

    feed = make_bt_feed(df)
    cerebro.adddata(feed, name=backtest_params.symbol)

    cerebro.addstrategy(
        TrendPullbackStrategy,
        strategy_params=strategy_params,
        signals_df=signals_df,
        stake=backtest_params.stake,
    )

    cerebro.addanalyzer(TradeListAnalyzer, _name="trade_list")
    cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity_curve")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)

    logger.info("Starting backtest…")
    results = cerebro.run()
    strat = results[0]

    trade_records = strat.analyzers.trade_list.get_analysis()
    equity_records = strat.analyzers.equity_curve.get_analysis()

    trades_df = build_trades_df(trade_records)
    equity_df = build_equity_df(equity_records)
    summary = compute_summary(trades_df, equity_df, backtest_params.initial_capital)

    if save and output_dir:
        save_results(trades_df, equity_df, summary, output_dir)

    if print_report:
        print_summary(summary)

    final_value = cerebro.broker.getvalue()
    logger.info("Final portfolio value: %.2f", final_value)

    return {
        "summary": summary,
        "trades_df": trades_df,
        "equity_df": equity_df,
    }


def run(config_path: str, output_dir: str) -> dict:
    cfg = load_config(config_path)
    sp = cfg.strategy
    bp = cfg.backtest

    logger.info("Config loaded: %s", config_path)
    logger.info("Symbol: %s Timeframe: %s", bp.symbol, bp.timeframe)
    logger.info("Capital: %.2f Commission: %.4f%%", bp.initial_capital, bp.commission_pct)

    df = load_ohlcv(bp.data_path)
    df = normalize_bt_datetime_index(df)

    logger.info("Data loaded: %d bars (%s → %s)", len(df), df.index[0], df.index[-1])

    result = run_backtest_df(
        df=df,
        strategy_params=sp,
        backtest_params=bp,
        output_dir=output_dir,
        save=True,
        print_report=True,
    )

    return result["summary"]


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run(args.config, args.output)


if __name__ == "__main__":
    main()