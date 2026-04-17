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

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtrader as bt

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


def run(config_path: str, output_dir: str) -> dict:
    """Execute a full backtest run.

    Args:
        config_path: Path to YAML config.
        output_dir:  Directory to write output files.

    Returns:
        Summary metrics dict.
    """
    cfg = load_config(config_path)
    sp  = cfg.strategy
    bp  = cfg.backtest

    logger.info("Config loaded: %s", config_path)
    logger.info("Symbol: %s  Timeframe: %s", bp.symbol, bp.timeframe)
    logger.info("Capital: %.2f  Commission: %.4f%%", bp.initial_capital, bp.commission_pct)

    # Load OHLCV data
    df = load_ohlcv(bp.data_path)
    logger.info("Data loaded: %d bars  (%s → %s)", len(df), df.index[0], df.index[-1])

    # Pre-compute all signals
    signals_df = compute_signals(df, sp)
    logger.info("Signals computed")

    # Build Cerebro
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(bp.initial_capital)
    cerebro.broker.setcommission(commission=bp.commission_pct / 100.0)

    feed = make_bt_feed(df)
    cerebro.adddata(feed, name=bp.symbol)

    cerebro.addstrategy(
        TrendPullbackStrategy,
        strategy_params=sp,
        signals_df=signals_df,
    )

    cerebro.addanalyzer(TradeListAnalyzer, _name="trade_list")
    cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity_curve")
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.0)

    # Run
    logger.info("Starting backtest…")
    results = cerebro.run()
    strat = results[0]

    # Collect results
    trade_records  = strat.analyzers.trade_list.get_analysis()
    equity_records = strat.analyzers.equity_curve.get_analysis()

    trades_df = build_trades_df(trade_records)
    equity_df = build_equity_df(equity_records)
    summary   = compute_summary(trades_df, equity_df, bp.initial_capital)

    # Save outputs
    save_results(trades_df, equity_df, summary, output_dir)
    print_summary(summary)

    final_value = cerebro.broker.getvalue()
    logger.info("Final portfolio value: %.2f", final_value)

    return summary


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run(args.config, args.output)


if __name__ == "__main__":
    main()
