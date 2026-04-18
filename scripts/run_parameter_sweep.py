"""
Parameter sweep runner for Trend Pullback Pro.

Iterates over a grid of key parameters, runs a backtest for each
combination, and saves aggregate results to output/sweep_results.csv.

Swept parameters:
  - fast_len
  - pull_len
  - atr_mul
  - rr

Usage:
    python scripts/run_parameter_sweep.py --config configs/base.yaml
    python scripts/run_parameter_sweep.py --config configs/base.yaml --output output/sweep
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import backtrader as bt
import pandas as pd

from trend_pullback.analyzers import EquityCurveAnalyzer, TradeListAnalyzer
from trend_pullback.config import AppConfig, load_config
from trend_pullback.datafeed import load_ohlcv, make_bt_feed
from trend_pullback.reporting import (
    build_equity_df,
    build_trades_df,
    compute_summary,
)
from trend_pullback.signal_engine import compute_signals
from trend_pullback.strategy import TrendPullbackStrategy
from trend_pullback.utils import setup_logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Parameter grid definition
# Edit these lists to adjust the sweep scope.
# ---------------------------------------------------------------------------

# FAST_LEN_VALUES = [21, 50, 89]
# PULL_LEN_VALUES = [9, 21, 34]
# ATR_MUL_VALUES  = [1.0, 1.5, 2.0]
# RR_VALUES       = [1.5, 2.0, 2.5]

FAST_LEN_VALUES = [34]
PULL_LEN_VALUES = [13]
ATR_MUL_VALUES  = [2.4]
RR_VALUES       = [2.1, 2.25, 2.3, 2.4, 2.5, 2.6]

# FAST_LEN_VALUES = [30, 34, 40, 45, 50, 55, 60]
# PULL_LEN_VALUES = [18, 21, 24, 26]
# ATR_MUL_VALUES  = [1.0, 1.25, 1.5, 1.75, 2.0]
# RR_VALUES       = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend Pullback — parameter sweep")
    parser.add_argument("--config", "-c", required=True, help="Base YAML config path")
    parser.add_argument("--output", "-o", default="output/sweep", help="Output directory")
    parser.add_argument("--log-level", default="WARNING", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def run_single(cfg: AppConfig, df: pd.DataFrame) -> dict:
    """Run one backtest for a given config and return summary dict."""
    sp = cfg.strategy
    bp = cfg.backtest

    signals_df = compute_signals(df, sp)

    cerebro = bt.Cerebro(stdstats=False)
    cerebro.broker.setcash(bp.initial_capital)
    cerebro.broker.setcommission(commission=bp.commission_pct / 100.0)

    feed = make_bt_feed(df)
    cerebro.adddata(feed)

    cerebro.addstrategy(
        TrendPullbackStrategy,
        strategy_params=sp,
        signals_df=signals_df,
        stake=bp.stake,
    )
    cerebro.addanalyzer(TradeListAnalyzer,   _name="trade_list")
    cerebro.addanalyzer(EquityCurveAnalyzer, _name="equity_curve")

    results = cerebro.run()
    strat = results[0]

    trade_records  = strat.analyzers.trade_list.get_analysis()
    equity_records = strat.analyzers.equity_curve.get_analysis()

    trades_df = build_trades_df(trade_records)
    equity_df = build_equity_df(equity_records)

    return compute_summary(trades_df, equity_df, bp.initial_capital)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    base_cfg = load_config(args.config)
    df = load_ohlcv(base_cfg.backtest.data_path)
    logger.info("Data loaded: %d bars", len(df))

    grid = list(itertools.product(
        FAST_LEN_VALUES,
        PULL_LEN_VALUES,
        ATR_MUL_VALUES,
        RR_VALUES,
    ))
    logger.info("Total combinations: %d", len(grid))

    rows: list[dict] = []

    for idx, (fast_len, pull_len, atr_mul, rr) in enumerate(grid, start=1):
        # Skip invalid combos (fast EMA >= slow EMA)
        if fast_len >= base_cfg.strategy.slow_len:
            logger.debug("Skipping fast_len=%d >= slow_len=%d", fast_len, base_cfg.strategy.slow_len)
            continue

        # Build a derived config for this combination
        cfg = deepcopy(base_cfg)
        cfg.strategy.fast_len = fast_len
        cfg.strategy.pull_len = pull_len
        cfg.strategy.atr_mul  = atr_mul
        cfg.strategy.rr       = rr

        print(
            f"[{idx}/{len(grid)}]  fast={fast_len}  pull={pull_len}"
            f"  atr_mul={atr_mul}  rr={rr}",
            end="  ",
            flush=True,
        )

        try:
            summary = run_single(cfg, df)
            row = {
                "fast_len":        fast_len,
                "pull_len":        pull_len,
                "atr_mul":         atr_mul,
                "rr":              rr,
                **summary,
            }
            rows.append(row)
            print(
                f"trades={summary['total_trades']}"
                f"  wr={summary['win_rate_pct']:.0f}%"
                f"  net%={summary['net_profit_pct']:.1f}%"
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Combo failed: fast=%d pull=%d atr_mul=%s rr=%s — %s",
                         fast_len, pull_len, atr_mul, rr, exc)
            print(f"ERROR: {exc}")

    if not rows:
        print("No results — check your data and config.")
        return

    results_df = pd.DataFrame(rows).sort_values("net_profit_pct", ascending=False)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "sweep_results.csv"
    results_df.to_csv(out_path, index=False)
    print(f"\nSweep complete — {len(results_df)} results saved to {out_path}")
    print("\nTop 5 by net profit %:")
    print(results_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
