"""
Reporting utilities for Trend Pullback Pro.

Converts analyzer outputs to pandas DataFrames and computes summary metrics.
Saves results to the output/ directory.

Outputs:
  output/summary.csv   — single-row summary of backtest metrics
  output/trades.csv    — all closed trades
  output/equity.csv    — bar-by-bar equity curve
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DataFrame builders
# ---------------------------------------------------------------------------

def build_equity_df(equity_records: list[dict]) -> pd.DataFrame:
    """Convert equity curve records to a DataFrame.

    Args:
        equity_records: List of dicts from EquityCurveAnalyzer.get_analysis().

    Returns:
        DataFrame with columns: datetime, equity.
    """
    if not equity_records:
        return pd.DataFrame(columns=["datetime", "equity"])
    df = pd.DataFrame(equity_records)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def build_trades_df(trade_records: list[dict]) -> pd.DataFrame:
    """Convert trade records to a DataFrame.

    Args:
        trade_records: List of dicts from TradeListAnalyzer.get_analysis().

    Returns:
        DataFrame with one row per trade.
    """
    if not trade_records:
        return pd.DataFrame()
    return pd.DataFrame(trade_records)


# ---------------------------------------------------------------------------
# Summary metrics
# ---------------------------------------------------------------------------

def compute_summary(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    initial_capital: float,
) -> dict:
    """Compute summary performance metrics.

    Args:
        trades_df:       Trades DataFrame from build_trades_df().
        equity_df:       Equity DataFrame from build_equity_df().
        initial_capital: Starting capital from config.

    Returns:
        Dict of summary metrics.
    """
    if trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "net_profit": 0.0,
            "net_profit_pct": 0.0,
            "avg_trade_net": 0.0,
            "max_drawdown_pct": 0.0,
        }

    total_trades = len(trades_df)
    winners = (trades_df["pnl_net"] > 0).sum()
    win_rate = winners / total_trades * 100.0
    net_profit = trades_df["pnl_net"].sum()
    net_profit_pct = net_profit / initial_capital * 100.0
    avg_trade = trades_df["pnl_net"].mean()

    max_dd = _compute_max_drawdown(equity_df["equity"]) if not equity_df.empty else 0.0

    return {
        "total_trades":    total_trades,
        "win_rate_pct":    round(win_rate, 2),
        "net_profit":      round(net_profit, 4),
        "net_profit_pct":  round(net_profit_pct, 2),
        "avg_trade_net":   round(avg_trade, 4),
        "max_drawdown_pct": round(max_dd, 2),
    }


def _compute_max_drawdown(equity: pd.Series) -> float:
    """Compute maximum drawdown percentage from equity curve."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max * 100.0
    return float(drawdown.min())


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary: dict,
    output_dir: str | Path = "output",
) -> None:
    """Save backtest results to CSV files.

    Creates output_dir if it does not exist.

    Args:
        trades_df:   Trades DataFrame.
        equity_df:   Equity DataFrame.
        summary:     Summary metrics dict.
        output_dir:  Directory to write output files.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    trades_path = out / "trades.csv"
    equity_path = out / "equity.csv"
    summary_path = out / "summary.csv"

    if not trades_df.empty:
        trades_df.to_csv(trades_path, index=False)
        logger.info("Saved %d trades → %s", len(trades_df), trades_path)
    else:
        logger.warning("No trades to save.")

    if not equity_df.empty:
        equity_df.to_csv(equity_path, index=False)
        logger.info("Saved equity curve (%d bars) → %s", len(equity_df), equity_path)

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(summary_path, index=False)
    logger.info("Saved summary → %s", summary_path)


def print_summary(summary: dict) -> None:
    """Print a formatted summary to stdout."""
    print("\n" + "=" * 48)
    print("  Trend Pullback Pro — Backtest Results")
    print("=" * 48)
    print(f"  Total Trades    : {summary['total_trades']}")
    print(f"  Win Rate        : {summary['win_rate_pct']:.1f}%")
    print(f"  Net Profit      : {summary['net_profit']:.2f}  ({summary['net_profit_pct']:.1f}%)")
    print(f"  Avg Trade (net) : {summary['avg_trade_net']:.4f}")
    print(f"  Max Drawdown    : {summary['max_drawdown_pct']:.2f}%")
    print("=" * 48 + "\n")
