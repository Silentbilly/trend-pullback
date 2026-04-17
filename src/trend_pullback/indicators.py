"""
Technical indicator implementations for Trend Pullback Pro.

Pure pandas/numpy functions — no TA-Lib dependency.
Each function:
  - accepts a pandas Series (or multiple Series for ATR)
  - returns a pandas Series of the same index
  - preserves NaN for the warmup period (same as Pine Script behaviour)

Design note:
  EMA uses the standard Wilder/TradingView formula: multiplier = 2 / (length + 1).
  RSI uses Wilder's smoothing (RMA), matching Pine Script ta.rsi().
  ATR uses Wilder's RMA on True Range, matching Pine Script ta.atr().
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average.

    Matches Pine Script ta.ema() — uses multiplier 2/(length+1).

    Args:
        series: Price series (typically close).
        length: EMA period.

    Returns:
        EMA series with NaN for the first (length - 1) bars.
    """
    if length < 1:
        raise ValueError(f"EMA length must be >= 1, got {length}")
    result = series.ewm(span=length, min_periods=length, adjust=False).mean()
    return result


def _rma(series: pd.Series, length: int) -> pd.Series:
    """Wilder's Running Moving Average (RMA).

    Used internally by rsi() and atr() to match Pine Script behaviour.
    alpha = 1 / length (Wilder smoothing).

    Args:
        series: Input series.
        length: Smoothing period.

    Returns:
        RMA series.
    """
    alpha = 1.0 / length
    result = series.ewm(alpha=alpha, min_periods=length, adjust=False).mean()
    return result


def rsi(series: pd.Series, length: int) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing.

    Matches Pine Script ta.rsi().

    Args:
        series: Price series (typically close).
        length: RSI period.

    Returns:
        RSI series (0–100) with NaN for warmup period.
    """
    if length < 1:
        raise ValueError(f"RSI length must be >= 1, got {length}")

    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    avg_gain = _rma(gain, length)
    avg_loss = _rma(loss, length)

    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi_val = 100.0 - (100.0 / (1.0 + rs))

    # When avg_loss == 0 and avg_gain > 0 → RSI = 100
    rsi_val = rsi_val.fillna(100.0)
    return rsi_val


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """Average True Range using Wilder's smoothing.

    Matches Pine Script ta.atr().

    Args:
        high: High price series.
        low:  Low price series.
        close: Close price series.
        length: ATR period.

    Returns:
        ATR series with NaN for warmup period.
    """
    if length < 1:
        raise ValueError(f"ATR length must be >= 1, got {length}")

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return _rma(tr, length)
