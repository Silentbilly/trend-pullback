"""
Signal engine for Trend Pullback Pro.

This module contains the pure, vectorised signal logic.
It is completely independent of Backtrader — it operates on a
pandas DataFrame and returns an enriched DataFrame with all
intermediate and final signal columns.

Column reference (output):
  ema_fast          — fast EMA
  ema_slow          — slow EMA
  ema_pull          — pullback EMA
  rsi               — RSI value
  atr               — ATR value
  fast_up / fast_down   — slope flags for fast EMA
  slow_up / slow_down   — slope flags for slow EMA
  up_base / down_base   — base trend condition (no slope)
  up_trend / down_trend — trend condition (with optional slope)
  pb_long_bar           — bar is in long pullback zone
  pb_short_bar          — bar is in short pullback zone
  pb_long_count         — consecutive long pullback bars so far
  pb_short_count        — consecutive short pullback bars so far
  pb_long_ready         — pullback count within [min_pb, max_pb]
  pb_short_ready        — same for short
  bull_trigger          — close > prev high
  bear_trigger          — close < prev low
  rsi_long_ok           — RSI passes long filter (or filter disabled)
  rsi_short_ok          — RSI passes short filter (or filter disabled)
  raw_long_signal       — trend + pullback ready + trigger + rsi
  raw_short_signal      — same for short
  long_signal           — raw signal after anti-duplicate gate
  short_signal          — raw signal after anti-duplicate gate
  long_stop_candidate   — proposed stop for long entry
  short_stop_candidate  — proposed stop for short entry

Design note on anti-duplicate (block_repeats):
  Pine Script uses mutable `var bool longUsed` that is reset when
  pbLongBar transitions from false → true (i.e. a fresh pullback starts).
  We replicate this with an explicit loop (iterate row-by-row) to avoid
  any ambiguity.  The loop is O(n) and acceptable for backtest data sizes.
"""

from __future__ import annotations

import pandas as pd

from trend_pullback.config import StrategyParams
from trend_pullback.indicators import atr, ema, rsi


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def compute_signals(df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
    """Compute all strategy signals for a price DataFrame.

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume.
            Index must be datetime-like and sorted ascending.
        params: Validated StrategyParams.

    Returns:
        New DataFrame containing all original columns plus all signal
        columns listed in the module docstring.
    """
    out = df.copy()

    # -----------------------------------------------------------------------
    # 1. Indicators
    # -----------------------------------------------------------------------
    out["ema_fast"] = ema(out["close"], params.fast_len)
    out["ema_slow"] = ema(out["close"], params.slow_len)
    out["ema_pull"] = ema(out["close"], params.pull_len)
    out["rsi"]      = rsi(out["close"], params.rsi_len)
    out["atr"]      = atr(out["high"], out["low"], out["close"], params.atr_len)

    # -----------------------------------------------------------------------
    # 2. Trend filter
    # -----------------------------------------------------------------------
    out["fast_up"]   = out["ema_fast"] > out["ema_fast"].shift(params.slope_lb)
    out["fast_down"] = out["ema_fast"] < out["ema_fast"].shift(params.slope_lb)
    out["slow_up"]   = out["ema_slow"] > out["ema_slow"].shift(params.slope_lb)
    out["slow_down"] = out["ema_slow"] < out["ema_slow"].shift(params.slope_lb)

    out["up_base"]   = (out["ema_fast"] > out["ema_slow"]) & (out["close"] > out["ema_slow"])
    out["down_base"] = (out["ema_fast"] < out["ema_slow"]) & (out["close"] < out["ema_slow"])

    if params.use_slope:
        out["up_trend"]   = out["up_base"]   & out["fast_up"]   & out["slow_up"]
        out["down_trend"] = out["down_base"] & out["fast_down"] & out["slow_down"]
    else:
        out["up_trend"]   = out["up_base"].copy()
        out["down_trend"] = out["down_base"].copy()

    # -----------------------------------------------------------------------
    # 3. Pullback bar detection
    # -----------------------------------------------------------------------
    out["pb_long_bar"]  = (out["low"]  <= out["ema_pull"]) & (out["close"] >= out["ema_slow"])
    out["pb_short_bar"] = (out["high"] >= out["ema_pull"]) & (out["close"] <= out["ema_slow"])

    # -----------------------------------------------------------------------
    # 4. Pullback consecutive counter (iterative — mirrors Pine var int)
    # -----------------------------------------------------------------------
    out["pb_long_count"]  = _compute_pb_count(out["pb_long_bar"])
    out["pb_short_count"] = _compute_pb_count(out["pb_short_bar"])

    out["pb_long_ready"]  = (
        out["pb_long_bar"]
        & (out["pb_long_count"]  >= params.min_pb)
        & (out["pb_long_count"]  <= params.max_pb)
    )
    out["pb_short_ready"] = (
        out["pb_short_bar"]
        & (out["pb_short_count"] >= params.min_pb)
        & (out["pb_short_count"] <= params.max_pb)
    )

    # -----------------------------------------------------------------------
    # 5. Trigger bars
    # -----------------------------------------------------------------------
    out["bull_trigger"] = out["close"] > out["high"].shift(1)
    out["bear_trigger"] = out["close"] < out["low"].shift(1)

    # -----------------------------------------------------------------------
    # 6. RSI filter
    # -----------------------------------------------------------------------
    if params.use_rsi:
        out["rsi_long_ok"]  = out["rsi"] >= params.rsi_long_min
        out["rsi_short_ok"] = out["rsi"] <= params.rsi_short_max
    else:
        out["rsi_long_ok"]  = True
        out["rsi_short_ok"] = True

    # -----------------------------------------------------------------------
    # 7. Raw signals (before anti-duplicate gate)
    # -----------------------------------------------------------------------
    out["raw_long_signal"]  = (
        out["up_trend"]
        & out["pb_long_ready"]
        & out["bull_trigger"]
        & out["rsi_long_ok"]
    )
    out["raw_short_signal"] = (
        out["down_trend"]
        & out["pb_short_ready"]
        & out["bear_trigger"]
        & out["rsi_short_ok"]
    )

    # -----------------------------------------------------------------------
    # 8. Anti-duplicate gate — one signal per pullback cycle
    # -----------------------------------------------------------------------
    if params.block_repeats:
        out["long_signal"], out["short_signal"] = _apply_block_repeats(
            out["raw_long_signal"],
            out["raw_short_signal"],
            out["pb_long_bar"],
            out["pb_short_bar"],
        )
    else:
        out["long_signal"]  = out["raw_long_signal"].copy()
        out["short_signal"] = out["raw_short_signal"].copy()

    # -----------------------------------------------------------------------
    # 9. Stop candidates (for reference / external use)
    # -----------------------------------------------------------------------
    out["long_stop_candidate"]  = (
        out[["low", "low"]].assign(prev_low=out["low"].shift(1))[["low", "prev_low"]].min(axis=1)
        - out["atr"] * params.atr_mul
    )
    out["short_stop_candidate"] = (
        out[["high", "high"]].assign(prev_high=out["high"].shift(1))[["high", "prev_high"]].max(axis=1)
        + out["atr"] * params.atr_mul
    )

    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_pb_count(pb_bar: pd.Series) -> pd.Series:
    """Compute consecutive pullback bar counter.

    Equivalent to Pine Script:
        pbCount := pbBar ? nz(pbCount[1]) + 1 : 0

    Args:
        pb_bar: Boolean Series indicating pullback bars.

    Returns:
        Integer Series with the consecutive run length.
    """
    values = pb_bar.to_numpy()
    counts = [0] * len(values)
    for i in range(len(values)):
        if values[i]:
            counts[i] = (counts[i - 1] if i > 0 else 0) + 1
        else:
            counts[i] = 0
    return pd.Series(counts, index=pb_bar.index, dtype="int64")


def _apply_block_repeats(
    raw_long: pd.Series,
    raw_short: pd.Series,
    pb_long_bar: pd.Series,
    pb_short_bar: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Apply one-signal-per-pullback-cycle gate.

    Replicates Pine Script logic exactly:
      - longUsed resets when a new long pullback starts
        (pb_long_bar transitions False → True)
      - Once a long signal fires, longUsed = True, further signals blocked
        until the next pullback start

    Args:
        raw_long:    Raw long signal Series.
        raw_short:   Raw short signal Series.
        pb_long_bar: Boolean Series for long pullback bars.
        pb_short_bar: Boolean Series for short pullback bars.

    Returns:
        Tuple (long_signal, short_signal) as boolean Series.
    """
    n = len(raw_long)
    raw_l = raw_long.to_numpy()
    raw_s = raw_short.to_numpy()
    pb_l  = pb_long_bar.to_numpy()
    pb_s  = pb_short_bar.to_numpy()

    long_out  = [False] * n
    short_out = [False] * n

    long_used  = False
    short_used = False

    prev_pb_l = False
    prev_pb_s = False

    for i in range(n):
        # Reset flags at the start of a new pullback cycle
        new_long_pb  = pb_l[i] and not prev_pb_l
        new_short_pb = pb_s[i] and not prev_pb_s

        if new_long_pb:
            long_used = False
        if new_short_pb:
            short_used = False

        # Gate: allow signal only if not already used this cycle
        if raw_l[i] and not long_used:
            long_out[i] = True
            long_used = True

        if raw_s[i] and not short_used:
            short_out[i] = True
            short_used = True

        prev_pb_l = bool(pb_l[i])
        prev_pb_s = bool(pb_s[i])

    return (
        pd.Series(long_out,  index=raw_long.index,  dtype=bool),
        pd.Series(short_out, index=raw_short.index, dtype=bool),
    )
