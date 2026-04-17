"""
Risk level calculations for Trend Pullback Pro.

Pure functions, no side effects, no Backtrader dependency.
These match the Pine Script order logic exactly:
  long:  stop = min(low, low[1]) - atr * atr_mul
         risk = close - stop
         take = close + risk * rr
  short: stop = max(high, high[1]) + atr * atr_mul
         risk = stop - close
         take = close - risk * rr
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LevelResult:
    """Entry levels for one trade."""
    stop: float
    risk: float
    take: float


def calc_long_levels(
    close: float,
    low: float,
    prev_low: float,
    atr_val: float,
    atr_mul: float,
    rr: float,
) -> LevelResult:
    """Calculate stop, risk and take-profit for a long entry.

    Args:
        close:    Close price of the signal bar.
        low:      Low of the signal bar.
        prev_low: Low of the previous bar.
        atr_val:  ATR value on the signal bar.
        atr_mul:  ATR stop multiplier.
        rr:       Risk-reward ratio.

    Returns:
        LevelResult with stop, risk, take.
    """
    stop = min(low, prev_low) - atr_val * atr_mul
    risk = close - stop
    take = close + risk * rr
    return LevelResult(stop=stop, risk=risk, take=take)


def calc_short_levels(
    close: float,
    high: float,
    prev_high: float,
    atr_val: float,
    atr_mul: float,
    rr: float,
) -> LevelResult:
    """Calculate stop, risk and take-profit for a short entry.

    Args:
        close:     Close price of the signal bar.
        high:      High of the signal bar.
        prev_high: High of the previous bar.
        atr_val:   ATR value on the signal bar.
        atr_mul:   ATR stop multiplier.
        rr:        Risk-reward ratio.

    Returns:
        LevelResult with stop, risk, take.
    """
    stop = max(high, prev_high) + atr_val * atr_mul
    risk = stop - close
    take = close - risk * rr
    return LevelResult(stop=stop, risk=risk, take=take)


def validate_risk(risk: float, min_tick: float = 1e-8) -> bool:
    """Validate that risk is positive and above minimum tick.

    Matches Pine Script condition: if longRisk > syminfo.mintick

    Args:
        risk:     Calculated risk amount.
        min_tick: Minimum acceptable risk (default: near-zero float).

    Returns:
        True if risk is acceptable, False otherwise.
    """
    return risk > min_tick
