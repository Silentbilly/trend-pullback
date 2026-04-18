"""
Configuration models for Trend Pullback Pro.

Uses Pydantic v2 for validation and type safety.
Load config via load_config(path) which reads a YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import yaml
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Strategy parameters
# ---------------------------------------------------------------------------

class StrategyParams(BaseModel):
    """All strategy parameters — mirrors Pine Script inputs 1:1."""

    # Trend EMAs
    fast_len: Annotated[int, Field(ge=1, description="Fast EMA period")] = 50
    slow_len: Annotated[int, Field(ge=1, description="Slow EMA period")] = 200
    pull_len: Annotated[int, Field(ge=1, description="Pullback EMA period")] = 21

    # Slope filter
    use_slope: bool = True
    slope_lb: Annotated[int, Field(ge=1, description="Bars lookback for slope check")] = 5

    # Pullback window
    min_pb: Annotated[int, Field(ge=1, description="Min consecutive pullback bars")] = 1
    max_pb: Annotated[int, Field(ge=1, description="Max consecutive pullback bars")] = 8

    # RSI filter
    use_rsi: bool = True
    rsi_len: Annotated[int, Field(ge=1, description="RSI period")] = 14
    rsi_long_min: Annotated[int, Field(ge=1, le=100, description="RSI long threshold")] = 45
    rsi_short_max: Annotated[int, Field(ge=1, le=100, description="RSI short threshold")] = 55

    # Anti-duplicate signal
    block_repeats: bool = True

    # Risk
    atr_len: Annotated[int, Field(ge=1, description="ATR period")] = 14
    atr_mul: Annotated[float, Field(gt=0.0, description="ATR stop multiplier")] = 1.5
    rr: Annotated[float, Field(ge=0.5, description="Risk-reward ratio")] = 2.0

    @model_validator(mode="after")
    def _validate_ema_order(self) -> "StrategyParams":
        if self.fast_len >= self.slow_len:
            raise ValueError(
                f"fast_len ({self.fast_len}) must be less than slow_len ({self.slow_len})"
            )
        if self.min_pb > self.max_pb:
            raise ValueError(
                f"min_pb ({self.min_pb}) must be <= max_pb ({self.max_pb})"
            )
        return self


# ---------------------------------------------------------------------------
# Backtest parameters
# ---------------------------------------------------------------------------

class BacktestParams(BaseModel):
    """Parameters for running a single backtest."""

    symbol: str = "BTCUSDT"
    timeframe: str = "1h"
    data_path: str = "data/raw/BTCUSDT_1h.csv"
    initial_capital: Annotated[float, Field(gt=0.0)] = 10_000.0
    commission_pct: Annotated[float, Field(ge=0.0, description="Commission percent per trade")] = 0.06
    stake: Annotated[float, Field(gt=0.0, description="Fixed position size (units or contracts). Use decimals for crypto (e.g. 0.01 BTC)")] = 0.01
    leverage: int = 1

# ---------------------------------------------------------------------------
# Root config
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    """Root configuration object loaded from YAML."""

    strategy: StrategyParams = StrategyParams()
    backtest: BacktestParams = BacktestParams()


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> AppConfig:
    """Load and validate configuration from a YAML file.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated AppConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        pydantic.ValidationError: If values fail validation.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        raw: dict = yaml.safe_load(f) or {}

    return AppConfig.model_validate(raw)
