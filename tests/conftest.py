"""Shared test fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_ohlcv() -> pd.DataFrame:
    """Generate a small synthetic OHLCV DataFrame for unit tests.

    300 bars of a simple uptrending price with slight noise.
    Close follows a linear ramp; high/low add a fixed band; volume is constant.
    """
    n = 300
    rng = np.random.default_rng(42)

    closes = np.linspace(100.0, 200.0, n) + rng.normal(0, 1.5, n)
    highs  = closes + rng.uniform(0.5, 2.0, n)
    lows   = closes - rng.uniform(0.5, 2.0, n)
    opens  = closes + rng.normal(0, 0.5, n)
    volume = np.full(n, 1000.0)

    index = pd.date_range("2023-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=index,
    )
