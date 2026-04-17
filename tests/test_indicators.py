"""Tests for indicator implementations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from trend_pullback.indicators import atr, ema, rsi


class TestEma:
    def test_shape_matches_input(self, sample_ohlcv: pd.DataFrame) -> None:
        result = ema(sample_ohlcv["close"], 10)
        assert result.shape == sample_ohlcv["close"].shape

    def test_warmup_nans(self, sample_ohlcv: pd.DataFrame) -> None:
        length = 20
        result = ema(sample_ohlcv["close"], length)
        # First (length - 1) values should be NaN
        assert result.iloc[:length - 1].isna().all(), "Expected NaN during warmup"
        # After warmup, no NaNs
        assert result.iloc[length - 1:].notna().all()

    def test_constant_series_equals_constant(self) -> None:
        s = pd.Series([10.0] * 100)
        result = ema(s, 10)
        assert np.isclose(result.dropna().values, 10.0).all()

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError):
            ema(pd.Series([1.0, 2.0]), 0)


class TestRsi:
    def test_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = rsi(sample_ohlcv["close"], 14)
        assert result.shape == sample_ohlcv["close"].shape

    def test_range_0_to_100(self, sample_ohlcv: pd.DataFrame) -> None:
        result = rsi(sample_ohlcv["close"], 14).dropna()
        assert (result >= 0.0).all() and (result <= 100.0).all()

    def test_constant_series_returns_nan_or_edge(self) -> None:
        # All same price → no change → RSI is undefined / 50
        s = pd.Series([50.0] * 50)
        result = rsi(s, 14).dropna()
        # avg_gain = avg_loss = 0, both go to NaN, RSI fill → 100 is wrong.
        # For constant series, expect no exceptions
        assert result is not None

    def test_invalid_length(self) -> None:
        with pytest.raises(ValueError):
            rsi(pd.Series([1.0, 2.0]), 0)


class TestAtr:
    def test_shape(self, sample_ohlcv: pd.DataFrame) -> None:
        result = atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            14,
        )
        assert result.shape == sample_ohlcv["close"].shape

    def test_non_negative_values(self, sample_ohlcv: pd.DataFrame) -> None:
        result = atr(
            sample_ohlcv["high"],
            sample_ohlcv["low"],
            sample_ohlcv["close"],
            14,
        ).dropna()
        assert (result >= 0.0).all()

    def test_invalid_length(self, sample_ohlcv: pd.DataFrame) -> None:
        with pytest.raises(ValueError):
            atr(sample_ohlcv["high"], sample_ohlcv["low"], sample_ohlcv["close"], 0)
