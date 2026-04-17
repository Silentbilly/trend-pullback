"""Tests for signal_engine — pullback counter and anti-duplicate logic."""

from __future__ import annotations

import pandas as pd
import pytest

from trend_pullback.config import StrategyParams
from trend_pullback.signal_engine import (
    _apply_block_repeats,
    _compute_pb_count,
    compute_signals,
)


class TestPbCounter:
    """Test consecutive pullback bar counter."""

    def test_simple_streak(self) -> None:
        # F F T T T F T → counts: 0 0 1 2 3 0 1
        bars = pd.Series([False, False, True, True, True, False, True])
        counts = _compute_pb_count(bars)
        assert list(counts) == [0, 0, 1, 2, 3, 0, 1]

    def test_all_false(self) -> None:
        bars = pd.Series([False] * 5)
        counts = _compute_pb_count(bars)
        assert list(counts) == [0, 0, 0, 0, 0]

    def test_all_true(self) -> None:
        bars = pd.Series([True] * 5)
        counts = _compute_pb_count(bars)
        assert list(counts) == [1, 2, 3, 4, 5]

    def test_single_bar(self) -> None:
        bars = pd.Series([True])
        counts = _compute_pb_count(bars)
        assert list(counts) == [1]

    def test_index_preserved(self) -> None:
        idx = pd.date_range("2023-01-01", periods=3, freq="1h")
        bars = pd.Series([True, True, False], index=idx)
        counts = _compute_pb_count(bars)
        assert list(counts.index) == list(idx)


class TestBlockRepeats:
    """Test anti-duplicate signal gate."""

    def _make_series(self, values: list[bool]) -> pd.Series:
        return pd.Series(values, dtype=bool)

    def test_only_first_signal_per_pullback(self) -> None:
        # pb_long:  F T T T F
        # raw_long: F T T T F  ← 3 potential signals in same pullback
        # expected: F T F F F  ← only first fires
        pb   = self._make_series([False, True,  True,  True,  False])
        raw  = self._make_series([False, True,  True,  True,  False])
        raw_s = self._make_series([False] * 5)
        pb_s  = self._make_series([False] * 5)

        long_out, short_out = _apply_block_repeats(raw, raw_s, pb, pb_s)
        assert list(long_out)  == [False, True, False, False, False]
        assert list(short_out) == [False] * 5

    def test_reset_on_new_pullback(self) -> None:
        # Two separate pullback cycles, each should allow one signal
        # pb:   F T T F F T T F
        # raw:  F T T F F T T F
        # exp:  F T F F F T F F
        pb   = self._make_series([False, True,  True,  False, False, True,  True,  False])
        raw  = self._make_series([False, True,  True,  False, False, True,  True,  False])
        raw_s = self._make_series([False] * 8)
        pb_s  = self._make_series([False] * 8)

        long_out, _ = _apply_block_repeats(raw, raw_s, pb, pb_s)
        assert list(long_out) == [False, True, False, False, False, True, False, False]

    def test_no_signal_fires_if_no_raw(self) -> None:
        pb   = self._make_series([True] * 5)
        raw  = self._make_series([False] * 5)
        raw_s = self._make_series([False] * 5)
        pb_s  = self._make_series([False] * 5)

        long_out, _ = _apply_block_repeats(raw, raw_s, pb, pb_s)
        assert not any(long_out)

    def test_long_and_short_independent(self) -> None:
        pb_l = self._make_series([False, True, True, False])
        pb_s = self._make_series([True,  True, False, False])
        raw_l = self._make_series([False, True, True, False])
        raw_s = self._make_series([True,  True, False, False])

        long_out, short_out = _apply_block_repeats(raw_l, raw_s, pb_l, pb_s)
        # Long: first signal at bar 1
        assert list(long_out)  == [False, True, False, False]
        # Short: first signal at bar 0
        assert list(short_out) == [True,  False, False, False]


class TestComputeSignals:
    """Integration test: compute_signals returns required columns."""

    REQUIRED_COLUMNS = [
        "ema_fast", "ema_slow", "ema_pull", "rsi", "atr",
        "up_trend", "down_trend",
        "pb_long_bar", "pb_short_bar",
        "pb_long_count", "pb_short_count",
        "bull_trigger", "bear_trigger",
        "raw_long_signal", "raw_short_signal",
        "long_signal", "short_signal",
        "long_stop_candidate", "short_stop_candidate",
    ]

    def test_all_columns_present(self, sample_ohlcv: pd.DataFrame) -> None:
        sp = StrategyParams()
        result = compute_signals(sample_ohlcv, sp)
        for col in self.REQUIRED_COLUMNS:
            assert col in result.columns, f"Missing column: {col}"

    def test_output_same_length(self, sample_ohlcv: pd.DataFrame) -> None:
        sp = StrategyParams()
        result = compute_signals(sample_ohlcv, sp)
        assert len(result) == len(sample_ohlcv)

    def test_signals_are_bool(self, sample_ohlcv: pd.DataFrame) -> None:
        sp = StrategyParams()
        result = compute_signals(sample_ohlcv, sp)
        assert result["long_signal"].dtype == bool
        assert result["short_signal"].dtype == bool

    def test_no_simultaneous_long_and_short(self, sample_ohlcv: pd.DataFrame) -> None:
        sp = StrategyParams()
        result = compute_signals(sample_ohlcv, sp)
        both = result["long_signal"] & result["short_signal"]
        # There should be no bar where both signals fire
        # (up_trend and down_trend are mutually exclusive)
        assert not both.any(), "Long and short signal cannot fire on the same bar"
