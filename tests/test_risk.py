"""Tests for risk level calculations."""

from __future__ import annotations

import pytest

from trend_pullback.risk import LevelResult, calc_long_levels, calc_short_levels, validate_risk


class TestCalcLongLevels:
    def test_basic_values(self) -> None:
        result = calc_long_levels(
            close=100.0, low=97.0, prev_low=98.0,
            atr_val=2.0, atr_mul=1.5, rr=2.0,
        )
        # stop = min(97, 98) - 2.0 * 1.5 = 97.0 - 3.0 = 94.0
        # risk = 100.0 - 94.0 = 6.0
        # take = 100.0 + 6.0 * 2.0 = 112.0
        assert isinstance(result, LevelResult)
        assert result.stop == pytest.approx(94.0)
        assert result.risk == pytest.approx(6.0)
        assert result.take == pytest.approx(112.0)

    def test_prev_low_is_lower(self) -> None:
        # Uses prev_low if it is lower
        result = calc_long_levels(
            close=100.0, low=98.0, prev_low=95.0,
            atr_val=1.0, atr_mul=1.0, rr=1.0,
        )
        # stop = min(98, 95) - 1.0 = 94.0
        assert result.stop == pytest.approx(94.0)
        assert result.risk == pytest.approx(6.0)

    def test_rr_scales_take(self) -> None:
        r1 = calc_long_levels(100.0, 97.0, 97.0, 1.0, 1.0, 1.0)
        r2 = calc_long_levels(100.0, 97.0, 97.0, 1.0, 1.0, 2.0)
        assert r2.take == pytest.approx(r1.take + r1.risk)  # additional 1R


class TestCalcShortLevels:
    def test_basic_values(self) -> None:
        result = calc_short_levels(
            close=100.0, high=103.0, prev_high=102.0,
            atr_val=2.0, atr_mul=1.5, rr=2.0,
        )
        # stop = max(103, 102) + 3.0 = 106.0
        # risk = 106.0 - 100.0 = 6.0
        # take = 100.0 - 12.0 = 88.0
        assert result.stop == pytest.approx(106.0)
        assert result.risk == pytest.approx(6.0)
        assert result.take == pytest.approx(88.0)

    def test_prev_high_is_higher(self) -> None:
        result = calc_short_levels(
            close=100.0, high=101.0, prev_high=105.0,
            atr_val=1.0, atr_mul=1.0, rr=1.0,
        )
        # stop = max(101, 105) + 1.0 = 106.0
        assert result.stop == pytest.approx(106.0)


class TestValidateRisk:
    def test_positive_risk_is_valid(self) -> None:
        assert validate_risk(5.0) is True

    def test_zero_risk_is_invalid(self) -> None:
        assert validate_risk(0.0) is False

    def test_negative_risk_is_invalid(self) -> None:
        assert validate_risk(-1.0) is False

    def test_tiny_positive_above_min_tick(self) -> None:
        assert validate_risk(1e-7) is True

    def test_tiny_positive_below_min_tick(self) -> None:
        assert validate_risk(1e-9) is False
