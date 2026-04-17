"""Tests for config loading and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from trend_pullback.config import AppConfig, BacktestParams, StrategyParams, load_config


class TestStrategyParams:
    def test_defaults_valid(self) -> None:
        sp = StrategyParams()
        assert sp.fast_len == 50
        assert sp.slow_len == 200
        assert sp.pull_len == 21

    def test_fast_must_be_less_than_slow(self) -> None:
        with pytest.raises(ValidationError, match="fast_len"):
            StrategyParams(fast_len=200, slow_len=50)

    def test_fast_equal_slow_raises(self) -> None:
        with pytest.raises(ValidationError, match="fast_len"):
            StrategyParams(fast_len=100, slow_len=100)

    def test_min_pb_gt_max_pb_raises(self) -> None:
        with pytest.raises(ValidationError, match="min_pb"):
            StrategyParams(min_pb=10, max_pb=5)

    def test_atr_mul_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            StrategyParams(atr_mul=0.0)

    def test_rr_minimum(self) -> None:
        with pytest.raises(ValidationError):
            StrategyParams(rr=0.4)

    def test_rsi_range(self) -> None:
        with pytest.raises(ValidationError):
            StrategyParams(rsi_long_min=0)
        with pytest.raises(ValidationError):
            StrategyParams(rsi_short_max=101)


class TestLoadConfig:
    def test_load_base_yaml(self) -> None:
        cfg = load_config("configs/base.yaml")
        assert isinstance(cfg, AppConfig)
        assert cfg.strategy.fast_len == 50
        assert cfg.backtest.initial_capital == 10_000.0

    def test_load_sample_yaml(self) -> None:
        cfg = load_config("configs/sample_backtest.yaml")
        assert cfg.strategy.use_rsi is False  # overridden in sample

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.yaml")

    def test_partial_override(self, tmp_path: Path) -> None:
        yaml_content = "strategy:\n  fast_len: 30\n  slow_len: 100\n"
        cfg_file = tmp_path / "partial.yaml"
        cfg_file.write_text(yaml_content)
        cfg = load_config(cfg_file)
        assert cfg.strategy.fast_len == 30
        # Unspecified fields use defaults
        assert cfg.strategy.pull_len == 21
