# Trend Pullback Pro — Python Backtest Engine

Python port of the **Trend Pullback Pro** Pine Script v6 strategy.
Uses **Backtrader** as the backtest engine with a clean, config-driven,
`src`-layout project structure designed for extension to live/paper trading.

---

## Project Structure

```
trend-pullback/
├─ pyproject.toml               # Build config + dependencies
├─ README.md
├─ .gitignore
│
├─ configs/
│  ├─ base.yaml                 # All parameters with defaults
│  └─ sample_backtest.yaml      # Example override config
│
├─ data/
│  ├─ raw/                      # Place your CSV files here
│  └─ processed/                # Reserved for future preprocessing
│
├─ output/                      # Auto-created by run_backtest.py
│  ├─ summary.csv
│  ├─ trades.csv
│  └─ equity.csv
│
├─ scripts/
│  ├─ run_backtest.py           # Single backtest CLI
│  └─ run_parameter_sweep.py    # Parameter grid runner
│
├─ src/
│  └─ trend_pullback/
│     ├─ __init__.py
│     ├─ config.py              # Pydantic models + YAML loader
│     ├─ indicators.py          # EMA, RSI, ATR (pure pandas/numpy)
│     ├─ signal_engine.py       # All strategy logic → signal DataFrame
│     ├─ risk.py                # Stop/take-profit calculations
│     ├─ datafeed.py            # CSV loader + Backtrader feed
│     ├─ strategy.py            # Backtrader Strategy class
│     ├─ analyzers.py           # Custom BT analyzers
│     ├─ reporting.py           # Metrics + CSV output
│     └─ utils.py               # Logging setup
│
└─ tests/
   ├─ conftest.py               # Shared fixtures
   ├─ test_config.py
   ├─ test_indicators.py
   ├─ test_signal_engine.py
   └─ test_risk.py
```

---

## Installation

Requires Python 3.11+.

```bash
# Clone / unzip the project, then from project root:
pip install -e ".[dev]"
```

This installs the package in editable mode with all runtime and dev dependencies.

---

## Data Format

Place OHLCV CSV files in `data/raw/`.

**Required columns** (header names, case-insensitive):

| Column     | Type   | Description         |
|------------|--------|---------------------|
| `datetime` | string | ISO-8601 or Unix ts |
| `open`     | float  | Open price          |
| `high`     | float  | High price          |
| `low`      | float  | Low price           |
| `close`    | float  | Close price         |
| `volume`   | float  | Volume              |

Example row:
```
datetime,open,high,low,close,volume
2023-01-01 00:00:00,16500.0,16600.0,16450.0,16550.0,1234.5
```

Update `backtest.data_path` in your config to point to the file.

---

## Running a Single Backtest

```bash
python scripts/run_backtest.py --config configs/sample_backtest.yaml
```

Options:
```
--config / -c   Path to YAML config (required)
--output / -o   Output directory (default: output/)
--log-level     DEBUG | INFO | WARNING | ERROR (default: INFO)
```

Results are saved to `output/`:
- `summary.csv`  — one-row performance summary
- `trades.csv`   — all closed trades
- `equity.csv`   — bar-by-bar equity curve

---

## Changing Parameters (YAML Config)

Copy `configs/base.yaml` and edit the values you want to change:

```yaml
strategy:
  fast_len: 50          # Fast EMA period
  slow_len: 200         # Slow EMA period
  pull_len: 21          # Pullback EMA period
  use_slope: true       # Require EMA slope in trend direction
  slope_lb: 5           # Bars for slope lookback
  min_pb: 1             # Min consecutive pullback bars
  max_pb: 8             # Max consecutive pullback bars
  use_rsi: true         # Enable RSI filter
  rsi_len: 14
  rsi_long_min: 45      # RSI >= this for long
  rsi_short_max: 55     # RSI <= this for short
  block_repeats: true   # One signal per pullback cycle
  atr_len: 14
  atr_mul: 1.5          # Stop = ATR * atr_mul beyond swing low/high
  rr: 2.0               # Take-profit = risk * rr

backtest:
  data_path: "data/raw/BTCUSDT_1h.csv"
  initial_capital: 10000.0
  commission_pct: 0.06
  stake: 1
```

Then run:
```bash
python scripts/run_backtest.py --config configs/my_config.yaml
```

---

## Parameter Sweep

```bash
python scripts/run_parameter_sweep.py --config configs/base.yaml
```

Edit the grid at the top of `scripts/run_parameter_sweep.py`:

```python
FAST_LEN_VALUES = [21, 50, 89]
PULL_LEN_VALUES = [9, 21, 34]
ATR_MUL_VALUES  = [1.0, 1.5, 2.0]
RR_VALUES       = [1.5, 2.0, 2.5]
```

Results saved to `output/sweep/sweep_results.csv`, sorted by net profit %.

---

## Running Tests

```bash
pytest
# or with coverage:
pytest --cov=src/trend_pullback
```

---

## What is Implemented

| Feature | Status |
|---|---|
| EMA fast / slow / pullback | ✅ |
| ATR (Wilder's RMA) | ✅ |
| RSI (Wilder's RMA) | ✅ |
| Trend filter (base + slope) | ✅ |
| Pullback bar detection | ✅ |
| Consecutive pullback counter | ✅ |
| Pullback window gate (min/max bars) | ✅ |
| Trigger bars (close > prev high/low) | ✅ |
| RSI filter (on/off) | ✅ |
| Anti-duplicate signal (one per pullback) | ✅ |
| Long entry: stop below swing low − ATR | ✅ |
| Short entry: stop above swing high + ATR | ✅ |
| Fixed take-profit (RR-based) | ✅ |
| Risk validation (positive risk check) | ✅ |
| Commission from config | ✅ |
| No pyramiding | ✅ |
| Trade logging | ✅ |
| summary / trades / equity CSV output | ✅ |
| Parameter sweep script | ✅ |
| Config-driven (YAML + Pydantic) | ✅ |

## What is NOT Implemented

| Feature | Reason |
|---|---|
| Breakeven / trailing stop | Excluded by design |
| Partial exits | Excluded by design |
| "Entry after pullback ends" filter | Excluded by design |
| Position sizing model | Fixed stake only (extend in risk.py) |
| Live / paper execution | Foundation ready; add broker adapter |
| Visual charts / notebooks | Out of scope for v1 |
| TA-Lib dependency | Not required |

---

## Design Decisions

### Pre-computed signals (option b)

The strategy uses **pre-computed signal columns** fed into the Backtrader `Strategy.next()` loop rather than re-computing indicators inside `next()`.

Rationale:
- All business logic lives in `signal_engine.py` — pure, testable, no Backtrader coupling.
- The exact same function is used for both backtesting and Pine vs Python comparison.
- Debugging is straightforward: inspect the signal DataFrame directly with pandas.
- Easy to extend: adding new columns to `compute_signals()` does not touch `strategy.py`.

---

## Execution Model Divergence — Pine Script vs Python

**This is the most important source of differences between TradingView and this engine.**

| Aspect | Pine Script | Backtrader (this project) |
|---|---|---|
| Order fill timing | `process_orders_on_close = true` → entry fills at the **close** of the signal bar | Default `Market` order → entry fills at the **open of the NEXT bar** |
| Signal computation | Calculated on every tick, then resolved at bar close | Computed once per completed bar (same bar-close logic) |
| Stop/Take placement | Set immediately at entry close | Set after entry fill confirmation in `notify_order()` — at next bar open |
| Commission model | `strategy.commission.percent` | Backtrader percent commission — same formula |

**Practical consequence:**  
Pine Script entries fill at the signal bar's close price. Backtrader entries fill at the next bar's open. For liquid instruments on hourly+ timeframes the gap is typically small, but it will produce different entry prices, stop levels, and P&L figures.

**To minimise divergence:**  
Use `exectype=bt.Order.Close` for entries (fills at bar close). This is a known Backtrader feature but requires care with order sequencing — it is left as an extension point.

---

## Extending to Live / Paper Trading

The architecture is designed for this:

1. `signal_engine.compute_signals()` works on any OHLCV DataFrame — connect a live data source.
2. `risk.calc_long_levels()` / `calc_short_levels()` are pure functions — call from any execution layer.
3. Replace `datafeed.py` with a streaming feed (CCXT, ByBit WS, etc.).
4. Replace Backtrader broker with a live broker adapter.
5. `config.py` / YAML config requires no changes.

---

## For Full Comparison with TradingView

> **Always compare execution assumptions before drawing conclusions from backtest results.**
>
> TradingView Pine Script uses `process_orders_on_close = true` and `calc_on_order_fills = true`.
> This Python engine fills at next bar open by default.
> Export TradingView trade list and compare entry/exit prices against `output/trades.csv`
> to quantify the divergence for your specific symbol and timeframe.
