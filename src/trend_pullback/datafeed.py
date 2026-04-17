"""
Data loading and Backtrader feed preparation for Trend Pullback Pro.

Expected CSV format:
    datetime,open,high,low,close,volume
    2023-01-01 00:00:00,16500.0,16600.0,16450.0,16550.0,1234.5
    ...

datetime can be:
  - ISO-8601 string (e.g. "2023-01-01 00:00:00", "2023-01-01T00:00:00")
  - Unix timestamp (integer seconds)

The loader returns a clean pandas DataFrame.
A separate function wraps it as a Backtrader PandasData feed.
"""

from __future__ import annotations

from pathlib import Path

import backtrader as bt
import pandas as pd

REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def load_ohlcv(path: str | Path) -> pd.DataFrame:
    """Load OHLCV data from a CSV file.

    Args:
        path: Path to the CSV file.

    Returns:
        DataFrame sorted by datetime ascending, with a DatetimeIndex.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing or data is empty.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    # Parse datetime — try column named 'datetime' or 'date' or 'timestamp'
    dt_col = _find_datetime_column(df)
    df[dt_col] = pd.to_datetime(df[dt_col], utc=False)
    df = df.rename(columns={dt_col: "datetime"})
    df = df.set_index("datetime")
    df = df.sort_index()

    if df.empty:
        raise ValueError(f"Data file is empty: {csv_path}")

    # Cast numeric columns
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Basic sanity check
    _validate_ohlcv(df)

    return df


def make_bt_feed(df: pd.DataFrame) -> bt.feeds.PandasData:
    """Wrap a loaded OHLCV DataFrame as a Backtrader PandasData feed.

    Args:
        df: DataFrame from load_ohlcv() — must have DatetimeIndex.

    Returns:
        Backtrader PandasData instance ready to be added to a Cerebro.
    """
    return bt.feeds.PandasData(
        dataname=df,
        datetime=None,   # use index as datetime
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        openinterest=-1, # not available
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_datetime_column(df: pd.DataFrame) -> str:
    """Find the datetime column by common names."""
    for candidate in ("datetime", "date", "timestamp", "time"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No datetime column found. Expected one of: datetime, date, timestamp, time. "
        f"Got: {list(df.columns)}"
    )


def _validate_ohlcv(df: pd.DataFrame) -> None:
    """Run basic OHLCV sanity checks and warn on issues."""
    # high >= low
    bad_hl = (df["high"] < df["low"]).sum()
    if bad_hl:
        import warnings
        warnings.warn(f"{bad_hl} bars have high < low — check your data.", stacklevel=3)

    # close within range
    bad_close = ((df["close"] > df["high"]) | (df["close"] < df["low"])).sum()
    if bad_close:
        import warnings
        warnings.warn(f"{bad_close} bars have close outside [low, high].", stacklevel=3)
