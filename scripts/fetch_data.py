import ccxt
import pandas as pd
from pathlib import Path

TF_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def get_exchange():
    exchange = ccxt.bybit({
        "enableRateLimit": True,
    })
    exchange.load_markets()
    return exchange


def validate_timeframe(timeframe: str):
    if timeframe not in TF_MS:
        raise ValueError(f"Unsupported timeframe: {timeframe}")


def build_params(category: str | None = None) -> dict:
    params = {}
    if category:
        params["category"] = category
    return params


def fetch_ohlcv_history(
    symbol: str = "ETH/USDT",
    timeframe: str = "15m",
    target: int = 40_000,
    limit: int = 1000,
    category: str | None = None,
) -> pd.DataFrame:
    validate_timeframe(timeframe)

    exchange = get_exchange()
    params = build_params(category)

    latest = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=1, params=params)
    if not latest:
        raise RuntimeError("Could not fetch latest OHLCV bar.")

    latest_ts = latest[-1][0]
    since = latest_ts - target * TF_MS[timeframe]

    all_ohlcv = []

    while len(all_ohlcv) < target:
        batch = exchange.fetch_ohlcv(
            symbol,
            timeframe=timeframe,
            since=since,
            limit=limit,
            params=params,
        )

        if not batch:
            break

        all_ohlcv.extend(batch)

        last_ts = batch[-1][0]
        next_since = last_ts + TF_MS[timeframe]

        if next_since <= since:
            break

        since = next_since
        print(f"Fetched {len(all_ohlcv)} bars...", end="\r")

        if len(batch) < limit:
            break

    df = pd.DataFrame(
        all_ohlcv,
        columns=["datetime", "open", "high", "low", "close", "volume"],
    )

    if df.empty:
        raise RuntimeError("Downloaded dataframe is empty.")

    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True)
    df = df.drop_duplicates(subset="datetime").sort_values("datetime").reset_index(drop=True)
    df = df.tail(target).reset_index(drop=True)

    numeric_cols = ["open", "high", "low", "close", "volume"]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna().reset_index(drop=True)

    return df


def check_gaps(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    expected = pd.Timedelta(milliseconds=TF_MS[timeframe])

    gaps = df["datetime"].diff()
    gap_rows = df.loc[gaps[gaps != expected].index].copy()

    if gap_rows.empty:
        print("No gaps detected.")
        return gap_rows

    gap_rows["prev_datetime"] = df["datetime"].shift(1).loc[gap_rows.index]
    gap_rows["gap"] = gap_rows["datetime"] - gap_rows["prev_datetime"]

    print(f"Detected {len(gap_rows)} gap(s).")
    return gap_rows[["prev_datetime", "datetime", "gap"]]


def save_dataframe(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    target: int,
    exchange_name: str = "bybit",
    category: str | None = None,
) -> Path:
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    symbol_slug = symbol.replace("/", "").replace(":", "_")
    market_tag = category if category else "spot"
    out_path = Path("data/raw") / f"{exchange_name}_{market_tag}_{symbol_slug}_{timeframe}_{target}.csv"

    df.to_csv(out_path, index=False)
    return out_path


def main():
    symbol = "ETH/USDT"
    timeframe = "15m"
    target = 40_000
    limit = 1000

    # Для spot оставь None
    # Для Bybit perpetual/futures обычно нужно "linear"
    category = None
    # category = "linear"

    df = fetch_ohlcv_history(
        symbol=symbol,
        timeframe=timeframe,
        target=target,
        limit=limit,
        category=category,
    )

    gap_report = check_gaps(df, timeframe)
    out_path = save_dataframe(
        df=df,
        symbol=symbol,
        timeframe=timeframe,
        target=target,
        exchange_name="bybit",
        category=category,
    )

    print(f"\nSaved {len(df)} bars → {out_path}")
    print(f"Range: {df['datetime'].iloc[0]}  →  {df['datetime'].iloc[-1]}")

    if not gap_report.empty:
        print("\nGap report:")
        print(gap_report.head(20).to_string(index=False))


if __name__ == "__main__":
    main()