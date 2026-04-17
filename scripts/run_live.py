"""
Live trading runner for Trend Pullback Pro.

Connects to Bybit (testnet or mainnet) via CCXT and runs the
strategy bar-by-bar on a closed-bar schedule.

Usage:
    # Testnet (default):
    python scripts/run_live.py --config configs/live_testnet.yaml

    # Mainnet:
    python scripts/run_live.py --config configs/live_mainnet.yaml

    # Override log level:
    python scripts/run_live.py --config configs/live_testnet.yaml --log-level DEBUG

Environment variables required:
    BYBIT_API_KEY      — Bybit API key
    BYBIT_API_SECRET   — Bybit API secret
    TELEGRAM_BOT_TOKEN — Telegram bot token (optional)
    TELEGRAM_CHAT_ID   — Telegram chat ID (optional)

Loop behaviour:
    1. Wait until the current bar closes (sleep until next bar boundary).
    2. Fetch the last N closed bars from Bybit.
    3. Run signal_engine.compute_signals() on the full DataFrame.
    4. Read signal for the latest closed bar.
    5. If in position: check if TP/SL orders are still open (detect fill).
    6. If no position + signal: place entry → TP + SL orders.
    7. Persist state. Sleep until next bar.

Execution model vs Pine Script:
    Live entry fills at the MARKET price of the next bar open —
    same divergence as the backtest engine. See README for details.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd

from trend_pullback.broker import BybitBroker, BrokerError
from trend_pullback.config import load_config
from trend_pullback.notifier import Notifier
from trend_pullback.risk import calc_long_levels, calc_short_levels, validate_risk
from trend_pullback.signal_engine import compute_signals
from trend_pullback.state import StateManager
from trend_pullback.utils import setup_logging

logger = logging.getLogger(__name__)

# How many bars to fetch for indicator warmup
WARMUP_BARS = 400

# Safety margin: wait this many extra seconds after bar close before fetching
BAR_CLOSE_BUFFER = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trend Pullback Pro — live trading")
    parser.add_argument("--config", "-c", required=True, help="Path to live YAML config")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


# ------------------------------------------------------------------
# Timeframe helpers
# ------------------------------------------------------------------

TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900,
    "30m": 1800, "1h": 3600, "4h": 14400, "1d": 86400,
}


def tf_to_seconds(timeframe: str) -> int:
    """Convert timeframe string to seconds."""
    if timeframe not in TF_SECONDS:
        raise ValueError(f"Unknown timeframe: {timeframe}. Supported: {list(TF_SECONDS)}")
    return TF_SECONDS[timeframe]


def seconds_until_next_bar(timeframe: str) -> float:
    """Return seconds until the next bar closes."""
    tf_sec = tf_to_seconds(timeframe)
    now_ts = time.time()
    next_bar_ts = (int(now_ts / tf_sec) + 1) * tf_sec
    return max(0.0, next_bar_ts - now_ts + BAR_CLOSE_BUFFER)


# ------------------------------------------------------------------
# OHLCV conversion
# ------------------------------------------------------------------

def ohlcv_to_df(raw: list[list]) -> pd.DataFrame:
    """Convert raw CCXT OHLCV list to a pandas DataFrame with DatetimeIndex."""
    df = pd.DataFrame(raw, columns=["datetime", "open", "high", "low", "close", "volume"])
    df["datetime"] = pd.to_datetime(df["datetime"], unit="ms", utc=True).dt.tz_localize(None)
    df = df.set_index("datetime").sort_index()
    return df


# ------------------------------------------------------------------
# Position monitoring
# ------------------------------------------------------------------

def check_exit_fills(
    broker: BybitBroker,
    state_mgr: StateManager,
    notifier: Notifier,
    symbol: str,
) -> bool:
    """Check if TP or SL order has been filled since last check.

    Returns:
        True if position was closed (fill detected), False otherwise.
    """
    st = state_mgr.state
    if not st.in_position:
        return False

    tp_status = broker.get_order_status(st.tp_order_id) if st.tp_order_id else "unknown"
    sl_status = broker.get_order_status(st.sl_order_id) if st.sl_order_id else "unknown"

    logger.debug("TP order %s: %s  |  SL order %s: %s",
                 st.tp_order_id, tp_status, st.sl_order_id, sl_status)

    if tp_status == "closed":
        logger.info("TP filled — closing position state")
        # Cancel the SL order
        if st.sl_order_id:
            broker.cancel_order(st.sl_order_id)
        # Estimate exit price from position entry + levels (best effort)
        notifier.on_tp_hit(symbol, st.entry_price, st.entry_price, pnl_pct=0)
        state_mgr.on_exit("TP")
        return True

    if sl_status in ("closed", "canceled"):
        logger.info("SL filled — closing position state")
        if st.tp_order_id:
            broker.cancel_order(st.tp_order_id)
        notifier.on_sl_hit(symbol, st.entry_price, st.entry_price, pnl_pct=0)
        state_mgr.on_exit("SL")
        return True

    return False


# ------------------------------------------------------------------
# Entry logic
# ------------------------------------------------------------------

def handle_signal(
    broker: BybitBroker,
    state_mgr: StateManager,
    notifier: Notifier,
    row: pd.Series,
    direction: str,
    params,
    symbol: str,
    stake: float,
) -> None:
    """Place entry + TP + SL orders for a given signal bar.

    Args:
        direction: "long" or "short".
    """
    close = float(row["close"])
    atr_val = float(row["atr"])
    bar_dt = str(row.name)

    if direction == "long":
        prev_low = float(row.get("prev_low", row["low"]))
        levels = calc_long_levels(
            close=close,
            low=float(row["low"]),
            prev_low=prev_low,
            atr_val=atr_val,
            atr_mul=params.atr_mul,
            rr=params.rr,
        )
        entry_side, tp_side, sl_side = "buy", "sell", "sell"
    else:
        prev_high = float(row.get("prev_high", row["high"]))
        levels = calc_short_levels(
            close=close,
            high=float(row["high"]),
            prev_high=prev_high,
            atr_val=atr_val,
            atr_mul=params.atr_mul,
            rr=params.rr,
        )
        entry_side, tp_side, sl_side = "sell", "buy", "buy"

    if not validate_risk(levels.risk):
        logger.warning("[%s] Signal skipped — invalid risk: %.6f", bar_dt, levels.risk)
        return

    logger.info(
        "[%s] %s  close=%.5f  stop=%.5f  take=%.5f  risk=%.5f",
        bar_dt, direction.upper(), close, levels.stop, levels.take, levels.risk,
    )

    try:
        # 1. Market entry
        entry_result = broker.place_market_order(entry_side, stake)
        logger.info("Entry order placed: %s", entry_result.order_id)

        # Small wait for the entry to fill on the exchange
        time.sleep(1.0)

        # 2. Take-profit limit order
        tp_result = broker.place_limit_order(tp_side, stake, levels.take)
        logger.info("TP order placed: %s @ %.5f", tp_result.order_id, levels.take)

        # 3. Stop-loss stop-market order
        sl_result = broker.place_stop_order(sl_side, stake, levels.stop)
        logger.info("SL order placed: %s @ %.5f", sl_result.order_id, levels.stop)

        # 4. Persist state
        state_mgr.on_entry(
            side=direction,
            size=stake,
            entry_price=close,
            tp_order_id=tp_result.order_id,
            sl_order_id=sl_result.order_id,
            signal_bar=bar_dt,
        )

        # 5. Notify
        if direction == "long":
            notifier.on_long_entry(symbol, close, levels.stop, levels.take, stake, bar_dt)
        else:
            notifier.on_short_entry(symbol, close, levels.stop, levels.take, stake, bar_dt)

    except BrokerError as exc:
        logger.error("Failed to place orders: %s", exc)
        notifier.on_error(symbol, f"Order placement failed: {exc}")


# ------------------------------------------------------------------
# Main loop
# ------------------------------------------------------------------

def run_loop(config_path: str) -> None:
    """Main live trading loop. Runs until interrupted."""
    cfg = load_config(config_path)
    sp  = cfg.strategy
    bp  = cfg.backtest

    symbol    = bp.symbol.replace("USDT", "/USDT")  # "ETHUSDT" → "ETH/USDT"
    timeframe = bp.timeframe
    stake     = bp.stake

    # Credentials from environment
    api_key    = os.environ.get("BYBIT_API_KEY", "")
    api_secret = os.environ.get("BYBIT_API_SECRET", "")
    testnet    = cfg.live.testnet if hasattr(cfg, "live") else True

    if not api_key or not api_secret:
        logger.error("BYBIT_API_KEY and BYBIT_API_SECRET must be set in environment")
        sys.exit(1)

    # Initialise components
    broker   = BybitBroker(api_key, api_secret, symbol, testnet=testnet)
    notifier = Notifier()
    state_mgr = StateManager(f"state/{bp.symbol.lower()}_state.json")

    notifier.on_start(symbol, timeframe, testnet)
    logger.info("Live bot started — symbol=%s  tf=%s  stake=%s  testnet=%s",
                symbol, timeframe, stake, testnet)

    # Graceful shutdown on Ctrl+C / SIGTERM
    running = True
    def _shutdown(sig, frame):
        nonlocal running
        logger.info("Shutdown signal received")
        running = False
    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # ----------------------------------------------------------------
    # Main loop
    # ----------------------------------------------------------------
    while running:
        try:
            # 1. Wait for bar close
            wait_sec = seconds_until_next_bar(timeframe)
            logger.info("Sleeping %.1f s until next bar close…", wait_sec)
            _interruptible_sleep(wait_sec, check=lambda: running)

            if not running:
                break

            bar_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")
            logger.info("=== Bar closed at %s ===", bar_time)

            # 2. Check if existing TP/SL filled
            if state_mgr.state.in_position:
                check_exit_fills(broker, state_mgr, notifier, symbol)

            # 3. Fetch OHLCV and compute signals
            raw = broker.fetch_ohlcv(timeframe, limit=WARMUP_BARS)
            if not raw:
                logger.warning("Empty OHLCV response — skipping bar")
                continue

            df = ohlcv_to_df(raw)

            # Add prev_low / prev_high columns needed by risk.py
            df["prev_low"]  = df["low"].shift(1)
            df["prev_high"] = df["high"].shift(1)

            signals_df = compute_signals(df, sp)

            # Latest closed bar
            latest = signals_df.iloc[-1]
            long_sig  = bool(latest["long_signal"])
            short_sig = bool(latest["short_signal"])

            logger.info(
                "Latest bar %s — long_sig=%s  short_sig=%s  in_position=%s",
                latest.name, long_sig, short_sig, state_mgr.state.in_position,
            )

            # 4. Skip if already in a position
            if state_mgr.state.in_position:
                logger.debug("In position — skipping entry check")
                continue

            # 5. Act on signals
            if long_sig:
                handle_signal(broker, state_mgr, notifier, latest,
                              "long", sp, symbol, stake)
            elif short_sig:
                handle_signal(broker, state_mgr, notifier, latest,
                              "short", sp, symbol, stake)
            else:
                logger.debug("No signal this bar")

        except BrokerError as exc:
            logger.error("BrokerError: %s", exc)
            notifier.on_error(symbol, str(exc))
            # Back off before retrying
            time.sleep(30)

        except Exception as exc:
            logger.exception("Unexpected error in main loop: %s", exc)
            notifier.on_error(symbol, f"Unexpected: {exc}")
            time.sleep(30)

    notifier.on_stop("Normal shutdown")
    logger.info("Bot stopped")


def _interruptible_sleep(seconds: float, check) -> None:
    """Sleep in 1-second chunks to allow clean shutdown."""
    elapsed = 0.0
    while elapsed < seconds and check():
        time.sleep(min(1.0, seconds - elapsed))
        elapsed += 1.0


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    run_loop(args.config)


if __name__ == "__main__":
    main()
