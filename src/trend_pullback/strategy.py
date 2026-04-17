"""
Backtrader Strategy class for Trend Pullback Pro.

Design choice — pre-computed signals approach (option b):
  The full signal DataFrame is computed by signal_engine.compute_signals()
  BEFORE Backtrader starts.  The strategy receives the signal DataFrame
  and reads per-bar values from it inside next().

Rationale:
  - Business logic lives in signal_engine.py (pure, testable, debuggable)
  - strategy.py is responsible only for order management and execution
  - Avoids duplicating or re-implementing indicator logic in Backtrader lines
  - Makes Pine vs Python comparison easy: compare signal_engine output against
    TradingView alerts/data export side-by-side

Execution model:
  - calc_on_every_tick = False (default in Backtrader)
  - Orders are evaluated on bar close, using the NEXT bar's open price.
  - Pine Script uses process_orders_on_close = true, which means entry
    fills at the close price of the signal bar.
  - This is the PRIMARY source of divergence: see README for details.
"""

from __future__ import annotations

import logging

import backtrader as bt
import pandas as pd

from trend_pullback.config import StrategyParams
from trend_pullback.risk import calc_long_levels, calc_short_levels, validate_risk

logger = logging.getLogger(__name__)


class TrendPullbackStrategy(bt.Strategy):
    """Trend Pullback Pro strategy for Backtrader.

    Parameters (passed via params kwarg or cerebro.optstrategy):
        strategy_params (StrategyParams): Validated config object.
        signals_df (pd.DataFrame): Pre-computed signal DataFrame from
            signal_engine.compute_signals().
    """

    params = (
        ("strategy_params", None),
        ("signals_df", None),
        ("stake", 1),
    )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        sp: StrategyParams = self.p.strategy_params
        if sp is None:
            raise ValueError("strategy_params must be provided")
        if self.p.signals_df is None:
            raise ValueError("signals_df must be provided")

        self._sp: StrategyParams = sp
        self._signals: pd.DataFrame = self.p.signals_df

        # Entry order reference
        self._entry_order: bt.Order | None = None

        # Exit bracket: take-profit and stop-loss orders
        self._tp_order: bt.Order | None = None
        self._sl_order: bt.Order | None = None

        # Pending levels stored until entry fills
        self._pending_levels = None

        # Set of entry order refs — used in notify_order to identify entry vs exit
        self._entry_refs: set[int] = set()

    def start(self) -> None:
        logger.info("Backtest started — %d bars", len(self.data))

    def stop(self) -> None:
        logger.info("Backtest finished")

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def next(self) -> None:
        """Called once per completed bar."""
        bar_dt = self.data.datetime.datetime(0)

        # Look up precomputed signals for this bar.
        # Use nearest-match to avoid microsecond/tz mismatch between
        # Backtrader datetime and pandas DatetimeIndex.
        idx = self._signals.index
        pos = idx.searchsorted(bar_dt, side="left")
        if pos >= len(idx):
            return
        # Allow up to 1-second tolerance for datetime matching
        # (guards against microsecond/tz drift between BT and pandas)
        matched_dt = idx[pos]
        bar_dt_pd = pd.Timestamp(bar_dt)
        delta = abs((matched_dt - bar_dt_pd).total_seconds())
        if delta > 1.0:
            return
        row = self._signals.iloc[pos]

        long_sig  = bool(row["long_signal"])
        short_sig = bool(row["short_signal"])

        in_market = self.position.size != 0
        pending_entry = self._entry_order is not None

        if in_market or pending_entry:
            return

        if long_sig:
            self._open_long(row, bar_dt)
        elif short_sig:
            self._open_short(row, bar_dt)

    # ------------------------------------------------------------------
    # Order creation
    # ------------------------------------------------------------------

    def _open_long(self, row: pd.Series, bar_dt: object) -> None:
        """Submit long entry + exit bracket."""
        close    = float(row["close"])
        low      = float(row["low"])
        prev_low = float(self.data.low[-1])
        atr_val  = float(row["atr"])

        levels = calc_long_levels(
            close=close,
            low=low,
            prev_low=prev_low,
            atr_val=atr_val,
            atr_mul=self._sp.atr_mul,
            rr=self._sp.rr,
        )

        if not validate_risk(levels.risk):
            logger.debug("[%s] Long signal skipped — zero/negative risk", bar_dt)
            return

        logger.info(
            "[%s] LONG  entry=%.5f  stop=%.5f  take=%.5f  risk=%.5f",
            bar_dt, close, levels.stop, levels.take, levels.risk,
        )

        # In Backtrader with default exectype=Market, the entry fills at
        # the NEXT bar's open.  This differs from Pine Script
        # (process_orders_on_close=true → fills at signal bar close).
        # See README: "Execution model divergence".
        self._entry_order = self.buy(size=self.p.stake)
        self._entry_refs.add(self._entry_order.ref)
        self._pending_levels = levels

    def _open_short(self, row: pd.Series, bar_dt: object) -> None:
        """Submit short entry + exit bracket."""
        close     = float(row["close"])
        high      = float(row["high"])
        prev_high = float(self.data.high[-1])
        atr_val   = float(row["atr"])

        levels = calc_short_levels(
            close=close,
            high=high,
            prev_high=prev_high,
            atr_val=atr_val,
            atr_mul=self._sp.atr_mul,
            rr=self._sp.rr,
        )

        if not validate_risk(levels.risk):
            logger.debug("[%s] Short signal skipped — zero/negative risk", bar_dt)
            return

        logger.info(
            "[%s] SHORT entry=%.5f  stop=%.5f  take=%.5f  risk=%.5f",
            bar_dt, close, levels.stop, levels.take, levels.risk,
        )

        self._entry_order = self.sell(size=self.p.stake)
        self._entry_refs.add(self._entry_order.ref)
        self._pending_levels = levels


    # ------------------------------------------------------------------
    # Order / trade callbacks
    # ------------------------------------------------------------------

    def notify_order(self, order: bt.Order) -> None:
        """Track order lifecycle; place exit after entry fills."""
        if order.status in (bt.Order.Submitted, bt.Order.Accepted):
            return

        if order.status == bt.Order.Completed:
            is_entry = order.ref in self._entry_refs

            if is_entry:
                self._entry_refs.discard(order.ref)
                direction = "Long" if order.isbuy() else "Short"
                fill_price = order.executed.price
                filled_size = abs(order.executed.size) or self.p.stake
                logger.info(
                    "ENTRY filled %-6s @ %.5f  size=%.6f  cost=%.2f  comm=%.4f",
                    direction, fill_price, filled_size,
                    order.executed.value, order.executed.comm,
                )
                levels = self._pending_levels
                if levels is not None:
                    # Place OCO-style bracket: TP limit + SL stop
                    # Backtrader has no native OCO — we cancel the sibling
                    # manually in notify_order when either leg fills.
                    if order.isbuy():
                        self._tp_order = self.sell(
                            size=filled_size,
                            exectype=bt.Order.Limit,
                            price=levels.take,
                        )
                        self._sl_order = self.sell(
                            size=filled_size,
                            exectype=bt.Order.Stop,
                            price=levels.stop,
                        )
                    else:
                        self._tp_order = self.buy(
                            size=filled_size,
                            exectype=bt.Order.Limit,
                            price=levels.take,
                        )
                        self._sl_order = self.buy(
                            size=filled_size,
                            exectype=bt.Order.Stop,
                            price=levels.stop,
                        )
                self._entry_order = None
                self._pending_levels = None

            else:
                # One of TP or SL filled — cancel the other immediately
                fill_price = order.executed.price
                filled_size = abs(order.executed.size) or self.p.stake
                is_tp = (self._tp_order is not None and order.ref == self._tp_order.ref)
                is_sl = (self._sl_order is not None and order.ref == self._sl_order.ref)
                exit_type = "TP" if is_tp else "SL" if is_sl else "EXIT"
                logger.info(
                    "EXIT [%s] filled @ %.5f  size=%.6f  comm=%.4f",
                    exit_type, fill_price, filled_size, order.executed.comm,
                )
                # Cancel the sibling order
                if is_tp and self._sl_order is not None:
                    self.cancel(self._sl_order)
                    self._sl_order = None
                elif is_sl and self._tp_order is not None:
                    self.cancel(self._tp_order)
                    self._tp_order = None
                self._tp_order = None
                self._sl_order = None

        elif order.status in (bt.Order.Cancelled, bt.Order.Expired, bt.Order.Rejected):
            # Log only unexpected cancellations (not our own cancel() calls)
            if order.ref in self._entry_refs:
                logger.warning("Entry order %s cancelled/rejected", order.ref)
                self._entry_refs.discard(order.ref)
                self._entry_order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """Log closed trade P&L."""
        if not trade.isclosed:
            return
        logger.info(
            "TRADE closed — PnL gross=%.2f  net=%.2f  bars=%d",
            trade.pnl, trade.pnlcomm, trade.barlen,
        )
