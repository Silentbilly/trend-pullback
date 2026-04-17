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

        # Pending exit order reference — tracked to cancel on new signals
        self._exit_order: bt.Order | None = None
        self._entry_order: bt.Order | None = None

        # State tracking
        self._entry_bar: int | None = None

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

        # Look up precomputed signals for this bar
        try:
            row = self._signals.loc[bar_dt]
        except KeyError:
            # Bar not in signals DataFrame (e.g. during warmup)
            return

        long_sig  = bool(row["long_signal"])
        short_sig = bool(row["short_signal"])

        in_market = self.position.size != 0
        pending_entry = self._entry_order is not None and not self._entry_order.status in (
            bt.Order.Completed, bt.Order.Cancelled, bt.Order.Expired, bt.Order.Rejected
        )

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
        # Exit order is placed in notify_order() after entry confirms
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
        self._pending_levels = levels

    # ------------------------------------------------------------------
    # Order / trade callbacks
    # ------------------------------------------------------------------

    def notify_order(self, order: bt.Order) -> None:
        """Track order lifecycle; place exit after entry fills."""
        if order.status in (bt.Order.Submitted, bt.Order.Accepted):
            return

        if order.status == bt.Order.Completed:
            is_entry = (order is self._entry_order)

            if is_entry:
                direction = "Long" if order.isbuy() else "Short"
                fill_price = order.executed.price
                logger.info(
                    "ENTRY filled %-6s @ %.5f  size=%d  cost=%.2f  comm=%.4f",
                    direction, fill_price, order.executed.size,
                    order.executed.value, order.executed.comm,
                )
                levels = getattr(self, "_pending_levels", None)
                if levels is not None:
                    # Place fixed stop + limit exit
                    if order.isbuy():
                        self._exit_order = self.sell(
                            size=order.executed.size,
                            exectype=bt.Order.StopLimit,
                            price=levels.take,
                            plimit=levels.take,
                            valid=None,
                        )
                        # Stop-loss as a separate order
                        self.sell(
                            size=order.executed.size,
                            exectype=bt.Order.Stop,
                            price=levels.stop,
                            valid=None,
                        )
                    else:
                        self._exit_order = self.buy(
                            size=order.executed.size,
                            exectype=bt.Order.StopLimit,
                            price=levels.take,
                            plimit=levels.take,
                            valid=None,
                        )
                        self.buy(
                            size=order.executed.size,
                            exectype=bt.Order.Stop,
                            price=levels.stop,
                            valid=None,
                        )
                self._entry_order = None

            else:
                # Exit order completed
                fill_price = order.executed.price
                logger.info(
                    "EXIT  filled @ %.5f  size=%d  comm=%.4f",
                    fill_price, order.executed.size, order.executed.comm,
                )
                self._exit_order = None

        elif order.status in (bt.Order.Cancelled, bt.Order.Expired, bt.Order.Rejected):
            logger.warning(
                "Order %s — status: %s", order.ref, order.getstatusname()
            )
            if order is self._entry_order:
                self._entry_order = None

    def notify_trade(self, trade: bt.Trade) -> None:
        """Log closed trade P&L."""
        if not trade.isclosed:
            return
        logger.info(
            "TRADE closed — PnL gross=%.2f  net=%.2f  bars=%d",
            trade.pnl, trade.pnlcomm, trade.barlen,
        )
