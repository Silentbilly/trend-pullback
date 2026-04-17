"""
Custom Backtrader analyzers for Trend Pullback Pro.

Provides:
  - TradeListAnalyzer: Collects per-trade details into a list of dicts.
  - EquityCurveAnalyzer: Records equity (portfolio value) after each bar.

Both are lightweight wrappers — the heavy lifting is in reporting.py.
"""

from __future__ import annotations

import backtrader as bt


class TradeListAnalyzer(bt.Analyzer):
    """Collects closed trade data for post-backtest reporting.

    Access results via:
        cerebro.run()[0].analyzers.trade_list.get_analysis()
    """

    def start(self) -> None:
        self.trades: list[dict] = []
        # Track last known price per trade ref for exit price approximation
        self._open_prices: dict[int, float] = {}

    def notify_trade(self, trade: bt.Trade) -> None:
        if trade.isopen:
            # Record entry price when trade opens
            self._open_prices[trade.ref] = trade.price
            return
        if not trade.isclosed:
            return
        entry_price = self._open_prices.pop(trade.ref, trade.price)
        # Exit price derived from P&L: exit = entry +/- pnl_gross / size
        # For long:  exit = entry + pnl_gross / size
        # For short: exit = entry - pnl_gross / size (size is negative for short)
        size = trade.history[0].event.size if trade.history else trade.size
        if size != 0:
            exit_price = entry_price + trade.pnl / abs(size)
        else:
            exit_price = entry_price
        self.trades.append(
            {
                "entry_bar":     trade.baropen,
                "exit_bar":      trade.barclose,
                "entry_price":   round(entry_price, 8),
                "exit_price":    round(exit_price, 8),
                "size":          abs(size),
                "pnl_gross":     round(trade.pnl, 6),
                "pnl_net":       round(trade.pnlcomm, 6),
                "commission":    round(trade.commission, 6),
                "duration_bars": trade.barlen,
            }
        )

    def get_analysis(self) -> list[dict]:
        return self.trades


class EquityCurveAnalyzer(bt.Analyzer):
    """Records portfolio value (equity) at each bar close.

    Access results via:
        cerebro.run()[0].analyzers.equity_curve.get_analysis()
    """

    def start(self) -> None:
        self.equity: list[dict] = []

    def next(self) -> None:
        self.equity.append(
            {
                "datetime": self.strategy.data.datetime.datetime(0),
                "equity":   round(self.strategy.broker.getvalue(), 4),
            }
        )

    def get_analysis(self) -> list[dict]:
        return self.equity
