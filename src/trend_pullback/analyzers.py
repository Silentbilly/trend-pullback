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

    def notify_trade(self, trade: bt.Trade) -> None:
        if not trade.isclosed:
            return
        self.trades.append(
            {
                "entry_bar":  trade.baropen,
                "exit_bar":   trade.barclose,
                "entry_price": trade.price,
                "exit_price":  trade.priceclosing,
                "size":        trade.size,
                "pnl_gross":   round(trade.pnl, 6),
                "pnl_net":     round(trade.pnlcomm, 6),
                "commission":  round(trade.commission, 6),
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
