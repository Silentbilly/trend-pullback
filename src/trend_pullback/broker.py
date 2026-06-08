"""
Bybit broker adapter for Trend Pullback Pro live trading.

Connects to Bybit via CCXT (unified API).
Supports mainnet, testnet, and Bybit demo trading — controlled by config.

Responsibilities:
  - Fetch closed OHLCV bars
  - Get current position
  - Place market entry orders
  - Place stop-loss and take-profit orders (as separate orders)
  - Cancel open orders
  - Query order status

Design notes:
  - All public methods raise BrokerError on unrecoverable failures.
  - Retryable network errors are retried internally (up to MAX_RETRIES).
  - No business logic here — only exchange communication.
  - Uses CCXT unified API so swapping to Binance/OKX requires minimal changes.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import ccxt

logger = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_DELAY = 2.0  # seconds between retries


class BrokerError(Exception):
    """Unrecoverable broker-level error."""


@dataclass
class PositionInfo:
    """Current position state returned by get_position()."""
    symbol: str
    side: str           # "long", "short", or "none"
    size: float         # absolute position size in base currency
    entry_price: float  # average entry price (0 if no position)


@dataclass
class OrderResult:
    """Result of a placed order."""
    order_id: str
    symbol: str
    side: str           # "buy" or "sell"
    order_type: str     # "market", "limit", "stop"
    price: float        # requested price (0 for market)
    size: float
    status: str         # "open", "closed", "canceled"


class BybitBroker:
    """Bybit broker adapter using CCXT.

    Args:
        api_key:       Bybit API key.
        api_secret:    Bybit API secret.
        symbol:        Trading symbol in CCXT format, e.g. "ETH/USDT:USDT".
        testnet:       If True, connects to Bybit testnet.
        demo_trading:  If True, connects to Bybit Demo Trading.
        leverage:      Leverage to set on init (default 1 = no leverage).

    Notes:
        - demo_trading and testnet are mutually exclusive.
        - Priority: demo_trading > testnet > mainnet.
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        symbol: str,
        testnet: bool = True,
        demo_trading: bool = False,
        leverage: int = 1,
    ) -> None:
        self.symbol = symbol
        self.testnet = testnet
        self.demo_trading = demo_trading
        self.leverage = leverage

        if self.demo_trading and self.testnet:
            raise BrokerError("Invalid config: demo_trading=true and testnet=true cannot be used together")

        self._exchange = ccxt.bybit(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "linear",
                    "defaultSubType": "linear",
                },
            }
        )

        self._configure_environment()
        self._exchange.load_markets()
        self._set_leverage()

    def _configure_environment(self) -> None:
        """Configure Bybit environment: demo, testnet, or mainnet."""
        if self.demo_trading:
            if hasattr(self._exchange, "enable_demo_trading"):
                self._exchange.enable_demo_trading(True)
                logger.info("Broker: Bybit DEMO TRADING  symbol=%s", self.symbol)
                return

            self._exchange.urls["api"] = "https://api-demo.bybit.com"
            logger.info("Broker: Bybit DEMO TRADING  symbol=%s", self.symbol)
            return

        if self.testnet:
            self._exchange.set_sandbox_mode(True)
            logger.info("Broker: Bybit TESTNET  symbol=%s", self.symbol)
        else:
            logger.info("Broker: Bybit MAINNET  symbol=%s", self.symbol)

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def fetch_ohlcv(self, timeframe: str, limit: int = 300) -> list[list]:
        """Fetch the most recent closed OHLCV bars."""
        raw = self._retry(
            lambda: self._exchange.fetch_ohlcv(
                self.symbol,
                timeframe,
                limit=limit + 1,
                params={"category": "linear"},
            )
        )
        return raw[:-1]

    # ------------------------------------------------------------------
    # Account / position
    # ------------------------------------------------------------------

    def get_position(self) -> PositionInfo:
        """Return the current open position for the symbol."""
        positions = self._retry(
            lambda: self._exchange.fetch_positions([self.symbol])
        )
        market = self._exchange.market(self.symbol)
        market_id = market.get("id")
        alt_symbols = {
            self.symbol,
            self.symbol.replace("/", ""),
            self.symbol.replace("/", "").replace(":", ""),
            market_id,
        }

        for pos in positions:
            info_symbol = pos.get("info", {}).get("symbol")
            pos_symbol = pos.get("symbol")
            if pos_symbol not in alt_symbols and info_symbol not in alt_symbols:
                continue

            size = float(pos.get("contracts", 0) or 0)
            if size > 0:
                side = (pos.get("side", "") or "").lower()
                entry = float(pos.get("entryPrice", 0) or 0)
                if side not in ("long", "short"):
                    side = "long"
                return PositionInfo(
                    symbol=self.symbol,
                    side=side,
                    size=size,
                    entry_price=entry,
                )

        return PositionInfo(
            symbol=self.symbol,
            side="none",
            size=0.0,
            entry_price=0.0,
        )

    def get_balance_usdt(self) -> float:
        """Return available USDT balance."""
        balance = self._retry(lambda: self._exchange.fetch_balance())
        return float(balance.get("USDT", {}).get("free", 0) or 0)

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_market_order(self, side: str, size: float) -> OrderResult:
        """Place a market entry order."""
        logger.info("Placing MARKET %s  size=%.6f  symbol=%s", side.upper(), size, self.symbol)
        order = self._retry(
            lambda: self._exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=size,
                params={"category": "linear"},
            )
        )
        return self._to_order_result(order)

    def place_limit_order(self, side: str, size: float, price: float) -> OrderResult:
        """Place a limit take-profit order."""
        logger.info(
            "Placing LIMIT %s  size=%.6f  price=%.5f  symbol=%s",
            side.upper(), size, price, self.symbol,
        )
        order = self._retry(
            lambda: self._exchange.create_order(
                symbol=self.symbol,
                type="limit",
                side=side,
                amount=size,
                price=price,
                params={"category": "linear", "reduceOnly": True},
            )
        )
        return self._to_order_result(order)

    def place_stop_order(self, side: str, size: float, stop_price: float) -> OrderResult:
        """Place a stop-loss order."""
        logger.info(
            "Placing STOP %s  size=%.6f  stop=%.5f  symbol=%s",
            side.upper(), size, stop_price, self.symbol,
        )

        params = {
            "category": "linear",
            "triggerPrice": stop_price,
            "triggerBy": "LastPrice",
            "reduceOnly": True,
        }

        order = self._retry(
            lambda: self._exchange.create_order(
                symbol=self.symbol,
                type="market",
                side=side,
                amount=size,
                params=params,
            )
        )
        return self._to_order_result(order)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order by ID."""
        try:
            self._exchange.cancel_order(order_id, self.symbol)
            logger.info("Cancelled order %s", order_id)
            return True
        except ccxt.OrderNotFound:
            logger.debug("Order %s not found (already filled?)", order_id)
            return False
        except Exception as exc:
            logger.warning("Failed to cancel order %s: %s", order_id, exc)
            return False

    def get_order_status(self, order_id: str) -> str:
        """Return order status string: 'open', 'closed', 'canceled'."""
        try:
            order = self._retry(
                lambda: self._exchange.fetch_order(order_id, self.symbol)
            )
            return order.get("status", "unknown")
        except ccxt.OrderNotFound:
            return "canceled"
        except Exception as exc:
            logger.warning("get_order_status(%s) failed: %s", order_id, exc)
            return "unknown"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _set_leverage(self) -> None:
        """Set leverage on the symbol."""
        try:
            self._exchange.set_leverage(self.leverage, self.symbol)
            logger.info("Leverage set to %dx for %s", self.leverage, self.symbol)
        except Exception as exc:
            logger.warning("Could not set leverage: %s", exc)

    def _retry(self, fn, retries: int = MAX_RETRIES):
        """Execute fn with retry on network errors."""
        last_exc = None
        for attempt in range(1, retries + 1):
            try:
                return fn()
            except (ccxt.NetworkError, ccxt.RequestTimeout) as exc:
                last_exc = exc
                logger.warning("Network error (attempt %d/%d): %s", attempt, retries, exc)
                time.sleep(RETRY_DELAY * attempt)
            except ccxt.AuthenticationError as exc:
                raise BrokerError(f"Authentication failed: {exc}") from exc
            except ccxt.InsufficientFunds as exc:
                raise BrokerError(f"Insufficient funds: {exc}") from exc
            except ccxt.InvalidOrder as exc:
                raise BrokerError(f"Invalid order: {exc}") from exc
            except Exception as exc:
                raise BrokerError(f"Unexpected broker error: {exc}") from exc
        raise BrokerError(f"Max retries exceeded: {last_exc}") from last_exc

    @staticmethod
    def _to_order_result(order: dict) -> OrderResult:
        return OrderResult(
            order_id=str(order.get("id", "")),
            symbol=order.get("symbol", ""),
            side=order.get("side", ""),
            order_type=order.get("type", ""),
            price=float(order.get("price") or 0),
            size=float(order.get("amount") or 0),
            status=order.get("status", "open"),
        )