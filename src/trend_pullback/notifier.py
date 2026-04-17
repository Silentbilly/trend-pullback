"""
Telegram notifier for Trend Pullback Pro live trading.

Sends alerts on:
  - Bot start / stop
  - Trade entry (Long / Short)
  - Trade exit (TP / SL)
  - Errors requiring attention

Configuration:
  Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in environment variables
  or pass directly to Notifier().

Getting credentials:
  1. Create a bot via @BotFather → get BOT_TOKEN
  2. Send any message to your bot
  3. GET https://api.telegram.org/bot<TOKEN>/getUpdates → find chat.id

Usage:
    notifier = Notifier(token="...", chat_id="...")
    notifier.send("Hello from bot")

If token/chat_id are not set, all send() calls are no-ops (silent).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import requests

logger = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"
TIMEOUT = 10  # seconds


class Notifier:
    """Telegram notification sender.

    Args:
        token:   Telegram bot token. Falls back to TELEGRAM_BOT_TOKEN env var.
        chat_id: Telegram chat ID. Falls back to TELEGRAM_CHAT_ID env var.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self._token   = token   or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self._chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID", "")
        self._enabled = bool(self._token and self._chat_id)

        if not self._enabled:
            logger.warning(
                "Telegram notifier disabled — set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID"
            )

    def send(self, text: str) -> None:
        """Send a plain-text message. Silently skips if not configured.

        Args:
            text: Message text (Markdown supported).
        """
        if not self._enabled:
            return
        try:
            url = TELEGRAM_API.format(token=self._token)
            resp = requests.post(
                url,
                json={
                    "chat_id": self._chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                },
                timeout=TIMEOUT,
            )
            if not resp.ok:
                logger.warning("Telegram send failed: %s %s", resp.status_code, resp.text[:200])
        except Exception as exc:
            logger.warning("Telegram send error: %s", exc)

    # ------------------------------------------------------------------
    # Semantic helpers
    # ------------------------------------------------------------------

    def on_start(self, symbol: str, timeframe: str, testnet: bool) -> None:
        mode = "🟡 TESTNET" if testnet else "🟢 MAINNET"
        self.send(
            f"*Trend Pullback Pro started*\n"
            f"Mode: {mode}\n"
            f"Symbol: `{symbol}`  TF: `{timeframe}`"
        )

    def on_stop(self, reason: str = "") -> None:
        self.send(f"⛔ *Bot stopped*\n{reason}")

    def on_long_entry(
        self,
        symbol: str,
        entry: float,
        stop: float,
        take: float,
        size: float,
        bar_dt: str,
    ) -> None:
        risk_pct = abs(entry - stop) / entry * 100
        self.send(
            f"📈 *LONG entry* `{symbol}`\n"
            f"Bar: `{bar_dt}`\n"
            f"Entry : `{entry:.5f}`\n"
            f"Stop  : `{stop:.5f}`  ({risk_pct:.2f}% risk)\n"
            f"Take  : `{take:.5f}`\n"
            f"Size  : `{size}`"
        )

    def on_short_entry(
        self,
        symbol: str,
        entry: float,
        stop: float,
        take: float,
        size: float,
        bar_dt: str,
    ) -> None:
        risk_pct = abs(stop - entry) / entry * 100
        self.send(
            f"📉 *SHORT entry* `{symbol}`\n"
            f"Bar: `{bar_dt}`\n"
            f"Entry : `{entry:.5f}`\n"
            f"Stop  : `{stop:.5f}`  ({risk_pct:.2f}% risk)\n"
            f"Take  : `{take:.5f}`\n"
            f"Size  : `{size}`"
        )

    def on_tp_hit(self, symbol: str, entry: float, exit_price: float, pnl_pct: float) -> None:
        self.send(
            f"✅ *Take-profit hit* `{symbol}`\n"
            f"Entry : `{entry:.5f}`\n"
            f"Exit  : `{exit_price:.5f}`\n"
            f"PnL   : `+{pnl_pct:.2f}%`"
        )

    def on_sl_hit(self, symbol: str, entry: float, exit_price: float, pnl_pct: float) -> None:
        self.send(
            f"❌ *Stop-loss hit* `{symbol}`\n"
            f"Entry : `{entry:.5f}`\n"
            f"Exit  : `{exit_price:.5f}`\n"
            f"PnL   : `{pnl_pct:.2f}%`"
        )

    def on_error(self, symbol: str, message: str) -> None:
        self.send(f"⚠️ *Error* `{symbol}`\n`{message}`")
