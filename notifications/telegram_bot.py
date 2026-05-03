"""
Phase 9 — Telegram Notifications
TG-01 through TG-05
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from loguru import logger
from telegram import Bot, BotCommand, InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from backend.claude_analyst import SignalResult
from backend.risk_manager import RiskDecision
from backend.signal_logger import SignalStats
from config import settings

# ---------------------------------------------------------------------------
# TG-01  TelegramNotifier — core class
# ---------------------------------------------------------------------------

GetStatusFn = Callable[[], Awaitable[dict[str, Any]]]
GetSymbolFn = Callable[[], str]
SetSymbolFn = Callable[[str], None]

# Symbol category icons
_SYMBOL_ICONS: dict[str, str] = {
    "BTC": "🪙", "ETH": "🪙", "SOL": "🪙",
    "XAU": "🥇", "XAG": "🥈",
    "EUR": "💱", "GBP": "💱", "USD": "💱", "JPY": "💱",
    "AUD": "💱", "NZD": "💱", "CAD": "💱", "CHF": "💱",
}


def _symbol_icon(symbol: str) -> str:
    for prefix, icon in _SYMBOL_ICONS.items():
        if symbol.upper().startswith(prefix):
            return icon
    return "📊"


class TelegramNotifier:
    """
    Manages all Telegram interactions for the MT5 AI Analyst Bot.

    Lifecycle: call start() in FastAPI lifespan startup,
               call stop()  in FastAPI lifespan shutdown.
    """

    def __init__(self) -> None:
        self._enabled: bool = (
            settings.telegram.get("enabled", False)
            and bool(settings.telegram_bot_token)
            and bool(settings.telegram_chat_id)
        )
        # Strip whitespace — a common .env copy-paste issue that causes silent mismatches
        self._chat_id: str = str(settings.telegram_chat_id).strip()
        self._app: Application | None = None
        self._get_status: GetStatusFn | None = None
        self._get_symbol: GetSymbolFn | None = None
        self._set_symbol: SetSymbolFn | None = None
        self._watchlist: list[str] = []

        if not self._enabled:
            logger.info("Telegram notifications disabled (check settings.yaml + .env)")
        else:
            logger.info(f"Telegram notifier init — chat_id={self._chat_id!r}")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        get_status_fn: GetStatusFn | None = None,
        get_symbol_fn: GetSymbolFn | None = None,
        set_symbol_fn: SetSymbolFn | None = None,
        watchlist: list[str] | None = None,
    ) -> None:
        """Initialise the bot and start polling for commands."""
        if not self._enabled:
            return

        self._get_status = get_status_fn
        self._get_symbol = get_symbol_fn
        self._set_symbol = set_symbol_fn
        self._watchlist = watchlist or settings.symbols.get("watchlist", [])

        self._app = Application.builder().token(settings.telegram_bot_token).build()

        # Groups, supergroups, private chats
        self._app.add_handler(CommandHandler("status", self._cmd_status))
        self._app.add_handler(CommandHandler("market", self._cmd_market))
        self._app.add_handler(CommandHandler("chatid", self._cmd_chatid))

        # Channels — channel_post updates are NOT handled by CommandHandler,
        # so we route them manually through a MessageHandler
        self._app.add_handler(
            MessageHandler(
                filters.UpdateType.CHANNEL_POSTS & filters.COMMAND,
                self._dispatch_channel_command,
            )
        )

        self._app.add_handler(
            CallbackQueryHandler(self._cb_market, pattern=r"^set_market:")
        )
        self._app.add_error_handler(self._error_handler)

        await self._app.initialize()

        # Register commands so they appear in the Telegram command menu
        await self._app.bot.set_my_commands([
            BotCommand("status", "Bot status and active market"),
            BotCommand("market", "Select trading symbol"),
            BotCommand("chatid", "Show this chat's ID (for setup/debug)"),
        ])

        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        active = self._get_symbol() if self._get_symbol else "—"
        logger.info(f"Telegram bot polling started — authorized chat_id={self._chat_id!r}")
        await self._send_raw(
            f"🟢 <b>AI Analyst Bot started</b>\n"
            f"Active market: <code>{active}</code>\n"
            f"Commands: /status  /market"
        )

    async def stop(self) -> None:
        """Gracefully shut down the Telegram application."""
        if self._app is None:
            return
        try:
            await self._send_raw("🔴 <b>AI Analyst Bot shutting down.</b>")
            await self._app.updater.stop()
            await self._app.stop()
            await self._app.shutdown()
        except Exception as e:
            logger.warning(f"Telegram shutdown error: {e}")
        finally:
            self._app = None

    # ------------------------------------------------------------------
    # TG-03  Signal alert
    # ------------------------------------------------------------------

    async def send_signal_alert(
        self, signal: SignalResult, risk: RiskDecision
    ) -> None:
        """Send a trade signal notification to the configured chat."""
        if not self._enabled:
            return
        if not settings.telegram.get("send_signal_alerts", True):
            return

        text = _format_signal_message(signal, risk)
        await self._send_raw(text)

    # ------------------------------------------------------------------
    # TG-04  Daily summary
    # ------------------------------------------------------------------

    async def send_agent_update(self, message: str) -> None:
        """Send a free-form agent decision update (HTML supported)."""
        await self._send_raw(message)

    async def send_daily_summary(self, stats: SignalStats) -> None:
        """Send the daily performance summary."""
        if not self._enabled:
            return
        if not settings.telegram.get("send_daily_summary", True):
            return

        text = _format_daily_summary(stats)
        await self._send_raw(text)

    # ------------------------------------------------------------------
    # TG-05  /status command handler
    # ------------------------------------------------------------------

    async def _error_handler(
        self, update: object, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        logger.error(f"Telegram handler exception: {context.error}", exc_info=context.error)
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "⚠️ An error occurred — check server logs."
                )
            except Exception:
                pass

    def _is_authorized(self, update: Update) -> bool:
        """Return True if the update comes from the configured chat."""
        chat = update.effective_chat
        if chat is None:
            return False
        incoming = str(chat.id).strip()
        if incoming != self._chat_id:
            logger.warning(
                f"Telegram: unauthorized message from chat_id={incoming!r} "
                f"(expected {self._chat_id!r}) — ignoring"
            )
            return False
        return True

    async def _cmd_status(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        if update.effective_message is None or not self._is_authorized(update):
            return

        if self._get_status is not None:
            try:
                data = await self._get_status()
                active = self._get_symbol() if self._get_symbol else None
                text = _format_status(data, active_symbol=active)
            except Exception as e:
                logger.error(f"Status handler error: {e}")
                text = "⚠️ Could not fetch status — check server logs."
        else:
            text = "⚠️ Status callback not registered."

        await update.effective_message.reply_text(text, parse_mode=ParseMode.HTML)

    # ------------------------------------------------------------------
    # /market command + callback
    # ------------------------------------------------------------------

    async def _dispatch_channel_command(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Route channel_post commands — CommandHandler doesn't fire for channels."""
        msg = update.effective_message
        if not msg or not msg.text:
            return
        # Parse /command or /command@botname
        cmd = msg.text.split()[0].lstrip("/").split("@")[0].lower()
        routes = {
            "status":  self._cmd_status,
            "market":  self._cmd_market,
            "chatid":  self._cmd_chatid,
        }
        handler = routes.get(cmd)
        if handler:
            await handler(update, context)

    async def _cmd_chatid(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """No-auth diagnostic: reply with the current chat ID."""
        msg = update.effective_message
        chat = update.effective_chat
        if msg is None or chat is None:
            return
        chat_id = chat.id
        matched = str(chat_id).strip() == self._chat_id
        match_str = "✅ matches TELEGRAM_CHAT_ID" if matched else f"❌ does NOT match (configured: <code>{self._chat_id}</code>)"
        chat_type = chat.type  # private / group / supergroup / channel
        await msg.reply_text(
            f"Chat ID:   <code>{chat_id}</code>\n"
            f"Chat type: <code>{chat_type}</code>\n"
            f"{match_str}\n\n"
            f"If not matched, set <code>TELEGRAM_CHAT_ID={chat_id}</code> in .env and restart.",
            parse_mode=ParseMode.HTML,
        )

    async def _cmd_market(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show inline keyboard listing all watchlist symbols."""
        if update.effective_message is None or not self._is_authorized(update):
            return

        current = self._get_symbol() if self._get_symbol else ""
        markup = self._build_market_keyboard(current)

        await update.effective_message.reply_text(
            f"🎯 <b>Select Trading Market</b>\n\n"
            f"Active now: <code>{current or '—'}</code>\n\n"
            f"Tap a symbol to switch. The bot will trade it from the next M15 bar.",
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _cb_market(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle symbol selection from inline keyboard."""
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer()
            return

        symbol = query.data.replace("set_market:", "")

        if self._set_symbol:
            self._set_symbol(symbol)
            logger.info(f"Telegram: active symbol changed to {symbol}")

        await query.answer(f"✅ Switched to {symbol}")

        markup = self._build_market_keyboard(symbol)
        await query.edit_message_text(
            f"✅ <b>Market switched to <code>{symbol}</code></b>\n\n"
            f"The agent will analyse <code>{symbol}</code> from the next M15 bar.\n"
            f"Use /market to change again.",
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    def _build_market_keyboard(self, active_symbol: str) -> InlineKeyboardMarkup:
        """Build the 2-column inline keyboard from the watchlist."""
        buttons: list[InlineKeyboardButton] = []
        for sym in self._watchlist:
            icon = _symbol_icon(sym)
            label = f"✅ {icon} {sym}" if sym == active_symbol else f"{icon} {sym}"
            buttons.append(InlineKeyboardButton(label, callback_data=f"set_market:{sym}"))

        # Arrange into rows of 2
        rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
        return InlineKeyboardMarkup(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send_raw(self, text: str) -> None:
        """Send a raw HTML message. Logs and swallows TelegramError."""
        if not self._enabled:
            return

        try:
            if self._app is not None:
                await self._app.bot.send_message(
                    chat_id=self._chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            else:
                # Fallback: use async context manager so aiohttp session is cleaned up
                async with Bot(token=settings.telegram_bot_token) as bot:
                    await bot.send_message(
                        chat_id=self._chat_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
        except TelegramError as e:
            logger.warning(f"Telegram send failed: {e}")


# ---------------------------------------------------------------------------
# TG-02  Message formatters
# ---------------------------------------------------------------------------

def _direction_icon(direction: str) -> str:
    return "🟢" if direction == "BUY" else "🔴" if direction == "SELL" else "⚪"


def _format_signal_message(signal: SignalResult, risk: RiskDecision) -> str:
    """TG-02 — Format signal alert as Telegram HTML message."""
    icon  = _direction_icon(signal.direction)
    title = f"{icon} <b>{signal.symbol} — {signal.direction}</b>"

    if not signal.is_actionable:
        reason = signal.reasoning[:200] if signal.reasoning else "No setup found."
        return (
            f"{title}\n\n"
            f"<i>No trade signal — {reason}</i>"
        )

    rr_str   = f"{signal.risk_reward:.2f}" if signal.risk_reward else "n/a"
    approved = "✅ Approved" if risk.approved else "❌ Blocked"
    conf_bar = _confidence_bar(signal.confidence)

    # Truncate reasoning + invalidation
    reasoning   = (signal.reasoning   or "")[:300]
    invalidation = (signal.invalidation or "")[:150]

    confluence = ""
    if signal.confluence_factors:
        factors = "\n".join(f"  • {f}" for f in signal.confluence_factors[:6])
        confluence = f"\n\n<b>Confluence:</b>\n{factors}"

    warnings = ""
    if risk.warnings:
        warnings = "\n⚠️ " + " | ".join(risk.warnings)

    rejection = ""
    if not risk.approved and risk.reasons:
        rejection = "\n🚫 " + " | ".join(risk.reasons)

    def _p(v: float | None) -> str:
        return f"{v:.5f}" if v is not None else "n/a"

    return (
        f"{title}\n"
        f"Confidence: {signal.confidence}% {conf_bar}\n\n"
        f"<b>Entry:</b>  <code>{_p(signal.entry)}</code>\n"
        f"<b>SL:</b>     <code>{_p(signal.sl)}</code>\n"
        f"<b>TP1:</b>    <code>{_p(signal.tp1)}</code>  (+1.5R)\n"
        f"<b>TP2:</b>    <code>{_p(signal.tp2)}</code>  (+2.5R)\n"
        f"<b>TP3:</b>    <code>{_p(signal.tp3)}</code>  (+4.0R)\n\n"
        f"R:R: <b>{rr_str}</b>  |  Lot: <b>{risk.lot_size:.2f}</b>  |  {approved}"
        f"{warnings}{rejection}"
        f"\n\n<b>Analysis:</b>\n<i>{reasoning}</i>"
        f"\n\n<b>Invalidation:</b> <i>{invalidation}</i>"
        f"{confluence}"
        f"\n\n<code>#{signal.symbol} #{signal.direction} #AIA</code>"
    )


def _format_daily_summary(stats: SignalStats) -> str:
    """TG-04 — Format daily summary as Telegram HTML message."""
    closed   = stats.wins + stats.losses
    wr_str   = f"{stats.win_rate_pct:.1f}%" if closed > 0 else "—"
    rr_str   = f"{stats.avg_rr:.2f}" if closed > 0 else "—"
    pnl_icon = "📈" if stats.total_pnl >= 0 else "📉"
    pnl_str  = f"${stats.total_pnl:+.2f}"

    period_str = ""
    if stats.period_start:
        period_str = f"\n<i>Period: {stats.period_start.strftime('%Y-%m-%d')} → today</i>"

    by_sym = ""
    if stats.by_symbol:
        rows = []
        for sym, d in list(stats.by_symbol.items())[:5]:
            sym_closed = d["wins"] + d["losses"]
            sym_wr = f"{d['win_rate']:.0f}%" if sym_closed > 0 else "—"
            rows.append(f"  <code>{sym:<8}</code> {d['total']} trades  {sym_wr} WR")
        by_sym = "\n\n<b>By Symbol:</b>\n" + "\n".join(rows)

    return (
        f"📅 <b>Daily Summary</b> — {datetime.now(timezone.utc).strftime('%Y-%m-%d UTC')}"
        f"{period_str}\n\n"
        f"📊 Total Signals:  <b>{stats.total_signals}</b>\n"
        f"   Actionable:     <b>{stats.actionable_signals}</b>\n"
        f"   No-trade:       <b>{stats.no_trade_signals}</b>\n\n"
        f"✅ Wins:    <b>{stats.wins}</b>\n"
        f"❌ Losses:  <b>{stats.losses}</b>\n"
        f"🟰 BE:      <b>{stats.breakevens}</b>\n\n"
        f"📈 Win Rate:  <b>{wr_str}</b>\n"
        f"📐 Avg R:R:   <b>{rr_str}</b>\n"
        f"{pnl_icon} P&L:      <b>{pnl_str}</b>\n"
        f"💸 API Cost:  <b>${stats.total_api_cost_usd:.4f}</b>"
        f"{by_sym}"
    )


def _format_status(data: dict, active_symbol: str | None = None) -> str:
    """TG-05 — Format /status response as Telegram HTML."""
    uptime_s   = int(data.get("uptime_seconds", 0))
    uptime_str = _fmt_uptime(uptime_s)
    zmq_alive  = data.get("zmq_listener_alive", False)
    zmq_icon   = "🟢" if zmq_alive else "🔴"
    fetcher    = data.get("mt5_fetcher_alive", False)
    fetch_icon = "🟢" if fetcher else "🔴"
    sigs       = data.get("signals_processed", 0)
    last_sig   = data.get("last_signal_at")

    last_str = "—"
    if last_sig:
        try:
            dt = datetime.fromisoformat(last_sig.replace("Z", "+00:00"))
            last_str = dt.strftime("%H:%M UTC")
        except ValueError:
            last_str = last_sig

    sym_line = f"🎯 Active Market:     <b><code>{active_symbol}</code></b>\n" if active_symbol else ""

    return (
        f"🤖 <b>AI Analyst Bot Status</b>\n\n"
        f"{sym_line}"
        f"⏱ Uptime:            <b>{uptime_str}</b>\n"
        f"{fetch_icon} MT5 Fetcher:      <b>{'Active' if fetcher else 'Inactive'}</b>\n"
        f"{zmq_icon} ZeroMQ:           <b>{'Active' if zmq_alive else 'Inactive'}</b>\n"
        f"📊 Cycles (session):  <b>{sigs}</b>\n"
        f"🕐 Last Cycle:        <b>{last_str}</b>\n\n"
        f"<i>Use /market to switch the trading symbol.</i>"
    )


def _confidence_bar(confidence: int) -> str:
    filled = max(0, min(10, round(confidence / 10)))
    return "█" * filled + "░" * (10 - filled)


def _fmt_uptime(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    if h > 0:
        return f"{h}h {m}m"
    return f"{m}m"


# ---------------------------------------------------------------------------
# Module-level singleton — imported by main.py
# ---------------------------------------------------------------------------

notifier = TelegramNotifier()
