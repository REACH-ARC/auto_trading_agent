"""
Phase 9 — Telegram Notifications
TG-01 through TG-05
"""
from __future__ import annotations

import html
import re
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
from backend.model_manager import AVAILABLE_MODELS, AVAILABLE_STRATEGIES
from config import settings

# ---------------------------------------------------------------------------
# TG-01  TelegramNotifier — core class
# ---------------------------------------------------------------------------

GetStatusFn = Callable[[], Awaitable[dict[str, Any]]]
GetSymbolFn = Callable[[], str]
SetSymbolFn = Callable[[str], None]
GetModelFn = Callable[[], str]
SetModelFn = Callable[[str], None]
GetWatchlistFn = Callable[[], list[str]]
GetStrategyFn = Callable[[], str]
SetStrategyFn = Callable[[str], None]
GetBoolFn = Callable[[], bool]
SetBoolFn = Callable[[bool], None]

# Telegram HTML mode only allows these tags; everything else must be stripped.
_TG_ALLOWED_TAGS: frozenset[str] = frozenset({
    "b", "strong", "i", "em", "u", "ins", "s", "strike", "del",
    "code", "pre", "a", "tg-spoiler", "blockquote",
})
_TAG_RE = re.compile(r"<(/?)\s*([a-zA-Z][a-zA-Z0-9-]*)((?:\s[^>]*)?)>")


def _sanitize_html(text: str) -> str:
    """Strip tags not supported by Telegram's HTML parser, keep allowed ones."""
    def _replace(m: re.Match) -> str:
        tag = m.group(2).lower()
        attrs = m.group(3)
        # <span class="tg-spoiler"> is the one valid span variant
        if tag == "span" and 'class="tg-spoiler"' in attrs:
            return m.group(0)
        if tag in _TG_ALLOWED_TAGS:
            return m.group(0)
        return ""  # strip unsupported tag
    return _TAG_RE.sub(_replace, text)


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
        self._signal_channel_id: str = str(settings.telegram_signal_channel_id).strip()
        self._app: Application | None = None
        self._get_status: GetStatusFn | None = None
        self._get_symbol: GetSymbolFn | None = None
        self._set_symbol: SetSymbolFn | None = None
        self._get_model: GetModelFn | None = None
        self._set_model: SetModelFn | None = None
        self._get_watchlist: GetWatchlistFn | None = None
        self._get_strategy: GetStrategyFn | None = None
        self._set_strategy: SetStrategyFn | None = None
        self._get_auto_trade: GetBoolFn | None = None
        self._set_auto_trade: SetBoolFn | None = None
        self._get_scan_all: GetBoolFn | None = None
        self._set_scan_all: SetBoolFn | None = None

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
        get_model_fn: GetModelFn | None = None,
        set_model_fn: SetModelFn | None = None,
        get_watchlist_fn: GetWatchlistFn | None = None,
        get_strategy_fn: GetStrategyFn | None = None,
        set_strategy_fn: SetStrategyFn | None = None,
        get_auto_trade_fn: GetBoolFn | None = None,
        set_auto_trade_fn: SetBoolFn | None = None,
        get_scan_all_fn: GetBoolFn | None = None,
        set_scan_all_fn: SetBoolFn | None = None,
    ) -> None:
        """Initialise the bot and start polling for commands."""
        if not self._enabled:
            return

        self._get_status = get_status_fn
        self._get_symbol = get_symbol_fn
        self._set_symbol = set_symbol_fn
        self._get_model = get_model_fn
        self._set_model = set_model_fn
        self._get_watchlist = get_watchlist_fn
        self._get_strategy = get_strategy_fn
        self._set_strategy = set_strategy_fn
        self._get_auto_trade = get_auto_trade_fn
        self._set_auto_trade = set_auto_trade_fn
        self._get_scan_all = get_scan_all_fn
        self._set_scan_all = set_scan_all_fn

        self._app = Application.builder().token(settings.telegram_bot_token).build()

        # Groups, supergroups, private chats
        self._app.add_handler(CommandHandler("status",   self._cmd_status))
        self._app.add_handler(CommandHandler("market",   self._cmd_market))
        self._app.add_handler(CommandHandler("model",    self._cmd_model))
        self._app.add_handler(CommandHandler("strategy", self._cmd_strategy))
        self._app.add_handler(CommandHandler("settings", self._cmd_settings))
        self._app.add_handler(CommandHandler("chatid",   self._cmd_chatid))

        # Channels — channel_post updates are NOT handled by CommandHandler,
        # so we route them manually through a MessageHandler
        self._app.add_handler(
            MessageHandler(
                filters.UpdateType.CHANNEL_POSTS & filters.COMMAND,
                self._dispatch_channel_command,
            )
        )

        # Debug catch-all: log every callback query before the specific handlers.
        # Helps diagnose whether button taps reach the bot at all.
        self._app.add_handler(CallbackQueryHandler(self._cb_debug), group=-1)

        self._app.add_handler(
            CallbackQueryHandler(self._cb_market, pattern=r"^set_market:")
        )
        self._app.add_handler(
            CallbackQueryHandler(self._cb_model, pattern=r"^set_model:")
        )
        self._app.add_handler(
            CallbackQueryHandler(self._cb_strategy, pattern=r"^set_strategy:")
        )
        self._app.add_handler(
            CallbackQueryHandler(self._cb_toggle_setting, pattern=r"^toggle_setting:")
        )
        self._app.add_error_handler(self._error_handler)

        await self._app.initialize()

        # Register commands so they appear in the Telegram command menu
        await self._app.bot.set_my_commands([
            BotCommand("status",   "Bot status and active market"),
            BotCommand("market",   "Select trading symbol"),
            BotCommand("model",    "Switch AI model (Claude / Ollama / Strategy)"),
            BotCommand("strategy", "Select rule-based strategy (when model=Strategy)"),
            BotCommand("settings", "Toggle auto-trade and multi-market scan ON/OFF"),
            BotCommand("chatid",   "Show this chat's ID (for setup/debug)"),
        ])

        await self._app.start()
        # Explicitly include callback_query so Telegram sends button-tap events.
        # Without this, Telegram reuses the previous session's allowed_updates
        # which may have been set before inline keyboards were added.
        await self._app.updater.start_polling(
            drop_pending_updates=True,
            allowed_updates=["message", "channel_post", "callback_query"],
        )

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
        """Send a trade signal notification to the configured chat and signal channel."""
        if not self._enabled:
            return
        if not settings.telegram.get("send_signal_alerts", True):
            return

        text = _format_signal_message(signal, risk)
        await self._send_raw(text)

        if self._signal_channel_id:
            await self._send_to(self._signal_channel_id, text)

    # ------------------------------------------------------------------
    # TG-04  Daily summary
    # ------------------------------------------------------------------

    async def send_agent_update(self, message: str) -> None:
        """Send a free-form agent decision update (HTML supported)."""
        await self._send_raw(_sanitize_html(message))

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
                model = self._get_model() if self._get_model else None
                strategy = self._get_strategy() if self._get_strategy else None
                auto_trade = self._get_auto_trade() if self._get_auto_trade else True
                scan_all = self._get_scan_all() if self._get_scan_all else False
                text = _format_status(
                    data,
                    active_symbol=active,
                    active_model=model,
                    active_strategy=strategy,
                    auto_trade=auto_trade,
                    scan_all=scan_all,
                )
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
            "status":    self._cmd_status,
            "market":    self._cmd_market,
            "model":     self._cmd_model,
            "strategy":  self._cmd_strategy,
            "settings":  self._cmd_settings,
            "chatid":    self._cmd_chatid,
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

    # ------------------------------------------------------------------
    # /model command + callback
    # ------------------------------------------------------------------

    async def _cb_debug(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Catch-all callback logger — fires for every button tap regardless of data."""
        query = update.callback_query
        if query is None:
            return
        chat_id = update.effective_chat.id if update.effective_chat else "unknown"
        logger.info(f"Telegram: [DEBUG] callback arrived — data={query.data!r} chat={chat_id}")

    async def _cmd_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show inline keyboard listing available AI models."""
        chat_id = update.effective_chat.id if update.effective_chat else "unknown"
        logger.info(f"Telegram: /model received from chat_id={chat_id}")

        if update.effective_message is None or not self._is_authorized(update):
            logger.warning(f"Telegram: /model rejected — not authorized (chat={chat_id})")
            return

        current = self._get_model() if self._get_model else "claude"
        markup = self._build_model_keyboard(current)

        await update.effective_message.reply_text(
            f"🤖 <b>Select AI Model</b>\n\n"
            f"Active now: <code>{AVAILABLE_MODELS.get(current, current)}</code>\n\n"
            f"Tap to switch. Takes effect on the next M15 bar.",
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _cb_model(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle model selection from inline keyboard."""
        query = update.callback_query
        if query is None:
            return

        chat_id = update.effective_chat.id if update.effective_chat else "unknown"
        logger.info(f"Telegram: model callback received — data={query.data!r} chat={chat_id}")

        if not self._is_authorized(update):
            logger.warning(f"Telegram: model callback rejected — not authorized (chat={chat_id})")
            await query.answer()
            return

        model_key = query.data.replace("set_model:", "")

        if self._set_model:
            try:
                self._set_model(model_key)
                logger.info(f"Telegram: AI model changed to {model_key}")
            except ValueError as e:
                await query.answer(f"Error: {e}", show_alert=True)
                return

        display = AVAILABLE_MODELS.get(model_key, model_key)
        await query.answer(f"✅ Switched to {display}")

        markup = self._build_model_keyboard(model_key)
        await query.edit_message_text(
            f"✅ <b>AI Model switched to <code>{display}</code></b>\n\n"
            f"The agent will use <b>{display}</b> from the next M15 bar.\n"
            f"Use /model to change again.",
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    # ------------------------------------------------------------------
    # /strategy command + callback
    # ------------------------------------------------------------------

    async def _cmd_strategy(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show inline keyboard to pick the active rule-based strategy."""
        if update.effective_message is None or not self._is_authorized(update):
            return

        active_model = self._get_model() if self._get_model else "ollama"
        current = self._get_strategy() if self._get_strategy else "ema_pullback"
        markup = self._build_strategy_keyboard(current)

        note = ""
        if active_model != "strategy":
            note = "\n\n⚠️ <i>Switch to Strategy mode first: /model → Strategy (No AI)</i>"

        await update.effective_message.reply_text(
            f"📐 <b>Select Rule-Based Strategy</b>\n\n"
            f"Active now: <code>{AVAILABLE_STRATEGIES.get(current, current)}</code>"
            f"{note}\n\n"
            f"Tap a strategy to select it.",
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _cb_strategy(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle strategy selection from inline keyboard."""
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer()
            return

        strategy_key = query.data.replace("set_strategy:", "")

        if self._set_strategy:
            try:
                self._set_strategy(strategy_key)
                logger.info(f"Telegram: active strategy changed to {strategy_key}")
            except ValueError as e:
                await query.answer(f"Error: {e}", show_alert=True)
                return

        display = AVAILABLE_STRATEGIES.get(strategy_key, strategy_key)
        await query.answer(f"✅ Strategy: {display}")

        markup = self._build_strategy_keyboard(strategy_key)
        await query.edit_message_text(
            f"✅ <b>Strategy set to <code>{display}</code></b>\n\n"
            f"Make sure /model is set to <b>Strategy (No AI)</b> to use it.\n"
            f"Use /strategy to change again.",
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    # ------------------------------------------------------------------
    # /settings command + toggle callback
    # ------------------------------------------------------------------

    async def _cmd_settings(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Show toggle buttons for auto-trade and multi-market scan."""
        if update.effective_message is None or not self._is_authorized(update):
            return

        auto_trade = self._get_auto_trade() if self._get_auto_trade else True
        scan_all   = self._get_scan_all()   if self._get_scan_all   else False
        markup = self._build_settings_keyboard(auto_trade, scan_all)

        await update.effective_message.reply_text(
            self._settings_text(auto_trade, scan_all),
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    async def _cb_toggle_setting(
        self, update: Update, context: ContextTypes.DEFAULT_TYPE
    ) -> None:
        """Handle toggle button taps from /settings keyboard."""
        query = update.callback_query
        if query is None:
            return
        if not self._is_authorized(update):
            await query.answer()
            return

        key = query.data.replace("toggle_setting:", "")

        if key == "auto_trade":
            current = self._get_auto_trade() if self._get_auto_trade else True
            new_val = not current
            if self._set_auto_trade:
                self._set_auto_trade(new_val)
            label = "ON ✅" if new_val else "OFF ❌"
            logger.info(f"Telegram: auto_trade toggled → {new_val}")
            await query.answer(f"Auto-Trade {label}")

        elif key == "scan_all":
            current = self._get_scan_all() if self._get_scan_all else False
            new_val = not current
            if self._set_scan_all:
                self._set_scan_all(new_val)
            label = "ON ✅" if new_val else "OFF ❌"
            logger.info(f"Telegram: scan_all_symbols toggled → {new_val}")
            await query.answer(f"Multi-Market Scan {label}")

        else:
            await query.answer("Unknown setting")
            return

        auto_trade = self._get_auto_trade() if self._get_auto_trade else True
        scan_all   = self._get_scan_all()   if self._get_scan_all   else False
        markup = self._build_settings_keyboard(auto_trade, scan_all)
        await query.edit_message_text(
            self._settings_text(auto_trade, scan_all),
            parse_mode=ParseMode.HTML,
            reply_markup=markup,
        )

    def _settings_text(self, auto_trade: bool, scan_all: bool) -> str:
        at_icon  = "✅ ON" if auto_trade else "❌ OFF"
        sa_icon  = "✅ ON" if scan_all   else "❌ OFF"
        scan_note = (
            "\n⚠️ <i>Multi-market scan is ON — bot will scan all watchlist symbols per bar and pick the best setup.</i>"
            if scan_all else
            "\n<i>Multi-market scan is OFF — bot trades the single active symbol only.</i>"
        )
        return (
            f"⚙️ <b>Bot Settings</b>\n\n"
            f"🤖 Auto-Trade:          <b>{at_icon}</b>\n"
            f"   When ON, approved signals are placed automatically via MT5.\n"
            f"   When OFF, signals are sent to Telegram only (alert mode).\n\n"
            f"🌐 Multi-Market Scan:   <b>{sa_icon}</b>\n"
            f"   When ON, every bar scans all watchlist symbols and picks the best signal.\n"
            f"   When OFF, only the active market is analysed."
            f"{scan_note}\n\n"
            f"<i>Tap a button to toggle. Changes take effect on the next bar.</i>"
        )

    def _build_settings_keyboard(self, auto_trade: bool, scan_all: bool) -> InlineKeyboardMarkup:
        at_label = f"🤖 Auto-Trade: {'✅ ON' if auto_trade else '❌ OFF'}  →  toggle"
        sa_label = f"🌐 Multi-Market: {'✅ ON' if scan_all else '❌ OFF'}  →  toggle"
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(at_label, callback_data="toggle_setting:auto_trade")],
            [InlineKeyboardButton(sa_label, callback_data="toggle_setting:scan_all")],
        ])

    def _build_strategy_keyboard(self, active_strategy: str) -> InlineKeyboardMarkup:
        """Build inline keyboard from available strategies."""
        buttons: list[InlineKeyboardButton] = []
        for key, display in AVAILABLE_STRATEGIES.items():
            label = f"✅ 📐 {display}" if key == active_strategy else f"📐 {display}"
            buttons.append(InlineKeyboardButton(label, callback_data=f"set_strategy:{key}"))
        rows = [buttons[i : i + 1] for i in range(0, len(buttons))]
        return InlineKeyboardMarkup(rows)

    def _build_model_keyboard(self, active_model: str) -> InlineKeyboardMarkup:
        """Build inline keyboard from available models."""
        buttons: list[InlineKeyboardButton] = []
        icons = {"claude": "🔵", "ollama": "🟢", "strategy": "📐"}
        for key, display in AVAILABLE_MODELS.items():
            icon = icons.get(key, "⚪")
            label = f"✅ {icon} {display}" if key == active_model else f"{icon} {display}"
            buttons.append(InlineKeyboardButton(label, callback_data=f"set_model:{key}"))
        rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
        return InlineKeyboardMarkup(rows)

    def _build_market_keyboard(self, active_symbol: str) -> InlineKeyboardMarkup:
        """Build the 2-column inline keyboard from the dynamic watchlist."""
        watchlist = self._get_watchlist() if self._get_watchlist else []
        buttons: list[InlineKeyboardButton] = []
        for sym in watchlist:
            icon = _symbol_icon(sym)
            label = f"✅ {icon} {sym}" if sym == active_symbol else f"{icon} {sym}"
            buttons.append(InlineKeyboardButton(label, callback_data=f"set_market:{sym}"))

        rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
        return InlineKeyboardMarkup(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _send_raw(self, text: str) -> None:
        """Send a raw HTML message to the main control chat."""
        await self._send_to(self._chat_id, text)

    async def _send_to(self, chat_id: str, text: str) -> None:
        """Send a raw HTML message to any chat ID. Logs and swallows TelegramError."""
        if not self._enabled or not chat_id:
            return

        try:
            if self._app is not None:
                await self._app.bot.send_message(
                    chat_id=chat_id,
                    text=text,
                    parse_mode=ParseMode.HTML,
                    disable_web_page_preview=True,
                )
            else:
                async with Bot(token=settings.telegram_bot_token) as bot:
                    await bot.send_message(
                        chat_id=chat_id,
                        text=text,
                        parse_mode=ParseMode.HTML,
                        disable_web_page_preview=True,
                    )
        except TelegramError as e:
            logger.warning(f"Telegram send failed (chat={chat_id}): {e}")


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
        reason = html.escape((signal.reasoning or "No setup found.")[:200])
        return (
            f"{title}\n\n"
            f"<i>No trade signal — {reason}</i>"
        )

    rr_str   = f"{signal.risk_reward:.2f}" if signal.risk_reward else "n/a"
    approved = "✅ Approved" if risk.approved else "❌ Blocked"
    conf_bar = _confidence_bar(signal.confidence)

    # Truncate and escape reasoning + invalidation (Claude output may contain < > &)
    reasoning    = html.escape((signal.reasoning    or "")[:300])
    invalidation = html.escape((signal.invalidation or "")[:150])

    confluence = ""
    if signal.confluence_factors:
        factors = "\n".join(f"  • {html.escape(f)}" for f in signal.confluence_factors[:6])
        confluence = f"\n\n<b>Confluence:</b>\n{factors}"

    warnings = ""
    if risk.warnings:
        warnings = "\n⚠️ " + " | ".join(html.escape(w) for w in risk.warnings)

    rejection = ""
    if not risk.approved and risk.reasons:
        rejection = "\n🚫 " + " | ".join(html.escape(r) for r in risk.reasons)

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


def _format_status(
    data: dict,
    active_symbol: str | None = None,
    active_model: str | None = None,
    active_strategy: str | None = None,
    auto_trade: bool = True,
    scan_all: bool = False,
) -> str:
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

    _model = active_model or "ollama"
    model_display = AVAILABLE_MODELS.get(_model, _model)
    model_icons = {"claude": "🔵", "ollama": "🟢", "strategy": "📐"}
    model_icon = model_icons.get(_model, "⚪")
    model_line = f"{model_icon} Mode:              <b>{model_display}</b>\n"

    strategy_line = ""
    if _model == "strategy" and active_strategy:
        strat_display = AVAILABLE_STRATEGIES.get(active_strategy, active_strategy)
        strategy_line = f"📐 Strategy:         <b>{strat_display}</b>\n"

    at_icon = "✅ ON" if auto_trade else "❌ OFF"
    sa_icon = "✅ ON" if scan_all   else "❌ OFF"

    return (
        f"🤖 <b>AI Analyst Bot Status</b>\n\n"
        f"{sym_line}"
        f"{model_line}"
        f"{strategy_line}"
        f"🤖 Auto-Trade:        <b>{at_icon}</b>\n"
        f"🌐 Multi-Market:      <b>{sa_icon}</b>\n"
        f"⏱ Uptime:            <b>{uptime_str}</b>\n"
        f"{fetch_icon} MT5 Fetcher:      <b>{'Active' if fetcher else 'Inactive'}</b>\n"
        f"{zmq_icon} ZeroMQ:           <b>{'Active' if zmq_alive else 'Inactive'}</b>\n"
        f"📊 Cycles (session):  <b>{sigs}</b>\n"
        f"🕐 Last Cycle:        <b>{last_str}</b>\n\n"
        f"<i>Use /market · /model · /strategy · /settings to configure.</i>"
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
