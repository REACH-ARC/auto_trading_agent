"""
Quick smoke-test for Telegram notifications.
Run: python test_telegram.py
"""
import asyncio
from datetime import datetime, timezone

from backend.claude_analyst import SignalResult
from backend.risk_manager import RiskDecision
from backend.signal_logger import SignalStats
from notifications.telegram_bot import notifier


def _make_signal() -> SignalResult:
    return SignalResult(
        symbol="EURUSD",
        direction="BUY",
        confidence=82,
        reasoning=(
            "H4 EMA alignment is bullish (20>50>200). Price bounced from the "
            "1.08500 support zone with RSI divergence on M15. MACD histogram "
            "turning positive on H1 confirms momentum shift."
        ),
        invalidation="4H close below 1.08420 invalidates the setup.",
        confluence_factors=[
            "H4 EMA stack bullish",
            "RSI divergence on M15",
            "MACD histogram positive H1",
            "Price at key support 1.08500",
            "London session active",
        ],
        entry=1.08650,
        sl=1.08380,
        tp1=1.09055,
        tp2=1.09325,
        tp3=1.09730,
    )


def _make_risk() -> RiskDecision:
    return RiskDecision(
        approved=True,
        lot_size=0.37,
        risk_amount=99.90,
        sl_distance=0.00270,
        warnings=["One trade slot remaining (2/3)"],
    )


def _make_stats() -> SignalStats:
    return SignalStats(
        total_signals=14,
        actionable_signals=9,
        no_trade_signals=5,
        pending_signals=1,
        wins=5,
        losses=3,
        breakevens=1,
        win_rate_pct=62.5,
        avg_rr=1.84,
        total_pnl=312.40,
        total_api_cost_usd=0.0182,
        by_symbol={
            "EURUSD": {"total": 6, "wins": 3, "losses": 2, "win_rate": 60.0, "avg_rr": 1.7, "total_pnl": 145.20},
            "XAUUSD": {"total": 5, "wins": 2, "losses": 1, "win_rate": 66.7, "avg_rr": 2.1, "total_pnl": 167.20},
        },
        by_direction={"BUY": 8, "SELL": 1, "NO_TRADE": 5},
        period_start=datetime(2026, 4, 22, 0, 0, tzinfo=timezone.utc),
        period_end=datetime.now(timezone.utc),
    )


async def main() -> None:
    print("Telegram notifier enabled:", notifier._enabled)
    if not notifier._enabled:
        print("\nNot enabled — check:")
        print("  1. TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are set in .env")
        print("  2. telegram.enabled = true in config/settings.yaml")
        return

    print("\n[1/3] Sending signal alert...")
    await notifier.send_signal_alert(_make_signal(), _make_risk())
    print("      Done.")

    print("[2/3] Sending daily summary...")
    await notifier.send_daily_summary(_make_stats())
    print("      Done.")

    print("[3/3] Sending status message...")
    status_data = {
        "status": "ok",
        "uptime_seconds": 7542,
        "signals_processed": 14,
        "last_signal_at": datetime.now(timezone.utc).isoformat(),
        "zmq_listener_alive": True,
    }
    from notifications.telegram_bot import _format_status
    from telegram import Bot
    from config import settings
    async with Bot(token=settings.telegram_bot_token) as bot:
        await bot.send_message(
            chat_id=settings.telegram_chat_id,
            text=_format_status(status_data),
            parse_mode="HTML",
        )
    print("      Done.\n")
    print("All 3 messages sent — check your Telegram chat.")


if __name__ == "__main__":
    asyncio.run(main())
