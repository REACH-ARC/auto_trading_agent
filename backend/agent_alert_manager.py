"""
Agent Alert Manager — stores price-level alerts set by the AI agent.

The agent calls set_price_alert() during a cycle to register levels it wants to watch.
The fetcher loop checks these every ~30s. When price crosses a level, the alert is
marked triggered and popped at the next bar close, injecting context into the agent prompt.

This lets the agent skip full market analysis (Step 3) every 15 min when a position is
open and nothing notable has happened — only run when something it specifically flagged occurs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from loguru import logger


@dataclass
class PriceAlert:
    symbol: str
    price: float
    condition: str      # "above" | "below"
    hint: str           # agent-written note on what to do when triggered
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AgentAlertManager:
    """In-memory store for agent-set price alerts. One singleton per process."""

    def __init__(self) -> None:
        self._pending: dict[str, list[PriceAlert]] = {}
        self._triggered: dict[str, list[PriceAlert]] = {}

    def set_alert(self, symbol: str, price: float, condition: str, hint: str) -> None:
        """Register a price alert. Silently deduplicates same price+condition."""
        alerts = self._pending.setdefault(symbol, [])
        # Remove any existing alert at the same price/condition before adding
        self._pending[symbol] = [
            a for a in alerts
            if not (abs(a.price - price) < 1e-9 and a.condition == condition)
        ]
        self._pending[symbol].append(PriceAlert(
            symbol=symbol, price=price, condition=condition, hint=hint,
        ))
        logger.info(
            f"AgentAlert [{symbol}]: registered {condition} {price:.5f} — {hint[:80]}"
        )

    def clear(self, symbol: str) -> None:
        """Remove all pending and triggered alerts for a symbol."""
        self._pending.pop(symbol, None)
        self._triggered.pop(symbol, None)
        logger.debug(f"AgentAlert [{symbol}]: all alerts cleared")

    def get_pending(self, symbol: str) -> list[PriceAlert]:
        """Return currently pending (not yet triggered) alerts for a symbol."""
        return list(self._pending.get(symbol, []))

    def check_price(self, symbol: str, current_price: float) -> list[PriceAlert]:
        """
        Called every tick. Returns newly triggered alerts and moves them to the
        triggered store. Consumed at bar close via pop_triggered().
        """
        pending = self._pending.get(symbol, [])
        if not pending:
            return []

        already_triggered_keys = {
            (a.price, a.condition) for a in self._triggered.get(symbol, [])
        }
        new_triggers: list[PriceAlert] = []
        still_pending: list[PriceAlert] = []

        for alert in pending:
            key = (alert.price, alert.condition)
            if key in already_triggered_keys:
                still_pending.append(alert)
                continue

            hit = (
                (alert.condition == "below" and current_price <= alert.price)
                or (alert.condition == "above" and current_price >= alert.price)
            )
            if hit:
                new_triggers.append(alert)
                already_triggered_keys.add(key)
                logger.info(
                    f"AgentAlert [{symbol}]: TRIGGERED {alert.condition} {alert.price:.5f} "
                    f"@ price={current_price:.5f} — {alert.hint[:80]}"
                )
            else:
                still_pending.append(alert)

        if new_triggers:
            self._pending[symbol] = still_pending
            self._triggered.setdefault(symbol, []).extend(new_triggers)

        return new_triggers

    def pop_triggered(self, symbol: str) -> list[PriceAlert]:
        """Return and clear all triggered alerts for a symbol. Call at bar close."""
        return self._triggered.pop(symbol, [])

    def summary(self, symbol: str) -> str:
        """One-line summary of active alerts for logging."""
        pending = self._pending.get(symbol, [])
        if not pending:
            return "no alerts"
        parts = [f"{a.condition} {a.price:.5f}" for a in pending]
        return ", ".join(parts)


# Module-level singleton — shared between fetcher_loop and tool dispatcher
_manager = AgentAlertManager()


def get_alert_manager() -> AgentAlertManager:
    return _manager
