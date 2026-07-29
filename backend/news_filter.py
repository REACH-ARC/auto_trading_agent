"""
Phase 4 — News Filter
NEWS-01 through NEWS-05
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
from loguru import logger

from config import settings

# ---------------------------------------------------------------------------
# NEWS-01  Types
# ---------------------------------------------------------------------------

ImpactLevel = Literal["LOW", "MEDIUM", "HIGH", "HOLIDAY"]

# Forex Factory JSON calendar (current week) — well-known community endpoint
_FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
_FF_NEXT_WEEK_URL = "https://nfs.faireconomy.media/ff_calendar_nextweek.json"

# Map trading symbol → affected currencies
_SYMBOL_CURRENCIES: dict[str, list[str]] = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "USDCHF": ["USD", "CHF"],
    "AUDUSD": ["AUD", "USD"],
    "NZDUSD": ["NZD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "XAUUSD": ["USD"],
    "XAGUSD": ["USD"],
    "US30":   ["USD"],
    "NAS100": ["USD"],
    "SPX500": ["USD"],
    "US500":  ["USD"],
    "GER40":  ["EUR"],
    "UK100":  ["GBP"],
}


@dataclass
class NewsEvent:
    title: str
    currency: str
    impact: ImpactLevel
    event_time: datetime  # UTC

    @property
    def block_start(self) -> datetime:
        minutes = settings.news_filter["block_minutes_before"]
        return self.event_time - timedelta(minutes=minutes)

    @property
    def block_end(self) -> datetime:
        minutes = settings.news_filter["block_minutes_after"]
        return self.event_time + timedelta(minutes=minutes)

    def is_blocking(self, dt: datetime) -> bool:
        return self.block_start <= dt <= self.block_end


# ---------------------------------------------------------------------------
# NEWS-05  In-memory cache with TTL
# ---------------------------------------------------------------------------

@dataclass
class _NewsCache:
    events: list[NewsEvent] = field(default_factory=list)
    fetched_at: datetime | None = None

    def is_stale(self) -> bool:
        if self.fetched_at is None:
            return True
        refresh_hours: int = settings.news_filter["calendar_refresh_hours"]
        age = datetime.now(timezone.utc) - self.fetched_at
        return age >= timedelta(hours=refresh_hours)

    def update(self, events: list[NewsEvent]) -> None:
        self.events = events
        self.fetched_at = datetime.now(timezone.utc)
        logger.info(f"News cache updated: {len(events)} events loaded")


_cache = _NewsCache()


# ---------------------------------------------------------------------------
# NEWS-02  Fetch economic calendar
# ---------------------------------------------------------------------------

def _parse_impact(raw: str) -> ImpactLevel:
    mapping = {
        "High":     "HIGH",
        "Medium":   "MEDIUM",
        "Low":      "LOW",
        "Holiday":  "HOLIDAY",
        "Non-Economic": "LOW",
    }
    return mapping.get(raw, "LOW")  # type: ignore


def _parse_ff_events(data: list[dict]) -> list[NewsEvent]:
    """Parse Forex Factory JSON list into NewsEvent objects (UTC times)."""
    events: list[NewsEvent] = []
    for item in data:
        try:
            raw_date = item.get("date", "")
            if not raw_date:
                continue

            # FF timestamps are already in UTC ISO format: "2024-01-15T08:30:00-0500"
            # We normalise to UTC aware datetime
            dt = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            dt = dt.astimezone(timezone.utc)

            impact = _parse_impact(item.get("impact", "Low"))
            currency = str(item.get("country", "")).upper()
            title = str(item.get("title", "Unknown Event"))

            events.append(NewsEvent(
                title=title,
                currency=currency,
                impact=impact,
                event_time=dt,
            ))
        except Exception as e:
            logger.debug(f"Skipping unparseable news event: {e} | raw={item}")

    return events


async def _fetch_url(client: httpx.AsyncClient, url: str) -> list[dict]:
    try:
        resp = await client.get(url, timeout=10.0)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            # Next-week calendar returns 404 until ForexFactory publishes it (~Thursday)
            logger.debug(f"Calendar not yet available (404): {url}")
        else:
            logger.warning(f"Failed to fetch calendar from {url}: {e}")
        return []
    except Exception as e:
        logger.warning(f"Failed to fetch calendar from {url}: {e}")
        return []


async def fetch_calendar() -> list[NewsEvent]:
    """
    NEWS-02 — Fetch this week + next week calendars from Forex Factory.
    Returns empty list (does NOT block trading) if the fetch fails.
    """
    async with httpx.AsyncClient() as client:
        this_week, next_week = await asyncio.gather(
            _fetch_url(client, _FF_CALENDAR_URL),
            _fetch_url(client, _FF_NEXT_WEEK_URL),
        )

    all_raw = this_week + next_week
    events = _parse_ff_events(all_raw)
    logger.info(f"Fetched {len(events)} news events ({len(this_week)} this week, {len(next_week)} next week)")
    return events


# ---------------------------------------------------------------------------
# NEWS-05  Cache refresh
# ---------------------------------------------------------------------------

async def refresh_cache_if_stale() -> None:
    """Refresh the in-memory cache if TTL has expired."""
    if _cache.is_stale():
        events = await fetch_calendar()
        _cache.update(events)


def get_upcoming_events(
    currency: str,
    dt: datetime,
    lookahead_hours: int = 2,
) -> list[NewsEvent]:
    """NEWS-03 — Return high-impact events for a currency in the next N hours."""
    blocked_impacts: list[str] = settings.news_filter["impact_levels"]
    window_end = dt + timedelta(hours=lookahead_hours)

    return [
        e for e in _cache.events
        if e.currency == currency
        and e.impact in blocked_impacts
        and dt <= e.event_time <= window_end
    ]


def get_all_upcoming_events(dt: datetime, lookahead_hours: float = 4.0) -> list[NewsEvent]:
    """Return all cached events (any currency) in the next N hours, sorted by event time."""
    blocked_impacts: list[str] = settings.news_filter["impact_levels"]
    window_end = dt + timedelta(hours=lookahead_hours)
    return sorted(
        (
            e for e in _cache.events
            if e.impact in blocked_impacts
            and dt <= e.event_time <= window_end
        ),
        key=lambda e: e.event_time,
    )


# ---------------------------------------------------------------------------
# NEWS-04  Block check — public API
# ---------------------------------------------------------------------------

def _currencies_for_symbol(symbol: str) -> list[str]:
    """Return the currencies affected by a trading symbol."""
    from backend import account_manager
    base_symbol = account_manager.strip_suffix(symbol).upper()
    
    if base_symbol in _SYMBOL_CURRENCIES:
        return _SYMBOL_CURRENCIES[base_symbol]
    # Fallback: try to derive from 6-char FX pair (e.g. EURJPY → EUR, JPY)
    if len(base_symbol) == 6 and base_symbol.isalpha():
        return [base_symbol[:3], base_symbol[3:]]
    return []


def is_news_blocked(symbol: str, dt: datetime | None = None) -> tuple[bool, list[NewsEvent]]:
    """
    NEWS-04 — Check whether trading is blocked for a symbol at the given time.

    Returns (blocked: bool, blocking_events: list[NewsEvent]).
    Always returns (False, []) if news_filter is disabled in settings.
    """
    if not settings.news_filter["enabled"]:
        return False, []

    if dt is None:
        dt = datetime.now(timezone.utc)

    blocked_impacts: list[str] = settings.news_filter["impact_levels"]
    currencies = _currencies_for_symbol(symbol)

    blocking: list[NewsEvent] = [
        e for e in _cache.events
        if e.currency in currencies
        and e.impact in blocked_impacts
        and e.is_blocking(dt)
    ]

    if blocking:
        names = ", ".join(f"{e.currency} '{e.title}'" for e in blocking)
        logger.warning(f"News block active for {symbol} at {dt.strftime('%H:%M UTC')}: {names}")
        return True, blocking

    return False, []


async def check_and_refresh(symbol: str, dt: datetime | None = None) -> tuple[bool, list[NewsEvent]]:
    """
    Convenience coroutine: refresh cache if stale, then check if trading is blocked.
    Call this from the backend server before each signal request.
    """
    await refresh_cache_if_stale()
    return is_news_blocked(symbol, dt)


def get_news_context_for_agent(
    symbol: str,
    dt: datetime | None = None,
    lookahead_minutes: int = 30,
) -> list[NewsEvent]:
    """
    Return high-impact news events relevant to an AI agent cycle:
    events currently in the blocking window OR releasing within the next lookahead_minutes.
    Used to inject news awareness into the agent prompt instead of hard-blocking.
    """
    if dt is None:
        dt = datetime.now(timezone.utc)

    currencies = _currencies_for_symbol(symbol)
    if not currencies:
        return []

    blocked_impacts: list[str] = settings.news_filter["impact_levels"]
    window_end = dt + timedelta(minutes=lookahead_minutes)

    return sorted(
        [
            e for e in _cache.events
            if e.currency in currencies
            and e.impact in blocked_impacts
            and (e.is_blocking(dt) or dt <= e.event_time <= window_end)
        ],
        key=lambda e: e.event_time,
    )
