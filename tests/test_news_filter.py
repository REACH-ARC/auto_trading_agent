"""Unit tests for backend/news_filter.py — NEWS-01 through NEWS-05"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.news_filter import (
    NewsEvent,
    _NewsCache,
    _currencies_for_symbol,
    _parse_ff_events,
    _parse_impact,
    check_and_refresh,
    fetch_calendar,
    get_upcoming_events,
    is_news_blocked,
    refresh_cache_if_stale,
    _cache,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc(hour: int, minute: int = 0, day: int = 15) -> datetime:
    return datetime(2024, 1, day, hour, minute, 0, tzinfo=timezone.utc)


def _make_event(
    currency: str = "USD",
    impact: str = "HIGH",
    hour: int = 14,
    minute: int = 30,
    title: str = "NFP",
) -> NewsEvent:
    return NewsEvent(
        title=title,
        currency=currency,
        impact=impact,  # type: ignore
        event_time=_utc(hour, minute),
    )


def _make_ff_raw(
    currency: str = "USD",
    impact: str = "High",
    date_str: str = "2024-01-15T14:30:00+00:00",
    title: str = "Non-Farm Payrolls",
) -> dict:
    return {
        "title": title,
        "country": currency,
        "impact": impact,
        "date": date_str,
    }


def _reset_cache() -> None:
    """Reset global cache between tests."""
    _cache.events.clear()
    _cache.fetched_at = None


# ---------------------------------------------------------------------------
# NEWS-01  NewsEvent dataclass
# ---------------------------------------------------------------------------

class TestNewsEvent:
    def test_block_start_before_event(self):
        event = _make_event(hour=14, minute=30)
        assert event.block_start < event.event_time

    def test_block_end_after_event(self):
        event = _make_event(hour=14, minute=30)
        assert event.block_end > event.event_time

    def test_is_blocking_exactly_at_event(self):
        event = _make_event(hour=14, minute=30)
        assert event.is_blocking(_utc(14, 30)) is True

    def test_is_blocking_inside_window(self):
        event = _make_event(hour=14, minute=30)
        # 15 minutes before — within default 30-min pre-window
        assert event.is_blocking(_utc(14, 15)) is True

    def test_is_blocking_after_window(self):
        event = _make_event(hour=14, minute=30)
        # 31 minutes before — outside window
        assert event.is_blocking(_utc(13, 59)) is False

    def test_is_blocking_well_after_event(self):
        event = _make_event(hour=14, minute=30)
        # 31 minutes after — outside post-window
        assert event.is_blocking(_utc(15, 1)) is False

    def test_not_blocking_outside_window(self):
        event = _make_event(hour=14, minute=30)
        assert event.is_blocking(_utc(10, 0)) is False


# ---------------------------------------------------------------------------
# NEWS-05  _NewsCache
# ---------------------------------------------------------------------------

class TestNewsCache:
    def test_is_stale_when_empty(self):
        cache = _NewsCache()
        assert cache.is_stale() is True

    def test_not_stale_immediately_after_update(self):
        cache = _NewsCache()
        cache.update([_make_event()])
        assert cache.is_stale() is False

    def test_stale_after_ttl(self):
        cache = _NewsCache()
        cache.update([_make_event()])
        # Backdate fetched_at beyond refresh window (4 hours)
        cache.fetched_at = datetime.now(timezone.utc) - timedelta(hours=5)
        assert cache.is_stale() is True

    def test_update_stores_events(self):
        cache = _NewsCache()
        events = [_make_event("EUR"), _make_event("USD")]
        cache.update(events)
        assert len(cache.events) == 2

    def test_update_replaces_old_events(self):
        cache = _NewsCache()
        cache.update([_make_event("EUR")])
        cache.update([_make_event("GBP"), _make_event("USD")])
        assert len(cache.events) == 2
        assert all(e.currency != "EUR" for e in cache.events)


# ---------------------------------------------------------------------------
# NEWS-02  _parse_ff_events
# ---------------------------------------------------------------------------

class TestParseFfEvents:
    def test_parses_single_event(self):
        raw = [_make_ff_raw()]
        events = _parse_ff_events(raw)
        assert len(events) == 1
        assert events[0].title == "Non-Farm Payrolls"
        assert events[0].currency == "USD"
        assert events[0].impact == "HIGH"

    def test_parses_multiple_events(self):
        raw = [_make_ff_raw("USD"), _make_ff_raw("EUR", "Medium")]
        events = _parse_ff_events(raw)
        assert len(events) == 2

    def test_impact_mapped_correctly(self):
        assert _parse_impact("High") == "HIGH"
        assert _parse_impact("Medium") == "MEDIUM"
        assert _parse_impact("Low") == "LOW"
        assert _parse_impact("Holiday") == "HOLIDAY"
        assert _parse_impact("Unknown") == "LOW"

    def test_skips_event_with_missing_date(self):
        raw = [{"title": "No date event", "country": "USD", "impact": "High", "date": ""}]
        events = _parse_ff_events(raw)
        assert len(events) == 0

    def test_skips_malformed_event(self):
        raw = [{"bad": "data"}, _make_ff_raw()]
        events = _parse_ff_events(raw)
        assert len(events) == 1

    def test_datetime_converted_to_utc(self):
        # Eastern time offset (-05:00), 09:30 ET = 14:30 UTC
        raw = [_make_ff_raw(date_str="2024-01-15T09:30:00-05:00")]
        events = _parse_ff_events(raw)
        assert events[0].event_time.hour == 14
        assert events[0].event_time.tzinfo == timezone.utc

    def test_empty_list_returns_empty(self):
        assert _parse_ff_events([]) == []


# ---------------------------------------------------------------------------
# NEWS-02  fetch_calendar (mocked HTTP)
# ---------------------------------------------------------------------------

class TestFetchCalendar:
    @pytest.mark.asyncio
    async def test_returns_events_on_success(self):
        this_week = [_make_ff_raw("USD"), _make_ff_raw("EUR", "Medium")]
        next_week = [_make_ff_raw("GBP", "High", date_str="2024-01-22T10:00:00+00:00")]

        async def mock_get(url, timeout):
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            if "thisweek" in url:
                resp.json.return_value = this_week
            else:
                resp.json.return_value = next_week
            return resp

        with patch("backend.news_filter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = mock_get

            events = await fetch_calendar()

        assert len(events) == 3

    @pytest.mark.asyncio
    async def test_returns_empty_list_on_http_failure(self):
        async def mock_get(url, timeout):
            raise httpx.ConnectError("network failure")

        with patch("backend.news_filter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = mock_get

            events = await fetch_calendar()

        assert events == []

    @pytest.mark.asyncio
    async def test_partial_failure_still_returns_successful_week(self):
        this_week = [_make_ff_raw("USD")]

        async def mock_get(url, timeout):
            if "nextweek" in url:
                raise httpx.ConnectError("fail")
            resp = MagicMock()
            resp.raise_for_status = MagicMock()
            resp.json.return_value = this_week
            return resp

        with patch("backend.news_filter.httpx.AsyncClient") as mock_client_cls:
            mock_client = AsyncMock()
            mock_client_cls.return_value.__aenter__.return_value = mock_client
            mock_client.get.side_effect = mock_get

            events = await fetch_calendar()

        assert len(events) == 1


# ---------------------------------------------------------------------------
# NEWS-03  get_upcoming_events
# ---------------------------------------------------------------------------

class TestGetUpcomingEvents:
    def setup_method(self):
        _reset_cache()

    def test_returns_upcoming_high_impact(self):
        now = _utc(13, 0)
        event = _make_event("USD", "HIGH", hour=14, minute=0)
        _cache.update([event])
        result = get_upcoming_events("USD", now)
        assert len(result) == 1
        assert result[0].title == "NFP"

    def test_excludes_past_events(self):
        now = _utc(15, 0)
        event = _make_event("USD", "HIGH", hour=13, minute=0)  # 2 hrs ago
        _cache.update([event])
        result = get_upcoming_events("USD", now)
        assert result == []

    def test_excludes_medium_impact(self):
        now = _utc(13, 0)
        event = _make_event("USD", "MEDIUM", hour=14, minute=0)
        _cache.update([event])
        result = get_upcoming_events("USD", now)
        # Only HIGH is in impact_levels by default
        assert result == []

    def test_excludes_different_currency(self):
        now = _utc(13, 0)
        _cache.update([_make_event("EUR", "HIGH", hour=14)])
        result = get_upcoming_events("USD", now)
        assert result == []

    def test_respects_lookahead_window(self):
        now = _utc(10, 0)
        far_event = _make_event("USD", "HIGH", hour=13, minute=0)  # 3 hrs away
        _cache.update([far_event])
        result = get_upcoming_events("USD", now, lookahead_hours=2)
        assert result == []

        result = get_upcoming_events("USD", now, lookahead_hours=4)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# NEWS-04  is_news_blocked
# ---------------------------------------------------------------------------

class TestIsNewsBlocked:
    def setup_method(self):
        _reset_cache()

    def test_blocked_during_event_window(self):
        event = _make_event("USD", "HIGH", hour=14, minute=30)
        _cache.update([event])
        blocked, events = is_news_blocked("EURUSD", _utc(14, 20))
        assert blocked is True
        assert len(events) == 1

    def test_not_blocked_outside_window(self):
        event = _make_event("USD", "HIGH", hour=14, minute=30)
        _cache.update([event])
        blocked, events = is_news_blocked("EURUSD", _utc(10, 0))
        assert blocked is False
        assert events == []

    def test_not_blocked_when_different_currency(self):
        event = _make_event("JPY", "HIGH", hour=14, minute=30)
        _cache.update([event])
        blocked, _ = is_news_blocked("EURUSD", _utc(14, 20))
        assert blocked is False

    def test_xauusd_blocked_by_usd_event(self):
        event = _make_event("USD", "HIGH", hour=14, minute=30)
        _cache.update([event])
        blocked, _ = is_news_blocked("XAUUSD", _utc(14, 20))
        assert blocked is True

    def test_gbpusd_blocked_by_gbp_event(self):
        event = _make_event("GBP", "HIGH", hour=10, minute=0)
        _cache.update([event])
        blocked, _ = is_news_blocked("GBPUSD", _utc(9, 45))
        assert blocked is True

    def test_not_blocked_medium_impact(self):
        event = _make_event("USD", "MEDIUM", hour=14, minute=30)
        _cache.update([event])
        blocked, _ = is_news_blocked("EURUSD", _utc(14, 20))
        assert blocked is False

    def test_not_blocked_when_cache_empty(self):
        # no events in cache → should not block (fail open)
        blocked, events = is_news_blocked("EURUSD", _utc(14, 0))
        assert blocked is False
        assert events == []

    def test_disabled_filter_never_blocks(self):
        event = _make_event("USD", "HIGH", hour=14, minute=30)
        _cache.update([event])
        with patch("backend.news_filter.settings") as mock_settings:
            mock_settings.news_filter = {"enabled": False}
            blocked, _ = is_news_blocked("EURUSD", _utc(14, 20))
        assert blocked is False

    def test_uses_now_when_dt_not_provided(self):
        # Should not raise — just checks current time
        blocked, _ = is_news_blocked("EURUSD")
        assert isinstance(blocked, bool)


# ---------------------------------------------------------------------------
# NEWS-04  _currencies_for_symbol
# ---------------------------------------------------------------------------

class TestCurrenciesForSymbol:
    def test_known_symbol(self):
        assert set(_currencies_for_symbol("EURUSD")) == {"EUR", "USD"}

    def test_case_insensitive(self):
        assert set(_currencies_for_symbol("eurusd")) == {"EUR", "USD"}

    def test_xauusd_maps_to_usd(self):
        assert "USD" in _currencies_for_symbol("XAUUSD")

    def test_indices_map_to_usd(self):
        assert "USD" in _currencies_for_symbol("US30")
        assert "USD" in _currencies_for_symbol("NAS100")

    def test_unknown_6char_pair_derived(self):
        # EURJPY not in table — derive from string
        result = _currencies_for_symbol("EURJPY")
        assert "EUR" in result and "JPY" in result

    def test_unknown_symbol_returns_empty(self):
        assert _currencies_for_symbol("UNKNOWN123") == []


# ---------------------------------------------------------------------------
# NEWS-05  refresh_cache_if_stale
# ---------------------------------------------------------------------------

class TestRefreshCacheIfStale:
    def setup_method(self):
        _reset_cache()

    @pytest.mark.asyncio
    async def test_fetches_when_stale(self):
        assert _cache.is_stale()
        events = [_make_event("USD"), _make_event("EUR")]

        with patch("backend.news_filter.fetch_calendar", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = events
            await refresh_cache_if_stale()

        mock_fetch.assert_called_once()
        assert not _cache.is_stale()
        assert len(_cache.events) == 2

    @pytest.mark.asyncio
    async def test_skips_fetch_when_fresh(self):
        _cache.update([_make_event("USD")])
        assert not _cache.is_stale()

        with patch("backend.news_filter.fetch_calendar", new_callable=AsyncMock) as mock_fetch:
            await refresh_cache_if_stale()

        mock_fetch.assert_not_called()


# ---------------------------------------------------------------------------
# check_and_refresh integration
# ---------------------------------------------------------------------------

class TestCheckAndRefresh:
    def setup_method(self):
        _reset_cache()

    @pytest.mark.asyncio
    async def test_refreshes_then_checks(self):
        event = _make_event("USD", "HIGH", hour=14, minute=30)

        with patch("backend.news_filter.fetch_calendar", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = [event]
            blocked, events = await check_and_refresh("EURUSD", _utc(14, 20))

        assert blocked is True
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_not_blocked_when_no_events(self):
        with patch("backend.news_filter.fetch_calendar", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = []
            blocked, events = await check_and_refresh("EURUSD", _utc(14, 20))

        assert blocked is False
        assert events == []
