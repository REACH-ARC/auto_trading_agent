"""
Phase 10 — Multi-Symbol Scanner
SCAN-01  async scanner loop
SCAN-02  symbol watchlist from settings.yaml
SCAN-03  fetch latest data for all symbols every N minutes
SCAN-04  confluence strength scoring
SCAN-05  top-ranked signal per scan cycle
SCAN-06  APScheduler integration (wired in main.py)
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from loguru import logger

from config import settings
from backend.indicators import IndicatorResult, OHLCVData, compute_indicators
from backend.claude_analyst import SignalResult, analyse
from backend.news_filter import check_and_refresh
from backend.risk_manager import AccountState, RiskDecision, evaluate
from backend.signal_logger import log_signal


# ---------------------------------------------------------------------------
# SCAN-04  Confluence score
# ---------------------------------------------------------------------------

def _confluence_score(signal: SignalResult, indicators: IndicatorResult) -> float:
    """
    Composite score = base confidence + bonuses for:
      - number of confirmed confluence factors (capped at 10)
      - high-liquidity session (London / NY active)
      - London–NY overlap (highest liquidity window)
    """
    score = float(signal.confidence)
    score += min(len(signal.confluence_factors), 10) * 2.0
    if indicators.session.is_high_liquidity:
        score += 5.0
    if indicators.session.is_overlap:
        score += 3.0
    return round(score, 2)


# ---------------------------------------------------------------------------
# ScanResult — output of one symbol analysis
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    """Fully processed result for one symbol in a scan cycle."""
    symbol: str
    signal: SignalResult
    risk: RiskDecision
    signal_id: int
    confluence_score: float
    scanned_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# SCAN-01  Symbol data cache
# ---------------------------------------------------------------------------

class _SymbolCache:
    """Async-safe store of the most recent OHLCVData per symbol."""

    def __init__(self) -> None:
        self._data: dict[str, OHLCVData] = {}
        self._lock = asyncio.Lock()

    async def update(self, ohlcv: OHLCVData) -> None:
        async with self._lock:
            self._data[ohlcv.symbol] = ohlcv
            logger.debug(f"Scanner cache updated: {ohlcv.symbol}")

    async def snapshot(self) -> dict[str, OHLCVData]:
        async with self._lock:
            return dict(self._data)

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()


_cache = _SymbolCache()

# Fallback account state when the scanner runs without live MT5 account data.
# main.py can inject a real AccountState via scan_cycle(account=...).
_DEFAULT_ACCOUNT = AccountState(
    equity=10_000.0,
    balance=10_000.0,
    open_trades=0,
    daily_pnl=0.0,
)


async def update_symbol_data(ohlcv: OHLCVData) -> None:
    """
    Called by the ZeroMQ listener every time MT5 pushes fresh bars.
    Keeps the scanner cache current so the next scheduled cycle has
    up-to-date data for every symbol that has been active.
    """
    await _cache.update(ohlcv)


# ---------------------------------------------------------------------------
# Per-symbol analysis (run concurrently via asyncio.gather)
# ---------------------------------------------------------------------------

async def _analyse_symbol(
    symbol: str,
    ohlcv: OHLCVData,
    account: AccountState,
    now: datetime,
) -> ScanResult | None:
    """
    Full analysis pipeline for one symbol.
    Returns None when no actionable trade signal is found.

    CPU-bound / blocking calls (indicators, Claude API) are offloaded to a
    thread pool so the event loop remains responsive during concurrent scans.
    """
    # News gate
    blocked, blocking_events = await check_and_refresh(symbol, now)
    if blocked:
        titles = ", ".join(e.title for e in blocking_events)
        logger.debug(f"Scanner: {symbol} blocked by news ({titles})")
        return None

    # Indicators (blocking numpy/pandas work — offload to thread)
    try:
        indicators: IndicatorResult = await asyncio.to_thread(compute_indicators, ohlcv)
    except Exception as exc:
        logger.warning(f"Scanner: indicator error for {symbol}: {exc}")
        return None

    # Claude analysis (blocking HTTP call — offload to thread)
    try:
        signal: SignalResult = await asyncio.to_thread(analyse, ohlcv, indicators)
    except Exception as exc:
        logger.warning(f"Scanner: Claude error for {symbol}: {exc}")
        return None

    if not signal.is_actionable:
        logger.debug(f"Scanner: {symbol} → NO_TRADE (conf={signal.confidence}%)")
        return None

    # Risk check
    atr_h4 = indicators.atr.get(
        "H4",
        indicators.atr.get(list(indicators.atr.keys())[-1], 0.0) if indicators.atr else 0.0,
    )
    risk: RiskDecision = evaluate(signal, account, atr_h4)

    signal_id = await log_signal(signal, risk)
    score = _confluence_score(signal, indicators)

    logger.info(
        f"Scanner: {symbol} {signal.direction} conf={signal.confidence}% "
        f"score={score} approved={risk.approved} id={signal_id}"
    )
    return ScanResult(
        symbol=symbol,
        signal=signal,
        risk=risk,
        signal_id=signal_id,
        confluence_score=score,
    )


# ---------------------------------------------------------------------------
# SCAN-03 / SCAN-05  One full scan cycle
# ---------------------------------------------------------------------------

async def scan_cycle(
    account: AccountState | None = None,
    *,
    stale_threshold_minutes: int = 60,
) -> list[ScanResult]:
    """
    Analyse every watchlist symbol that has fresh cached data.

    Returns results sorted by confluence_score (highest first).
    Only the first `scanner.top_signals_per_cycle` entries are meant to be
    published — callers must slice accordingly.

    Args:
        account: Live account state from MT5 (falls back to defaults if None).
        stale_threshold_minutes: Symbols whose last bar is older than this
            are skipped — their data is considered too stale to trade on.
    """
    cfg_scanner = settings.scanner
    watchlist: list[str] = settings.symbols.get("watchlist", [])

    if not watchlist:
        logger.warning("Scanner: watchlist is empty — check settings.yaml symbols.watchlist")
        return []

    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(minutes=stale_threshold_minutes)
    data_snapshot = await _cache.snapshot()
    effective_account = account or _DEFAULT_ACCOUNT

    # Build per-symbol coroutines for all symbols with fresh data
    tasks: list[asyncio.Task] = []
    queued_symbols: list[str] = []

    for symbol in watchlist:
        ohlcv = data_snapshot.get(symbol)
        if ohlcv is None:
            logger.debug(f"Scanner: {symbol} — no cached data, skipping")
            continue
        if ohlcv.received_at < stale_cutoff:
            age_min = (now - ohlcv.received_at).total_seconds() / 60
            logger.debug(f"Scanner: {symbol} — stale data ({age_min:.0f} min old), skipping")
            continue
        tasks.append(asyncio.ensure_future(_analyse_symbol(symbol, ohlcv, effective_account, now)))
        queued_symbols.append(symbol)

    if not tasks:
        logger.info("Scanner: no fresh symbol data available for this cycle")
        return []

    logger.info(f"Scanner: analysing {len(tasks)} symbols — {queued_symbols}")
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results: list[ScanResult] = []
    for symbol, outcome in zip(queued_symbols, raw_results):
        if isinstance(outcome, Exception):
            logger.error(f"Scanner: unhandled error for {symbol}: {outcome}")
        elif outcome is not None:
            results.append(outcome)

    # SCAN-05  Rank by confluence score
    results.sort(key=lambda r: r.confluence_score, reverse=True)

    top_n: int = cfg_scanner.get("top_signals_per_cycle", 1)
    top = results[:top_n]

    logger.info(
        f"Scanner cycle complete — {len(watchlist)} watchlist, "
        f"{len(queued_symbols)} fresh, {len(results)} signals found. "
        f"Top {top_n}: "
        + (", ".join(f"{r.symbol}({r.confluence_score})" for r in top) or "none")
    )

    return results  # full sorted list; callers slice to top_n


# ---------------------------------------------------------------------------
# SCAN-06  APScheduler job entry-point
# ---------------------------------------------------------------------------

OnSignalCallback = Callable[[ScanResult], Awaitable[None]]


async def scheduled_scan(
    on_signal_fn: OnSignalCallback | None = None,
    account: AccountState | None = None,
) -> None:
    """
    APScheduler job called every `scanner.interval_minutes` minutes.

    Runs a full scan cycle, picks the top-ranked signals, and calls
    `on_signal_fn` for each (e.g. to publish via ZeroMQ or Telegram).
    """
    cfg_scanner = settings.scanner
    top_n: int = cfg_scanner.get("top_signals_per_cycle", 1)

    logger.info("Scanner: scheduled scan cycle starting")
    try:
        all_results = await scan_cycle(account=account)
        top = all_results[:top_n]

        if not top:
            logger.info("Scanner: no actionable signals this cycle")
            return

        if on_signal_fn is not None:
            for result in top:
                try:
                    await on_signal_fn(result)
                except Exception as exc:
                    logger.error(f"Scanner: on_signal_fn failed for {result.symbol}: {exc}")

    except Exception as exc:
        logger.error(f"Scanner: unhandled error in scheduled_scan: {exc}")
