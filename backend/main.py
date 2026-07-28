"""
Phase 7 — FastAPI Backend Server   (SRV-01, SRV-03–SRV-09)
Phase 10 — Multi-Symbol Scanner integration (SCAN-06)
"""
from __future__ import annotations

import asyncio
import sys

# ZMQ requires SelectorEventLoop on Windows (Proactor is the default in Python 3.8+)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

from config import settings
from backend.indicators import OHLCVData, compute_indicators
from backend.claude_analyst import SignalResult, analyse
from backend.news_filter import check_and_refresh
from backend.risk_manager import AccountState, RiskDecision, evaluate
from backend.signal_logger import (
    SignalStats,
    get_pending_signals,
    get_signal,
    get_stats,
    init_db,
    log_signal,
    update_outcome,
)
from backend.mt5_bridge import parse_mt5_message, serialize_signal
from backend.mt5_tools import get_symbol_info as _mt5_symbol_info, get_open_positions as _mt5_open_positions
from backend.scanner import ScanResult, scheduled_scan, update_symbol_data
from backend.mt5_fetcher import fetcher_loop
from backend.claude_agent import run_agent
from backend.strategy_engine import run_strategy
from backend.trade_manager import trade_manager_loop
from backend.backtester import BacktestConfig, run_backtest
from backend import model_manager, account_manager
from notifications.telegram_bot import notifier as telegram

# ---------------------------------------------------------------------------
# SRV-09  Logging setup
# ---------------------------------------------------------------------------

_logging_configured = False


def _setup_logging() -> None:
    global _logging_configured
    if _logging_configured:
        return

    log_cfg = settings.logging

    # Ensure log directory exists before adding the file sink
    from pathlib import Path
    Path(log_cfg["file"]).parent.mkdir(parents=True, exist_ok=True)

    logger.remove()  # remove default handler
    logger.add(
        sys.stdout,
        level=log_cfg["level"],
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
               "<level>{message}</level>",
        colorize=True,
    )
    logger.add(
        log_cfg["file"],
        level=log_cfg["level"],
        rotation=log_cfg["rotation"],
        retention=log_cfg["retention"],
        encoding="utf-8",
    )
    _logging_configured = True
    logger.info("Logging initialised")


# ---------------------------------------------------------------------------
# Shared app state
# ---------------------------------------------------------------------------

class _AppState:
    zmq_task: asyncio.Task | None = None
    mt5_fetcher_task: asyncio.Task | None = None
    trade_manager_task: asyncio.Task | None = None
    started_at: datetime = datetime.now(timezone.utc)
    signals_processed: int = 0
    last_signal_at: datetime | None = None
    scheduler: AsyncIOScheduler | None = None
    scanner_cycles: int = 0
    last_scan_at: datetime | None = None
    last_scan_top_symbol: str | None = None
    active_symbol: str = account_manager.strip_suffix(settings.mt5_fetcher.get("symbol", "XAUUSD"))


_state = _AppState()

# Tracks news events we've already warned about (avoids duplicate Telegram alerts)
_warned_events: set[str] = set()

# Cache tick info per symbol so we don't fetch on every bar
_tick_info_cache: dict[str, dict] = {}


async def _fetch_tick_info(symbol: str) -> dict | None:
    """Fetch symbol tick/lot info from MT5 (cached per symbol per session)."""
    if not settings.risk.get("use_fixed_sl_amount", False):
        return None
    if symbol in _tick_info_cache:
        return _tick_info_cache[symbol]
    info = await asyncio.to_thread(_mt5_symbol_info, symbol)
    if "error" in info:
        logger.warning(f"tick_info fetch failed for {symbol}: {info['error']}")
        return None
    _tick_info_cache[symbol] = info
    logger.debug(f"tick_info cached for {symbol}: tick_size={info['tick_size']} tick_value={info['tick_value']}")
    return info


# ---------------------------------------------------------------------------
# SRV-03/04/05  Core pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    ohlcv: OHLCVData,
    account: AccountState,
    level_hits=None,  # accepted but unused — level context is agent-only
) -> tuple[SignalResult, RiskDecision, int]:
    """
    Full analysis pipeline:
      news filter → indicators → Claude → risk manager → log

    Returns (signal, risk_decision, db_signal_id).
    Always returns — never raises (errors produce a NO_TRADE signal).
    """
    symbol = ohlcv.symbol

    # SRV-04  News filter
    blocked, blocking_events = await check_and_refresh(symbol, ohlcv.received_at)
    if blocked:
        titles = ", ".join(e.title for e in blocking_events)
        no_trade = SignalResult(
            symbol=symbol,
            direction="NO_TRADE",
            confidence=0,
            reasoning=f"News block active: {titles}",
            invalidation="",
            confluence_factors=[],
        )
        risk = RiskDecision(
            approved=False, lot_size=0.0,
            risk_amount=0.0, sl_distance=0.0,
            reasons=[f"News block: {titles}"],
        )
        signal_id = await log_signal(no_trade, risk)
        return no_trade, risk, signal_id

    # Indicator engine
    try:
        indicators = compute_indicators(ohlcv)
    except Exception as e:
        logger.error(f"Indicator error for {symbol}: {e}")
        no_trade = SignalResult(
            symbol=symbol, direction="NO_TRADE", confidence=0,
            reasoning=f"Indicator error: {e}", invalidation="",
            confluence_factors=[],
        )
        risk = RiskDecision(approved=False, lot_size=0.0, risk_amount=0.0, sl_distance=0.0)
        signal_id = await log_signal(no_trade, risk)
        return no_trade, risk, signal_id

    # Analysis — only call Claude when it is the active model.
    # When Ollama is selected the full agent loop (run_agent) handles analysis;
    # calling analyse() here would silently spend Claude credits.
    if model_manager.get_active_model() != "claude":
        logger.info(
            f"run_pipeline: skipping Claude analyse() — active model="
            f"{model_manager.get_active_model()} (use MT5 fetcher agent loop)"
        )
        no_trade = SignalResult(
            symbol=symbol, direction="NO_TRADE", confidence=0,
            reasoning=f"Pipeline analysis skipped — model={model_manager.get_active_model()} uses agent loop.",
            invalidation="", confluence_factors=[],
        )
        risk = RiskDecision(approved=False, lot_size=0.0, risk_amount=0.0, sl_distance=0.0)
        signal_id = await log_signal(no_trade, risk)
        return no_trade, risk, signal_id

    try:
        signal = analyse(ohlcv, indicators)
    except Exception as e:
        logger.error(f"Claude error for {symbol}: {e}")
        no_trade = SignalResult(
            symbol=symbol, direction="NO_TRADE", confidence=0,
            reasoning=f"Claude error: {e}", invalidation="",
            confluence_factors=[],
        )
        risk = RiskDecision(approved=False, lot_size=0.0, risk_amount=0.0, sl_distance=0.0)
        signal_id = await log_signal(no_trade, risk)
        return no_trade, risk, signal_id

    # SRV-05  Risk manager — use H4 ATR for SL validation
    atr_h4 = indicators.atr.get("H4", indicators.atr.get(list(indicators.atr.keys())[-1], 0.0))
    tick_info = await _fetch_tick_info(ohlcv.symbol)
    risk = evaluate(signal, account, atr_h4, tick_info=tick_info)

    # Log to DB
    signal_id = await log_signal(signal, risk)

    _state.signals_processed += 1
    _state.last_signal_at = datetime.now(timezone.utc)

    logger.info(
        f"Pipeline complete: {symbol} {signal.direction} "
        f"conf={signal.confidence}% approved={risk.approved} id={signal_id}"
    )

    # Telegram alert for actionable signals
    if signal.is_actionable:
        await telegram.send_signal_alert(signal, risk)

    return signal, risk, signal_id


# ---------------------------------------------------------------------------
# SRV-02  ZeroMQ listener loop
# ---------------------------------------------------------------------------

async def _zmq_listener() -> None:
    """Background task: receive MT5 data via ZeroMQ, run pipeline, send back."""
    try:
        import zmq
        import zmq.asyncio as azmq
    except ImportError:
        logger.warning("pyzmq not installed — ZeroMQ listener disabled")
        return

    ctx = azmq.Context()
    pull = ctx.socket(zmq.PULL)
    pub  = ctx.socket(zmq.PUB)

    pull_port: int = settings.server["zmq_pull_port"]
    pub_port:  int = settings.server["zmq_pub_port"]

    pull.bind(f"tcp://127.0.0.1:{pull_port}")
    pub.bind(f"tcp://127.0.0.1:{pub_port}")

    logger.info(f"ZeroMQ PULL listening on :{pull_port}, PUB broadcasting on :{pub_port}")

    try:
        while True:
            raw = await pull.recv_string()
            logger.debug(f"ZeroMQ received {len(raw)} bytes")

            try:
                ohlcv, account = parse_mt5_message(raw)
            except ValueError as e:
                logger.warning(f"Bad MT5 message: {e}")
                continue

            # Update scanner cache with fresh data for this symbol
            await update_symbol_data(ohlcv)

            # Respect the active symbol — skip pipeline for symbols that don't match
            active = account_manager.apply_suffix(_state.active_symbol)
            if ohlcv.symbol != active:
                logger.debug(
                    f"ZMQ: skipping pipeline for {ohlcv.symbol} "
                    f"(active symbol is {active})"
                )
                continue

            signal, risk, signal_id = await run_pipeline(ohlcv, account)
            response = serialize_signal(signal, risk, signal_id)
            await pub.send_string(response)
            logger.debug(f"ZeroMQ sent signal id={signal_id}")

    except asyncio.CancelledError:
        logger.info("ZeroMQ listener shutting down")
    finally:
        pull.close()
        pub.close()
        ctx.term()


# ---------------------------------------------------------------------------
# SRV-01  FastAPI app + lifespan
# ---------------------------------------------------------------------------

async def _send_daily_summary() -> None:
    """APScheduler job — fetch today's stats and push to Telegram."""
    today_start = datetime.combine(datetime.now(timezone.utc).date(), datetime.min.time()).replace(tzinfo=timezone.utc)
    stats = await get_stats(since=today_start)
    await telegram.send_daily_summary(stats)
    logger.info("Daily summary sent to Telegram")


async def _refresh_news_cache() -> None:
    """APScheduler job — proactively refresh the Forex Factory news cache."""
    from backend.news_filter import fetch_calendar, _cache as _news_cache
    try:
        events = await fetch_calendar()
        _news_cache.update(events)
    except Exception as e:
        logger.warning(f"Proactive news cache refresh failed: {e}")


async def _check_news_warning() -> None:
    """APScheduler job — send Telegram alert when high-impact news is approaching."""
    from backend.news_filter import get_all_upcoming_events
    now = datetime.now(timezone.utc)
    block_before: int = settings.news_filter.get("block_minutes_before", 30)
    upcoming = get_all_upcoming_events(now, lookahead_hours=block_before / 60)

    new_warnings = []
    for event in upcoming:
        key = f"{event.event_time.isoformat()}|{event.currency}|{event.title}"
        if key not in _warned_events:
            _warned_events.add(key)
            new_warnings.append(event)

    # Prune warned entries for events that passed more than 2 hours ago
    cutoff = now - timedelta(hours=2)
    _warned_events.difference_update(
        k for k in list(_warned_events)
        if datetime.fromisoformat(k.split("|")[0]) < cutoff
    )

    if new_warnings:
        logger.info(f"News warning: {len(new_warnings)} approaching event(s)")
        await telegram.send_news_warning(new_warnings)


async def _on_scanner_signal(result: ScanResult) -> None:
    """
    Callback invoked by the scanner for each top-ranked signal per cycle.
    Publishes via ZeroMQ (if the socket is available) and Telegram.
    """
    _state.scanner_cycles += 1
    _state.last_scan_at = datetime.now(timezone.utc)
    _state.last_scan_top_symbol = result.symbol

    logger.info(
        f"Scanner top signal: {result.symbol} {result.signal.direction} "
        f"conf={result.signal.confidence}% score={result.confluence_score}"
    )

    if result.signal.is_actionable:
        await telegram.send_signal_alert(result.signal, result.risk)


async def _run_scanner_cycle() -> None:
    """APScheduler job wrapper — calls scheduled_scan with the shared callback."""
    await scheduled_scan(on_signal_fn=_on_scanner_signal)


async def _run_strategy_cycle(ohlcv: OHLCVData, account: AccountState) -> None:
    """
    Execute one strategy bar cycle.
    - If scan_all_symbols=true: scans every watchlist symbol and picks the best signal.
    - If scan_all_symbols=false: runs only on the provided symbol.
    - If auto_trade=true and risk approved: places the order via MT5.
    """
    from backend.indicators import compute_indicators
    from backend.risk_manager import evaluate
    from backend.mt5_fetcher import fetch_ohlcv, fetch_account

    auto_trade: bool = model_manager.get_auto_trade()
    scan_all:   bool = model_manager.get_scan_all_symbols()
    bars: int        = int(settings.mt5_fetcher.get("bars_per_tf", 100))

    candidates: list[tuple] = []   # (signal, risk, indicators)

    async def _evaluate_symbol(sym: str) -> None:
        # Skip if a position is already open on this symbol
        existing = await asyncio.to_thread(_mt5_open_positions, sym)
        if existing.get("positions"):
            logger.debug(f"Strategy cycle: skipping {sym} — position already open")
            return
        ohlcv_s = await asyncio.to_thread(fetch_ohlcv, sym, bars)
        if ohlcv_s is None:
            return
        ind_s = compute_indicators(ohlcv_s)
        sig_s = run_strategy(ohlcv_s, ind_s)
        if not sig_s.is_actionable:
            return
        acc_s = await asyncio.to_thread(fetch_account)
        atr_h4 = ind_s.atr.get("H4", ind_s.atr.get(list(ind_s.atr.keys())[-1], 0.0))
        tick_s = await _fetch_tick_info(sym)
        risk_s = evaluate(sig_s, acc_s, atr_h4, tick_info=tick_s)
        candidates.append((sig_s, risk_s, ind_s))

    if scan_all:
        watchlist = account_manager.get_watchlist()
        logger.info(f"Strategy multi-symbol scan: {len(watchlist)} symbols")
        await asyncio.gather(*[_evaluate_symbol(sym) for sym in watchlist])
    else:
        indicators = compute_indicators(ohlcv)
        signal     = run_strategy(ohlcv, indicators)
        if signal.is_actionable:
            atr_h4 = indicators.atr.get("H4", indicators.atr.get(list(indicators.atr.keys())[-1], 0.0))
            tick_info = await _fetch_tick_info(ohlcv.symbol)
            risk   = evaluate(signal, account, atr_h4, tick_info=tick_info)
            candidates.append((signal, risk, indicators))
        else:
            signal_id = await log_signal(signal, evaluate(signal, account, 0.0))
            logger.info(
                f"Strategy cycle: {ohlcv.symbol} NO_TRADE conf=0% id={signal_id}"
            )
            return

    if not candidates:
        logger.info("Strategy cycle: no actionable signals this bar")
        return

    # Pick highest-confidence approved signal; fall back to highest-confidence overall
    approved  = [(s, r, i) for s, r, i in candidates if r.approved]
    best_signal, best_risk, _ = max(
        approved if approved else candidates,
        key=lambda x: x[0].confidence,
    )

    signal_id = await log_signal(best_signal, best_risk)
    logger.info(
        f"Strategy cycle: {best_signal.symbol} {best_signal.direction} "
        f"conf={best_signal.confidence}% approved={best_risk.approved} id={signal_id}"
    )

    await telegram.send_signal_alert(best_signal, best_risk)

    # Place order when auto_trade is on and risk approved
    if auto_trade and best_risk.approved:
        from backend.mt5_tools import place_order
        strategy_name = model_manager.get_active_strategy()
        result = await asyncio.to_thread(
            place_order,
            best_signal.symbol,
            best_signal.direction,
            best_risk.lot_size,
            best_signal.sl,
            best_signal.tp1,
            best_signal.tp2,
            best_signal.tp3,
            f"Strategy-{strategy_name}"[:31],
        )
        if "error" in result:
            logger.error(f"Strategy order failed: {result['error']}")
            await telegram.send_agent_update(
                f"⚠️ <b>Strategy order failed</b>\n"
                f"<code>{best_signal.symbol} {best_signal.direction}</code>\n"
                f"Reason: {result['error']}"
            )
        else:
            logger.info(
                f"Strategy order placed ✅ ticket={result.get('ticket')} "
                f"{best_signal.symbol} {best_signal.direction} lot={best_risk.lot_size}"
            )
            await telegram.send_agent_update(
                f"✅ <b>Strategy order placed</b>\n"
                f"<code>{best_signal.symbol} {best_signal.direction}</code>  "
                f"lot={best_risk.lot_size}  ticket={result.get('ticket')}\n"
                f"Entry={best_signal.entry:.5f}  SL={best_signal.sl:.5f}  "
                f"TP1={best_signal.tp1:.5f}"
            )


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _setup_logging()
    await init_db()
    _state.zmq_task = asyncio.create_task(_zmq_listener())

    def _get_active_symbol() -> str:
        return account_manager.apply_suffix(_state.active_symbol)

    def _set_active_symbol(symbol: str) -> None:
        _state.active_symbol = account_manager.strip_suffix(symbol)
        logger.info(f"Active trading symbol changed to {symbol}")

    async def _tg_run_backtest(symbol: str, strategy: str, days: int):
        cfg = BacktestConfig(symbol=symbol, strategy=strategy, days=days)
        return await asyncio.to_thread(run_backtest, cfg)

    # Telegram bot
    await telegram.start(
        get_status_fn=health,
        get_symbol_fn=_get_active_symbol,
        set_symbol_fn=_set_active_symbol,
        get_model_fn=model_manager.get_active_model,
        set_model_fn=model_manager.set_active_model,
        get_watchlist_fn=account_manager.get_watchlist,
        get_strategy_fn=model_manager.get_active_strategy,
        set_strategy_fn=model_manager.set_active_strategy,
        get_auto_trade_fn=model_manager.get_auto_trade,
        set_auto_trade_fn=model_manager.set_auto_trade,
        get_scan_all_fn=model_manager.get_scan_all_symbols,
        set_scan_all_fn=model_manager.set_scan_all_symbols,
        run_backtest_fn=_tg_run_backtest,
    )

    # APScheduler — daily summary + multi-symbol scanner
    _state.scheduler = AsyncIOScheduler(timezone="UTC")

    summary_time: str = settings.telegram.get("daily_summary_time", "00:00")
    hour, minute = (int(x) for x in summary_time.split(":"))
    _state.scheduler.add_job(
        _send_daily_summary,
        trigger="cron",
        hour=hour,
        minute=minute,
        id="daily_summary",
    )

    # SCAN-06  Scanner job — runs every scanner.interval_minutes
    scanner_cfg = settings.scanner
    if scanner_cfg.get("enabled", True):
        interval_min: int = int(scanner_cfg.get("interval_minutes", 15))
        _state.scheduler.add_job(
            _run_scanner_cycle,
            trigger="interval",
            minutes=interval_min,
            id="multi_symbol_scanner",
            next_run_time=None,  # don't run immediately on startup
        )
        logger.info(f"Scanner scheduled — every {interval_min} min")
    else:
        logger.info("Scanner disabled in settings.yaml (scanner.enabled=false)")

    # News cache proactive refresh job
    news_refresh_hours: int = settings.news_filter.get("calendar_refresh_hours", 4)
    _state.scheduler.add_job(
        _refresh_news_cache,
        trigger="interval",
        hours=news_refresh_hours,
        id="news_cache_refresh",
        next_run_time=None,
    )
    logger.info(f"News cache refresh scheduled — every {news_refresh_hours}h")

    # Pre-news warning job — runs every warning_check_interval_min minutes
    warn_interval: int = settings.news_filter.get("warning_check_interval_min", 15)
    _state.scheduler.add_job(
        _check_news_warning,
        trigger="interval",
        minutes=warn_interval,
        id="news_warning_check",
    )
    logger.info(f"News warning check scheduled — every {warn_interval} min")

    _state.scheduler.start()
    logger.info(f"APScheduler started — daily summary at {summary_time} UTC")

    # MT5 Direct Fetcher — drives the agent / strategy loop
    if settings.mt5_fetcher.get("enabled", False):
        use_agent = settings.mt5_fetcher.get("use_agent", True)
        if use_agent:
            async def _agent_cycle(
                ohlcv: OHLCVData,
                account: AccountState,
                level_hits=None,
                alert_hits=None,
                skip_analysis: bool = False,
            ) -> None:
                active = model_manager.get_active_model()
                if active == "strategy":
                    await _run_strategy_cycle(ohlcv, account)
                else:
                    from backend.news_filter import refresh_cache_if_stale, get_news_context_for_agent
                    await refresh_cache_if_stale()
                    news_events = get_news_context_for_agent(ohlcv.symbol, ohlcv.received_at)
                    if news_events:
                        names = ", ".join(f"{e.currency} '{e.title}'" for e in news_events)
                        logger.info(f"Agent cycle — news context injected for {ohlcv.symbol}: {names}")
                    # News overrides skip_analysis — always run full cycle during news
                    effective_skip = skip_analysis and not news_events
                    await run_agent(
                        ohlcv.symbol,
                        account,
                        telegram_notifier=telegram,
                        level_hits=level_hits,
                        news_events=news_events or None,
                        alert_hits=alert_hits or None,
                        skip_analysis=effective_skip,
                    )
                _state.signals_processed += 1
                _state.last_signal_at = datetime.now(timezone.utc)

            _state.mt5_fetcher_task = asyncio.create_task(
                fetcher_loop(_agent_cycle, get_symbol_fn=_get_active_symbol)
            )
            logger.info("MT5 fetcher started — agent/strategy mode")
        else:
            _state.mt5_fetcher_task = asyncio.create_task(
                fetcher_loop(run_pipeline, get_symbol_fn=_get_active_symbol)
            )
            logger.info("MT5 fetcher started — analyst mode (signal-only)")
    else:
        logger.info("MT5 fetcher disabled in settings.yaml (mt5_fetcher.enabled=false)")

    # Trade manager — manages SL for all open positions
    _state.trade_manager_task = asyncio.create_task(trade_manager_loop())

    logger.info("MT5 Analyst Bot server started")

    yield

    # Shutdown
    if _state.scheduler:
        _state.scheduler.shutdown(wait=False)
    await telegram.stop()
    for task in (
        _state.trade_manager_task,
        _state.mt5_fetcher_task,
        _state.zmq_task,
    ):
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    logger.info("Server shut down cleanly")


app = FastAPI(
    title="MT5 AI Analyst Bot",
    description="Real-time trading signal generator powered by Claude AI",
    version="1.0.0",
    lifespan=_lifespan,
)


# ---------------------------------------------------------------------------
# SRV-08  GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["monitoring"])
async def health() -> dict:
    """Server liveness check — returns uptime and processing stats."""
    uptime = (datetime.now(timezone.utc) - _state.started_at).total_seconds()
    return {
        "status":               "ok",
        "uptime_seconds":       int(uptime),
        "signals_processed":    _state.signals_processed,
        "last_signal_at":       _state.last_signal_at.isoformat() if _state.last_signal_at else None,
        "mt5_fetcher_alive":    _state.mt5_fetcher_task is not None and not _state.mt5_fetcher_task.done(),
        "zmq_listener_alive":   _state.zmq_task is not None and not _state.zmq_task.done(),
        "scanner_cycles":       _state.scanner_cycles,
        "last_scan_at":         _state.last_scan_at.isoformat() if _state.last_scan_at else None,
        "last_scan_top":        _state.last_scan_top_symbol,
    }


# ---------------------------------------------------------------------------
# GET /scanner/status — scanner health + cycle info
# ---------------------------------------------------------------------------

@app.get("/scanner/status", tags=["monitoring"])
async def scanner_status() -> dict:
    """Return scanner configuration and runtime statistics."""
    from backend.scanner import _cache
    snapshot = await _cache.snapshot()
    cfg = settings.scanner
    watchlist: list[str] = account_manager.get_watchlist()

    symbol_data_ages = {}
    now = datetime.now(timezone.utc)
    for sym in watchlist:
        ohlcv = snapshot.get(sym)
        if ohlcv:
            age_sec = int((now - ohlcv.received_at).total_seconds())
            symbol_data_ages[sym] = age_sec
        else:
            symbol_data_ages[sym] = None

    return {
        "enabled":             cfg.get("enabled", True),
        "interval_minutes":    cfg.get("interval_minutes", 15),
        "top_signals_per_cycle": cfg.get("top_signals_per_cycle", 1),
        "watchlist":           watchlist,
        "cached_symbols":      list(snapshot.keys()),
        "symbol_data_age_sec": symbol_data_ages,
        "scanner_cycles":      _state.scanner_cycles,
        "last_scan_at":        _state.last_scan_at.isoformat() if _state.last_scan_at else None,
        "last_scan_top_symbol": _state.last_scan_top_symbol,
    }


# ---------------------------------------------------------------------------
# GET /news — upcoming high-impact news from Forex Factory calendar
# ---------------------------------------------------------------------------

@app.get("/news", tags=["monitoring"])
async def upcoming_news(hours: int = 4) -> dict:
    """Return upcoming high/medium-impact news events from the Forex Factory calendar."""
    from backend.news_filter import get_all_upcoming_events, refresh_cache_if_stale
    if hours < 1 or hours > 48:
        raise HTTPException(status_code=422, detail="hours must be between 1 and 48")
    await refresh_cache_if_stale()
    now = datetime.now(timezone.utc)
    events = get_all_upcoming_events(now, lookahead_hours=float(hours))
    return {
        "hours_ahead": hours,
        "count": len(events),
        "checked_at": now.isoformat(),
        "events": [
            {
                "title":          e.title,
                "currency":       e.currency,
                "impact":         e.impact,
                "event_time":     e.event_time.isoformat(),
                "block_start":    e.block_start.isoformat(),
                "block_end":      e.block_end.isoformat(),
                "is_blocking_now": e.is_blocking(now),
            }
            for e in events
        ],
    }


# ---------------------------------------------------------------------------
# POST /scanner/trigger — manually trigger one scan cycle (for testing)
# ---------------------------------------------------------------------------

@app.post("/scanner/trigger", tags=["signals"])
async def trigger_scanner() -> dict:
    """
    Manually fire one scanner cycle immediately.
    Useful for testing without waiting for the scheduled interval.
    Returns the top-ranked signals found in the cycle.
    """
    from backend.scanner import scan_cycle
    cfg = settings.scanner
    top_n: int = cfg.get("top_signals_per_cycle", 1)

    all_results = await scan_cycle()
    top = all_results[:top_n]

    _state.scanner_cycles += 1
    _state.last_scan_at = datetime.now(timezone.utc)
    if top:
        _state.last_scan_top_symbol = top[0].symbol

    return {
        "cycle_symbols_analysed": len(all_results),
        "top_n": top_n,
        "results": [
            {
                "symbol":           r.symbol,
                "direction":        r.signal.direction,
                "confidence":       r.signal.confidence,
                "confluence_score": r.confluence_score,
                "entry":            r.signal.entry,
                "sl":               r.signal.sl,
                "tp1":              r.signal.tp1,
                "lot_size":         r.risk.lot_size,
                "approved":         r.risk.approved,
                "risk_reward":      r.signal.risk_reward,
                "reasoning":        r.signal.reasoning,
                "confluence_factors": r.signal.confluence_factors,
                "signal_id":        r.signal_id,
                "scanned_at":       r.scanned_at.isoformat(),
            }
            for r in top
        ],
    }


# ---------------------------------------------------------------------------
# POST /backtest — walk-forward strategy backtest on MT5 historical data
# ---------------------------------------------------------------------------

class BacktestRequest(BaseModel):
    symbol: str = ""                    # defaults to active symbol
    strategy: str = "ema_pullback"      # ema_pullback | asian_breakout | sr_bounce | amd_fvg
    days: int = 90                      # how many days of history to test
    initial_balance: float = 10_000.0
    risk_pct: float = 1.0
    max_trades: int = 0                 # 0 = unlimited; >0 stops after N closed trades


@app.post("/backtest", tags=["signals"])
async def backtest(req: BacktestRequest) -> dict:
    """
    Run a walk-forward backtest of a rule-based strategy on MT5 historical data.
    MT5 terminal must be open and logged in.
    """
    symbol = req.symbol or _state.active_symbol
    if not symbol:
        raise HTTPException(status_code=422, detail="symbol is required")

    if req.days < 7 or req.days > 365:
        raise HTTPException(status_code=422, detail="days must be between 7 and 365")

    cfg = BacktestConfig(
        symbol=symbol,
        strategy=req.strategy,
        days=req.days,
        initial_balance=req.initial_balance,
        risk_pct=req.risk_pct,
        max_trades=req.max_trades,
    )

    try:
        result = await asyncio.to_thread(run_backtest, cfg)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        raise HTTPException(status_code=500, detail=f"Backtest failed: {e}")

    return {
        "symbol":            result.symbol,
        "strategy":          result.strategy,
        "period_start":      result.period_start.strftime("%Y-%m-%d"),
        "period_end":        result.period_end.strftime("%Y-%m-%d"),
        "total_trades":      result.total_trades,
        "wins":              result.wins,
        "losses":            result.losses,
        "open_trades":       result.open_trades,
        "win_rate_pct":      result.win_rate_pct,
        "profit_factor":     result.profit_factor,
        "max_drawdown_pct":  result.max_drawdown_pct,
        "net_pnl_r":         result.net_pnl_r,
        "net_pnl_usd":       result.net_pnl_usd,
        "initial_balance":   result.initial_balance,
        "final_balance":     result.final_balance,
        "avg_bars_held":     result.avg_bars_held,
        "avg_confidence":    result.avg_confidence,
        "bars_tested":       result.bars_tested,
        "no_trade_count":    result.no_trade_count,
        "sample_no_trade_reasons": result.sample_no_trade_reasons,
        "trades": [
            {
                "entry_time":  t.entry_time.strftime("%Y-%m-%d %H:%M"),
                "close_time":  t.close_time.strftime("%Y-%m-%d %H:%M") if t.close_time else None,
                "direction":   t.direction,
                "entry":       t.entry,
                "sl":          t.sl,
                "tp1":         t.tp1,
                "confidence":  t.confidence,
                "outcome":     t.outcome,
                "bars_held":   t.bars_held,
                "pnl_r":       t.pnl_r,
                "pnl_usd":     round(t.pnl_usd, 2),
                "reasoning":   t.reasoning,
            }
            for t in result.trades
        ],
    }


# ---------------------------------------------------------------------------
# SRV-06  POST /signal — manual trigger for testing
# ---------------------------------------------------------------------------

class SignalRequest(BaseModel):
    symbol: str
    equity: float = 10_000.0
    balance: float = 10_000.0
    open_trades: int = 0
    daily_pnl: float = 0.0
    bars_m5: list[dict] = []
    bars_h1:  list[dict] = []
    bars_h4:  list[dict] = []
    bars_d1:  list[dict] = []


def _bars_to_ohlcv(symbol: str, req: SignalRequest) -> OHLCVData:
    """Convert REST request payload into OHLCVData."""
    from backend.indicators import Bar

    def _parse(bars: list[dict]) -> list[Bar]:
        result = []
        for b in bars:
            dt = datetime.fromisoformat(
                str(b.get("time", "")).replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            result.append(Bar(
                time=dt,
                open=float(b["open"]), high=float(b["high"]),
                low=float(b["low"]),   close=float(b["close"]),
                volume=float(b.get("volume", 0)),
            ))
        return result

    timeframes = {}
    for tf, bars in [("M5", req.bars_m5), ("H1", req.bars_h1),
                     ("H4", req.bars_h4), ("D1", req.bars_d1)]:
        parsed = _parse(bars)
        if parsed:
            timeframes[tf] = parsed

    if not timeframes:
        raise ValueError("No bar data provided — include at least one timeframe")

    return OHLCVData(
        symbol=symbol,
        timeframes=timeframes,
        received_at=datetime.now(timezone.utc),
    )


@app.post("/signal", tags=["signals"])
async def request_signal(req: SignalRequest) -> dict:
    """
    Manual signal request — for testing without MT5.
    Send OHLCV bars and account state; receive full signal + risk output.
    """
    try:
        ohlcv = _bars_to_ohlcv(req.symbol, req)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    account = AccountState(
        equity=req.equity,
        balance=req.balance,
        open_trades=req.open_trades,
        daily_pnl=req.daily_pnl,
    )

    signal, risk, signal_id = await run_pipeline(ohlcv, account)

    return {
        "signal_id":  signal_id,
        "symbol":     signal.symbol,
        "direction":  signal.direction,
        "confidence": signal.confidence,
        "entry":      signal.entry,
        "sl":         signal.sl,
        "tp1":        signal.tp1,
        "tp2":        signal.tp2,
        "tp3":        signal.tp3,
        "lot_size":   risk.lot_size,
        "approved":   risk.approved,
        "risk_reward": signal.risk_reward,
        "reasoning":  signal.reasoning,
        "invalidation": signal.invalidation,
        "confluence_factors": signal.confluence_factors,
        "rejection_reasons":  risk.reasons,
        "warnings":           risk.warnings,
        "estimated_cost_usd": signal.estimated_cost_usd,
    }


# ---------------------------------------------------------------------------
# SRV-07  GET /stats
# ---------------------------------------------------------------------------

@app.get("/stats", tags=["monitoring"])
async def stats(symbol: str | None = None) -> dict:
    """Return aggregated signal performance statistics."""
    s: SignalStats = await get_stats(symbol)
    return {
        "total_signals":      s.total_signals,
        "actionable_signals": s.actionable_signals,
        "no_trade_signals":   s.no_trade_signals,
        "pending_signals":    s.pending_signals,
        "wins":               s.wins,
        "losses":             s.losses,
        "breakevens":         s.breakevens,
        "win_rate_pct":       s.win_rate_pct,
        "avg_rr":             s.avg_rr,
        "total_pnl":          s.total_pnl,
        "total_api_cost_usd": s.total_api_cost_usd,
        "by_symbol":          s.by_symbol,
        "by_direction":       s.by_direction,
        "period_start":       s.period_start.isoformat() if s.period_start else None,
        "period_end":         s.period_end.isoformat()   if s.period_end   else None,
        "filter_symbol":      symbol,
    }


# ---------------------------------------------------------------------------
# POST /outcome/{signal_id} — update trade result from MT5
# ---------------------------------------------------------------------------

class OutcomeRequest(BaseModel):
    outcome: str          # WIN | LOSS | BREAKEVEN
    close_price: float
    actual_pnl: float | None = None


@app.post("/outcome/{signal_id}", tags=["signals"])
async def record_outcome(signal_id: int, req: OutcomeRequest) -> dict:
    """Called by MT5 EA when a trade closes — updates the signal log."""
    valid_outcomes = {"WIN", "LOSS", "BREAKEVEN"}
    if req.outcome.upper() not in valid_outcomes:
        raise HTTPException(
            status_code=422,
            detail=f"outcome must be one of {valid_outcomes}",
        )
    updated = await update_outcome(
        signal_id,
        outcome=req.outcome.upper(),  # type: ignore[arg-type]
        close_price=req.close_price,
        actual_pnl=req.actual_pnl,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
    record = await get_signal(signal_id)
    return {
        "signal_id":  signal_id,
        "outcome":    record.outcome,
        "close_price": record.close_price,
        "actual_rr":  record.actual_rr,
        "actual_pnl": record.actual_pnl,
    }


# ---------------------------------------------------------------------------
# GET /signal/{signal_id} — fetch a single signal record
# ---------------------------------------------------------------------------

@app.get("/signal/{signal_id}", tags=["signals"])
async def get_signal_record(signal_id: int) -> dict:
    """Fetch a single signal record by id."""
    record = await get_signal(signal_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Signal {signal_id} not found")
    return {
        "id":           record.id,
        "symbol":       record.symbol,
        "direction":    record.direction,
        "confidence":   record.confidence,
        "entry":        record.entry,
        "sl":           record.sl,
        "tp1":          record.tp1,
        "lot_size":     record.lot_size,
        "outcome":      record.outcome,
        "actual_rr":    record.actual_rr,
        "actual_pnl":   record.actual_pnl,
        "created_at":   record.created_at.isoformat(),
        "closed_at":    record.closed_at.isoformat() if record.closed_at else None,
        "estimated_cost_usd": record.estimated_cost_usd,
    }
