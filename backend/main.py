"""
Phase 7 — FastAPI Backend Server   (SRV-01, SRV-03–SRV-09)
Phase 10 — Multi-Symbol Scanner integration (SCAN-06)
"""
from __future__ import annotations

import asyncio
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
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
from backend.scanner import ScanResult, scheduled_scan, update_symbol_data
from backend.mt5_fetcher import fetcher_loop
from backend.claude_agent import run_agent
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
    started_at: datetime = datetime.now(timezone.utc)
    signals_processed: int = 0
    last_signal_at: datetime | None = None
    scheduler: AsyncIOScheduler | None = None
    scanner_cycles: int = 0
    last_scan_at: datetime | None = None
    last_scan_top_symbol: str | None = None
    active_symbol: str = settings.mt5_fetcher.get("symbol", "XAUUSD")


_state = _AppState()


# ---------------------------------------------------------------------------
# SRV-03/04/05  Core pipeline
# ---------------------------------------------------------------------------

async def run_pipeline(
    ohlcv: OHLCVData,
    account: AccountState,
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

    # Claude analysis
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
    risk = evaluate(signal, account, atr_h4)

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
    from datetime import date
    today_start = datetime.combine(date.today(), datetime.min.time()).replace(tzinfo=timezone.utc)
    stats = await get_stats(since=today_start)
    await telegram.send_daily_summary(stats)
    logger.info("Daily summary sent to Telegram")


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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    _setup_logging()
    await init_db()
    _state.zmq_task = asyncio.create_task(_zmq_listener())

    def _get_active_symbol() -> str:
        return _state.active_symbol

    def _set_active_symbol(symbol: str) -> None:
        _state.active_symbol = symbol
        logger.info(f"Active trading symbol changed to {symbol}")

    # Telegram bot
    await telegram.start(
        get_status_fn=health,
        get_symbol_fn=_get_active_symbol,
        set_symbol_fn=_set_active_symbol,
        watchlist=settings.symbols.get("watchlist", []),
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

    _state.scheduler.start()
    logger.info(f"APScheduler started — daily summary at {summary_time} UTC")

    # MT5 Direct Fetcher — drives the Claude agent loop
    if settings.mt5_fetcher.get("enabled", False):
        use_agent = settings.mt5_fetcher.get("use_agent", True)
        if use_agent:
            async def _agent_cycle(ohlcv: OHLCVData, account: AccountState) -> None:
                await run_agent(ohlcv.symbol, account, telegram_notifier=telegram)
                _state.signals_processed += 1
                _state.last_signal_at = datetime.now(timezone.utc)

            _state.mt5_fetcher_task = asyncio.create_task(
                fetcher_loop(_agent_cycle, get_symbol_fn=_get_active_symbol)
            )
            logger.info("MT5 fetcher started — Claude agent mode (full tool use)")
        else:
            _state.mt5_fetcher_task = asyncio.create_task(
                fetcher_loop(run_pipeline, get_symbol_fn=_get_active_symbol)
            )
            logger.info("MT5 fetcher started — analyst mode (signal-only)")
    else:
        logger.info("MT5 fetcher disabled in settings.yaml (mt5_fetcher.enabled=false)")

    logger.info("MT5 Analyst Bot server started")

    yield

    # Shutdown
    if _state.scheduler:
        _state.scheduler.shutdown(wait=False)
    await telegram.stop()
    if _state.mt5_fetcher_task:
        _state.mt5_fetcher_task.cancel()
        try:
            await _state.mt5_fetcher_task
        except asyncio.CancelledError:
            pass
    if _state.zmq_task:
        _state.zmq_task.cancel()
        try:
            await _state.zmq_task
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
    watchlist: list[str] = settings.symbols.get("watchlist", [])

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
# SRV-06  POST /signal — manual trigger for testing
# ---------------------------------------------------------------------------

class SignalRequest(BaseModel):
    symbol: str
    equity: float = 10_000.0
    balance: float = 10_000.0
    open_trades: int = 0
    daily_pnl: float = 0.0
    bars_m15: list[dict] = []
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
    for tf, bars in [("M15", req.bars_m15), ("H1", req.bars_h1),
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
