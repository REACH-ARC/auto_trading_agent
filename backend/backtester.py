"""
Walk-Forward Backtester — simulates any rule-based strategy on MT5 historical data.

Design principles:
  - Zero lookahead: at each M5 bar the strategy only sees bars up to that point.
  - Multi-TP trail: SL moves to entry after TP1, to TP1 after TP2, closes at TP3.
    Outcomes: LOSS (-1R) | BREAKEVEN (0R) | WIN_TP1 (+1.5R) | WIN_FULL (+2.5R)
  - One trade at a time (max_open_trades=1 for clean P&L tracking).
  - All data fetched directly from the MT5 terminal (must be connected + logged in).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from loguru import logger

from backend.indicators import Bar, OHLCVData, compute_indicators
from backend.strategy_engine import run_strategy
from backend import model_manager
from config import settings

_TF_NAMES = ["M5", "H1", "H4", "D1"]

# Minimum M5 bars needed before indicators are reliable (EMA200 warmup)
_WARMUP_BARS = 220


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BacktestConfig:
    symbol: str
    strategy: str = "ema_pullback"
    days: int = 90
    initial_balance: float = 10_000.0
    risk_pct: float = 1.0
    bars_per_tf: int = 100          # lookback window fed to strategy per bar
    max_trades: int = 0             # 0 = unlimited; >0 stops after N closed trades


@dataclass
class BacktestTrade:
    entry_time: datetime
    close_time: datetime | None
    direction: str
    entry: float
    sl: float
    tp1: float
    confidence: int
    outcome: Literal["WIN_FULL", "WIN_TP1", "BREAKEVEN", "LOSS", "OPEN"]
    bars_held: int
    pnl_r: float                    # e.g. +2.5, +1.5, 0.0, or -1.0
    pnl_usd: float
    tp2: float | None = None
    tp3: float | None = None
    reasoning: str = ""


@dataclass
class BacktestResult:
    symbol: str
    strategy: str
    period_start: datetime
    period_end: datetime
    total_trades: int
    wins: int           # WIN_FULL + WIN_TP1
    losses: int
    breakevens: int
    open_trades: int
    win_rate_pct: float
    profit_factor: float
    max_drawdown_pct: float
    net_pnl_r: float
    net_pnl_usd: float
    initial_balance: float
    final_balance: float
    avg_bars_held: float
    avg_confidence: float
    bars_tested: int = 0
    no_trade_count: int = 0
    sample_no_trade_reasons: list[str] = field(default_factory=list)
    trades: list[BacktestTrade] = field(default_factory=list)


# ---------------------------------------------------------------------------
# MT5 data fetching
# ---------------------------------------------------------------------------

def _fetch_rates(symbol: str, tf_name: str, date_from: datetime, date_to: datetime) -> list[Bar]:
    import MetaTrader5 as mt5
    tf_map = {
        "M5": mt5.TIMEFRAME_M5,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }
    rates = mt5.copy_rates_range(symbol, tf_map[tf_name], date_from, date_to)
    if rates is None or len(rates) == 0:
        logger.warning(f"Backtest: no {tf_name} data for {symbol} ({mt5.last_error()})")
        return []
    return [
        Bar(
            time=datetime.fromtimestamp(r["time"], tz=timezone.utc),
            open=float(r["open"]),
            high=float(r["high"]),
            low=float(r["low"]),
            close=float(r["close"]),
            volume=float(r["tick_volume"]),
        )
        for r in rates
    ]


def _bars_up_to(bars: list[Bar], cutoff: datetime, n: int) -> list[Bar]:
    """Return last N bars with bar.time <= cutoff (no lookahead)."""
    eligible = [b for b in bars if b.time <= cutoff]
    return eligible[-n:] if len(eligible) >= n else eligible


# ---------------------------------------------------------------------------
# Outcome simulation — multi-TP trail
# ---------------------------------------------------------------------------

def _bar_hits(bar: Bar, direction: str, level: float, side: str) -> bool:
    """Check if a bar touches a level. side='sl' or 'tp'."""
    if side == "sl":
        return bar.low <= level if direction == "BUY" else bar.high >= level
    return bar.high >= level if direction == "BUY" else bar.low <= level


def _simulate_bar(
    bar: Bar,
    direction: str,
    current_sl: float,
    next_tp: float,
) -> Literal["HIT_SL", "HIT_TP", "OPEN"]:
    """Conservative: SL takes priority if both are within the bar range."""
    sl_hit = _bar_hits(bar, direction, current_sl, "sl")
    tp_hit = _bar_hits(bar, direction, next_tp,    "tp")
    if sl_hit:
        return "HIT_SL"
    if tp_hit:
        return "HIT_TP"
    return "OPEN"


# ---------------------------------------------------------------------------
# Core walk-forward engine
# ---------------------------------------------------------------------------

def run_backtest(config: BacktestConfig) -> BacktestResult:
    """
    Run a full walk-forward backtest. Returns BacktestResult with all trades and metrics.
    Raises RuntimeError if MT5 is not connected or data is unavailable.
    """
    import MetaTrader5 as mt5

    if not mt5.initialize():
        raise RuntimeError(f"MT5 not connected: {mt5.last_error()}")

    now      = datetime.now(timezone.utc)
    date_end = now
    # Extra warmup time so indicators are stable from the very first test bar
    date_from_fetch = now - timedelta(days=config.days + 30)
    date_start_test = now - timedelta(days=config.days)

    logger.info(
        f"Backtest: {config.symbol} | strategy={config.strategy} | "
        f"period={date_start_test.date()} → {date_end.date()} ({config.days} days)"
    )

    # Temporarily switch strategy
    original_strategy = model_manager.get_active_strategy()
    model_manager.set_active_strategy(config.strategy)

    try:
        # Fetch full history for all timeframes
        all_bars: dict[str, list[Bar]] = {}
        for tf in _TF_NAMES:
            all_bars[tf] = _fetch_rates(config.symbol, tf, date_from_fetch, date_end)
            logger.info(f"Backtest: fetched {len(all_bars[tf])} {tf} bars")

        if not all_bars.get("M5"):
            raise RuntimeError(f"No M5 data returned for {config.symbol}")

        m5_bars = all_bars["M5"]

        # Risk per trade in USD
        risk_usd_per_trade = config.initial_balance * config.risk_pct / 100.0
        tp1_rr = float(settings.risk.get("tp1_rr_ratio", 1.5))
        tp2_rr = float(settings.risk.get("tp2_rr_ratio", 2.0))
        tp3_rr = float(settings.risk.get("tp3_rr_ratio", 2.5))

        trades: list[BacktestTrade] = []
        balance   = config.initial_balance
        peak_bal  = balance
        max_dd    = 0.0

        open_trade: BacktestTrade | None = None
        open_trade_start_idx: int = 0
        # Multi-TP trail state for the current open trade
        trail_sl:    float = 0.0   # current (possibly moved) stop loss
        trail_phase: int   = 0     # 0=to TP1, 1=to TP2 (SL at entry), 2=to TP3 (SL at TP1)

        bars_tested      = 0
        no_trade_count   = 0
        no_trade_reasons: list[str] = []   # sample first 5

        for i, m5_bar in enumerate(m5_bars):
            bar_time = m5_bar.time

            # Skip warmup period
            if i < _WARMUP_BARS:
                continue
            if bar_time < date_start_test:
                continue

            # ----------------------------------------------------------------
            # 1. Check if open trade closed on this bar (multi-TP trail)
            # ----------------------------------------------------------------
            if open_trade is not None:
                closed = False
                pnl_r  = 0.0

                if trail_phase == 0:
                    # Waiting for TP1 (full risk)
                    result = _simulate_bar(m5_bar, open_trade.direction, trail_sl, open_trade.tp1)
                    if result == "HIT_SL":
                        pnl_r  = -1.0
                        open_trade.outcome = "LOSS"
                        closed = True
                    elif result == "HIT_TP":
                        # TP1 hit — move SL to entry, chase TP2
                        trail_sl    = open_trade.entry
                        trail_phase = 1
                        if open_trade.tp2 is None:
                            pnl_r = tp1_rr
                            open_trade.outcome = "WIN_FULL"
                            closed = True

                elif trail_phase == 1:
                    # Waiting for TP2 (SL at entry — breakeven guaranteed)
                    tp2 = open_trade.tp2 or open_trade.tp1
                    result = _simulate_bar(m5_bar, open_trade.direction, trail_sl, tp2)
                    if result == "HIT_SL":
                        pnl_r  = 0.0
                        open_trade.outcome = "BREAKEVEN"
                        closed = True
                    elif result == "HIT_TP":
                        # TP2 hit — move SL to TP1, chase TP3
                        trail_sl    = open_trade.tp1
                        trail_phase = 2
                        if open_trade.tp3 is None:
                            pnl_r = tp2_rr
                            open_trade.outcome = "WIN_FULL"
                            closed = True

                elif trail_phase == 2:
                    # Waiting for TP3 (SL at TP1 — +1.5R locked)
                    tp3 = open_trade.tp3 or open_trade.tp2 or open_trade.tp1
                    result = _simulate_bar(m5_bar, open_trade.direction, trail_sl, tp3)
                    if result == "HIT_SL":
                        pnl_r  = tp1_rr
                        open_trade.outcome = "WIN_TP1"
                        closed = True
                    elif result == "HIT_TP":
                        pnl_r  = tp3_rr
                        open_trade.outcome = "WIN_FULL"
                        closed = True

                if closed:
                    pnl_usd = risk_usd_per_trade * pnl_r
                    bars_held = i - open_trade_start_idx

                    open_trade.close_time = bar_time
                    open_trade.bars_held  = bars_held
                    open_trade.pnl_r      = round(pnl_r, 2)
                    open_trade.pnl_usd    = round(pnl_usd, 2)

                    balance += pnl_usd
                    peak_bal = max(peak_bal, balance)
                    dd = (peak_bal - balance) / peak_bal * 100 if peak_bal > 0 else 0.0
                    max_dd = max(max_dd, dd)

                    logger.debug(
                        f"Backtest trade closed: {open_trade.direction} {open_trade.outcome} "
                        f"R={pnl_r:+.2f} USD={pnl_usd:+.2f} bal={balance:.2f}"
                    )
                    trades.append(open_trade)
                    open_trade   = None
                    trail_sl     = 0.0
                    trail_phase  = 0
                    if config.max_trades > 0 and len(trades) >= config.max_trades:
                        break
                else:
                    continue  # trade still open — don't look for new signal

            # ----------------------------------------------------------------
            # 2. Build no-lookahead OHLCVData and run strategy
            # ----------------------------------------------------------------
            n = config.bars_per_tf
            timeframes: dict = {}
            for tf in _TF_NAMES:
                sliced = _bars_up_to(all_bars[tf], bar_time, n)
                if sliced:
                    timeframes[tf] = sliced

            if "M5" not in timeframes or len(timeframes["M5"]) < 20:
                continue

            ohlcv = OHLCVData(
                symbol=config.symbol,
                timeframes=timeframes,
                received_at=bar_time,
            )

            bars_tested += 1
            try:
                indicators = compute_indicators(ohlcv)
                signal     = run_strategy(ohlcv, indicators)
            except Exception as e:
                logger.debug(f"Backtest bar {bar_time}: strategy error — {e}")
                no_trade_count += 1
                if len(no_trade_reasons) < 5:
                    no_trade_reasons.append(f"[{bar_time:%Y-%m-%d %H:%M}] ERROR: {e}")
                continue

            if not signal.is_actionable or signal.entry is None or signal.sl is None or signal.tp1 is None:
                no_trade_count += 1
                if len(no_trade_reasons) < 5:
                    no_trade_reasons.append(
                        f"[{bar_time:%Y-%m-%d %H:%M}] {signal.reasoning[:100]}"
                    )
                continue

            # ----------------------------------------------------------------
            # 3. Open virtual trade
            # ----------------------------------------------------------------
            open_trade = BacktestTrade(
                entry_time=bar_time,
                close_time=None,
                direction=signal.direction,
                entry=signal.entry,
                sl=signal.sl,
                tp1=signal.tp1,
                tp2=signal.tp2,
                tp3=signal.tp3,
                confidence=signal.confidence,
                outcome="OPEN",
                bars_held=0,
                pnl_r=0.0,
                pnl_usd=0.0,
                reasoning=signal.reasoning[:120] if signal.reasoning else "",
            )
            open_trade_start_idx = i
            trail_sl    = signal.sl
            trail_phase = 0
            if config.max_trades > 0 and len(trades) >= config.max_trades:
                break
            logger.debug(
                f"Backtest trade opened: {bar_time.strftime('%Y-%m-%d %H:%M')} "
                f"{signal.direction} conf={signal.confidence}% "
                f"entry={signal.entry:.5f} sl={signal.sl:.5f} tp1={signal.tp1:.5f}"
            )

        # Mark any still-open trade at end of test
        if open_trade is not None:
            open_trade.outcome = "OPEN"
            open_trade.bars_held = len(m5_bars) - open_trade_start_idx
            trades.append(open_trade)

    finally:
        model_manager.set_active_strategy(original_strategy)

    # ----------------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------------
    closed     = [t for t in trades if t.outcome != "OPEN"]
    wins       = [t for t in closed if t.outcome in ("WIN_FULL", "WIN_TP1")]
    losses     = [t for t in closed if t.outcome == "LOSS"]
    breakevens = [t for t in closed if t.outcome == "BREAKEVEN"]
    open_t     = [t for t in trades if t.outcome == "OPEN"]

    win_rate    = len(wins) / len(closed) * 100 if closed else 0.0
    net_pnl_r   = sum(t.pnl_r   for t in closed)
    net_pnl_usd = sum(t.pnl_usd for t in closed)

    gross_win  = sum(t.pnl_usd for t in wins)
    gross_loss = abs(sum(t.pnl_usd for t in losses))
    pf = round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf")

    avg_bars = (
        sum(t.bars_held for t in closed) / len(closed) if closed else 0.0
    )
    avg_conf = (
        sum(t.confidence for t in trades) / len(trades) if trades else 0.0
    )

    logger.info(
        f"Backtest complete: {len(trades)} trades | "
        f"WR={win_rate:.1f}% | PF={pf} | "
        f"net={net_pnl_r:+.1f}R / ${net_pnl_usd:+.2f} | "
        f"max_dd={max_dd:.1f}% | "
        f"bars_tested={bars_tested} no_trade={no_trade_count}"
    )
    if no_trade_count > 0 and not trades:
        logger.info(f"Backtest sample NO_TRADE reasons: {no_trade_reasons}")

    return BacktestResult(
        symbol=config.symbol,
        strategy=config.strategy,
        period_start=date_start_test,
        period_end=date_end,
        total_trades=len(trades),
        wins=len(wins),
        losses=len(losses),
        breakevens=len(breakevens),
        open_trades=len(open_t),
        win_rate_pct=round(win_rate, 1),
        profit_factor=pf,
        max_drawdown_pct=round(max_dd, 2),
        net_pnl_r=round(net_pnl_r, 2),
        net_pnl_usd=round(net_pnl_usd, 2),
        initial_balance=config.initial_balance,
        final_balance=round(balance, 2),
        avg_bars_held=round(avg_bars, 1),
        avg_confidence=round(avg_conf, 1),
        bars_tested=bars_tested,
        no_trade_count=no_trade_count,
        sample_no_trade_reasons=no_trade_reasons,
        trades=trades,
    )
