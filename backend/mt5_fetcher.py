"""
MT5 Direct Fetcher — replaces the desktop EA.
Connects to a running MT5 terminal via the MetaTrader5 Python package,
detects each new M5 bar, fetches OHLCV + account state, and runs the pipeline.

Requirements:
  - MT5 terminal open on the same Windows PC (minimized is fine)
  - Logged into your MT5 account (cent or standard)
  - pip install MetaTrader5
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from loguru import logger

from config import settings
from backend.indicators import Bar, OHLCVData
from backend.risk_manager import AccountState

_TF_NAMES = ["M5", "H1", "H4", "D1"]


def _tf_const(name: str):
    import MetaTrader5 as mt5
    return {
        "M5": mt5.TIMEFRAME_M5,
        "H1":  mt5.TIMEFRAME_H1,
        "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,
    }[name]


def connect() -> bool:
    """Connect to the running MT5 terminal. Returns True on success."""
    try:
        import MetaTrader5 as mt5
    except ImportError:
        logger.error("MetaTrader5 package not installed — run: pip install MetaTrader5")
        return False

    if not mt5.initialize():
        logger.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False

    info = mt5.account_info()
    if info is None:
        logger.error("MT5 connected but no account — is the terminal logged in?")
        mt5.shutdown()
        return False

    from backend import account_manager
    account_manager.set_currency(str(info.currency))

    logger.info(
        f"MT5 connected — account #{info.login} | {info.server} | "
        f"balance={info.balance:.2f} {info.currency} "
        f"({'cent' if account_manager.is_cent() else 'standard'})"
    )
    return True


def disconnect() -> None:
    try:
        import MetaTrader5 as mt5
        mt5.shutdown()
        logger.info("MT5 disconnected")
    except ImportError:
        pass


def fetch_ohlcv(symbol: str, bars: int = 100, silent: bool = False) -> OHLCVData | None:
    """Fetch M5/H1/H4/D1 bars for symbol from the MT5 terminal."""
    import MetaTrader5 as mt5

    timeframes: dict = {}
    for tf_name in _TF_NAMES:
        rates = mt5.copy_rates_from_pos(symbol, _tf_const(tf_name), 0, bars)
        if rates is None or len(rates) == 0:
            if not silent:
                logger.warning(f"MT5: no {tf_name} data for {symbol} — {mt5.last_error()}")
            continue
        timeframes[tf_name] = [
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

    if not timeframes:
        if not silent:
            logger.error(f"MT5: could not fetch any data for {symbol}")
        return None

    return OHLCVData(
        symbol=symbol,
        timeframes=timeframes,
        received_at=datetime.now(timezone.utc),
    )


def fetch_account() -> AccountState:
    """Return live account state from MT5, including account currency."""
    try:
        import MetaTrader5 as mt5
        info = mt5.account_info()
        if info is None:
            return AccountState(equity=10_000, balance=10_000, open_trades=0, daily_pnl=0)
        return AccountState(
            equity=float(info.equity),
            balance=float(info.balance),
            open_trades=int(mt5.positions_total() or 0),
            daily_pnl=float(info.equity) - float(info.balance),
            currency=str(info.currency),
        )
    except ImportError:
        return AccountState(equity=10_000, balance=10_000, open_trades=0, daily_pnl=0)


def _ensure_symbol(symbol: str) -> bool:
    """Ensure symbol is in Market Watch so data calls work."""
    import MetaTrader5 as mt5
    if not mt5.symbol_select(symbol, True):
        logger.warning(f"MT5: symbol_select({symbol}) failed — {mt5.last_error()}")
        return False
    return True


def _current_m5_time(symbol: str) -> datetime | None:
    """Return the open timestamp of the current M5 bar."""
    import MetaTrader5 as mt5
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, 1)
    if rates is None or len(rates) == 0:
        logger.debug(f"MT5: copy_rates_from_pos({symbol}, M5) returned nothing — {mt5.last_error()}")
        return None
    return datetime.fromtimestamp(rates[0]["time"], tz=timezone.utc)


def _current_price(symbol: str) -> float | None:
    """Return current mid price (bid+ask)/2 for the symbol."""
    import MetaTrader5 as mt5
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        return None
    return (tick.bid + tick.ask) / 2.0


async def fetcher_loop(run_pipeline_fn, get_symbol_fn=None) -> None:
    """
    Background coroutine — checks every 30s for a new M5 bar.
    On a new bar: fetch data → run pipeline → Claude → Telegram.

    Between bar closes the loop monitors price against S/R levels (Option 3).
    Any level touches during a candle are passed to the agent at bar close so
    Claude can factor in confirmation context without reacting to mid-candle wicks.

    get_symbol_fn: optional callable () -> str that returns the active symbol.
    When provided, the loop reads the symbol on every tick so Telegram
    market-selection changes take effect without a restart.
    """
    from backend.scanner import update_symbol_data
    from backend.level_watcher import LevelWatcher
    from backend.agent_alert_manager import get_alert_manager

    from backend import account_manager

    cfg = settings.mt5_fetcher
    check_sec: int = int(cfg.get("check_interval_sec", 30))
    bars: int = int(cfg.get("bars_per_tf", 100))

    def _active_symbol() -> str:
        base = get_symbol_fn() if get_symbol_fn else cfg.get("symbol", "XAUUSD")
        return account_manager.apply_suffix(account_manager.strip_suffix(base))

    logger.info(f"MT5 fetcher starting — check_every={check_sec}s")

    connected = await asyncio.to_thread(connect)
    if not connected:
        logger.error(
            "MT5 fetcher: could not connect. "
            "Make sure MT5 terminal is open and logged in, then restart."
        )
        return

    # account_manager.set_currency() was called inside connect() — suffix is now correct
    initial_symbol = _active_symbol()
    logger.info(f"MT5 fetcher: resolved symbol={initial_symbol} (account={account_manager.get_currency()})")

    # Ensure initial symbol is in Market Watch
    sym_ok = await asyncio.to_thread(_ensure_symbol, initial_symbol)
    if not sym_ok:
        logger.error(
            f"MT5 fetcher: symbol '{initial_symbol}' not found in Market Watch. "
            f"Account type: {account_manager.get_currency()} — "
            f"check that the symbol exists on your broker."
        )
        await asyncio.to_thread(disconnect)
        return

    last_bar: datetime | None = None
    last_symbol: str = initial_symbol
    watcher = LevelWatcher()
    alert_mgr = get_alert_manager()

    try:
        while True:
            await asyncio.sleep(check_sec)

            symbol = _active_symbol()

            # Symbol changed via Telegram — re-validate and reset bar tracking
            if symbol != last_symbol:
                logger.info(f"MT5 fetcher: symbol switched {last_symbol!r} → {symbol!r}")
                ok = await asyncio.to_thread(_ensure_symbol, symbol)
                if not ok:
                    logger.warning(
                        f"MT5 fetcher: '{symbol}' not in Market Watch — reverting to {last_symbol!r}"
                    )
                    symbol = last_symbol
                else:
                    last_bar = None
                    watcher.clear_symbol(last_symbol)
                    alert_mgr.clear(last_symbol)
                    last_symbol = symbol

            # Check current price against watched S/R levels and agent alerts every tick
            price = await asyncio.to_thread(_current_price, symbol)
            if price is not None:
                watcher.check_price(symbol, price)
                alert_mgr.check_price(symbol, price)

            bar_time = await asyncio.to_thread(_current_m5_time, symbol)
            if bar_time is None:
                logger.warning(
                    f"MT5 fetcher: cannot read M5 bar for {symbol} — "
                    "market may be closed, or symbol dropped from Market Watch"
                )
                continue

            if bar_time == last_bar:
                continue  # same bar — level monitoring continues above

            # New M5 bar: collect level hits and triggered agent alerts from the closed candle
            level_hits = watcher.pop_hits(symbol)
            alert_hits = alert_mgr.pop_triggered(symbol)

            if level_hits:
                logger.info(
                    f"MT5 fetcher: {len(level_hits)} level hit(s) from closed candle — "
                    f"{[f'{h.level_type} {h.level_price}' for h in level_hits]}"
                )
            if alert_hits:
                logger.info(
                    f"MT5 fetcher: {len(alert_hits)} agent alert(s) triggered — "
                    f"{[f'{a.condition} {a.price:.5f}' for a in alert_hits]}"
                )

            last_bar = bar_time
            logger.info(f"MT5 fetcher: new M5 bar {bar_time.strftime('%H:%M UTC')} — fetching {symbol}")

            ohlcv = await asyncio.to_thread(fetch_ohlcv, symbol, bars)
            if ohlcv is None:
                continue

            # Update watcher with fresh S/R levels from the new bar's data
            try:
                from backend.indicators import compute_indicators
                indicators = compute_indicators(ohlcv)
                watcher.update_levels(symbol, indicators.support_resistance)
            except Exception as exc:
                logger.warning(f"MT5 fetcher: level update failed for {symbol} — {exc}")

            account = await asyncio.to_thread(fetch_account)

            # Skip full market analysis when a position is open and nothing notable happened.
            # The agent will still run Steps 1+2+5 (position management) every bar.
            skip_analysis = (
                account.open_trades > 0
                and not level_hits
                and not alert_hits
            )
            if skip_analysis:
                logger.debug(
                    f"MT5 fetcher: skip_analysis=True for {symbol} "
                    f"(open_trades={account.open_trades}, no level/alert hits)"
                )

            # Keep scanner cache current
            await update_symbol_data(ohlcv)

            # Run full analysis pipeline, passing level hit and alert context for the agent
            try:
                await run_pipeline_fn(
                    ohlcv, account,
                    level_hits=level_hits,
                    alert_hits=alert_hits,
                    skip_analysis=skip_analysis,
                )
            except Exception as exc:
                logger.error(f"MT5 fetcher: pipeline error — {exc}")

    except asyncio.CancelledError:
        logger.info("MT5 fetcher: shutting down")
    finally:
        await asyncio.to_thread(disconnect)
