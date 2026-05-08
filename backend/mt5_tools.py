"""
MT5 tool implementations for the Claude trading agent.
All functions are synchronous — called via asyncio.to_thread in the agent loop.
Hard risk guards are enforced here, never left to Claude's discretion.
"""
from __future__ import annotations

from datetime import datetime, timezone

from loguru import logger

from config import settings


def _mt5():
    import MetaTrader5 as mt5
    return mt5


# ---------------------------------------------------------------------------
# Account & position queries
# ---------------------------------------------------------------------------

def get_account_info() -> dict:
    """Return live account state including daily loss headroom."""
    try:
        mt5 = _mt5()
        info = mt5.account_info()
        if info is None:
            return {"error": "Cannot read account info — is MT5 logged in?"}

        positions = mt5.positions_get() or []
        daily_pnl = float(info.equity) - float(info.balance)
        loss_limit = float(info.balance) * float(settings.risk.get("max_daily_loss_percent", 3.0)) / 100

        return {
            "login": info.login,
            "server": info.server,
            "equity": round(float(info.equity), 2),
            "balance": round(float(info.balance), 2),
            "free_margin": round(float(info.margin_free), 2),
            "margin_used": round(float(info.margin), 2),
            "open_trades": len(positions),
            "daily_pnl": round(daily_pnl, 2),
            "daily_loss_limit": round(loss_limit, 2),
            "daily_loss_remaining": round(loss_limit + daily_pnl, 2),
            "currency": info.currency,
            "leverage": info.leverage,
        }
    except Exception as e:
        return {"error": str(e)}


def get_open_positions(symbol: str | None = None) -> dict:
    """List open positions, optionally filtered by symbol."""
    try:
        mt5 = _mt5()
        positions = (mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()) or []

        result = []
        for p in positions:
            entry_price = p.price_open
            current_price = p.price_current
            sl = p.sl
            # Calculate R moved (how far price has moved toward/away from target)
            sl_distance = abs(entry_price - sl) if sl else 0
            price_move = current_price - entry_price if p.type == 0 else entry_price - current_price
            r_moved = round(price_move / sl_distance, 2) if sl_distance > 0 else 0

            result.append({
                "ticket": p.ticket,
                "symbol": p.symbol,
                "direction": "BUY" if p.type == 0 else "SELL",
                "lot_size": p.volume,
                "entry_price": p.price_open,
                "current_price": p.price_current,
                "sl": p.sl,
                "tp": p.tp,
                "profit_usd": round(p.profit, 2),
                "r_moved": r_moved,
                "open_time": datetime.fromtimestamp(p.time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
                "comment": p.comment,
            })

        return {"positions": result, "count": len(result)}
    except Exception as e:
        return {"error": str(e)}


def get_market_snapshot(symbol: str) -> dict:
    """
    Fetch OHLCV data, compute all indicators, and return a formatted snapshot.
    This is the primary analysis tool — always call before any trade decision.
    """
    try:
        from backend.mt5_fetcher import fetch_ohlcv
        from backend.indicators import compute_indicators
        from backend.claude_analyst import format_market_snapshot
        from backend.pattern_analyst import detect_patterns

        bars = int(settings.mt5_fetcher.get("bars_per_tf", 100))
        ohlcv = fetch_ohlcv(symbol, bars)
        if ohlcv is None:
            return {"error": f"No data for {symbol} — is it in Market Watch?"}

        indicators = compute_indicators(ohlcv)
        patterns   = detect_patterns(ohlcv)
        snapshot_text = format_market_snapshot(ohlcv, indicators, patterns)

        return {
            "symbol": symbol,
            "snapshot": snapshot_text,
            "trend_bias": indicators.trend_bias,
            "active_sessions": indicators.session.active_sessions,
            "is_high_liquidity": indicators.session.is_high_liquidity,
            "is_overlap": indicators.session.is_overlap,
        }
    except Exception as e:
        return {"error": str(e)}


def get_symbol_info(symbol: str) -> dict:
    """Return trading specs: bid/ask, tick value, lot constraints."""
    try:
        mt5 = _mt5()
        info = mt5.symbol_info(symbol)
        if info is None:
            return {"error": f"Symbol {symbol} not found in Market Watch"}

        tick = mt5.symbol_info_tick(symbol)
        return {
            "symbol": symbol,
            "bid": round(tick.bid, info.digits) if tick else None,
            "ask": round(tick.ask, info.digits) if tick else None,
            "spread": round((tick.ask - tick.bid) / info.point, 1) if tick else None,
            "digits": info.digits,
            "point": info.point,
            "tick_size": info.trade_tick_size,
            "tick_value": round(info.trade_tick_value, 6),
            "contract_size": info.trade_contract_size,
            "min_lot": info.volume_min,
            "max_lot": info.volume_max,
            "lot_step": info.volume_step,
            "currency_profit": info.currency_profit,
            "swap_long": info.swap_long,
            "swap_short": info.swap_short,
        }
    except Exception as e:
        return {"error": str(e)}


# ---------------------------------------------------------------------------
# Trade execution
# ---------------------------------------------------------------------------

def place_order(
    symbol: str,
    direction: str,
    lot_size: float,
    sl: float,
    tp1: float,
    tp2: float | None = None,
    tp3: float | None = None,
    comment: str = "ClaudeAgent",
    order_type: str = "MARKET",
    limit_price: float | None = None,
) -> dict:
    """
    Place a trade order with hard risk guards enforced before execution.

    Guards (cannot be bypassed by Claude):
    - lot_size <= max_lot_size in config
    - open_trades < max_open_trades in config
    - daily P&L has not breached max_daily_loss_percent
    - SL must be at least 10 points from entry
    """
    try:
        mt5 = _mt5()
        risk_cfg = settings.risk

        # Guard: lot size range — read limits for the detected account type
        acct_info = mt5.account_info()
        _currency = str(acct_info.currency) if acct_info else "USD"
        _acct_type = "cent" if _currency.upper() == "USC" else "standard"
        _type_cfg = risk_cfg.get(_acct_type, {})
        min_lot = float(_type_cfg.get("min_lot_size", 0.01))
        max_lot = float(_type_cfg.get("max_lot_size", 1.0))
        if lot_size < min_lot:
            lot_size = min_lot  # silently raise to minimum rather than rejecting
        if lot_size > max_lot:
            return {"error": f"Lot {lot_size} exceeds max_lot_size={max_lot} for {_acct_type} account. Reduce lot size."}

        # Guard: no duplicate position on the same symbol
        existing = mt5.positions_get(symbol=symbol) or []
        if existing:
            return {
                "error": (
                    f"Position already open on {symbol} "
                    f"(ticket={existing[0].ticket}, {('BUY' if existing[0].type == 0 else 'SELL')}). "
                    f"Only one position per symbol allowed."
                )
            }

        # Guard: max open trades
        open_positions = mt5.positions_get() or []
        max_trades = int(risk_cfg.get("max_open_trades", 3))
        if len(open_positions) >= max_trades:
            return {"error": f"Max open trades ({max_trades}) reached. Close a position first."}

        # Guard: daily loss limit
        account = mt5.account_info()
        if account:
            daily_pnl = float(account.equity) - float(account.balance)
            loss_limit = float(account.balance) * float(risk_cfg.get("max_daily_loss_percent", 3.0)) / 100
            if daily_pnl < -loss_limit:
                return {
                    "error": (
                        f"Daily loss limit reached (P&L={daily_pnl:.2f}, "
                        f"limit=-{loss_limit:.2f}). No new trades today."
                    )
                }

        # Symbol validation
        sym_info = mt5.symbol_info(symbol)
        if sym_info is None:
            return {"error": f"Symbol {symbol} not available"}

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return {"error": f"Cannot get live price for {symbol}"}

        # Entry price
        if order_type == "MARKET":
            price = tick.ask if direction == "BUY" else tick.bid
        elif limit_price is not None:
            price = limit_price
        else:
            return {"error": "limit_price required for LIMIT/STOP orders"}

        # Guard: minimum SL distance
        min_distance = sym_info.point * 10
        if abs(price - sl) < min_distance:
            return {"error": f"SL too close to entry (min distance: {min_distance:.5f})"}

        # Guard: max SL distance in pips (forex pairs only — skip for gold/crypto)
        is_metal_or_crypto = any(x in symbol.upper() for x in ["XAU", "XAG", "BTC", "ETH", "LTC", "SOL"])
        if not is_metal_or_crypto and sym_info.digits >= 3:
            pip_value = sym_info.point * 10
            sl_pips = abs(price - sl) / pip_value
            max_sl_pips = int(risk_cfg.get("max_sl_pips", 100))
            if sl_pips > max_sl_pips:
                return {"error": f"SL too wide: {sl_pips:.0f} pips exceeds max {max_sl_pips} pips for {symbol}. Use tighter structure."}

        # Round lot to broker step
        step = sym_info.volume_step
        lot_size = round(round(lot_size / step) * step, 8)
        lot_size = max(sym_info.volume_min, min(sym_info.volume_max, lot_size))

        # Build MT5 request
        if order_type == "MARKET":
            mt5_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
            mt5_action = mt5.TRADE_ACTION_DEAL
        elif order_type == "LIMIT":
            mt5_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "BUY" else mt5.ORDER_TYPE_SELL_LIMIT
            mt5_action = mt5.TRADE_ACTION_PENDING
        elif order_type == "STOP":
            mt5_type = mt5.ORDER_TYPE_BUY_STOP if direction == "BUY" else mt5.ORDER_TYPE_SELL_STOP
            mt5_action = mt5.TRADE_ACTION_PENDING
        else:
            return {"error": f"Unknown order_type: {order_type}"}

        # Use the furthest TP as the hard MT5 close; intermediate levels are
        # managed by the trade manager (SL stepped to TP1 then TP2 as each is hit).
        hard_tp = tp3 or tp2 or tp1

        request = {
            "action": mt5_action,
            "symbol": symbol,
            "volume": lot_size,
            "type": mt5_type,
            "price": price,
            "sl": sl,
            "tp": hard_tp,
            "comment": comment[:31],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            code = result.retcode if result else "N/A"
            msg = result.comment if result else "no result"
            if code == 10027:
                msg = "AutoTrading disabled — enable it in MT5: toolbar → AutoTrading button (or Alt+T)"
            logger.error(f"Order rejected: {msg} (retcode={code})")
            return {"error": f"Order rejected: {msg} (retcode={code})"}

        logger.info(
            f"Agent placed {direction} {lot_size} {symbol} @ {result.price} "
            f"SL={sl} TP1={tp1} ticket={result.order}"
        )
        return {
            "success": True,
            "ticket": result.order,
            "symbol": symbol,
            "direction": direction,
            "lot_size": lot_size,
            "entry_price": result.price,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
        }

    except Exception as e:
        logger.error(f"place_order error: {e}")
        return {"error": str(e)}


def modify_position(ticket: int, sl: float | None = None, tp: float | None = None) -> dict:
    """Modify the SL and/or TP of an existing position."""
    try:
        mt5 = _mt5()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"error": f"Position {ticket} not found"}

        pos = positions[0]
        new_sl = sl if sl is not None else pos.sl
        new_tp = tp if tp is not None else pos.tp

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "sl": new_sl,
            "tp": new_tp,
            "position": ticket,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = result.comment if result else "no result"
            return {"error": f"Modify failed: {msg} (retcode={result.retcode if result else 'N/A'})"}

        logger.info(f"Agent modified position {ticket}: SL={new_sl} TP={new_tp}")
        return {"success": True, "ticket": ticket, "new_sl": new_sl, "new_tp": new_tp}

    except Exception as e:
        logger.error(f"modify_position error: {e}")
        return {"error": str(e)}


def close_position(ticket: int, lot_size: float | None = None) -> dict:
    """Close a position fully or partially."""
    try:
        mt5 = _mt5()
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return {"error": f"Position {ticket} not found"}

        pos = positions[0]
        close_vol = min(lot_size if lot_size is not None else pos.volume, pos.volume)

        tick = mt5.symbol_info_tick(pos.symbol)
        if tick is None:
            return {"error": f"Cannot get price for {pos.symbol}"}

        close_price = tick.bid if pos.type == 0 else tick.ask

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": pos.symbol,
            "volume": close_vol,
            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
            "position": ticket,
            "price": close_price,
            "comment": "ClaudeAgent-Close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            msg = result.comment if result else "no result"
            return {"error": f"Close failed: {msg} (retcode={result.retcode if result else 'N/A'})"}

        logger.info(f"Agent closed {close_vol} lots of position {ticket} @ {close_price}")
        return {
            "success": True,
            "ticket": ticket,
            "closed_volume": close_vol,
            "close_price": close_price,
            "remaining_volume": round(pos.volume - close_vol, 8),
        }

    except Exception as e:
        logger.error(f"close_position error: {e}")
        return {"error": str(e)}
