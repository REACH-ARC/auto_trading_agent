"""
Trade Manager — background loop that applies breakeven and trailing SL rules
to all open positions, regardless of which mode placed them.

Rules (configurable via settings.yaml):
  1. r_moved ≥ breakeven_at_r (1.0R) → SL → entry          (no-loss guarantee)
  2. r_moved ≥ tp1_rr_ratio   (1.5R) → SL → TP1 price      (lock TP1, ride to TP2)
  3. r_moved ≥ tp2_rr_ratio   (2.0R) → SL → TP2 price      (lock TP2, ride to TP3)

TP1/TP2 prices are computed from entry + direction × orig_sl_dist × rr_ratio so
no extra per-ticket storage is needed beyond the original SL.

R-moved uses the ORIGINAL SL distance (recorded at first observation) so rules
stay accurate after SL adjustments.
"""
from __future__ import annotations

import asyncio
from loguru import logger

from config import settings

# ticket → original SL recorded at first observation
_original_sl: dict[int, float] = {}


async def trade_manager_loop() -> None:
    from backend.news_filter import refresh_cache_if_stale
    cfg = settings._yaml.get("trade_manager", {})
    if not cfg.get("enabled", True):
        logger.info("Trade manager disabled in settings.yaml")
        return

    interval: int = int(cfg.get("check_interval_sec", 30))
    logger.info(f"Trade manager started — checking positions every {interval}s")

    try:
        while True:
            await asyncio.sleep(interval)
            try:
                await refresh_cache_if_stale()
                await asyncio.to_thread(_manage_all_positions)
                await _sync_closed_signals()
            except Exception as e:
                logger.error(f"Trade manager error: {e}")
    except asyncio.CancelledError:
        logger.info("Trade manager shutting down")


def _manage_all_positions() -> None:
    from backend.mt5_tools import get_open_positions, modify_position, close_position
    from backend.news_filter import get_imminent_news_events
    import re
    import time
    from datetime import datetime, timezone

    cfg        = settings._yaml.get("trade_manager", {})
    be_r: float  = float(cfg.get("breakeven_at_r", 1.0))
    tp1_rr: float = float(settings.risk.get("tp1_rr_ratio", 1.5))
    tp2_rr: float = float(settings.risk.get("tp2_rr_ratio", 2.5))
    
    news_cfg = settings._yaml.get("news_filter", {})
    close_mins: int = int(news_cfg.get("close_open_positions_minutes", 15))

    result = get_open_positions()
    if "error" in result:
        logger.warning(f"Trade manager: cannot read positions — {result['error']}")
        return

    positions: list[dict] = result.get("positions", [])
    open_tickets = {p["ticket"] for p in positions}

    # Purge entries for positions that have closed
    for ticket in list(_original_sl.keys()):
        if ticket not in open_tickets:
            logger.debug(f"Trade manager: ticket {ticket} closed — removing from tracking")
            del _original_sl[ticket]

    for pos in positions:
        ticket        = pos["ticket"]
        direction     = pos["direction"]
        entry         = float(pos["entry_price"])
        cur_sl        = float(pos.get("sl") or 0.0)
        current_price = float(pos["current_price"])
        symbol        = pos["symbol"]
        comment       = pos.get("comment", "")
        time_setup    = pos.get("time_setup", 0)

        # -------------------------------------------------------------
        # Time Stop Check
        # -------------------------------------------------------------
        if comment:
            match = re.search(r'expr:(\d+)h', comment)
            if match and time_setup > 0:
                expr_hours = int(match.group(1))
                age_seconds = time.time() - time_setup
                if age_seconds > expr_hours * 3600:
                    logger.info(f"Time Stop triggered for ticket {ticket} ({age_seconds/3600:.1f}h > {expr_hours}h). Closing position.")
                    close_position(ticket)
                    continue

        # -------------------------------------------------------------
        # News Stop Check
        # -------------------------------------------------------------
        if close_mins > 0:
            now = datetime.now(timezone.utc)
            imminent_news = get_imminent_news_events(symbol, now, minutes=close_mins)
            if imminent_news:
                names = ", ".join(f"{e.currency} '{e.title}'" for e in imminent_news)
                logger.info(f"News Stop triggered for ticket {ticket}. Imminent news in <={close_mins}m: {names}. Closing position.")
                close_position(ticket)
                continue

        if cur_sl == 0.0:
            continue  # position has no SL — skip

        # Record original SL the first time we see this ticket
        if ticket not in _original_sl:
            _original_sl[ticket] = cur_sl
            logger.debug(
                f"Trade manager: now tracking {symbol} ticket={ticket} "
                f"entry={entry:.5f} orig_sl={cur_sl:.5f}"
            )

        orig_sl      = _original_sl[ticket]
        orig_sl_dist = abs(entry - orig_sl)
        if orig_sl_dist < 1e-8:
            continue

        # R-moved using original SL distance (direction-aware)
        if direction == "BUY":
            r_moved = (current_price - entry) / orig_sl_dist
        else:
            r_moved = (entry - current_price) / orig_sl_dist

        if r_moved < 0:
            continue  # position in loss — nothing to protect yet

        mult      = 1.0 if direction == "BUY" else -1.0
        tp1_price = entry + mult * orig_sl_dist * tp1_rr
        tp2_price = entry + mult * orig_sl_dist * tp2_rr

        # Step SL up through milestones — highest threshold wins
        if r_moved >= tp2_rr:
            target_sl = tp2_price
            label = f"SL → TP2 ({tp2_price:.5f}), riding to TP3"
        elif r_moved >= tp1_rr:
            target_sl = tp1_price
            label = f"SL → TP1 ({tp1_price:.5f}), riding to TP2"
        elif r_moved >= be_r:
            target_sl = entry
            label = f"SL → breakeven ({entry:.5f})"
        else:
            continue  # not enough profit yet

        # Only move SL in the favorable direction — never widen the stop
        if direction == "BUY"  and target_sl <= cur_sl:
            continue
        if direction == "SELL" and target_sl >= cur_sl:
            continue

        logger.info(
            f"Trade manager: {symbol} ticket={ticket} {direction} "
            f"r_moved={r_moved:.2f} → {label} "
            f"(SL {cur_sl:.5f} → {target_sl:.5f})"
        )

        res = modify_position(ticket, sl=round(target_sl, 5))
        if "error" in res:
            logger.error(f"Trade manager: SL update failed — {res['error']}")
        else:
            logger.info(
                f"Trade manager: ✅ SL updated — {symbol} ticket={ticket} new_sl={target_sl:.5f}"
            )


async def _sync_closed_signals() -> None:
    from backend.signal_logger import get_pending_signals, update_outcome
    from backend.mt5_tools import get_open_positions, get_recent_closed_deals
    
    pending = await get_pending_signals()
    if not pending:
        return
        
    open_res = await asyncio.to_thread(get_open_positions)
    if "error" in open_res:
        logger.warning(f"Trade manager sync: cannot read positions — {open_res['error']}")
        return
    
    live_positions = open_res.get("positions", [])
    
    closed_res = await asyncio.to_thread(get_recent_closed_deals, 72)
    closed_deals = closed_res.get("deals", []) if "error" not in closed_res else []
    
    for sig in pending:
        is_open = any(p["symbol"] == sig.symbol and p["direction"] == sig.direction for p in live_positions)
        if is_open:
            continue
            
        matching_deals = [
            d for d in closed_deals 
            if d["symbol"] == sig.symbol and d["original_direction"] == sig.direction
        ]
        
        if not matching_deals:
            logger.warning(f"Signal {sig.id} ({sig.symbol} {sig.direction}) is closed but no MT5 deal found in last 72h.")
            continue
            
        latest_deal = sorted(matching_deals, key=lambda x: x["time"], reverse=True)[0]
        pnl = latest_deal["profit"]
        close_price = latest_deal["price"]
        
        if pnl > 0:
            outcome = "WIN"
        elif pnl < 0:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"
            
        logger.info(f"Syncing closed signal {sig.id} ({sig.symbol}) -> {outcome} (PNL: {pnl:.2f})")
        await update_outcome(
            signal_id=sig.id,
            outcome=outcome,
            close_price=close_price,
            actual_pnl=pnl
        )
