"""
Asian Range Breakout Strategy for XAUUSD.

Logic:
  1. Asian session (00:00–07:00 UTC) defines the consolidation range.
     high = highest high, low = lowest low across all M15 bars in that window.
  2. Only fire during the London open window (07:00–10:00 UTC) — the period
     when gold typically breaks out of the Asian range with volume.
  3. Breakout condition:
       BUY  — current close > asian_high + breakout_buffer × ATR
       SELL — current close < asian_low  - breakout_buffer × ATR
  4. Range filter — skip if range > max_range_atr_mult × ATR (too wide = choppy Asia)
  5. H4 trend alignment adds confidence (bonus, not required).
  6. London/NY session active adds confidence.

Confidence scoring (max 100):
  Clean breakout confirmed                  +35
  Range is tight (< 1.0× ATR)              +20   (tight range → stronger break)
  Range is medium (1.0–1.5× ATR)           +10
  H4 trend aligned with break direction    +25
  London session active                    +20   (capped at 100)
"""
from __future__ import annotations

from datetime import time, timezone

from loguru import logger

from backend.claude_analyst import SignalResult
from backend.indicators import IndicatorResult, OHLCVData
from config import settings

# Asian session window (UTC)
_ASIAN_START = time(0, 0)
_ASIAN_END   = time(7, 0)

# London open fire window (UTC) — extended to catch late-London breakouts too
_FIRE_START = time(7, 0)
_FIRE_END   = time(11, 0)


def analyse(ohlcv: OHLCVData, indicators: IndicatorResult) -> SignalResult:
    cfg = settings._yaml.get("strategy", {}).get("asian_breakout", {})
    max_range_mult: float = float(cfg.get("max_range_atr_mult", 1.5))
    min_conf: int         = int(cfg.get("min_confidence", 65))
    breakout_buffer: float = float(cfg.get("breakout_buffer_atr", 0.1))

    symbol = ohlcv.symbol
    tfs    = list(ohlcv.timeframes.keys())
    current_price: float = ohlcv.timeframes[tfs[0]][-1].close
    now_utc = ohlcv.received_at.astimezone(timezone.utc).time()

    # ------------------------------------------------------------------ #
    # 1. Fire-window check — only trade during London open
    # ------------------------------------------------------------------ #
    if not (_FIRE_START <= now_utc <= _FIRE_END):
        return _no_trade(
            symbol,
            f"Outside London open fire window ({_FIRE_START}–{_FIRE_END} UTC), "
            f"current time={now_utc.strftime('%H:%M')} UTC"
        )

    # ------------------------------------------------------------------ #
    # 2. Build Asian session range from M15 bars
    # ------------------------------------------------------------------ #
    m15_bars = ohlcv.timeframes.get("M15") or ohlcv.timeframes.get(tfs[0])
    asian_bars = [
        b for b in m15_bars
        if _ASIAN_START <= b.time.astimezone(timezone.utc).time() < _ASIAN_END
    ]

    if len(asian_bars) < 4:
        return _no_trade(symbol, f"Insufficient Asian session bars ({len(asian_bars)} < 4)")

    asian_high = max(b.high  for b in asian_bars)
    asian_low  = min(b.low   for b in asian_bars)
    asian_range = asian_high - asian_low

    # ------------------------------------------------------------------ #
    # 3. ATR reference (H1 preferred, fallback to first available)
    # ------------------------------------------------------------------ #
    atr = indicators.atr.get("H1") or indicators.atr.get("H4") or indicators.atr.get(tfs[0], 1.0)

    # ------------------------------------------------------------------ #
    # 4. Range filter — skip if Asian range is too wide
    # ------------------------------------------------------------------ #
    if asian_range > max_range_mult * atr:
        return _no_trade(
            symbol,
            f"Asian range too wide: {asian_range:.5f} > {max_range_mult}×ATR ({max_range_mult * atr:.5f})"
        )

    # ------------------------------------------------------------------ #
    # 5. Breakout detection
    # ------------------------------------------------------------------ #
    buffer  = breakout_buffer * atr
    buy_trigger  = asian_high + buffer
    sell_trigger = asian_low  - buffer

    if current_price > buy_trigger:
        direction = "BUY"
    elif current_price < sell_trigger:
        direction = "SELL"
    else:
        return _no_trade(
            symbol,
            f"No breakout: price={current_price:.5f} inside range "
            f"[{asian_low:.5f} – {asian_high:.5f}] ±buffer={buffer:.5f}"
        )

    # ------------------------------------------------------------------ #
    # 6. Confidence scoring
    # ------------------------------------------------------------------ #
    confidence = 35   # confirmed breakout
    confluence: list[str] = [
        f"Price broke {'above' if direction == 'BUY' else 'below'} Asian range "
        f"({'high' if direction == 'BUY' else 'low'}={asian_high if direction == 'BUY' else asian_low:.5f})"
    ]

    range_ratio = asian_range / atr
    if range_ratio < 1.0:
        confidence += 20
        confluence.append(f"Tight Asian range ({asian_range:.5f} < 1.0×ATR)")
    elif range_ratio < max_range_mult:
        confidence += 10
        confluence.append(f"Medium Asian range ({asian_range:.5f}, {range_ratio:.1f}×ATR)")

    h4_emas  = indicators.ema.get("H4", {})
    h4_ema200 = h4_emas.get(200, current_price)
    h4_aligned = (direction == "BUY" and current_price > h4_ema200) or \
                 (direction == "SELL" and current_price < h4_ema200)
    if h4_aligned:
        confidence += 25
        confluence.append(f"H4 EMA200 supports {direction} direction")

    if indicators.session.is_high_liquidity:
        confidence += 20
        confluence.append(f"High-liquidity session: {', '.join(indicators.session.active_sessions)}")

    confidence = min(confidence, 100)

    if confidence < min_conf:
        return _no_trade(
            symbol,
            f"Confidence {confidence}% below minimum {min_conf}%"
        )

    # ------------------------------------------------------------------ #
    # 7. Entry, SL, TP
    # ------------------------------------------------------------------ #
    entry = current_price

    # SL sits just inside the opposite edge of the Asian range
    if direction == "BUY":
        sl = asian_low - 0.3 * atr
    else:
        sl = asian_high + 0.3 * atr

    if direction == "BUY" and sl >= entry:
        return _no_trade(symbol, f"Invalid SL for BUY: sl={sl:.5f} >= entry={entry:.5f}")
    if direction == "SELL" and sl <= entry:
        return _no_trade(symbol, f"Invalid SL for SELL: sl={sl:.5f} <= entry={entry:.5f}")

    sl_distance = abs(entry - sl)
    if sl_distance < 1e-8:
        return _no_trade(symbol, "SL distance is zero")

    risk_cfg = settings.risk
    mult = 1 if direction == "BUY" else -1
    tp1 = entry + mult * sl_distance * float(risk_cfg.get("tp1_rr_ratio", 1.5))
    tp2 = entry + mult * sl_distance * float(risk_cfg.get("tp2_rr_ratio", 2.5))
    tp3 = entry + mult * sl_distance * float(risk_cfg.get("tp3_rr_ratio", 4.0))
    rr  = round(abs(tp1 - entry) / sl_distance, 2)

    reasoning = (
        f"Asian Range Breakout ({direction}): price {current_price:.5f} broke "
        f"{'above' if direction == 'BUY' else 'below'} Asian range "
        f"[{asian_low:.5f}–{asian_high:.5f}] (range={asian_range:.5f}, {range_ratio:.1f}×ATR). "
        f"Confidence {confidence}% from {len(confluence)} factor(s)."
    )
    invalidation = (
        f"Close back {'below' if direction == 'BUY' else 'above'} Asian range "
        f"{'high' if direction == 'BUY' else 'low'} "
        f"({asian_high if direction == 'BUY' else asian_low:.5f})."
    )

    logger.info(
        f"Asian Breakout signal: {symbol} {direction} conf={confidence}% "
        f"entry={entry:.5f} sl={sl:.5f} tp1={tp1:.5f}"
    )

    return SignalResult(
        symbol=symbol,
        direction=direction,  # type: ignore[arg-type]
        confidence=confidence,
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        reasoning=reasoning,
        invalidation=invalidation,
        confluence_factors=confluence,
    )


def _no_trade(symbol: str, reason: str) -> SignalResult:
    logger.info(f"Asian Breakout — NO_TRADE ({symbol}): {reason}")
    return SignalResult(
        symbol=symbol,
        direction="NO_TRADE",
        confidence=0,
        reasoning=reason,
        invalidation="",
        confluence_factors=[],
    )
