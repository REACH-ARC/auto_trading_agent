"""
EMA Pullback Strategy for XAUUSD (and other trending instruments).

Logic:
  1. Trend filter  — H4 EMA200: price above → BUY bias, below → SELL bias.
                     Stronger signal when EMA20 > EMA50 > EMA200 (bull stack).
  2. Entry trigger — Price pulls back to within N×ATR of H1 EMA20.
  3. RSI filter    — RSI(H1) must be in neutral zone (not already stretched).
  4. Candle confirm — Current M15 bar must close in trade direction (bullish for BUY,
                      bearish for SELL) to confirm the pullback is holding.
  5. MACD confirm  — H1 MACD histogram bias aligns with direction (+confidence).
  6. Session bonus — London or NY active adds confidence.
  7. D1 alignment  — D1 EMA200 same side as H4 adds confidence.

Confidence scoring (max 100):
  H4 EMA full stack aligned        +25
  H4 EMA200 only (partial)         +15
  Price near H1 EMA20              +20
  RSI in favorable zone            +15
  Candle confirms direction        +10
  H1 MACD aligned                  +15
  London or NY session active      +10
  D1 EMA200 aligned                +15   (capped at 100 total)
"""
from __future__ import annotations

from loguru import logger

from backend.claude_analyst import SignalResult
from backend.indicators import IndicatorResult, OHLCVData
from config import settings


def analyse(ohlcv: OHLCVData, indicators: IndicatorResult) -> SignalResult:
    cfg = settings._yaml.get("strategy", {}).get("ema_pullback", {})
    trend_tf: str = cfg.get("trend_tf", "H4")
    entry_tf: str = cfg.get("entry_tf", "H1")
    rsi_min: float = float(cfg.get("rsi_min", 40))
    rsi_max: float = float(cfg.get("rsi_max", 60))
    sl_atr_buf: float = float(cfg.get("sl_atr_buffer", 0.5))
    prox_atr: float = float(cfg.get("ema_proximity_atr", 0.8))
    min_conf: int = int(cfg.get("min_confidence", 65))

    symbol = ohlcv.symbol
    tfs = list(ohlcv.timeframes.keys())
    m15_bars = ohlcv.timeframes.get("M15") or ohlcv.timeframes[tfs[0]]
    current_bar = m15_bars[-1]
    current_price: float = current_bar.close

    # ------------------------------------------------------------------ #
    # 1. Trend filter — H4 EMAs
    # ------------------------------------------------------------------ #
    h4_emas = indicators.ema.get(trend_tf, {})
    h4_ema20  = h4_emas.get(20,  current_price)
    h4_ema50  = h4_emas.get(50,  current_price)
    h4_ema200 = h4_emas.get(200, current_price)
    h4_atr    = indicators.atr.get(trend_tf, 1.0)

    bull_stack = h4_ema20 > h4_ema50 > h4_ema200
    bear_stack = h4_ema20 < h4_ema50 < h4_ema200
    bull_ema200 = current_price > h4_ema200
    bear_ema200 = current_price < h4_ema200

    if bull_stack:
        direction = "BUY"
    elif bear_stack:
        direction = "SELL"
    else:
        return _no_trade(
            symbol,
            "H4 EMA not fully stacked (need EMA20 > EMA50 > EMA200 or reverse) — "
            "partial EMA200-only bias rejected"
        )

    # ------------------------------------------------------------------ #
    # 2. D1 trend alignment — required hard filter
    # ------------------------------------------------------------------ #
    d1_emas   = indicators.ema.get("D1", {})
    d1_ema200 = d1_emas.get(200, current_price)
    d1_aligned = (direction == "BUY" and current_price > d1_ema200) or \
                 (direction == "SELL" and current_price < d1_ema200)

    if not d1_aligned:
        return _no_trade(
            symbol,
            f"D1 EMA200 ({d1_ema200:.3f}) not aligned with {direction} — "
            f"price={current_price:.3f} trading against daily trend"
        )

    # ------------------------------------------------------------------ #
    # 3. Entry trigger — price near H1 EMA20
    # ------------------------------------------------------------------ #
    h1_emas  = indicators.ema.get(entry_tf, {})
    h1_ema20 = h1_emas.get(20, current_price)
    h1_atr   = indicators.atr.get(entry_tf, h4_atr)

    distance_to_ema20 = abs(current_price - h1_ema20)
    near_ema20 = distance_to_ema20 <= prox_atr * h1_atr

    # ------------------------------------------------------------------ #
    # 4. RSI filter
    # ------------------------------------------------------------------ #
    h1_rsi = indicators.rsi.get(entry_tf, 50.0)
    if direction == "BUY":
        rsi_ok = rsi_min <= h1_rsi <= 65
    else:
        rsi_ok = 35 <= h1_rsi <= rsi_max

    # ------------------------------------------------------------------ #
    # 5. Candle confirmation — M15 bar must close in trade direction
    # ------------------------------------------------------------------ #
    candle_bullish = current_bar.close > current_bar.open
    candle_ok = (direction == "BUY" and candle_bullish) or \
                (direction == "SELL" and not candle_bullish)

    # ------------------------------------------------------------------ #
    # 6. MACD H1 alignment
    # ------------------------------------------------------------------ #
    h1_macd = indicators.macd.get(entry_tf, {})
    macd_bull = h1_macd.get("is_bullish", False)
    macd_aligned = (direction == "BUY" and macd_bull) or (direction == "SELL" and not macd_bull)

    # ------------------------------------------------------------------ #
    # 7. Session
    # ------------------------------------------------------------------ #
    session_active = indicators.session.is_high_liquidity

    # ------------------------------------------------------------------ #
    # Confidence score
    # ------------------------------------------------------------------ #
    confidence = 0
    confluence: list[str] = []

    # H4 full stack is now required — always +25 at this point
    confidence += 25
    confluence.append(f"H4 EMA full stack {'bullish' if direction == 'BUY' else 'bearish'}")

    # D1 alignment is now required — always +15 at this point
    confidence += 15
    confluence.append(f"D1 EMA200 ({d1_ema200:.3f}) confirms {direction} bias")

    if near_ema20:
        confidence += 20
        confluence.append(f"Price near H1 EMA20 ({distance_to_ema20:.5f} vs {prox_atr * h1_atr:.5f} ATR)")
    else:
        return _no_trade(
            symbol,
            f"Price not near H1 EMA20 (distance={distance_to_ema20:.5f}, threshold={prox_atr * h1_atr:.5f})"
        )

    if rsi_ok:
        confidence += 15
        confluence.append(f"H1 RSI {h1_rsi:.1f} in favorable zone")
    else:
        return _no_trade(symbol, f"H1 RSI {h1_rsi:.1f} not in favorable zone for {direction}")

    if candle_ok:
        confidence += 10
        confluence.append(
            f"M15 candle confirms {'bullish' if direction == 'BUY' else 'bearish'} "
            f"(close={current_bar.close:.3f} {'>' if direction == 'BUY' else '<'} open={current_bar.open:.3f})"
        )
    else:
        return _no_trade(
            symbol,
            f"M15 candle not confirming {direction} "
            f"(close={current_bar.close:.3f}, open={current_bar.open:.3f})"
        )

    if macd_aligned:
        confidence += 15
        confluence.append(f"H1 MACD aligned with {direction}")

    if session_active:
        confidence += 10
        confluence.append(f"High-liquidity session active: {', '.join(indicators.session.active_sessions)}")

    confidence = min(confidence, 100)

    if confidence < min_conf:
        return _no_trade(
            symbol,
            f"Confidence {confidence}% below minimum {min_conf}% — "
            f"confluence factors: {len(confluence)}"
        )

    # ------------------------------------------------------------------ #
    # Entry, SL, TP calculation
    # ------------------------------------------------------------------ #
    entry = current_price

    if direction == "BUY":
        sl = h1_ema20 - sl_atr_buf * h1_atr
    else:
        sl = h1_ema20 + sl_atr_buf * h1_atr

    if direction == "BUY" and sl >= entry:
        return _no_trade(symbol, f"Invalid SL for BUY: sl={sl:.5f} >= entry={entry:.5f} — price below EMA20 buffer")
    if direction == "SELL" and sl <= entry:
        return _no_trade(symbol, f"Invalid SL for SELL: sl={sl:.5f} <= entry={entry:.5f} — price above EMA20 buffer")

    sl_distance = abs(entry - sl)
    if sl_distance < 1e-8:
        return _no_trade(symbol, "SL distance is zero — cannot calculate R:R")

    risk_cfg = settings.risk
    tp1 = entry + (sl_distance * float(risk_cfg.get("tp1_rr_ratio", 1.5))) * (1 if direction == "BUY" else -1)
    tp2 = entry + (sl_distance * float(risk_cfg.get("tp2_rr_ratio", 2.5))) * (1 if direction == "BUY" else -1)
    tp3 = entry + (sl_distance * float(risk_cfg.get("tp3_rr_ratio", 4.0))) * (1 if direction == "BUY" else -1)

    reasoning = (
        f"EMA Pullback ({direction}): price pulled back to H1 EMA20 ({h1_ema20:.5f}) "
        f"with H4 EMA {'bullish stack' if bull_stack else 'bearish stack' if bear_stack else 'EMA200 bias'}. "
        f"H1 RSI={h1_rsi:.1f}, MACD {'aligned' if macd_aligned else 'diverging'}. "
        f"M15 candle confirmed. Confidence {confidence}% from {len(confluence)} confluence factor(s)."
    )
    invalidation = (
        f"{'Close below' if direction == 'BUY' else 'Close above'} H1 EMA20 ({h1_ema20:.5f}) "
        f"or H4 EMA200 ({h4_ema200:.5f})."
    )

    logger.info(
        f"EMA Pullback signal: {symbol} {direction} conf={confidence}% "
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
    logger.info(f"EMA Pullback — NO_TRADE ({symbol}): {reason}")
    return SignalResult(
        symbol=symbol,
        direction="NO_TRADE",
        confidence=0,
        reasoning=reason,
        invalidation="",
        confluence_factors=[],
    )
