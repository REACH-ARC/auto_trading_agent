"""
Support/Resistance Bounce + RSI Confirmation Strategy.

Logic:
  1. Find the nearest S/R level to current price (from indicators.support_resistance).
  2. Price must be within proximity_atr × ATR of that level, approaching from the
     correct side: at/below resistance for SELL, at/above support for BUY.
  3. RSI confirmation (H1):
       BUY  at support    — RSI < rsi_oversold  (price oversold near support)
       SELL at resistance — RSI > rsi_overbought (price overbought near resistance)
  4. Candle confirmation — M15 bar must close in trade direction (bullish for BUY,
     bearish for SELL) to confirm the rejection is starting.
  5. S/R level strength filter — level must have been touched >= min_strength times.
  6. H4 trend alignment is a bonus (not required).
  7. MACD H1 histogram aligning with the bounce direction adds confidence.

Confidence scoring (max 100):
  Price within 0.3×ATR of level (very close)   +30
  Price within proximity×ATR of level (close)  +20
  RSI confirms (oversold/overbought)            +20
  Candle confirms rejection direction           +10
  Level strength >= 3                           +15
  Level strength == 2                           +10
  H4 trend aligned with bounce                 +20
  H1 MACD turning in bounce direction          +15   (capped at 100)
"""
from __future__ import annotations

from loguru import logger

from backend.claude_analyst import SignalResult
from backend.indicators import IndicatorResult, OHLCVData, SRLevel
from config import settings


def analyse(ohlcv: OHLCVData, indicators: IndicatorResult) -> SignalResult:
    cfg = settings._yaml.get("strategy", {}).get("sr_bounce", {})
    proximity_atr: float  = float(cfg.get("proximity_atr", 0.5))
    rsi_oversold: float   = float(cfg.get("rsi_oversold", 30))
    rsi_overbought: float = float(cfg.get("rsi_overbought", 70))
    min_strength: int     = int(cfg.get("min_strength", 2))
    min_conf: int         = int(cfg.get("min_confidence", 65))

    symbol = ohlcv.symbol
    tfs    = list(ohlcv.timeframes.keys())
    m15_bars = ohlcv.timeframes.get("M15") or ohlcv.timeframes[tfs[0]]
    current_bar  = m15_bars[-1]
    current_price: float = current_bar.close

    sr_levels = indicators.support_resistance
    if not sr_levels:
        return _no_trade(symbol, "No S/R levels available in indicator data")

    atr    = indicators.atr.get("H1") or indicators.atr.get("H4") or indicators.atr.get(tfs[0], 1.0)
    h1_rsi = indicators.rsi.get("H1", indicators.rsi.get(tfs[0], 50.0))

    # ------------------------------------------------------------------ #
    # 1. Find nearest S/R level that meets strength requirement
    # ------------------------------------------------------------------ #
    qualified = [lvl for lvl in sr_levels if lvl.strength >= min_strength]
    if not qualified:
        return _no_trade(
            symbol,
            f"No S/R levels with strength >= {min_strength} (found {len(sr_levels)} total)"
        )

    nearest: SRLevel = min(qualified, key=lambda lvl: abs(lvl.price - current_price))
    distance  = abs(current_price - nearest.price)
    threshold = proximity_atr * atr

    # ------------------------------------------------------------------ #
    # 2. Proximity + directional side check
    # ------------------------------------------------------------------ #
    if distance > threshold:
        return _no_trade(
            symbol,
            f"Nearest S/R level {nearest.price:.5f} ({nearest.level_type}) too far: "
            f"distance={distance:.5f} > {proximity_atr}×ATR ({threshold:.5f})"
        )

    # Ensure price is approaching from the correct side:
    #   support → price should be at or slightly below (not far above — that's a breakout)
    #   resistance → price should be at or slightly above (not far below — approaching only)
    side_buffer = 0.3 * atr
    if nearest.level_type == "support" and current_price > nearest.price + side_buffer:
        return _no_trade(
            symbol,
            f"Price {current_price:.5f} too far above support {nearest.price:.5f} — looks like breakout not bounce"
        )
    if nearest.level_type == "resistance" and current_price < nearest.price - side_buffer:
        return _no_trade(
            symbol,
            f"Price {current_price:.5f} too far below resistance {nearest.price:.5f} — approaching only, wait for touch"
        )

    # ------------------------------------------------------------------ #
    # 3. Determine bounce direction + RSI check
    # ------------------------------------------------------------------ #
    if nearest.level_type == "support":
        direction = "BUY"
        rsi_ok = h1_rsi <= rsi_oversold
    else:
        direction = "SELL"
        rsi_ok = h1_rsi >= rsi_overbought

    if not rsi_ok:
        return _no_trade(
            symbol,
            f"RSI {h1_rsi:.1f} not confirming {direction} bounce at "
            f"{nearest.level_type} {nearest.price:.5f} "
            f"(need {'<=' + str(rsi_oversold) if direction == 'BUY' else '>=' + str(rsi_overbought)})"
        )

    # ------------------------------------------------------------------ #
    # 4. Candle confirmation — M15 bar must close in trade direction
    # ------------------------------------------------------------------ #
    candle_bullish = current_bar.close > current_bar.open
    candle_ok = (direction == "BUY" and candle_bullish) or \
                (direction == "SELL" and not candle_bullish)

    if not candle_ok:
        return _no_trade(
            symbol,
            f"M15 candle not confirming {direction} rejection at {nearest.level_type} "
            f"(close={current_bar.close:.3f}, open={current_bar.open:.3f})"
        )

    # ------------------------------------------------------------------ #
    # 5. Confidence scoring
    # ------------------------------------------------------------------ #
    confidence = 0
    confluence: list[str] = []

    if distance <= 0.3 * atr:
        confidence += 30
        confluence.append(f"Price very close to {nearest.level_type} {nearest.price:.5f} (within 0.3×ATR)")
    else:
        confidence += 20
        confluence.append(f"Price near {nearest.level_type} {nearest.price:.5f} (within {proximity_atr}×ATR)")

    confidence += 20
    confluence.append(f"H1 RSI {h1_rsi:.1f} confirms {'oversold' if direction == 'BUY' else 'overbought'} at level")

    confidence += 10
    confluence.append(
        f"M15 candle confirms {'bullish' if direction == 'BUY' else 'bearish'} rejection"
    )

    if nearest.strength >= 3:
        confidence += 15
        confluence.append(f"Strong S/R level ({nearest.strength} touches)")
    else:
        confidence += 10
        confluence.append(f"S/R level ({nearest.strength} touches)")

    h4_emas   = indicators.ema.get("H4", {})
    h4_ema200 = h4_emas.get(200, current_price)
    h4_aligned = (direction == "BUY" and current_price > h4_ema200) or \
                 (direction == "SELL" and current_price < h4_ema200)
    if h4_aligned:
        confidence += 20
        confluence.append("H4 EMA200 supports bounce direction")

    h1_macd = indicators.macd.get("H1", {})
    hist = h1_macd.get("histogram", 0.0)
    macd_turning = (direction == "BUY" and hist > 0) or (direction == "SELL" and hist < 0)
    if macd_turning:
        confidence += 15
        confluence.append(f"H1 MACD histogram supports {direction}")

    confidence = min(confidence, 100)

    if confidence < min_conf:
        return _no_trade(
            symbol,
            f"Confidence {confidence}% below minimum {min_conf}% "
            f"({len(confluence)} confluence factor(s))"
        )

    # ------------------------------------------------------------------ #
    # 6. Entry, SL, TP
    # ------------------------------------------------------------------ #
    entry = current_price

    if direction == "BUY":
        sl = nearest.price - 0.5 * atr
    else:
        sl = nearest.price + 0.5 * atr

    if direction == "BUY" and sl >= entry:
        return _no_trade(symbol, f"Invalid SL for BUY: sl={sl:.5f} >= entry={entry:.5f}")
    if direction == "SELL" and sl <= entry:
        return _no_trade(symbol, f"Invalid SL for SELL: sl={sl:.5f} <= entry={entry:.5f}")

    sl_distance = abs(entry - sl)
    if sl_distance < 1e-8:
        return _no_trade(symbol, "SL distance is zero — price is exactly at level")

    risk_cfg = settings.risk
    mult = 1 if direction == "BUY" else -1
    tp1 = entry + mult * sl_distance * float(risk_cfg.get("tp1_rr_ratio", 1.5))
    tp2 = entry + mult * sl_distance * float(risk_cfg.get("tp2_rr_ratio", 2.5))
    tp3 = entry + mult * sl_distance * float(risk_cfg.get("tp3_rr_ratio", 4.0))

    reasoning = (
        f"S/R Bounce ({direction}): price {current_price:.5f} at "
        f"{nearest.level_type} {nearest.price:.5f} (strength={nearest.strength}, "
        f"distance={distance:.5f}). H1 RSI={h1_rsi:.1f} {'oversold' if direction == 'BUY' else 'overbought'}. "
        f"M15 candle confirmed. Confidence {confidence}% from {len(confluence)} factor(s)."
    )
    invalidation = (
        f"Close {'below' if direction == 'BUY' else 'above'} {nearest.level_type} level "
        f"{nearest.price:.5f} by more than 0.5×ATR ({0.5 * atr:.5f})."
    )

    logger.info(
        f"S/R Bounce signal: {symbol} {direction} conf={confidence}% "
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
    logger.info(f"S/R Bounce — NO_TRADE ({symbol}): {reason}")
    return SignalResult(
        symbol=symbol,
        direction="NO_TRADE",
        confidence=0,
        reasoning=reason,
        invalidation="",
        confluence_factors=[],
    )
