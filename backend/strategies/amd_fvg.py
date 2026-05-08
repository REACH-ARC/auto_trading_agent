"""
AMD + FVG Strategy (ICT / Smart Money Concepts)

Phases:
  Accumulation  — price consolidates, building liquidity pools above/below.
  Manipulation  — a liquidity sweep (stop hunt) breaks a recent swing high/low
                  then closes back inside the range — the "fake-out" move.
  Distribution  — price reverses in the real direction; the impulse candles
                  from this reversal leave a Fair Value Gap (FVG).

Entry:
  Wait for price to retrace INTO the FVG left by the distribution impulse.
  Enter from the FVG zone with SL beyond the sweep extreme.

Logic:
  1. Bias       — H4 EMA200: price above → BUY bias, below → SELL bias.
  2. Sweep      — H1: find a bar within the last sweep_lookback candles whose
                  wick exceeded a recent swing high/low but CLOSED back inside.
  3. FVG        — H1: find a bullish/bearish FVG in the post-sweep reversal bars.
  4. Entry      — M15 price is currently inside the FVG zone.
  5. Confirm    — M15 bar closes in trade direction inside/at-the-edge of FVG.
  6. SL         — beyond the sweep extreme + ATR buffer.
  7. TP         — 1.5R / 2.0R / 2.5R.

Confidence scoring (max 100):
  H4 EMA200 supports bias                  +20
  D1 EMA200 confirms bias                  +15
  AMD sweep detected                       +20
  FVG found in post-sweep move             +15
  Price currently inside FVG               +15
  M15 confirmation candle                  +10
  High-liquidity session active            +10
  H4 MACD aligned                         +10   (capped at 100)
"""
from __future__ import annotations

from loguru import logger

from backend.claude_analyst import SignalResult
from backend.indicators import Bar, IndicatorResult, OHLCVData
from config import settings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_sweep(
    bars: list[Bar],
    bias: str,
    lookback: int,
    ref_lookback: int,
    atr: float,
) -> tuple[float | None, int | None]:
    """
    Scan the last `lookback` H1 bars for a liquidity sweep.

    Bullish sweep: bar.low < recent_swing_low  AND  bar.close > recent_swing_low
    Bearish sweep: bar.high > recent_swing_high AND  bar.close < recent_swing_high

    Returns (sweep_extreme, bar_index) where bar_index is the absolute index
    in `bars` where the sweep occurred, or (None, None) if not found.
    """
    n = len(bars)
    # Don't check the very last bar (still forming) or beyond lookback
    search_start = max(0, n - lookback - 1)
    search_end   = n - 1

    for i in range(search_end - 1, search_start - 1, -1):
        bar = bars[i]
        # Reference range: up to ref_lookback bars before bar[i]
        ref = bars[max(0, i - ref_lookback):i]
        if len(ref) < 4:
            continue

        if bias == "BUY":
            swing_low = min(b.low for b in ref)
            # Wick below swing low, close back above it
            if bar.low < swing_low and bar.close > swing_low:
                return bar.low, i

        else:  # SELL
            swing_high = max(b.high for b in ref)
            if bar.high > swing_high and bar.close < swing_high:
                return bar.high, i

    return None, None


def _find_fvg(
    bars: list[Bar],
    direction: str,
) -> tuple[float, float] | None:
    """
    Scan a slice of bars for the most recent FVG matching `direction`.

    Bullish FVG : bars[i].high < bars[i+2].low
    Bearish FVG : bars[i].low  > bars[i+2].high

    Returns (fvg_low, fvg_high) of the MOST RECENT match, or None.
    """
    n = len(bars)
    if n < 3:
        return None

    # Scan from newest to oldest so we return the most recent FVG
    for i in range(n - 3, -1, -1):
        c1 = bars[i]
        c3 = bars[i + 2]

        if direction == "BUY" and c1.high < c3.low:
            return (c1.high, c3.low)
        if direction == "SELL" and c1.low > c3.high:
            return (c3.high, c1.low)

    return None


# ---------------------------------------------------------------------------
# Main strategy
# ---------------------------------------------------------------------------

def analyse(ohlcv: OHLCVData, indicators: IndicatorResult) -> SignalResult:
    cfg          = settings._yaml.get("strategy", {}).get("amd_fvg", {})
    sweep_lkb    = int(cfg.get("sweep_lookback", 12))
    ref_lkb      = int(cfg.get("ref_lookback", 15))
    min_conf     = int(cfg.get("min_confidence", 65))

    symbol   = ohlcv.symbol
    tfs      = list(ohlcv.timeframes.keys())
    m15_bars = ohlcv.timeframes.get("M15") or ohlcv.timeframes[tfs[0]]
    h1_bars  = ohlcv.timeframes.get("H1",  [])

    if not m15_bars:
        return _no_trade(symbol, "No M15 data available")
    if len(h1_bars) < 20:
        return _no_trade(symbol, f"Insufficient H1 data ({len(h1_bars)} bars)")

    current_bar   = m15_bars[-1]
    current_price = float(current_bar.close)

    h1_atr  = indicators.atr.get("H1")  or indicators.atr.get("H4", 1.0)
    h4_atr  = indicators.atr.get("H4",  h1_atr)

    # ------------------------------------------------------------------ #
    # 1. Trend bias — H4 EMA200                                           #
    # ------------------------------------------------------------------ #
    h4_emas   = indicators.ema.get("H4", {})
    h4_ema200 = h4_emas.get(200, current_price)
    h4_ema50  = h4_emas.get(50,  current_price)
    h4_ema20  = h4_emas.get(20,  current_price)

    if current_price > h4_ema200:
        bias = "BUY"
    elif current_price < h4_ema200:
        bias = "SELL"
    else:
        return _no_trade(symbol, "Price at H4 EMA200 — no directional bias")

    # ------------------------------------------------------------------ #
    # 2. Liquidity sweep (Manipulation phase)                             #
    # ------------------------------------------------------------------ #
    sweep_extreme, sweep_idx = _find_sweep(h1_bars, bias, sweep_lkb, ref_lkb, h1_atr)
    if sweep_extreme is None or sweep_idx is None:
        return _no_trade(symbol, f"No AMD liquidity sweep found in last {sweep_lkb} H1 bars")

    # ------------------------------------------------------------------ #
    # 3. FVG in the post-sweep distribution move                          #
    # ------------------------------------------------------------------ #
    post_sweep_bars = h1_bars[sweep_idx + 1:]
    if len(post_sweep_bars) < 3:
        return _no_trade(symbol, "Not enough bars after sweep to form FVG")

    fvg = _find_fvg(post_sweep_bars, bias)
    if fvg is None:
        return _no_trade(symbol, "No FVG found in post-sweep distribution bars")

    fvg_low, fvg_high = fvg
    fvg_mid = (fvg_low + fvg_high) / 2

    # ------------------------------------------------------------------ #
    # 4. Entry condition — current M15 price inside the FVG               #
    # ------------------------------------------------------------------ #
    price_in_fvg = fvg_low <= current_price <= fvg_high

    if not price_in_fvg:
        direction_word = "above" if current_price > fvg_high else "below"
        return _no_trade(
            symbol,
            f"Price {current_price:.5f} is {direction_word} FVG zone "
            f"({fvg_low:.5f}–{fvg_high:.5f}) — waiting for retracement"
        )

    # ------------------------------------------------------------------ #
    # 5. M15 candle confirmation                                          #
    # ------------------------------------------------------------------ #
    candle_confirms = (
        (bias == "BUY"  and current_bar.close > current_bar.open) or
        (bias == "SELL" and current_bar.close < current_bar.open)
    )

    # ------------------------------------------------------------------ #
    # 6. Supporting confluence                                             #
    # ------------------------------------------------------------------ #
    d1_emas   = indicators.ema.get("D1", {})
    d1_ema200 = d1_emas.get(200, current_price)
    d1_aligned = (bias == "BUY" and current_price > d1_ema200) or \
                 (bias == "SELL" and current_price < d1_ema200)

    h4_macd      = indicators.macd.get("H4", {})
    h4_macd_bull = h4_macd.get("is_bullish", False)
    macd_aligned = (bias == "BUY" and h4_macd_bull) or (bias == "SELL" and not h4_macd_bull)

    session_active = indicators.session.is_high_liquidity

    # ------------------------------------------------------------------ #
    # Confidence scoring                                                   #
    # ------------------------------------------------------------------ #
    confidence = 0
    confluence: list[str] = []

    # Bias (required)
    confidence += 20
    confluence.append(f"H4 EMA200 {'bullish' if bias == 'BUY' else 'bearish'} bias")

    if d1_aligned:
        confidence += 15
        confluence.append(f"D1 EMA200 confirms {bias} bias")

    # AMD sweep detected (already verified above)
    confidence += 20
    confluence.append(
        f"Liquidity sweep at {sweep_extreme:.5f} "
        f"({len(h1_bars) - sweep_idx - 1} H1 bars ago)"
    )

    # FVG found (already verified above)
    confidence += 15
    confluence.append(
        f"FVG zone {fvg_low:.5f}–{fvg_high:.5f} "
        f"(mid={fvg_mid:.5f}) on H1 post-sweep"
    )

    # Price in FVG (already verified above)
    confidence += 15
    confluence.append(
        f"Price {current_price:.5f} inside FVG "
        f"({'above' if current_price > fvg_mid else 'below'} midpoint)"
    )

    if candle_confirms:
        confidence += 10
        confluence.append(
            f"M15 candle confirms {bias} "
            f"(close={current_bar.close:.5f} {'>' if bias == 'BUY' else '<'} open={current_bar.open:.5f})"
        )

    if session_active:
        confidence += 10
        confluence.append(f"High-liquidity session: {', '.join(indicators.session.active_sessions)}")

    if macd_aligned:
        confidence += 10
        confluence.append(f"H4 MACD aligned with {bias}")

    confidence = min(confidence, 100)

    if confidence < min_conf:
        return _no_trade(
            symbol,
            f"AMD+FVG setup found but confidence {confidence}% < {min_conf}% — "
            f"factors: {len(confluence)}"
        )

    # ------------------------------------------------------------------ #
    # 7. Entry, SL, TP                                                    #
    # ------------------------------------------------------------------ #
    entry = current_price

    # SL beyond the sweep extreme + ATR buffer
    atr_buffer = h1_atr * 0.3
    if bias == "BUY":
        sl = sweep_extreme - atr_buffer
    else:
        sl = sweep_extreme + atr_buffer

    if bias == "BUY" and sl >= entry:
        return _no_trade(symbol, f"Invalid SL for BUY: sl={sl:.5f} >= entry={entry:.5f}")
    if bias == "SELL" and sl <= entry:
        return _no_trade(symbol, f"Invalid SL for SELL: sl={sl:.5f} <= entry={entry:.5f}")

    sl_distance = abs(entry - sl)
    if sl_distance < 1e-8:
        return _no_trade(symbol, "SL distance too small — cannot calculate R:R")

    risk_cfg = settings.risk
    mult = 1 if bias == "BUY" else -1
    tp1 = entry + mult * sl_distance * float(risk_cfg.get("tp1_rr_ratio", 1.5))
    tp2 = entry + mult * sl_distance * float(risk_cfg.get("tp2_rr_ratio", 2.0))
    tp3 = entry + mult * sl_distance * float(risk_cfg.get("tp3_rr_ratio", 2.5))

    reasoning = (
        f"AMD+FVG ({bias}): liquidity sweep at {sweep_extreme:.5f} took stop-side liquidity "
        f"({len(h1_bars) - sweep_idx - 1} H1 bars ago). Distribution impulse left FVG "
        f"{fvg_low:.5f}–{fvg_high:.5f}. Price has retraced into the FVG at {current_price:.5f}. "
        f"Confidence {confidence}% from {len(confluence)} confluence factor(s)."
    )
    invalidation = (
        f"{'Close below' if bias == 'BUY' else 'Close above'} sweep extreme "
        f"{sweep_extreme:.5f}, or FVG fully mitigated without continuation."
    )

    logger.info(
        f"AMD+FVG signal: {symbol} {bias} conf={confidence}% "
        f"entry={entry:.5f} sl={sl:.5f} fvg={fvg_low:.5f}-{fvg_high:.5f}"
    )

    return SignalResult(
        symbol=symbol,
        direction=bias,  # type: ignore[arg-type]
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
    logger.info(f"AMD+FVG — NO_TRADE ({symbol}): {reason}")
    return SignalResult(
        symbol=symbol,
        direction="NO_TRADE",
        confidence=0,
        reasoning=reason,
        invalidation="",
        confluence_factors=[],
    )
