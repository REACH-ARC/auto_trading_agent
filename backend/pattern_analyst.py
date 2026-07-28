"""
Pattern Analyst — Candlestick and Chart Pattern Detection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from backend.indicators import OHLCVData

PatternDirection = Literal["BULLISH", "BEARISH", "NEUTRAL"]
PatternStrength  = Literal["STRONG", "MODERATE", "WEAK"]
Confidence       = Literal["HIGH", "MEDIUM", "LOW"]

_CANDLE_TFS = ["M5", "H1", "H4"]
_CHART_TFS  = ["H4", "D1"]
_FVG_TFS    = ["M5", "H1", "H4"]


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class CandlePattern:
    name:      str
    timeframe: str
    direction: PatternDirection
    strength:  PatternStrength


@dataclass
class ChartPattern:
    name:       str
    timeframe:  str
    direction:  PatternDirection
    key_level:  float   # neckline / resistance / support at pattern boundary
    confidence: Confidence


@dataclass
class FVGPattern:
    """Fair Value Gap — 3-candle imbalance zone (ICT concept)."""
    timeframe:  str
    direction:  PatternDirection   # BULLISH or BEARISH
    fvg_high:   float              # upper boundary of the gap
    fvg_low:    float              # lower boundary of the gap
    midpoint:   float              # 50% level (equilibrium / optimal entry)
    mitigated:  bool               # True if price has already passed through 50%+


@dataclass
class PatternResult:
    candle_patterns: list[CandlePattern] = field(default_factory=list)
    chart_patterns:  list[ChartPattern]  = field(default_factory=list)
    fvg_patterns:    list[FVGPattern]    = field(default_factory=list)

    @property
    def has_patterns(self) -> bool:
        return bool(self.candle_patterns or self.chart_patterns or self.fvg_patterns)


# ---------------------------------------------------------------------------
# Candlestick helpers
# ---------------------------------------------------------------------------

def _body(o: float, c: float) -> float:
    return abs(c - o)

def _rng(h: float, l: float) -> float:
    return max(h - l, 1e-10)

def _upper_wick(o: float, c: float, h: float) -> float:
    return h - max(o, c)

def _lower_wick(o: float, c: float, l: float) -> float:
    return min(o, c) - l


# ---------------------------------------------------------------------------
# Candlestick pattern detection
# ---------------------------------------------------------------------------

def _detect_candle_patterns(df: pd.DataFrame, tf: str) -> list[CandlePattern]:
    if len(df) < 3:
        return []

    patterns: list[CandlePattern] = []

    rows = []
    for i in [-3, -2, -1]:
        r = df.iloc[i]
        o, h, l, c = float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])
        body = _body(o, c)
        rng  = _rng(h, l)
        rows.append({
            "o": o, "h": h, "l": l, "c": c,
            "body": body, "rng": rng,
            "br": body / rng,
            "uw": _upper_wick(o, c, h),
            "lw": _lower_wick(o, c, l),
            "bull": c >= o,
        })

    b0, b1, b2 = rows  # oldest → newest; b2 is the latest completed bar

    # ------------------------------------------------------------------ #
    # Single-bar patterns (latest bar b2)                                 #
    # ------------------------------------------------------------------ #

    if b2["br"] < 0.10:
        patterns.append(CandlePattern("Doji", tf, "NEUTRAL", "WEAK"))

    elif b2["br"] > 0.90:
        name = "Bullish Marubozu" if b2["bull"] else "Bearish Marubozu"
        d    = "BULLISH"          if b2["bull"] else "BEARISH"
        patterns.append(CandlePattern(name, tf, d, "STRONG"))

    elif b2["lw"] >= 2.0 * b2["body"] and b2["uw"] <= 0.5 * b2["body"] and b2["br"] < 0.40:
        patterns.append(CandlePattern("Hammer", tf, "BULLISH", "MODERATE"))

    elif b2["uw"] >= 2.0 * b2["body"] and b2["lw"] <= 0.5 * b2["body"] and b2["br"] < 0.40:
        recent_high = max(b0["h"], b1["h"])
        if b2["c"] >= recent_high * 0.998:
            patterns.append(CandlePattern("Shooting Star",    tf, "BEARISH", "MODERATE"))
        else:
            patterns.append(CandlePattern("Inverted Hammer",  tf, "BULLISH", "MODERATE"))

    # ------------------------------------------------------------------ #
    # Two-bar patterns (b1 + b2)                                          #
    # ------------------------------------------------------------------ #

    # Engulfing
    if (not b1["bull"] and b2["bull"]
            and b2["o"] <= b1["c"] and b2["c"] >= b1["o"]
            and b2["body"] > b1["body"]):
        patterns.append(CandlePattern("Bullish Engulfing", tf, "BULLISH", "STRONG"))

    elif (b1["bull"] and not b2["bull"]
            and b2["o"] >= b1["c"] and b2["c"] <= b1["o"]
            and b2["body"] > b1["body"]):
        patterns.append(CandlePattern("Bearish Engulfing", tf, "BEARISH", "STRONG"))

    # Harami (b2 body inside b1 body)
    elif (not b1["bull"] and b2["bull"]
            and b2["o"] >= b1["c"] and b2["c"] <= b1["o"]
            and b2["body"] < 0.50 * b1["body"]):
        patterns.append(CandlePattern("Bullish Harami", tf, "BULLISH", "MODERATE"))

    elif (b1["bull"] and not b2["bull"]
            and b2["o"] <= b1["c"] and b2["c"] >= b1["o"]
            and b2["body"] < 0.50 * b1["body"]):
        patterns.append(CandlePattern("Bearish Harami", tf, "BEARISH", "MODERATE"))

    # Piercing Line / Dark Cloud Cover
    elif (not b1["bull"] and b2["bull"]
            and b2["o"] < b1["c"]
            and b2["c"] > (b1["o"] + b1["c"]) / 2
            and b2["c"] < b1["o"]):
        patterns.append(CandlePattern("Piercing Line",    tf, "BULLISH", "MODERATE"))

    elif (b1["bull"] and not b2["bull"]
            and b2["o"] > b1["c"]
            and b2["c"] < (b1["o"] + b1["c"]) / 2
            and b2["c"] > b1["o"]):
        patterns.append(CandlePattern("Dark Cloud Cover", tf, "BEARISH", "MODERATE"))

    # Tweezer
    tol = b2["rng"] * 0.002
    if abs(b1["l"] - b2["l"]) <= tol and not b1["bull"] and b2["bull"]:
        patterns.append(CandlePattern("Tweezer Bottom", tf, "BULLISH", "MODERATE"))
    elif abs(b1["h"] - b2["h"]) <= tol and b1["bull"] and not b2["bull"]:
        patterns.append(CandlePattern("Tweezer Top",    tf, "BEARISH", "MODERATE"))

    # ------------------------------------------------------------------ #
    # Three-bar patterns (b0 + b1 + b2)                                  #
    # ------------------------------------------------------------------ #

    # Morning Star
    if (not b0["bull"] and b0["br"] > 0.50
            and b1["br"] < 0.30
            and b2["bull"] and b2["br"] > 0.50
            and b2["c"] > (b0["o"] + b0["c"]) / 2):
        patterns.append(CandlePattern("Morning Star", tf, "BULLISH", "STRONG"))

    # Evening Star
    elif (b0["bull"] and b0["br"] > 0.50
            and b1["br"] < 0.30
            and not b2["bull"] and b2["br"] > 0.50
            and b2["c"] < (b0["o"] + b0["c"]) / 2):
        patterns.append(CandlePattern("Evening Star", tf, "BEARISH", "STRONG"))

    # Three White Soldiers
    elif (b0["bull"] and b1["bull"] and b2["bull"]
            and b1["c"] > b0["c"] and b2["c"] > b1["c"]
            and b0["br"] > 0.50 and b1["br"] > 0.50 and b2["br"] > 0.50):
        patterns.append(CandlePattern("Three White Soldiers", tf, "BULLISH", "STRONG"))

    # Three Black Crows
    elif (not b0["bull"] and not b1["bull"] and not b2["bull"]
            and b1["c"] < b0["c"] and b2["c"] < b1["c"]
            and b0["br"] > 0.50 and b1["br"] > 0.50 and b2["br"] > 0.50):
        patterns.append(CandlePattern("Three Black Crows", tf, "BEARISH", "STRONG"))

    # Three Inside Up (Harami b0+b1 + bullish confirmation b2)
    elif (not b0["bull"] and b1["bull"]
            and b1["o"] >= b0["c"] and b1["c"] <= b0["o"]
            and b1["body"] < 0.50 * b0["body"]
            and b2["bull"] and b2["c"] > b0["o"]):
        patterns.append(CandlePattern("Three Inside Up",   tf, "BULLISH", "MODERATE"))

    # Three Inside Down
    elif (b0["bull"] and not b1["bull"]
            and b1["o"] <= b0["c"] and b1["c"] >= b0["o"]
            and b1["body"] < 0.50 * b0["body"]
            and not b2["bull"] and b2["c"] < b0["o"]):
        patterns.append(CandlePattern("Three Inside Down", tf, "BEARISH", "MODERATE"))

    return patterns


# ---------------------------------------------------------------------------
# Chart pattern helpers
# ---------------------------------------------------------------------------

def _find_swings(df: pd.DataFrame, window: int = 3) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return (swing_highs, swing_lows) as lists of (bar_index, price)."""
    highs = df["high"].values
    lows  = df["low"].values
    n = len(highs)
    sh: list[tuple[int, float]] = []
    sl: list[tuple[int, float]] = []

    for i in range(window, n - window):
        if all(highs[i] >= highs[i - j] for j in range(1, window + 1)) and \
           all(highs[i] >= highs[i + j] for j in range(1, window + 1)):
            sh.append((i, float(highs[i])))
        if all(lows[i]  <= lows[i - j]  for j in range(1, window + 1)) and \
           all(lows[i]  <= lows[i + j]  for j in range(1, window + 1)):
            sl.append((i, float(lows[i])))

    return sh, sl


def _slope(pts: list[tuple[int, float]]) -> float:
    """Price change per bar index unit from first to last point."""
    if len(pts) < 2:
        return 0.0
    return (pts[-1][1] - pts[0][1]) / max(pts[-1][0] - pts[0][0], 1)


# ---------------------------------------------------------------------------
# Chart pattern detection
# ---------------------------------------------------------------------------

def _detect_chart_patterns(df: pd.DataFrame, tf: str) -> list[ChartPattern]:
    if len(df) < 20:
        return []

    patterns: list[ChartPattern] = []
    recent = df.tail(80)
    sh, sl = _find_swings(recent, window=3)

    if not sh or not sl:
        return []

    # ------------------------------------------------------------------ #
    # Double Top                                                           #
    # ------------------------------------------------------------------ #
    if len(sh) >= 2:
        h1i, h1p = sh[-2]
        h2i, h2p = sh[-1]
        tol = h1p * 0.003
        if h2i > h1i and abs(h1p - h2p) <= tol:
            troughs = [l for l in sl if h1i < l[0] < h2i]
            if troughs:
                neckline = min(l[1] for l in troughs)
                conf: Confidence = "HIGH" if abs(h1p - h2p) <= tol * 0.5 else "MEDIUM"
                patterns.append(ChartPattern("Double Top", tf, "BEARISH", round(neckline, 6), conf))

    # ------------------------------------------------------------------ #
    # Double Bottom                                                        #
    # ------------------------------------------------------------------ #
    if len(sl) >= 2:
        l1i, l1p = sl[-2]
        l2i, l2p = sl[-1]
        tol = l1p * 0.003
        if l2i > l1i and abs(l1p - l2p) <= tol:
            peaks = [h for h in sh if l1i < h[0] < l2i]
            if peaks:
                neckline = max(h[1] for h in peaks)
                conf = "HIGH" if abs(l1p - l2p) <= tol * 0.5 else "MEDIUM"
                patterns.append(ChartPattern("Double Bottom", tf, "BULLISH", round(neckline, 6), conf))

    # ------------------------------------------------------------------ #
    # Head and Shoulders                                                   #
    # ------------------------------------------------------------------ #
    if len(sh) >= 3:
        ls, hd, rs = sh[-3], sh[-2], sh[-1]  # left shoulder, head, right shoulder
        if hd[1] > ls[1] and hd[1] > rs[1]:
            tol = hd[1] * 0.05
            if abs(ls[1] - rs[1]) <= tol:
                t1 = [l for l in sl if ls[0] < l[0] < hd[0]]
                t2 = [l for l in sl if hd[0] < l[0] < rs[0]]
                if t1 and t2:
                    neckline = (min(l[1] for l in t1) + min(l[1] for l in t2)) / 2
                    patterns.append(ChartPattern(
                        "Head and Shoulders", tf, "BEARISH", round(neckline, 6), "HIGH"
                    ))

    # ------------------------------------------------------------------ #
    # Inverse Head and Shoulders                                           #
    # ------------------------------------------------------------------ #
    if len(sl) >= 3:
        ls, hd, rs = sl[-3], sl[-2], sl[-1]
        if hd[1] < ls[1] and hd[1] < rs[1]:
            tol = abs(hd[1]) * 0.05
            if abs(ls[1] - rs[1]) <= tol:
                p1 = [h for h in sh if ls[0] < h[0] < hd[0]]
                p2 = [h for h in sh if hd[0] < h[0] < rs[0]]
                if p1 and p2:
                    neckline = (max(h[1] for h in p1) + max(h[1] for h in p2)) / 2
                    patterns.append(ChartPattern(
                        "Inverse Head and Shoulders", tf, "BULLISH", round(neckline, 6), "HIGH"
                    ))

    # ------------------------------------------------------------------ #
    # Triangle and Wedge patterns (need 3 recent swings each side)        #
    # ------------------------------------------------------------------ #
    if len(sh) >= 3 and len(sl) >= 3:
        rh = sh[-3:]
        rl = sl[-3:]

        h_falling = rh[-1][1] < rh[-2][1] < rh[-3][1]
        h_rising  = rh[-1][1] > rh[-2][1] > rh[-3][1]
        l_rising  = rl[-1][1] > rl[-2][1] > rl[-3][1]
        l_falling = rl[-1][1] < rl[-2][1] < rl[-3][1]

        h_range = max(h[1] for h in rh) - min(h[1] for h in rh)
        l_range = max(l[1] for l in rl) - min(l[1] for l in rl)
        h_flat  = h_range / rh[-1][1] < 0.005
        l_flat  = l_range / rl[-1][1] < 0.005

        resistance = rh[-1][1]
        support    = rl[-1][1]

        if h_falling and l_rising:
            patterns.append(ChartPattern(
                "Symmetrical Triangle", tf, "NEUTRAL",
                round((resistance + support) / 2, 6), "MEDIUM"
            ))
        elif h_flat and l_rising:
            patterns.append(ChartPattern(
                "Ascending Triangle", tf, "BULLISH", round(resistance, 6), "MEDIUM"
            ))
        elif l_flat and h_falling:
            patterns.append(ChartPattern(
                "Descending Triangle", tf, "BEARISH", round(support, 6), "MEDIUM"
            ))
        elif h_rising and l_rising:
            sh_slope = _slope(rh)
            sl_slope = _slope(rl)
            if sl_slope > sh_slope > 0:   # lows rising faster → converging upward
                patterns.append(ChartPattern(
                    "Rising Wedge", tf, "BEARISH", round(support, 6), "MEDIUM"
                ))
        elif h_falling and l_falling:
            sh_slope = _slope(rh)
            sl_slope = _slope(rl)
            if sh_slope < sl_slope < 0:   # highs falling faster → converging downward
                patterns.append(ChartPattern(
                    "Falling Wedge", tf, "BULLISH", round(resistance, 6), "MEDIUM"
                ))

    return patterns


# ---------------------------------------------------------------------------
# Fair Value Gap detection
# ---------------------------------------------------------------------------

def _detect_fvgs(df: pd.DataFrame, tf: str, lookback: int = 30) -> list[FVGPattern]:
    """
    Detect Fair Value Gaps (3-candle imbalance zones) in recent bars.

    Bullish FVG : candle[i].high < candle[i+2].low  — gap above C1's high
    Bearish FVG : candle[i].low  > candle[i+2].high — gap below C1's low
    """
    if len(df) < 3:
        return []

    patterns: list[FVGPattern] = []
    current_price = float(df.iloc[-1]["close"])
    recent = df.tail(lookback + 2)
    n = len(recent)

    for i in range(n - 2):
        c1_high = float(recent.iloc[i]["high"])
        c1_low  = float(recent.iloc[i]["low"])
        c3_high = float(recent.iloc[i + 2]["high"])
        c3_low  = float(recent.iloc[i + 2]["low"])

        if c1_high < c3_low:
            fvg_low  = c1_high
            fvg_high = c3_low
            mid = (fvg_low + fvg_high) / 2
            # Mitigated once price retraces past the 50% level from above
            patterns.append(FVGPattern(
                timeframe=tf, direction="BULLISH",
                fvg_high=round(fvg_high, 6), fvg_low=round(fvg_low, 6),
                midpoint=round(mid, 6), mitigated=(current_price < mid),
            ))

        elif c1_low > c3_high:
            fvg_high = c1_low
            fvg_low  = c3_high
            mid = (fvg_low + fvg_high) / 2
            # Mitigated once price pushes past the 50% level from below
            patterns.append(FVGPattern(
                timeframe=tf, direction="BEARISH",
                fvg_high=round(fvg_high, 6), fvg_low=round(fvg_low, 6),
                midpoint=round(mid, 6), mitigated=(current_price > mid),
            ))

    # Return only the 3 most recent per direction (most relevant)
    bullish = [p for p in patterns if p.direction == "BULLISH"][-3:]
    bearish = [p for p in patterns if p.direction == "BEARISH"][-3:]
    return bullish + bearish


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_patterns(ohlcv: OHLCVData) -> PatternResult:
    """Detect candlestick, chart, and FVG patterns across all available timeframes."""
    result = PatternResult()

    for tf in _CANDLE_TFS:
        if tf not in ohlcv.timeframes:
            continue
        df = ohlcv.to_df(tf)  # type: ignore[arg-type]
        result.candle_patterns.extend(_detect_candle_patterns(df, tf))

    for tf in _CHART_TFS:
        if tf not in ohlcv.timeframes:
            continue
        df = ohlcv.to_df(tf)  # type: ignore[arg-type]
        result.chart_patterns.extend(_detect_chart_patterns(df, tf))

    for tf in _FVG_TFS:
        if tf not in ohlcv.timeframes:
            continue
        df = ohlcv.to_df(tf)  # type: ignore[arg-type]
        result.fvg_patterns.extend(_detect_fvgs(df, tf))

    return result
