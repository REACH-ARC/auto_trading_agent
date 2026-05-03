from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from typing import Literal

import numpy as np
import pandas as pd

from config import settings

# ---------------------------------------------------------------------------
# IND-01  Data types
# ---------------------------------------------------------------------------

Timeframe = Literal["M15", "H1", "H4", "D1"]
Direction = Literal["BUY", "SELL", "NEUTRAL"]


@dataclass
class Bar:
    time: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class OHLCVData:
    """Multi-timeframe OHLCV payload received from MT5."""
    symbol: str
    timeframes: dict[Timeframe, list[Bar]]  # e.g. {"M15": [...], "H1": [...]}
    received_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_df(self, tf: Timeframe) -> pd.DataFrame:
        bars = self.timeframes[tf]
        return pd.DataFrame(
            {
                "time": [b.time for b in bars],
                "open": [b.open for b in bars],
                "high": [b.high for b in bars],
                "low": [b.low for b in bars],
                "close": [b.close for b in bars],
                "volume": [b.volume for b in bars],
            }
        ).set_index("time")


@dataclass
class SRLevel:
    price: float
    level_type: Literal["support", "resistance"]
    strength: int  # number of touches / bounces


@dataclass
class SessionInfo:
    active_sessions: list[str]
    is_high_liquidity: bool  # True when London or NY is active
    is_overlap: bool         # True when London + NY overlap (12:00-16:00 UTC)


@dataclass
class IndicatorResult:
    """All computed indicators for a single symbol, all timeframes."""
    symbol: str
    rsi: dict[Timeframe, float]
    macd: dict[Timeframe, dict]          # {"value", "signal", "histogram"}
    atr: dict[Timeframe, float]
    ema: dict[Timeframe, dict[int, float]]  # {tf: {period: value}}
    support_resistance: list[SRLevel]
    session: SessionInfo
    trend_bias: Direction                 # derived from EMA alignment on H4


# ---------------------------------------------------------------------------
# IND-02  RSI
# ---------------------------------------------------------------------------

def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

    last_gain = float(avg_gain.iloc[-1])
    last_loss = float(avg_loss.iloc[-1])

    if np.isnan(last_gain) or np.isnan(last_loss):
        return 50.0  # insufficient data
    if last_loss == 0:
        return 100.0 if last_gain > 0 else 50.0  # pure uptrend or flat
    return round(100 - (100 / (1 + last_gain / last_loss)), 2)


def calc_rsi(df: pd.DataFrame, period: int | None = None) -> float:
    period = period or settings.indicators["rsi_period"]
    return _rsi(df["close"], period)


# ---------------------------------------------------------------------------
# IND-03  MACD
# ---------------------------------------------------------------------------

def calc_macd(df: pd.DataFrame) -> dict:
    cfg = settings.indicators
    fast = cfg["macd_fast"]
    slow = cfg["macd_slow"]
    signal_period = cfg["macd_signal"]

    close = df["close"]
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return {
        "value": round(float(macd_line.iloc[-1]), 6),
        "signal": round(float(signal_line.iloc[-1]), 6),
        "histogram": round(float(histogram.iloc[-1]), 6),
        "is_bullish": float(histogram.iloc[-1]) > 0,
    }


# ---------------------------------------------------------------------------
# IND-04  ATR
# ---------------------------------------------------------------------------

def calc_atr(df: pd.DataFrame, period: int | None = None) -> float:
    period = period or settings.indicators["atr_period"]
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = tr.ewm(com=period - 1, min_periods=period).mean()
    return round(float(atr.iloc[-1]), 6)


# ---------------------------------------------------------------------------
# IND-05  EMA
# ---------------------------------------------------------------------------

def calc_emas(df: pd.DataFrame) -> dict[int, float]:
    periods: list[int] = settings.indicators["ema_periods"]
    result: dict[int, float] = {}
    close = df["close"]
    for p in periods:
        ema = close.ewm(span=p, adjust=False).mean()
        result[p] = round(float(ema.iloc[-1]), 6)
    return result


def _ema_trend_bias(emas: dict[int, float]) -> Direction:
    """EMA 20 > EMA 50 > EMA 200 → BUY bias, reversed → SELL, else NEUTRAL."""
    e20 = emas.get(20, 0)
    e50 = emas.get(50, 0)
    e200 = emas.get(200, 0)
    if e20 > e50 > e200:
        return "BUY"
    if e20 < e50 < e200:
        return "SELL"
    return "NEUTRAL"


# ---------------------------------------------------------------------------
# IND-06  Support & Resistance
# ---------------------------------------------------------------------------

def calc_support_resistance(df: pd.DataFrame) -> list[SRLevel]:
    """Detect swing-high (resistance) and swing-low (support) levels."""
    lookback: int = settings.indicators["sr_lookback_bars"]
    window = df.tail(lookback).copy()

    highs = window["high"].values
    lows = window["low"].values
    levels: list[SRLevel] = []

    for i in range(2, len(window) - 2):
        # swing high
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] \
                and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            price = round(float(highs[i]), 6)
            strength = _count_touches(window["high"], price)
            levels.append(SRLevel(price=price, level_type="resistance", strength=strength))

        # swing low
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] \
                and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            price = round(float(lows[i]), 6)
            strength = _count_touches(window["low"], price)
            levels.append(SRLevel(price=price, level_type="support", strength=strength))

    # deduplicate levels within 0.1% of each other, keep strongest
    levels = _deduplicate_levels(levels, tolerance_pct=0.001)

    # sort by strength descending, return top 10
    return sorted(levels, key=lambda x: x.strength, reverse=True)[:10]


def _count_touches(series: pd.Series, price: float, tolerance_pct: float = 0.001) -> int:
    band = price * tolerance_pct
    return int(((series - price).abs() <= band).sum())


def _deduplicate_levels(levels: list[SRLevel], tolerance_pct: float) -> list[SRLevel]:
    if not levels:
        return levels
    levels = sorted(levels, key=lambda x: x.price)

    # Group all levels within tolerance_pct of each other, keep the strongest per group
    deduped: list[SRLevel] = []
    group: list[SRLevel] = [levels[0]]

    for lvl in levels[1:]:
        ref_price = group[0].price
        if abs(lvl.price - ref_price) / ref_price < tolerance_pct:
            group.append(lvl)
        else:
            deduped.append(max(group, key=lambda x: x.strength))
            group = [lvl]

    deduped.append(max(group, key=lambda x: x.strength))
    return deduped


# ---------------------------------------------------------------------------
# IND-07  Market Session Detector
# ---------------------------------------------------------------------------

def calc_session(dt: datetime | None = None) -> SessionInfo:
    """Determine which trading sessions are active for a given UTC time."""
    if dt is None:
        dt = datetime.now(timezone.utc)

    utc_time = dt.time()
    sessions_cfg: dict = settings.sessions

    def _in_session(open_str: str, close_str: str, t: time) -> bool:
        open_t = time.fromisoformat(open_str)
        close_t = time.fromisoformat(close_str)
        if open_t <= close_t:
            return open_t <= t <= close_t
        # overnight session (e.g. Sydney 21:00 → 06:00)
        return t >= open_t or t <= close_t

    active: list[str] = []
    for name in ("sydney", "tokyo", "london", "new_york"):
        cfg = sessions_cfg.get(name, {})
        if cfg and _in_session(cfg["open"], cfg["close"], utc_time):
            active.append(name.replace("_", " ").title())

    is_overlap = "London" in active and "New York" in active
    is_high_liquidity = "London" in active or "New York" in active

    return SessionInfo(
        active_sessions=active,
        is_high_liquidity=is_high_liquidity,
        is_overlap=is_overlap,
    )


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute_indicators(data: OHLCVData) -> IndicatorResult:
    """Run all indicators across all timeframes and return IndicatorResult."""
    tfs: list[Timeframe] = list(data.timeframes.keys())  # type: ignore

    rsi: dict[Timeframe, float] = {}
    macd_all: dict[Timeframe, dict] = {}
    atr: dict[Timeframe, float] = {}
    ema_all: dict[Timeframe, dict[int, float]] = {}
    dfs: dict[Timeframe, pd.DataFrame] = {}

    for tf in tfs:
        df = data.to_df(tf)
        dfs[tf] = df
        rsi[tf] = calc_rsi(df)
        macd_all[tf] = calc_macd(df)
        atr[tf] = calc_atr(df)
        ema_all[tf] = calc_emas(df)

    # S/R from H4 (best balance of noise vs significance) — reuse cached df
    sr_tf: Timeframe = "H4" if "H4" in tfs else tfs[-1]
    sr_levels = calc_support_resistance(dfs[sr_tf])

    # Use data.received_at so session reflects when MT5 sent the data, not processing time
    session = calc_session(data.received_at)

    # Trend bias from H4 EMAs (fallback to last available tf)
    h4_emas = ema_all.get("H4", ema_all.get(tfs[-1], {}))
    trend_bias = _ema_trend_bias(h4_emas)

    return IndicatorResult(
        symbol=data.symbol,
        rsi=rsi,
        macd=macd_all,
        atr=atr,
        ema=ema_all,
        support_resistance=sr_levels,
        session=session,
        trend_bias=trend_bias,
    )
