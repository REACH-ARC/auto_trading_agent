"""Unit tests for backend/indicators.py — IND-08"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from backend.indicators import (
    Bar,
    OHLCVData,
    SRLevel,
    calc_atr,
    calc_emas,
    calc_macd,
    calc_rsi,
    calc_session,
    calc_support_resistance,
    compute_indicators,
    _ema_trend_bias,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(closes: list[float], highs: list[float] | None = None,
             lows: list[float] | None = None) -> pd.DataFrame:
    n = len(closes)
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame(
        {
            "open": closes,
            "high": highs if highs else [c + 0.5 for c in closes],
            "low": lows if lows else [c - 0.5 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        },
        index=dates,
    )


def _make_ohlcv(symbol: str = "EURUSD") -> OHLCVData:
    """Create OHLCVData with simple sine-wave prices."""
    n = 150
    t = np.linspace(0, 4 * np.pi, n)
    closes = (1.1000 + 0.01 * np.sin(t)).tolist()
    highs = [c + 0.0005 for c in closes]
    lows = [c - 0.0005 for c in closes]
    dates = pd.date_range("2024-01-01", periods=n, freq="h")

    bars = [
        Bar(time=dates[i].to_pydatetime(), open=closes[i],
            high=highs[i], low=lows[i], close=closes[i], volume=1000.0)
        for i in range(n)
    ]

    return OHLCVData(
        symbol=symbol,
        timeframes={"M15": bars, "H1": bars, "H4": bars, "D1": bars},
    )


# ---------------------------------------------------------------------------
# IND-02  RSI
# ---------------------------------------------------------------------------

class TestRSI:
    def test_value_in_range(self):
        df = _make_df([float(i) for i in range(1, 51)])  # steadily rising
        rsi = calc_rsi(df)
        assert 0 <= rsi <= 100

    def test_overbought_on_strong_uptrend(self):
        closes = [1.0 + i * 0.1 for i in range(50)]
        df = _make_df(closes)
        assert calc_rsi(df) > 70

    def test_oversold_on_strong_downtrend(self):
        closes = [10.0 - i * 0.1 for i in range(50)]
        df = _make_df(closes)
        assert calc_rsi(df) < 30

    def test_neutral_on_flat_data(self):
        df = _make_df([1.1000] * 50)
        rsi = calc_rsi(df)
        # flat close → no gain, no loss → returns neutral 50
        assert rsi == 50.0


# ---------------------------------------------------------------------------
# IND-03  MACD
# ---------------------------------------------------------------------------

class TestMACD:
    def test_returns_expected_keys(self):
        df = _make_df([float(i) for i in range(1, 60)])
        result = calc_macd(df)
        assert {"value", "signal", "histogram", "is_bullish"} == set(result.keys())

    def test_bullish_on_uptrend(self):
        closes = [1.0 + i * 0.05 for i in range(60)]
        df = _make_df(closes)
        assert calc_macd(df)["is_bullish"] is True

    def test_bearish_on_downtrend(self):
        closes = [10.0 - i * 0.05 for i in range(60)]
        df = _make_df(closes)
        assert calc_macd(df)["is_bullish"] is False

    def test_histogram_equals_macd_minus_signal(self):
        df = _make_df([float(i % 10) for i in range(60)])
        r = calc_macd(df)
        assert abs(r["histogram"] - (r["value"] - r["signal"])) < 1e-6


# ---------------------------------------------------------------------------
# IND-04  ATR
# ---------------------------------------------------------------------------

class TestATR:
    def test_positive(self):
        df = _make_df([1.1] * 50, highs=[1.105] * 50, lows=[1.095] * 50)
        assert calc_atr(df) > 0

    def test_high_volatility_greater_than_low(self):
        low_vol = _make_df([1.1] * 50, highs=[1.101] * 50, lows=[1.099] * 50)
        high_vol = _make_df([1.1] * 50, highs=[1.120] * 50, lows=[1.080] * 50)
        assert calc_atr(high_vol) > calc_atr(low_vol)

    def test_consistent_range(self):
        # ATR should equal candle range on perfectly uniform bars
        df = _make_df([1.1] * 50, highs=[1.110] * 50, lows=[1.090] * 50)
        atr = calc_atr(df)
        assert 0.018 < atr < 0.022  # ~0.020 range


# ---------------------------------------------------------------------------
# IND-05  EMA
# ---------------------------------------------------------------------------

class TestEMA:
    def test_returns_all_periods(self):
        df = _make_df([float(i) for i in range(1, 250)])
        emas = calc_emas(df)
        assert 20 in emas and 50 in emas and 200 in emas

    def test_uptrend_ema_order(self):
        closes = [float(i) for i in range(1, 250)]
        df = _make_df(closes)
        emas = calc_emas(df)
        assert emas[20] > emas[50] > emas[200]

    def test_downtrend_ema_order(self):
        closes = [250.0 - float(i) for i in range(250)]
        df = _make_df(closes)
        emas = calc_emas(df)
        assert emas[20] < emas[50] < emas[200]


class TestEMATrendBias:
    def test_buy_bias(self):
        assert _ema_trend_bias({20: 1.15, 50: 1.10, 200: 1.05}) == "BUY"

    def test_sell_bias(self):
        assert _ema_trend_bias({20: 1.05, 50: 1.10, 200: 1.15}) == "SELL"

    def test_neutral_bias(self):
        assert _ema_trend_bias({20: 1.10, 50: 1.15, 200: 1.05}) == "NEUTRAL"


# ---------------------------------------------------------------------------
# IND-06  Support & Resistance
# ---------------------------------------------------------------------------

class TestSupportResistance:
    def _make_swing_df(self) -> pd.DataFrame:
        # create obvious swing high at bar 10, swing low at bar 20
        closes = [1.1000] * 50
        highs = [1.1005] * 50
        lows = [1.0995] * 50
        highs[10] = 1.1100   # clear swing high
        lows[20] = 1.0900    # clear swing low
        return _make_df(closes, highs=highs, lows=lows)

    def test_returns_list_of_sr_levels(self):
        df = self._make_swing_df()
        levels = calc_support_resistance(df)
        assert isinstance(levels, list)
        assert all(isinstance(l, SRLevel) for l in levels)

    def test_detects_swing_high_as_resistance(self):
        df = self._make_swing_df()
        levels = calc_support_resistance(df)
        resistances = [l for l in levels if l.level_type == "resistance"]
        assert len(resistances) > 0
        prices = [r.price for r in resistances]
        assert any(abs(p - 1.1100) < 0.001 for p in prices)

    def test_detects_swing_low_as_support(self):
        df = self._make_swing_df()
        levels = calc_support_resistance(df)
        supports = [l for l in levels if l.level_type == "support"]
        assert len(supports) > 0
        prices = [s.price for s in supports]
        assert any(abs(p - 1.0900) < 0.001 for p in prices)

    def test_max_10_levels_returned(self):
        df = self._make_swing_df()
        levels = calc_support_resistance(df)
        assert len(levels) <= 10


# ---------------------------------------------------------------------------
# IND-07  Session Detector
# ---------------------------------------------------------------------------

class TestSession:
    def _utc(self, hour: int, minute: int = 0) -> datetime:
        return datetime(2024, 1, 15, hour, minute, 0, tzinfo=timezone.utc)

    def test_london_session_active(self):
        session = calc_session(self._utc(10))  # 10:00 UTC
        assert "London" in session.active_sessions

    def test_new_york_session_active(self):
        session = calc_session(self._utc(15))  # 15:00 UTC
        assert "New York" in session.active_sessions

    def test_london_ny_overlap(self):
        session = calc_session(self._utc(14))  # 14:00 UTC — both open
        assert session.is_overlap is True

    def test_high_liquidity_during_london(self):
        session = calc_session(self._utc(10))
        assert session.is_high_liquidity is True

    def test_low_liquidity_asian_only(self):
        session = calc_session(self._utc(3))   # 03:00 UTC — Tokyo only
        assert session.is_high_liquidity is False

    def test_tokyo_session_active(self):
        session = calc_session(self._utc(3))
        assert "Tokyo" in session.active_sessions


# ---------------------------------------------------------------------------
# Integration: compute_indicators
# ---------------------------------------------------------------------------

class TestComputeIndicators:
    def test_full_pipeline_runs(self):
        data = _make_ohlcv("EURUSD")
        result = compute_indicators(data)
        assert result.symbol == "EURUSD"
        assert "M15" in result.rsi
        assert "H4" in result.macd
        assert "H1" in result.atr
        assert result.trend_bias in ("BUY", "SELL", "NEUTRAL")

    def test_all_timeframes_covered(self):
        data = _make_ohlcv()
        result = compute_indicators(data)
        for tf in ("M15", "H1", "H4", "D1"):
            assert tf in result.rsi
            assert tf in result.macd
            assert tf in result.atr
            assert tf in result.ema

    def test_sr_levels_populated(self):
        data = _make_ohlcv()
        result = compute_indicators(data)
        assert isinstance(result.support_resistance, list)

    def test_session_info_present(self):
        data = _make_ohlcv()
        result = compute_indicators(data)
        assert isinstance(result.session.active_sessions, list)
