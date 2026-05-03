"""Unit tests for backend/mt5_bridge.py — SRV-02"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from backend.mt5_bridge import parse_mt5_message, serialize_signal
from backend.claude_analyst import SignalResult
from backend.risk_manager import RiskDecision


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar(hour: int = 10) -> dict:
    return {
        "time":   f"2024-01-15T{hour:02d}:00:00+00:00",
        "open":   1.1000,
        "high":   1.1010,
        "low":    1.0990,
        "close":  1.1005,
        "volume": 1000,
    }


def _valid_msg(symbol: str = "EURUSD", n_bars: int = 10) -> str:
    bars = [_bar(i) for i in range(n_bars)]
    return json.dumps({
        "symbol": symbol,
        "account": {
            "equity":      10_000.0,
            "balance":     10_000.0,
            "open_trades": 1,
            "daily_pnl":   -50.0,
        },
        "timeframes": {
            "M15": bars,
            "H1":  bars,
            "H4":  bars,
            "D1":  bars,
        },
    })


def _buy_signal() -> SignalResult:
    return SignalResult(
        symbol="EURUSD", direction="BUY", confidence=78,
        reasoning="test", invalidation="test", confluence_factors=["EMA"],
        entry=1.1010, sl=1.0980, tp1=1.1055, tp2=1.1085, tp3=1.1130,
    )


def _approved_risk() -> RiskDecision:
    return RiskDecision(approved=True, lot_size=0.33,
                        risk_amount=99.0, sl_distance=0.003)


# ---------------------------------------------------------------------------
# parse_mt5_message
# ---------------------------------------------------------------------------

class TestParseMt5Message:
    def test_parses_valid_message(self):
        ohlcv, account = parse_mt5_message(_valid_msg())
        assert ohlcv.symbol == "EURUSD"
        assert account.equity == 10_000.0
        assert account.open_trades == 1
        assert account.daily_pnl == -50.0

    def test_all_timeframes_parsed(self):
        ohlcv, _ = parse_mt5_message(_valid_msg())
        assert set(ohlcv.timeframes.keys()) == {"M15", "H1", "H4", "D1"}

    def test_bar_count_matches(self):
        ohlcv, _ = parse_mt5_message(_valid_msg(n_bars=20))
        assert len(ohlcv.timeframes["M15"]) == 20

    def test_bar_fields_parsed(self):
        ohlcv, _ = parse_mt5_message(_valid_msg())
        bar = ohlcv.timeframes["M15"][0]
        assert bar.open   == pytest.approx(1.1000)
        assert bar.high   == pytest.approx(1.1010)
        assert bar.low    == pytest.approx(1.0990)
        assert bar.close  == pytest.approx(1.1005)
        assert bar.volume == pytest.approx(1000.0)

    def test_bar_time_converted_to_utc(self):
        ohlcv, _ = parse_mt5_message(_valid_msg())
        bar = ohlcv.timeframes["M15"][0]
        assert bar.time.tzinfo == timezone.utc

    def test_accepts_bytes_input(self):
        raw = _valid_msg().encode("utf-8")
        ohlcv, _ = parse_mt5_message(raw)
        assert ohlcv.symbol == "EURUSD"

    def test_raises_on_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_mt5_message("not-json")

    def test_raises_on_missing_symbol(self):
        msg = json.dumps({"account": {}, "timeframes": {"M15": [_bar()]}})
        with pytest.raises(ValueError, match="symbol"):
            parse_mt5_message(msg)

    def test_raises_on_missing_timeframes(self):
        msg = json.dumps({"symbol": "EURUSD", "account": {}, "timeframes": {}})
        with pytest.raises(ValueError, match="timeframes"):
            parse_mt5_message(msg)

    def test_received_at_is_utc_aware(self):
        ohlcv, _ = parse_mt5_message(_valid_msg())
        assert ohlcv.received_at.tzinfo is not None

    def test_account_defaults_to_zero_when_missing(self):
        msg = json.dumps({
            "symbol": "EURUSD",
            "timeframes": {"M15": [_bar()]},
        })
        _, account = parse_mt5_message(msg)
        assert account.equity == 0.0
        assert account.open_trades == 0


# ---------------------------------------------------------------------------
# serialize_signal
# ---------------------------------------------------------------------------

class TestSerializeSignal:
    def test_returns_valid_json(self):
        out = serialize_signal(_buy_signal(), _approved_risk(), signal_id=42)
        data = json.loads(out)
        assert isinstance(data, dict)

    def test_contains_required_fields(self):
        data = json.loads(serialize_signal(_buy_signal(), _approved_risk(), 42))
        for key in ("symbol", "direction", "entry", "sl", "tp1", "tp2", "tp3",
                    "confidence", "lot_size", "approved", "reasoning",
                    "signal_id", "timestamp"):
            assert key in data, f"Missing field: {key}"

    def test_signal_id_stored(self):
        data = json.loads(serialize_signal(_buy_signal(), _approved_risk(), 99))
        assert data["signal_id"] == 99

    def test_direction_preserved(self):
        data = json.loads(serialize_signal(_buy_signal(), _approved_risk(), 1))
        assert data["direction"] == "BUY"

    def test_lot_size_from_risk(self):
        data = json.loads(serialize_signal(_buy_signal(), _approved_risk(), 1))
        assert data["lot_size"] == pytest.approx(0.33)

    def test_approved_flag_from_risk(self):
        risk = RiskDecision(approved=False, lot_size=0.0,
                            risk_amount=0.0, sl_distance=0.0)
        data = json.loads(serialize_signal(_buy_signal(), risk, 1))
        assert data["approved"] is False

    def test_risk_reward_included(self):
        data = json.loads(serialize_signal(_buy_signal(), _approved_risk(), 1))
        assert "risk_reward" in data

    def test_no_trade_serializes_nulls(self):
        no_trade = SignalResult(
            symbol="EURUSD", direction="NO_TRADE", confidence=0,
            reasoning="no setup", invalidation="", confluence_factors=[],
        )
        risk = RiskDecision(approved=False, lot_size=0.0,
                            risk_amount=0.0, sl_distance=0.0)
        data = json.loads(serialize_signal(no_trade, risk, 5))
        assert data["entry"] is None
        assert data["direction"] == "NO_TRADE"
