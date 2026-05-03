"""Unit tests for backend/claude_analyst.py — AI-01 through AI-08"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.claude_analyst import (
    SignalResult,
    _check_confidence,
    _extract_json,
    _parse_signal,
    analyse,
    format_market_snapshot,
)
from backend.indicators import (
    Bar,
    IndicatorResult,
    OHLCVData,
    SessionInfo,
    SRLevel,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bars(n: int = 150, base: float = 1.1000) -> list[Bar]:
    t = np.linspace(0, 4 * np.pi, n)
    closes = (base + 0.01 * np.sin(t)).tolist()
    dates = [
        datetime(2024, 1, 15, i % 24, 0, 0, tzinfo=timezone.utc)
        for i in range(n)
    ]
    return [
        Bar(time=dates[i], open=closes[i], high=closes[i] + 0.0005,
            low=closes[i] - 0.0005, close=closes[i], volume=1000.0)
        for i in range(n)
    ]


@pytest.fixture
def ohlcv() -> OHLCVData:
    bars = _make_bars()
    return OHLCVData(
        symbol="EURUSD",
        timeframes={"M15": bars, "H1": bars, "H4": bars, "D1": bars},
        received_at=datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
    )


@pytest.fixture
def indicators(ohlcv) -> IndicatorResult:
    sr_levels = [
        SRLevel(price=1.1100, level_type="resistance", strength=5),
        SRLevel(price=1.0900, level_type="support", strength=4),
    ]
    return IndicatorResult(
        symbol="EURUSD",
        rsi={"M15": 58.0, "H1": 62.0, "H4": 55.0, "D1": 60.0},
        macd={
            tf: {"value": 0.0002, "signal": 0.0001, "histogram": 0.0001, "is_bullish": True}
            for tf in ("M15", "H1", "H4", "D1")
        },
        atr={"M15": 0.0008, "H1": 0.0015, "H4": 0.0030, "D1": 0.0060},
        ema={
            tf: {20: 1.1020, 50: 1.1010, 200: 1.0990}
            for tf in ("M15", "H1", "H4", "D1")
        },
        support_resistance=sr_levels,
        session=SessionInfo(
            active_sessions=["London"],
            is_high_liquidity=True,
            is_overlap=False,
        ),
        trend_bias="BUY",
    )


def _make_claude_response(payload: dict) -> MagicMock:
    """Build a mock anthropic Message object from a dict."""
    content_block = MagicMock()
    content_block.text = json.dumps(payload)

    usage = MagicMock()
    usage.input_tokens = 800
    usage.output_tokens = 120
    usage.cache_read_input_tokens = 600
    usage.cache_creation_input_tokens = 0

    response = MagicMock()
    response.content = [content_block]
    response.usage = usage
    return response


_VALID_BUY_PAYLOAD = {
    "direction": "BUY",
    "entry": 1.1010,
    "sl": 1.0980,
    "tp1": 1.1055,
    "tp2": 1.1085,
    "tp3": 1.1130,
    "confidence": 78,
    "reasoning": "H4 trend is bullish with EMA alignment. Price bounced off 1.0980 support.",
    "invalidation": "Close below 1.0975 on H1.",
    "confluence_factors": ["H4 bullish trend", "RSI not overbought", "London session active"],
}

_VALID_NO_TRADE_PAYLOAD = {
    "direction": "NO_TRADE",
    "entry": None,
    "sl": None,
    "tp1": None,
    "tp2": None,
    "tp3": None,
    "confidence": 40,
    "reasoning": "No clear confluence. Mixed signals across timeframes.",
    "invalidation": "N/A",
    "confluence_factors": [],
}


# ---------------------------------------------------------------------------
# AI-03  format_market_snapshot
# ---------------------------------------------------------------------------

class TestFormatMarketSnapshot:
    def test_contains_symbol(self, ohlcv, indicators):
        text = format_market_snapshot(ohlcv, indicators)
        assert "EURUSD" in text

    def test_contains_all_timeframes(self, ohlcv, indicators):
        text = format_market_snapshot(ohlcv, indicators)
        for tf in ("M15", "H1", "H4", "D1"):
            assert f"[{tf}]" in text

    def test_contains_rsi(self, ohlcv, indicators):
        text = format_market_snapshot(ohlcv, indicators)
        assert "RSI" in text

    def test_contains_macd(self, ohlcv, indicators):
        text = format_market_snapshot(ohlcv, indicators)
        assert "MACD" in text

    def test_contains_sr_levels(self, ohlcv, indicators):
        text = format_market_snapshot(ohlcv, indicators)
        assert "RESISTANCE" in text or "SUPPORT" in text

    def test_contains_session_info(self, ohlcv, indicators):
        text = format_market_snapshot(ohlcv, indicators)
        assert "London" in text

    def test_contains_trend_bias(self, ohlcv, indicators):
        text = format_market_snapshot(ohlcv, indicators)
        assert "BUY" in text


# ---------------------------------------------------------------------------
# AI-05  _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_clean_json(self):
        raw = '{"direction": "BUY", "confidence": 80}'
        result = _extract_json(raw)
        assert result["direction"] == "BUY"

    def test_strips_markdown_fences(self):
        raw = '```json\n{"direction": "SELL"}\n```'
        result = _extract_json(raw)
        assert result["direction"] == "SELL"

    def test_strips_plain_fences(self):
        raw = '```\n{"confidence": 75}\n```'
        result = _extract_json(raw)
        assert result["confidence"] == 75

    def test_json_embedded_in_text(self):
        raw = 'Here is my analysis:\n{"direction": "NO_TRADE", "confidence": 30}\nDone.'
        result = _extract_json(raw)
        assert result["direction"] == "NO_TRADE"

    def test_raises_on_no_json(self):
        with pytest.raises((ValueError, Exception)):
            _extract_json("Sorry, I cannot provide a signal.")


# ---------------------------------------------------------------------------
# AI-05  _parse_signal
# ---------------------------------------------------------------------------

class TestParseSignal:
    def _usage(self) -> dict:
        return {
            "input_tokens": 800, "output_tokens": 120,
            "cache_read_input_tokens": 600, "cache_creation_input_tokens": 0,
        }

    def test_parses_buy_signal(self):
        signal = _parse_signal(_VALID_BUY_PAYLOAD, "EURUSD", self._usage())
        assert signal.direction == "BUY"
        assert signal.entry == pytest.approx(1.1010)
        assert signal.sl == pytest.approx(1.0980)
        assert signal.confidence == 78
        assert signal.symbol == "EURUSD"

    def test_parses_no_trade(self):
        signal = _parse_signal(_VALID_NO_TRADE_PAYLOAD, "XAUUSD", self._usage())
        assert signal.direction == "NO_TRADE"
        assert signal.entry is None

    def test_confidence_clamped_to_100(self):
        payload = {**_VALID_BUY_PAYLOAD, "confidence": 150}
        signal = _parse_signal(payload, "EURUSD", self._usage())
        assert signal.confidence == 100

    def test_confidence_clamped_to_zero(self):
        payload = {**_VALID_BUY_PAYLOAD, "confidence": -10}
        signal = _parse_signal(payload, "EURUSD", self._usage())
        assert signal.confidence == 0

    def test_invalid_direction_becomes_no_trade(self):
        payload = {**_VALID_BUY_PAYLOAD, "direction": "LONG"}
        signal = _parse_signal(payload, "EURUSD", self._usage())
        assert signal.direction == "NO_TRADE"

    def test_token_counts_stored(self):
        signal = _parse_signal(_VALID_BUY_PAYLOAD, "EURUSD", self._usage())
        assert signal.input_tokens == 800
        assert signal.output_tokens == 120
        assert signal.cache_read_tokens == 600

    def test_confluence_factors_list(self):
        signal = _parse_signal(_VALID_BUY_PAYLOAD, "EURUSD", self._usage())
        assert isinstance(signal.confluence_factors, list)
        assert len(signal.confluence_factors) == 3


# ---------------------------------------------------------------------------
# AI-01  SignalResult properties
# ---------------------------------------------------------------------------

class TestSignalResult:
    def _buy_signal(self) -> SignalResult:
        return SignalResult(
            symbol="EURUSD",
            direction="BUY",
            confidence=78,
            reasoning="Test",
            invalidation="Test",
            confluence_factors=[],
            entry=1.1010,
            sl=1.0980,
            tp1=1.1055,
            tp2=1.1085,
            tp3=1.1130,
            input_tokens=800,
            output_tokens=120,
            cache_read_tokens=600,
            cache_write_tokens=0,
        )

    def test_is_actionable_true_for_buy(self):
        assert self._buy_signal().is_actionable is True

    def test_is_actionable_false_for_no_trade(self):
        s = self._buy_signal()
        s.direction = "NO_TRADE"
        assert s.is_actionable is False

    def test_risk_reward_calculation(self):
        # entry=1.1010, sl=1.0980 → risk=0.0030; tp1=1.1055 → reward=0.0045; RR=1.5
        rr = self._buy_signal().risk_reward
        assert rr == pytest.approx(1.5, rel=0.01)

    def test_risk_reward_none_for_no_trade(self):
        s = self._buy_signal()
        s.entry = None
        assert s.risk_reward is None

    def test_estimated_cost_positive(self):
        assert self._buy_signal().estimated_cost_usd > 0

    def test_estimated_cost_lower_with_cache(self):
        s = self._buy_signal()
        s_no_cache = self._buy_signal()
        s_no_cache.cache_read_tokens = 0
        s_no_cache.input_tokens = s.input_tokens + s.cache_read_tokens
        assert s.estimated_cost_usd < s_no_cache.estimated_cost_usd


# ---------------------------------------------------------------------------
# AI-06  _check_confidence
# ---------------------------------------------------------------------------

class TestCheckConfidence:
    def _signal(self, direction="BUY", confidence=78) -> SignalResult:
        return SignalResult(
            symbol="EURUSD", direction=direction, confidence=confidence,
            reasoning="r", invalidation="i", confluence_factors=[],
            entry=1.1010, sl=1.0980, tp1=1.1055, tp2=1.1085, tp3=1.1130,
        )

    def test_high_confidence_passes(self):
        s = _check_confidence(self._signal(confidence=80))
        assert s.direction == "BUY"
        assert s.entry is not None

    def test_low_confidence_filtered_to_no_trade(self):
        s = _check_confidence(self._signal(confidence=50))
        assert s.direction == "NO_TRADE"
        assert s.entry is None

    def test_exactly_at_threshold_passes(self):
        # threshold is 70 per settings.yaml
        s = _check_confidence(self._signal(confidence=70))
        assert s.direction == "BUY"

    def test_no_trade_unaffected_by_filter(self):
        s = _check_confidence(self._signal(direction="NO_TRADE", confidence=10))
        assert s.direction == "NO_TRADE"

    def test_reasoning_updated_when_filtered(self):
        s = _check_confidence(self._signal(confidence=30))
        assert "Filtered" in s.reasoning


# ---------------------------------------------------------------------------
# AI-04 + AI-07  analyse() — full pipeline with mocked Claude
# ---------------------------------------------------------------------------

class TestAnalyse:
    @patch("backend.claude_analyst._get_client")
    def test_buy_signal_end_to_end(self, mock_get_client, ohlcv, indicators):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_claude_response(_VALID_BUY_PAYLOAD)

        signal = analyse(ohlcv, indicators)

        assert signal.symbol == "EURUSD"
        assert signal.direction == "BUY"
        assert signal.confidence == 78
        assert signal.entry == pytest.approx(1.1010)
        mock_client.messages.create.assert_called_once()

    @patch("backend.claude_analyst._get_client")
    def test_no_trade_signal_end_to_end(self, mock_get_client, ohlcv, indicators):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_claude_response(_VALID_NO_TRADE_PAYLOAD)

        signal = analyse(ohlcv, indicators)
        assert signal.direction == "NO_TRADE"
        assert signal.entry is None

    @patch("backend.claude_analyst._get_client")
    def test_low_confidence_filtered(self, mock_get_client, ohlcv, indicators):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        payload = {**_VALID_BUY_PAYLOAD, "confidence": 45}
        mock_client.messages.create.return_value = _make_claude_response(payload)

        signal = analyse(ohlcv, indicators)
        assert signal.direction == "NO_TRADE"

    @patch("backend.claude_analyst._get_client")
    def test_prompt_caching_param_sent(self, mock_get_client, ohlcv, indicators):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _make_claude_response(_VALID_BUY_PAYLOAD)

        analyse(ohlcv, indicators)

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    @patch("backend.claude_analyst._get_client")
    def test_malformed_json_returns_no_trade(self, mock_get_client, ohlcv, indicators):
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="Sorry, I cannot help with that.")]
        usage = MagicMock()
        usage.input_tokens = 100
        usage.output_tokens = 10
        usage.cache_read_input_tokens = 0
        usage.cache_creation_input_tokens = 0
        bad_response.usage = usage
        mock_client.messages.create.return_value = bad_response

        signal = analyse(ohlcv, indicators)
        assert signal.direction == "NO_TRADE"
        assert "Parse error" in signal.reasoning

    @patch("backend.claude_analyst.time_mod.sleep")
    @patch("backend.claude_analyst._get_client")
    def test_retries_on_rate_limit(self, mock_get_client, mock_sleep, ohlcv, indicators):
        import anthropic as _anthropic
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # First 2 calls raise RateLimitError, 3rd succeeds
        rate_err = _anthropic.RateLimitError(
            message="rate limit", response=MagicMock(status_code=429), body={}
        )
        mock_client.messages.create.side_effect = [
            rate_err,
            rate_err,
            _make_claude_response(_VALID_BUY_PAYLOAD),
        ]

        signal = analyse(ohlcv, indicators)
        assert signal.direction == "BUY"
        assert mock_client.messages.create.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("backend.claude_analyst.time_mod.sleep")
    @patch("backend.claude_analyst._get_client")
    def test_raises_after_all_retries_exhausted(self, mock_get_client, mock_sleep, ohlcv, indicators):
        import anthropic as _anthropic
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        rate_err = _anthropic.RateLimitError(
            message="rate limit", response=MagicMock(status_code=429), body={}
        )
        mock_client.messages.create.side_effect = rate_err

        with pytest.raises(RuntimeError, match="Claude API failed after"):
            analyse(ohlcv, indicators)
