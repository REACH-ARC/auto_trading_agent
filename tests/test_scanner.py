"""
Unit tests for backend/scanner.py — Phase 10 Multi-Symbol Scanner.
Tests cover: cache, confluence scoring, scan cycle logic, and scheduled_scan callback.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch, call

import numpy as np
import pytest
import pytest_asyncio

from backend.claude_analyst import SignalResult
from backend.indicators import (
    Bar,
    IndicatorResult,
    OHLCVData,
    SessionInfo,
    SRLevel,
)
from backend.risk_manager import AccountState, RiskDecision
from backend.scanner import (
    ScanResult,
    _SymbolCache,
    _confluence_score,
    scan_cycle,
    scheduled_scan,
    update_symbol_data,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _make_bars(n: int = 150, base: float = 1.1000) -> list[Bar]:
    t = np.linspace(0, 4 * np.pi, n)
    closes = (base + 0.01 * np.sin(t)).tolist()
    dates = [datetime(2024, 1, 15, i % 24, 0, 0, tzinfo=timezone.utc) for i in range(n)]
    return [
        Bar(time=dates[i], open=closes[i], high=closes[i] + 0.0005,
            low=closes[i] - 0.0005, close=closes[i], volume=1000.0)
        for i in range(n)
    ]


def _make_ohlcv(symbol: str = "EURUSD", age_minutes: int = 0) -> OHLCVData:
    received_at = _now() - timedelta(minutes=age_minutes)
    return OHLCVData(
        symbol=symbol,
        timeframes={"M15": _make_bars(), "H1": _make_bars(),
                    "H4": _make_bars(), "D1": _make_bars()},
        received_at=received_at,
    )


def _make_indicators(
    trend_bias: str = "BUY",
    is_high_liquidity: bool = True,
    is_overlap: bool = False,
) -> IndicatorResult:
    return IndicatorResult(
        symbol="EURUSD",
        rsi={"M15": 58.0, "H1": 62.0, "H4": 55.0, "D1": 60.0},
        macd={
            tf: {"value": 0.0002, "signal": 0.0001, "histogram": 0.0001, "is_bullish": True}
            for tf in ("M15", "H1", "H4", "D1")
        },
        atr={"M15": 0.0008, "H1": 0.0015, "H4": 0.0030, "D1": 0.0060},
        ema={tf: {20: 1.1020, 50: 1.1010, 200: 1.0990} for tf in ("M15", "H1", "H4", "D1")},
        support_resistance=[
            SRLevel(price=1.1100, level_type="resistance", strength=5),
            SRLevel(price=1.0900, level_type="support", strength=4),
        ],
        session=SessionInfo(
            active_sessions=["London"],
            is_high_liquidity=is_high_liquidity,
            is_overlap=is_overlap,
        ),
        trend_bias=trend_bias,  # type: ignore[arg-type]
    )


def _make_buy_signal(symbol: str = "EURUSD", confidence: int = 78) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        direction="BUY",
        confidence=confidence,
        reasoning="Test setup",
        invalidation="Close below 1.0975",
        confluence_factors=["EMA alignment", "RSI not overbought", "London session"],
        entry=1.1010,
        sl=1.0980,
        tp1=1.1055,
        tp2=1.1085,
        tp3=1.1130,
        input_tokens=800,
        output_tokens=120,
    )


def _make_no_trade_signal(symbol: str = "EURUSD") -> SignalResult:
    return SignalResult(
        symbol=symbol,
        direction="NO_TRADE",
        confidence=40,
        reasoning="No confluence",
        invalidation="",
        confluence_factors=[],
    )


def _approved_risk() -> RiskDecision:
    return RiskDecision(approved=True, lot_size=0.33, risk_amount=99.0, sl_distance=0.003)


def _rejected_risk(reason: str = "SL too tight") -> RiskDecision:
    return RiskDecision(approved=False, lot_size=0.0, risk_amount=0.0,
                        sl_distance=0.001, reasons=[reason])


@pytest_asyncio.fixture(autouse=True)
async def reset_cache():
    """Clear the module-level _SymbolCache before each test."""
    from backend.scanner import _cache
    await _cache.clear()
    yield
    await _cache.clear()


@pytest.fixture(autouse=True)
def tmp_db(tmp_path: Path):
    """Redirect signal logger DB to temp file for isolation."""
    db_file = tmp_path / "scanner_test.db"
    with patch("backend.signal_logger._DB_PATH", db_file):
        yield db_file


# ---------------------------------------------------------------------------
# _SymbolCache
# ---------------------------------------------------------------------------

class TestSymbolCache:
    @pytest.mark.asyncio
    async def test_update_and_snapshot(self):
        cache = _SymbolCache()
        ohlcv = _make_ohlcv("EURUSD")
        await cache.update(ohlcv)
        snap = await cache.snapshot()
        assert "EURUSD" in snap
        assert snap["EURUSD"].symbol == "EURUSD"

    @pytest.mark.asyncio
    async def test_snapshot_is_copy(self):
        cache = _SymbolCache()
        await cache.update(_make_ohlcv("EURUSD"))
        snap1 = await cache.snapshot()
        snap1["EURUSD"] = None  # mutate the copy
        snap2 = await cache.snapshot()
        assert snap2["EURUSD"] is not None  # original unaffected

    @pytest.mark.asyncio
    async def test_update_overwrites_stale_data(self):
        cache = _SymbolCache()
        old = _make_ohlcv("EURUSD", age_minutes=30)
        fresh = _make_ohlcv("EURUSD", age_minutes=0)
        await cache.update(old)
        await cache.update(fresh)
        snap = await cache.snapshot()
        assert snap["EURUSD"].received_at == fresh.received_at

    @pytest.mark.asyncio
    async def test_multiple_symbols(self):
        cache = _SymbolCache()
        for sym in ("EURUSD", "XAUUSD", "GBPUSD"):
            await cache.update(_make_ohlcv(sym))
        snap = await cache.snapshot()
        assert set(snap.keys()) == {"EURUSD", "XAUUSD", "GBPUSD"}

    @pytest.mark.asyncio
    async def test_clear_empties_cache(self):
        cache = _SymbolCache()
        await cache.update(_make_ohlcv("EURUSD"))
        await cache.clear()
        snap = await cache.snapshot()
        assert snap == {}

    @pytest.mark.asyncio
    async def test_empty_snapshot_on_init(self):
        cache = _SymbolCache()
        snap = await cache.snapshot()
        assert snap == {}


# ---------------------------------------------------------------------------
# update_symbol_data (public API)
# ---------------------------------------------------------------------------

class TestUpdateSymbolData:
    @pytest.mark.asyncio
    async def test_populates_global_cache(self):
        from backend.scanner import _cache
        ohlcv = _make_ohlcv("EURUSD")
        await update_symbol_data(ohlcv)
        snap = await _cache.snapshot()
        assert "EURUSD" in snap

    @pytest.mark.asyncio
    async def test_multiple_symbols_stored_independently(self):
        from backend.scanner import _cache
        for sym in ("EURUSD", "XAUUSD"):
            await update_symbol_data(_make_ohlcv(sym))
        snap = await _cache.snapshot()
        assert "EURUSD" in snap
        assert "XAUUSD" in snap


# ---------------------------------------------------------------------------
# _confluence_score
# ---------------------------------------------------------------------------

class TestConfluenceScore:
    def test_base_score_equals_confidence(self):
        signal = _make_no_trade_signal()
        signal.confluence_factors = []
        indicators = _make_indicators(is_high_liquidity=False, is_overlap=False)
        score = _confluence_score(signal, indicators)
        assert score == pytest.approx(float(signal.confidence))

    def test_each_factor_adds_two_points(self):
        signal = _make_buy_signal(confidence=70)
        signal.confluence_factors = ["a", "b", "c"]
        indicators = _make_indicators(is_high_liquidity=False, is_overlap=False)
        score = _confluence_score(signal, indicators)
        assert score == pytest.approx(70.0 + 3 * 2.0)

    def test_high_liquidity_adds_bonus(self):
        signal = _make_buy_signal(confidence=70)
        signal.confluence_factors = []
        low_liq = _make_indicators(is_high_liquidity=False, is_overlap=False)
        high_liq = _make_indicators(is_high_liquidity=True, is_overlap=False)
        assert _confluence_score(signal, high_liq) > _confluence_score(signal, low_liq)

    def test_overlap_adds_extra_bonus(self):
        signal = _make_buy_signal(confidence=70)
        signal.confluence_factors = []
        no_overlap = _make_indicators(is_high_liquidity=True, is_overlap=False)
        overlap = _make_indicators(is_high_liquidity=True, is_overlap=True)
        assert _confluence_score(signal, overlap) > _confluence_score(signal, no_overlap)

    def test_factors_capped_at_10(self):
        signal = _make_buy_signal(confidence=70)
        signal.confluence_factors = ["f"] * 20  # 20 factors, only 10 should score
        indicators = _make_indicators(is_high_liquidity=False, is_overlap=False)
        score = _confluence_score(signal, indicators)
        assert score == pytest.approx(70.0 + 10 * 2.0)

    def test_higher_confidence_gives_higher_score(self):
        ind = _make_indicators(is_high_liquidity=False, is_overlap=False)
        low = _make_buy_signal(confidence=60)
        low.confluence_factors = []
        high = _make_buy_signal(confidence=90)
        high.confluence_factors = []
        assert _confluence_score(high, ind) > _confluence_score(low, ind)


# ---------------------------------------------------------------------------
# scan_cycle
# ---------------------------------------------------------------------------

class TestScanCycle:
    """Tests for scan_cycle() with all external I/O mocked."""

    @pytest_asyncio.fixture(autouse=True)
    async def _init_db(self, tmp_db):
        from backend.signal_logger import init_db
        await init_db()

    def _patch_analysis(self, signal, indicators, risk, news_blocked=False):
        """Return a context manager stack that patches the full analysis chain."""
        from contextlib import AsyncExitStack, ExitStack
        import contextlib

        async def _fake_check_refresh(sym, dt=None):
            return news_blocked, []

        return (
            patch("backend.scanner.check_and_refresh", side_effect=_fake_check_refresh),
            patch("backend.scanner.compute_indicators", return_value=indicators),
            patch("backend.scanner.analyse", return_value=signal),
            patch("backend.scanner.evaluate", return_value=risk),
        )

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_cached_data(self):
        results = await scan_cycle()
        assert results == []

    @pytest.mark.asyncio
    async def test_skips_stale_data(self):
        # Insert data older than 60 minutes
        stale = _make_ohlcv("EURUSD", age_minutes=90)
        await update_symbol_data(stale)
        results = await scan_cycle(stale_threshold_minutes=60)
        assert results == []

    @pytest.mark.asyncio
    async def test_returns_result_for_actionable_signal(self):
        await update_symbol_data(_make_ohlcv("EURUSD"))
        signal = _make_buy_signal("EURUSD", confidence=80)
        indicators = _make_indicators()
        risk = _approved_risk()

        patches = self._patch_analysis(signal, indicators, risk)
        with patches[0], patches[1], patches[2], patches[3]:
            results = await scan_cycle()

        assert len(results) == 1
        assert results[0].symbol == "EURUSD"
        assert results[0].signal.direction == "BUY"

    @pytest.mark.asyncio
    async def test_no_trade_signal_excluded_from_results(self):
        await update_symbol_data(_make_ohlcv("EURUSD"))
        signal = _make_no_trade_signal("EURUSD")
        indicators = _make_indicators()
        risk = _rejected_risk()

        patches = self._patch_analysis(signal, indicators, risk)
        with patches[0], patches[1], patches[2], patches[3]:
            results = await scan_cycle()

        assert results == []

    @pytest.mark.asyncio
    async def test_news_blocked_symbol_excluded(self):
        await update_symbol_data(_make_ohlcv("EURUSD"))

        patches = self._patch_analysis(
            _make_buy_signal(), _make_indicators(), _approved_risk(),
            news_blocked=True,
        )
        with patches[0], patches[1], patches[2], patches[3]:
            results = await scan_cycle()

        assert results == []

    @pytest.mark.asyncio
    async def test_results_sorted_by_confluence_score_descending(self):
        for sym in ("EURUSD", "XAUUSD", "GBPUSD"):
            await update_symbol_data(_make_ohlcv(sym))

        confidence_map = {"EURUSD": 80, "XAUUSD": 90, "GBPUSD": 70}

        async def _fake_check(*args, **kwargs):
            return False, []

        def _fake_compute(ohlcv):
            return _make_indicators(is_high_liquidity=False, is_overlap=False)

        def _fake_analyse(ohlcv, ind):
            return _make_buy_signal(ohlcv.symbol, confidence=confidence_map[ohlcv.symbol])

        def _fake_evaluate(signal, state, atr):
            return _approved_risk()

        with patch("backend.scanner.check_and_refresh", side_effect=_fake_check), \
             patch("backend.scanner.compute_indicators", side_effect=_fake_compute), \
             patch("backend.scanner.analyse", side_effect=_fake_analyse), \
             patch("backend.scanner.evaluate", side_effect=_fake_evaluate):
            results = await scan_cycle()

        assert len(results) == 3
        scores = [r.confluence_score for r in results]
        assert scores == sorted(scores, reverse=True)
        assert results[0].symbol == "XAUUSD"  # highest confidence = highest score

    @pytest.mark.asyncio
    async def test_indicator_error_skips_symbol(self):
        await update_symbol_data(_make_ohlcv("EURUSD"))

        async def _fake_check(*args, **kwargs):
            return False, []

        with patch("backend.scanner.check_and_refresh", side_effect=_fake_check), \
             patch("backend.scanner.compute_indicators", side_effect=RuntimeError("bad data")):
            results = await scan_cycle()

        assert results == []

    @pytest.mark.asyncio
    async def test_claude_error_skips_symbol(self):
        await update_symbol_data(_make_ohlcv("EURUSD"))
        indicators = _make_indicators()

        async def _fake_check(*args, **kwargs):
            return False, []

        with patch("backend.scanner.check_and_refresh", side_effect=_fake_check), \
             patch("backend.scanner.compute_indicators", return_value=indicators), \
             patch("backend.scanner.analyse", side_effect=RuntimeError("API down")):
            results = await scan_cycle()

        assert results == []

    @pytest.mark.asyncio
    async def test_scan_result_has_signal_id(self):
        await update_symbol_data(_make_ohlcv("EURUSD"))
        signal = _make_buy_signal()
        indicators = _make_indicators()
        risk = _approved_risk()

        patches = self._patch_analysis(signal, indicators, risk)
        with patches[0], patches[1], patches[2], patches[3]:
            results = await scan_cycle()

        assert results[0].signal_id > 0

    @pytest.mark.asyncio
    async def test_fresh_data_respected_over_stale_threshold(self):
        fresh = _make_ohlcv("EURUSD", age_minutes=5)
        await update_symbol_data(fresh)
        signal = _make_buy_signal()
        indicators = _make_indicators()
        risk = _approved_risk()

        patches = self._patch_analysis(signal, indicators, risk)
        with patches[0], patches[1], patches[2], patches[3]:
            results = await scan_cycle(stale_threshold_minutes=60)

        assert len(results) == 1


# ---------------------------------------------------------------------------
# scheduled_scan
# ---------------------------------------------------------------------------

class TestScheduledScan:
    @pytest_asyncio.fixture(autouse=True)
    async def _init_db(self, tmp_db):
        from backend.signal_logger import init_db
        await init_db()

    @pytest.mark.asyncio
    async def test_callback_called_for_top_signal(self):
        await update_symbol_data(_make_ohlcv("EURUSD"))
        signal = _make_buy_signal()
        indicators = _make_indicators()
        risk = _approved_risk()

        callback = AsyncMock()

        async def _fake_check(*args, **kwargs):
            return False, []

        with patch("backend.scanner.check_and_refresh", side_effect=_fake_check), \
             patch("backend.scanner.compute_indicators", return_value=indicators), \
             patch("backend.scanner.analyse", return_value=signal), \
             patch("backend.scanner.evaluate", return_value=risk):
            await scheduled_scan(on_signal_fn=callback)

        callback.assert_called_once()
        result_arg = callback.call_args[0][0]
        assert isinstance(result_arg, ScanResult)
        assert result_arg.symbol == "EURUSD"

    @pytest.mark.asyncio
    async def test_callback_not_called_when_no_signals(self):
        # No data in cache → no signals
        callback = AsyncMock()
        await scheduled_scan(on_signal_fn=callback)
        callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_scheduled_scan_without_callback_does_not_raise(self):
        await update_symbol_data(_make_ohlcv("EURUSD"))
        signal = _make_no_trade_signal()
        indicators = _make_indicators()
        risk = _rejected_risk()

        async def _fake_check(*args, **kwargs):
            return False, []

        with patch("backend.scanner.check_and_refresh", side_effect=_fake_check), \
             patch("backend.scanner.compute_indicators", return_value=indicators), \
             patch("backend.scanner.analyse", return_value=signal), \
             patch("backend.scanner.evaluate", return_value=risk):
            await scheduled_scan(on_signal_fn=None)  # should not raise

    @pytest.mark.asyncio
    async def test_only_top_n_signals_dispatched(self):
        """scanner.top_signals_per_cycle=1 → callback called exactly once even with 3 symbols."""
        for sym in ("EURUSD", "XAUUSD", "GBPUSD"):
            await update_symbol_data(_make_ohlcv(sym))

        async def _fake_check(*args, **kwargs):
            return False, []

        def _fake_compute(ohlcv):
            return _make_indicators(is_high_liquidity=False, is_overlap=False)

        def _fake_analyse(ohlcv, ind):
            return _make_buy_signal(ohlcv.symbol, confidence=78)

        def _fake_evaluate(signal, state, atr):
            return _approved_risk()

        callback = AsyncMock()

        with patch("backend.scanner.check_and_refresh", side_effect=_fake_check), \
             patch("backend.scanner.compute_indicators", side_effect=_fake_compute), \
             patch("backend.scanner.analyse", side_effect=_fake_analyse), \
             patch("backend.scanner.evaluate", side_effect=_fake_evaluate):
            await scheduled_scan(on_signal_fn=callback)

        # settings.scanner.top_signals_per_cycle = 1
        assert callback.call_count == 1

    @pytest.mark.asyncio
    async def test_callback_exception_does_not_abort_scan(self):
        """If on_signal_fn raises, scheduled_scan should catch it and not propagate."""
        await update_symbol_data(_make_ohlcv("EURUSD"))
        signal = _make_buy_signal()
        indicators = _make_indicators()
        risk = _approved_risk()

        async def _bad_callback(result):
            raise RuntimeError("callback failed")

        async def _fake_check(*args, **kwargs):
            return False, []

        with patch("backend.scanner.check_and_refresh", side_effect=_fake_check), \
             patch("backend.scanner.compute_indicators", return_value=indicators), \
             patch("backend.scanner.analyse", return_value=signal), \
             patch("backend.scanner.evaluate", return_value=risk):
            await scheduled_scan(on_signal_fn=_bad_callback)  # must not raise
