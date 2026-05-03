"""Unit tests for backend/signal_logger.py — LOG-01 through LOG-05"""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from backend.claude_analyst import SignalResult
from backend.risk_manager import RiskDecision
from backend.signal_logger import (
    SignalRecord,
    SignalStats,
    get_pending_signals,
    get_signal,
    get_stats,
    init_db,
    log_signal,
    update_outcome,
)


# ---------------------------------------------------------------------------
# Fixtures — use a temp DB per test to guarantee isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def tmp_db(tmp_path: Path):
    """Redirect DB_PATH to a fresh temp file for every test."""
    db_file = tmp_path / "test_signals.db"
    with patch("backend.signal_logger._DB_PATH", db_file):
        yield db_file


@pytest_asyncio.fixture
async def db(tmp_db):
    """Initialised (schema-ready) temp DB."""
    await init_db()
    return tmp_db


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _buy_signal(
    symbol: str = "EURUSD",
    confidence: int = 78,
    direction: str = "BUY",
) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        reasoning="H4 bullish confluence",
        invalidation="Close below 1.0975",
        confluence_factors=["EMA alignment", "RSI not overbought"],
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


def _no_trade_signal(symbol: str = "EURUSD") -> SignalResult:
    return SignalResult(
        symbol=symbol,
        direction="NO_TRADE",
        confidence=35,
        reasoning="No confluence",
        invalidation="",
        confluence_factors=[],
    )


def _approved_risk(lot_size: float = 0.33) -> RiskDecision:
    return RiskDecision(
        approved=True,
        lot_size=lot_size,
        risk_amount=99.0,
        sl_distance=0.003,
    )


def _rejected_risk() -> RiskDecision:
    return RiskDecision(
        approved=False,
        lot_size=0.0,
        risk_amount=0.0,
        sl_distance=0.003,
        reasons=["Max open trades reached"],
    )


# ---------------------------------------------------------------------------
# LOG-02  init_db
# ---------------------------------------------------------------------------

class TestInitDb:
    @pytest.mark.asyncio
    async def test_creates_signals_table(self, db):
        import aiosqlite
        async with aiosqlite.connect(db) as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'"
            ) as cur:
                row = await cur.fetchone()
        assert row is not None

    @pytest.mark.asyncio
    async def test_idempotent_multiple_calls(self, db):
        # Calling init_db twice must not raise
        await init_db()
        await init_db()

    @pytest.mark.asyncio
    async def test_creates_indexes(self, db):
        import aiosqlite
        async with aiosqlite.connect(db) as conn:
            async with conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ) as cur:
                rows = await cur.fetchall()
        index_names = {r[0] for r in rows}
        assert "idx_signals_symbol"    in index_names
        assert "idx_signals_outcome"   in index_names
        assert "idx_signals_created"   in index_names
        assert "idx_signals_direction" in index_names


# ---------------------------------------------------------------------------
# LOG-03  log_signal
# ---------------------------------------------------------------------------

class TestLogSignal:
    @pytest.mark.asyncio
    async def test_returns_positive_id(self, db):
        row_id = await log_signal(_buy_signal(), _approved_risk())
        assert isinstance(row_id, int)
        assert row_id > 0

    @pytest.mark.asyncio
    async def test_buy_signal_stored_as_pending(self, db):
        row_id = await log_signal(_buy_signal(), _approved_risk())
        record = await get_signal(row_id)
        assert record is not None
        assert record.outcome == "PENDING"
        assert record.direction == "BUY"

    @pytest.mark.asyncio
    async def test_no_trade_stored_as_skipped(self, db):
        row_id = await log_signal(_no_trade_signal(), _approved_risk())
        record = await get_signal(row_id)
        assert record.outcome == "SKIPPED"

    @pytest.mark.asyncio
    async def test_rejected_risk_stored_as_skipped(self, db):
        row_id = await log_signal(_buy_signal(), _rejected_risk())
        record = await get_signal(row_id)
        assert record.outcome == "SKIPPED"

    @pytest.mark.asyncio
    async def test_all_fields_persisted(self, db):
        signal = _buy_signal()
        risk   = _approved_risk(lot_size=0.33)
        row_id = await log_signal(signal, risk)
        record = await get_signal(row_id)

        assert record.symbol     == "EURUSD"
        assert record.confidence == 78
        assert record.entry      == pytest.approx(1.1010)
        assert record.sl         == pytest.approx(1.0980)
        assert record.tp1        == pytest.approx(1.1055)
        assert record.lot_size   == pytest.approx(0.33)
        assert record.risk_amount == pytest.approx(99.0)
        assert record.input_tokens  == 800
        assert record.cache_read_tokens == 600
        assert record.estimated_cost_usd > 0

    @pytest.mark.asyncio
    async def test_confluence_factors_round_trip(self, db):
        signal = _buy_signal()
        row_id = await log_signal(signal, _approved_risk())
        record = await get_signal(row_id)
        assert record.confluence_factors == ["EMA alignment", "RSI not overbought"]

    @pytest.mark.asyncio
    async def test_multiple_signals_get_unique_ids(self, db):
        id1 = await log_signal(_buy_signal("EURUSD"), _approved_risk())
        id2 = await log_signal(_buy_signal("XAUUSD"), _approved_risk())
        assert id1 != id2

    @pytest.mark.asyncio
    async def test_no_trade_entry_is_none(self, db):
        row_id = await log_signal(_no_trade_signal(), _approved_risk())
        record = await get_signal(row_id)
        assert record.entry is None
        assert record.sl    is None


# ---------------------------------------------------------------------------
# LOG-04  update_outcome
# ---------------------------------------------------------------------------

class TestUpdateOutcome:
    @pytest.mark.asyncio
    async def test_update_to_win(self, db):
        row_id = await log_signal(_buy_signal(), _approved_risk())
        ok = await update_outcome(row_id, "WIN", close_price=1.1055, actual_pnl=45.0)
        assert ok is True
        record = await get_signal(row_id)
        assert record.outcome     == "WIN"
        assert record.close_price == pytest.approx(1.1055)
        assert record.actual_pnl  == pytest.approx(45.0)
        assert record.closed_at   is not None

    @pytest.mark.asyncio
    async def test_update_to_loss(self, db):
        row_id = await log_signal(_buy_signal(), _approved_risk())
        ok = await update_outcome(row_id, "LOSS", close_price=1.0975)
        assert ok is True
        record = await get_signal(row_id)
        assert record.outcome == "LOSS"

    @pytest.mark.asyncio
    async def test_update_to_breakeven(self, db):
        row_id = await log_signal(_buy_signal(), _approved_risk())
        ok = await update_outcome(row_id, "BREAKEVEN", close_price=1.1010)
        assert ok is True

    @pytest.mark.asyncio
    async def test_returns_false_for_missing_id(self, db):
        ok = await update_outcome(9999, "WIN", close_price=1.1055)
        assert ok is False

    @pytest.mark.asyncio
    async def test_actual_rr_calculated_for_win(self, db):
        # entry=1.1010, sl=1.0980 → risk=0.003; close=1.1055 → reward=0.0045; RR=1.5
        row_id = await log_signal(_buy_signal(), _approved_risk())
        await update_outcome(row_id, "WIN", close_price=1.1055)
        record = await get_signal(row_id)
        assert record.actual_rr == pytest.approx(1.5, rel=0.01)

    @pytest.mark.asyncio
    async def test_actual_rr_negative_for_loss(self, db):
        # close below entry on BUY → negative RR
        row_id = await log_signal(_buy_signal(), _approved_risk())
        await update_outcome(row_id, "LOSS", close_price=1.0990)
        record = await get_signal(row_id)
        assert record.actual_rr is not None
        assert record.actual_rr < 0

    @pytest.mark.asyncio
    async def test_actual_rr_zero_for_breakeven(self, db):
        # close at entry → reward=0 → RR=0
        row_id = await log_signal(_buy_signal(), _approved_risk())
        await update_outcome(row_id, "BREAKEVEN", close_price=1.1010)
        record = await get_signal(row_id)
        assert record.actual_rr == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# LOG-05  get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    async def _seed(self, db, n_win=2, n_loss=1, n_no_trade=1):
        """Seed the DB with a mix of outcomes."""
        ids = []
        for _ in range(n_win):
            rid = await log_signal(_buy_signal("EURUSD"), _approved_risk())
            await update_outcome(rid, "WIN", close_price=1.1055, actual_pnl=45.0)
            ids.append(rid)
        for _ in range(n_loss):
            rid = await log_signal(_buy_signal("EURUSD"), _approved_risk())
            await update_outcome(rid, "LOSS", close_price=1.0975, actual_pnl=-30.0)
            ids.append(rid)
        for _ in range(n_no_trade):
            await log_signal(_no_trade_signal("EURUSD"), _approved_risk())
        return ids

    @pytest.mark.asyncio
    async def test_empty_db_returns_zero_stats(self, db):
        stats = await get_stats()
        assert stats.total_signals == 0
        assert stats.wins == 0
        assert stats.win_rate_pct == 0.0

    @pytest.mark.asyncio
    async def test_win_rate_calculation(self, db):
        await self._seed(db, n_win=2, n_loss=1, n_no_trade=0)
        stats = await get_stats()
        # 2 wins / (2+1) closed = 66.7%
        assert stats.win_rate_pct == pytest.approx(66.7, abs=0.1)

    @pytest.mark.asyncio
    async def test_total_signals_count(self, db):
        await self._seed(db, n_win=2, n_loss=1, n_no_trade=1)
        stats = await get_stats()
        assert stats.total_signals == 4

    @pytest.mark.asyncio
    async def test_no_trade_counted_separately(self, db):
        await self._seed(db, n_win=1, n_loss=0, n_no_trade=2)
        stats = await get_stats()
        assert stats.no_trade_signals == 2

    @pytest.mark.asyncio
    async def test_total_pnl_summed(self, db):
        await self._seed(db, n_win=2, n_loss=1)
        stats = await get_stats()
        # 2 × 45 - 1 × 30 = 60
        assert stats.total_pnl == pytest.approx(60.0)

    @pytest.mark.asyncio
    async def test_avg_rr_computed(self, db):
        rid1 = await log_signal(_buy_signal("EURUSD"), _approved_risk())
        await update_outcome(rid1, "WIN", close_price=1.1055)   # RR ≈ 1.5
        rid2 = await log_signal(_buy_signal("EURUSD"), _approved_risk())
        await update_outcome(rid2, "LOSS", close_price=1.0990)  # close 10 pips below entry (loss)

        stats = await get_stats()
        # avg of 1.5 and some negative RR
        assert stats.avg_rr < 1.5

    @pytest.mark.asyncio
    async def test_by_symbol_breakdown(self, db):
        await log_signal(_buy_signal("EURUSD"), _approved_risk())
        await log_signal(_buy_signal("XAUUSD"), _approved_risk())
        await log_signal(_buy_signal("XAUUSD"), _approved_risk())
        stats = await get_stats()
        assert "EURUSD" in stats.by_symbol
        assert "XAUUSD" in stats.by_symbol
        assert stats.by_symbol["XAUUSD"]["total"] == 2

    @pytest.mark.asyncio
    async def test_by_direction_counts(self, db):
        await log_signal(_buy_signal("EURUSD", direction="BUY"),  _approved_risk())
        await log_signal(_buy_signal("EURUSD", direction="SELL"), _approved_risk())
        await log_signal(_no_trade_signal(), _approved_risk())
        stats = await get_stats()
        assert stats.by_direction.get("BUY",      0) == 1
        assert stats.by_direction.get("SELL",     0) == 1
        assert stats.by_direction.get("NO_TRADE", 0) == 1

    @pytest.mark.asyncio
    async def test_symbol_filter(self, db):
        await log_signal(_buy_signal("EURUSD"), _approved_risk())
        await log_signal(_buy_signal("XAUUSD"), _approved_risk())
        stats = await get_stats(symbol="XAUUSD")
        assert stats.total_signals == 1
        assert "EURUSD" not in stats.by_symbol

    @pytest.mark.asyncio
    async def test_api_cost_accumulated(self, db):
        await log_signal(_buy_signal(), _approved_risk())
        await log_signal(_buy_signal(), _approved_risk())
        stats = await get_stats()
        assert stats.total_api_cost_usd > 0

    @pytest.mark.asyncio
    async def test_pending_count(self, db):
        await log_signal(_buy_signal(), _approved_risk())   # PENDING
        await log_signal(_no_trade_signal(), _approved_risk())  # SKIPPED
        stats = await get_stats()
        assert stats.pending_signals == 1

    @pytest.mark.asyncio
    async def test_period_dates_set(self, db):
        await log_signal(_buy_signal(), _approved_risk())
        stats = await get_stats()
        assert stats.period_start is not None
        assert stats.period_end   is not None


# ---------------------------------------------------------------------------
# get_pending_signals
# ---------------------------------------------------------------------------

class TestGetPendingSignals:
    @pytest.mark.asyncio
    async def test_returns_pending_only(self, db):
        rid1 = await log_signal(_buy_signal("EURUSD"), _approved_risk())
        rid2 = await log_signal(_buy_signal("XAUUSD"), _approved_risk())
        await update_outcome(rid1, "WIN", close_price=1.1055)

        pending = await get_pending_signals()
        assert len(pending) == 1
        assert pending[0].id == rid2

    @pytest.mark.asyncio
    async def test_empty_when_all_closed(self, db):
        rid = await log_signal(_buy_signal(), _approved_risk())
        await update_outcome(rid, "WIN", close_price=1.1055)
        pending = await get_pending_signals()
        assert pending == []

    @pytest.mark.asyncio
    async def test_skipped_signals_not_in_pending(self, db):
        await log_signal(_no_trade_signal(), _approved_risk())
        pending = await get_pending_signals()
        assert pending == []
