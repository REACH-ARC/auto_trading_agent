"""Unit tests for backend/risk_manager.py — RISK-01 through RISK-06"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from backend.claude_analyst import SignalResult
from backend.risk_manager import (
    AccountState,
    RiskDecision,
    _CONTRACT_SIZE,
    calc_position_size,
    check_daily_loss,
    check_max_open_trades,
    evaluate,
    validate_sl_distance,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(
    equity: float = 10_000.0,
    open_trades: int = 0,
    daily_pnl: float = 0.0,
    balance: float = 10_000.0,
) -> AccountState:
    return AccountState(equity=equity, open_trades=open_trades,
                        daily_pnl=daily_pnl, balance=balance)


def _buy_signal(
    symbol: str = "EURUSD",
    entry: float = 1.1010,
    sl: float = 1.0980,
    tp1: float = 1.1055,
    confidence: int = 78,
) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        direction="BUY",
        confidence=confidence,
        reasoning="test",
        invalidation="test",
        confluence_factors=[],
        entry=entry,
        sl=sl,
        tp1=tp1,
        tp2=1.1085,
        tp3=1.1130,
    )


def _no_trade_signal() -> SignalResult:
    return SignalResult(
        symbol="EURUSD",
        direction="NO_TRADE",
        confidence=30,
        reasoning="no setup",
        invalidation="",
        confluence_factors=[],
    )


# ---------------------------------------------------------------------------
# RISK-01  AccountState
# ---------------------------------------------------------------------------

class TestAccountState:
    def test_daily_loss_pct_zero_on_no_loss(self):
        state = _state(daily_pnl=0.0)
        assert state.daily_loss_pct == 0.0

    def test_daily_loss_pct_calculated_correctly(self):
        state = _state(equity=10_000.0, daily_pnl=-300.0)
        assert state.daily_loss_pct == pytest.approx(3.0)

    def test_daily_loss_pct_ignores_profit(self):
        state = _state(equity=10_000.0, daily_pnl=500.0)
        assert state.daily_loss_pct == 0.0

    def test_daily_loss_pct_zero_on_zero_equity(self):
        state = _state(equity=0.0, daily_pnl=-100.0)
        assert state.daily_loss_pct == 0.0


# ---------------------------------------------------------------------------
# RISK-06  RiskDecision
# ---------------------------------------------------------------------------

class TestRiskDecision:
    def test_summary_approved(self):
        d = RiskDecision(approved=True, lot_size=0.10, risk_amount=100.0, sl_distance=0.003)
        assert "APPROVED" in d.summary
        assert "0.10" in d.summary

    def test_summary_rejected(self):
        d = RiskDecision(approved=False, lot_size=0.0, risk_amount=0.0,
                         sl_distance=0.0, reasons=["Max trades reached"])
        assert "REJECTED" in d.summary
        assert "Max trades reached" in d.summary


# ---------------------------------------------------------------------------
# RISK-02  calc_position_size
# ---------------------------------------------------------------------------

class TestCalcPositionSize:
    def test_basic_eurusd(self):
        # equity=10k, risk=1%, sl=30 pips (0.0030), pip_value=10 → lots=0.33
        lot, risk, dist = calc_position_size("EURUSD", 1.1010, 1.0980, 10_000.0, 1.0)
        assert lot == pytest.approx(0.33, abs=0.01)
        assert risk == pytest.approx(100.0)
        assert dist == pytest.approx(0.003, rel=0.01)

    def test_lot_rounded_down_to_two_decimals(self):
        lot, _, _ = calc_position_size("EURUSD", 1.1010, 1.0980, 10_000.0, 1.0)
        assert round(lot, 2) == lot

    def test_minimum_lot_is_001(self):
        # Very small equity — should return min 0.01
        lot, _, _ = calc_position_size("EURUSD", 1.1010, 1.0980, 10.0, 1.0)
        assert lot == 0.01

    def test_zero_lot_when_entry_equals_sl(self):
        lot, risk, dist = calc_position_size("EURUSD", 1.1010, 1.1010, 10_000.0, 1.0)
        assert lot == 0.0
        assert dist == 0.0

    def test_xauusd_uses_correct_pip_value(self):
        # XAUUSD pip_value=100; equity=10k, risk=1%, sl=$2 → lots = 100/200 = 0.5
        lot, risk, dist = calc_position_size("XAUUSD", 2000.0, 1998.0, 10_000.0, 1.0)
        assert lot == pytest.approx(0.50, abs=0.01)
        assert risk == pytest.approx(100.0)

    def test_unknown_symbol_uses_default_pip_value(self):
        lot, _, _ = calc_position_size("BTCUSD", 50000.0, 49000.0, 10_000.0, 1.0)
        assert lot > 0

    def test_higher_risk_pct_gives_larger_lot(self):
        lot_1pct, _, _ = calc_position_size("EURUSD", 1.1010, 1.0980, 10_000.0, 1.0)
        lot_2pct, _, _ = calc_position_size("EURUSD", 1.1010, 1.0980, 10_000.0, 2.0)
        assert lot_2pct > lot_1pct

    def test_wider_sl_gives_smaller_lot(self):
        lot_tight, _, _ = calc_position_size("EURUSD", 1.1010, 1.0990, 10_000.0, 1.0)  # 20 pip
        lot_wide, _, _  = calc_position_size("EURUSD", 1.1010, 1.0960, 10_000.0, 1.0)  # 50 pip
        assert lot_tight > lot_wide

    def test_sell_signal_sl_above_entry(self):
        # SL above entry for a sell trade — distance is positive regardless
        lot, _, dist = calc_position_size("EURUSD", 1.1010, 1.1040, 10_000.0, 1.0)
        assert lot > 0
        assert dist == pytest.approx(0.003, rel=0.01)

    def test_uses_settings_default_when_risk_pct_not_given(self):
        # settings.risk["account_risk_percent"] = 1.0
        lot_explicit, _, _ = calc_position_size("EURUSD", 1.1010, 1.0980, 10_000.0, 1.0)
        lot_default, _, _ = calc_position_size("EURUSD", 1.1010, 1.0980, 10_000.0)
        assert lot_explicit == lot_default


# ---------------------------------------------------------------------------
# RISK-03  check_max_open_trades
# ---------------------------------------------------------------------------

class TestCheckMaxOpenTrades:
    def test_ok_when_below_limit(self):
        ok, reason = check_max_open_trades(0)
        assert ok is True
        assert reason == ""

    def test_ok_at_one_below_limit(self):
        # max = 3 from settings
        ok, _ = check_max_open_trades(2)
        assert ok is True

    def test_blocked_at_limit(self):
        ok, reason = check_max_open_trades(3)
        assert ok is False
        assert "Max open trades" in reason

    def test_blocked_above_limit(self):
        ok, reason = check_max_open_trades(10)
        assert ok is False


# ---------------------------------------------------------------------------
# RISK-04  check_daily_loss
# ---------------------------------------------------------------------------

class TestCheckDailyLoss:
    def test_ok_when_no_loss(self):
        ok, reason = check_daily_loss(_state(daily_pnl=0.0))
        assert ok is True

    def test_ok_when_profitable(self):
        ok, _ = check_daily_loss(_state(equity=10_000.0, daily_pnl=500.0))
        assert ok is True

    def test_ok_just_below_limit(self):
        # limit = 3.0%, loss = 2.9%
        ok, _ = check_daily_loss(_state(equity=10_000.0, daily_pnl=-290.0))
        assert ok is True

    def test_blocked_at_limit(self):
        # exactly 3.0% loss
        ok, reason = check_daily_loss(_state(equity=10_000.0, daily_pnl=-300.0))
        assert ok is False
        assert "Daily loss limit" in reason

    def test_blocked_above_limit(self):
        ok, reason = check_daily_loss(_state(equity=10_000.0, daily_pnl=-500.0))
        assert ok is False

    def test_reason_contains_loss_info(self):
        _, reason = check_daily_loss(_state(equity=10_000.0, daily_pnl=-300.0))
        assert "300" in reason or "3.00" in reason


# ---------------------------------------------------------------------------
# RISK-05  validate_sl_distance
# ---------------------------------------------------------------------------

class TestValidateSlDistance:
    def _atr(self) -> float:
        return 0.0020  # 20-pip ATR (typical for EURUSD H1)

    def test_valid_sl_in_range(self):
        # min=0.5×ATR=0.001, max=3×ATR=0.006; sl_dist=0.003 → valid
        ok, reason = validate_sl_distance(0.003, self._atr())
        assert ok is True
        assert reason == ""

    def test_too_tight_sl(self):
        # 0.0005 < 0.5 × 0.002 = 0.001
        ok, reason = validate_sl_distance(0.0005, self._atr())
        assert ok is False
        assert "tight" in reason.lower()

    def test_too_wide_sl(self):
        # 0.008 > 3 × 0.002 = 0.006
        ok, reason = validate_sl_distance(0.008, self._atr())
        assert ok is False
        assert "wide" in reason.lower()

    def test_exactly_at_min_multiplier(self):
        # exactly 0.5 × ATR = 0.001
        ok, _ = validate_sl_distance(0.001, self._atr())
        assert ok is True

    def test_exactly_at_max_multiplier(self):
        # exactly 3 × ATR = 0.006
        ok, _ = validate_sl_distance(0.006, self._atr())
        assert ok is True

    def test_zero_atr_always_passes(self):
        # No ATR data — allow through (fail open)
        ok, _ = validate_sl_distance(0.0001, 0.0)
        assert ok is True

    def test_reason_references_atr(self):
        _, reason = validate_sl_distance(0.0001, self._atr())
        assert "ATR" in reason


# ---------------------------------------------------------------------------
# RISK-06  evaluate — full pipeline
# ---------------------------------------------------------------------------

class TestEvaluate:
    def _atr(self) -> float:
        return 0.0020

    def test_approved_normal_trade(self):
        signal = _buy_signal(entry=1.1010, sl=1.0980)  # 30-pip SL
        decision = evaluate(signal, _state(), self._atr())
        assert decision.approved is True
        assert decision.lot_size > 0
        assert decision.risk_amount > 0

    def test_rejected_no_trade_signal(self):
        decision = evaluate(_no_trade_signal(), _state(), self._atr())
        assert decision.approved is False
        assert decision.lot_size == 0.0
        assert "NO_TRADE" in decision.reasons[0]

    def test_rejected_daily_loss_hit(self):
        signal = _buy_signal()
        state = _state(equity=10_000.0, daily_pnl=-300.0)  # 3% loss
        decision = evaluate(signal, state, self._atr())
        assert decision.approved is False
        assert any("Daily loss" in r for r in decision.reasons)

    def test_rejected_max_open_trades(self):
        signal = _buy_signal()
        state = _state(open_trades=3)  # at limit
        decision = evaluate(signal, state, self._atr())
        assert decision.approved is False
        assert any("Max open trades" in r for r in decision.reasons)

    def test_rejected_sl_too_tight(self):
        # SL only 1 pip away — way below 0.5×ATR
        signal = _buy_signal(entry=1.1010, sl=1.1009)
        decision = evaluate(signal, _state(), self._atr())
        assert decision.approved is False
        assert any("tight" in r.lower() for r in decision.reasons)

    def test_rejected_sl_too_wide(self):
        # SL 200 pips away — above 3×ATR
        signal = _buy_signal(entry=1.1010, sl=1.0810)
        decision = evaluate(signal, _state(), self._atr())
        assert decision.approved is False
        assert any("wide" in r.lower() for r in decision.reasons)

    def test_lot_size_zero_when_rejected(self):
        state = _state(open_trades=3)
        decision = evaluate(_buy_signal(), state, self._atr())
        assert decision.lot_size == 0.0

    def test_multiple_blocks_reported(self):
        # Both daily loss AND max trades hit
        state = _state(equity=10_000.0, daily_pnl=-300.0, open_trades=3)
        decision = evaluate(_buy_signal(), state, self._atr())
        assert decision.approved is False
        assert len(decision.reasons) >= 2

    def test_warning_when_approaching_daily_limit(self):
        # 2.3% loss = 76% of 3% limit → should produce a warning
        state = _state(equity=10_000.0, daily_pnl=-230.0)
        signal = _buy_signal(entry=1.1010, sl=1.0980)
        decision = evaluate(signal, state, self._atr())
        assert decision.approved is True  # not blocked
        assert any("Approaching" in w for w in decision.warnings)

    def test_warning_when_one_trade_slot_left(self):
        state = _state(open_trades=2)  # 1 below max of 3
        signal = _buy_signal(entry=1.1010, sl=1.0980)
        decision = evaluate(signal, state, self._atr())
        assert decision.approved is True
        assert any("slot" in w for w in decision.warnings)

    def test_sell_signal_approved(self):
        # SELL: entry below SL
        signal = _buy_signal(entry=1.1010, sl=1.1040)
        signal.direction = "SELL"
        decision = evaluate(signal, _state(), self._atr())
        assert decision.approved is True

    def test_xauusd_position_sizing(self):
        # Gold: entry=2000, SL=1998 ($2 distance, pip_value=100)
        signal = _buy_signal(symbol="XAUUSD", entry=2000.0, sl=1998.0, tp1=2003.0)
        atr = 2.5  # $2.5 ATR for gold
        decision = evaluate(signal, _state(), atr)
        assert decision.approved is True
        assert decision.lot_size >= 0.01
