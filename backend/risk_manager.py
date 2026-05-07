"""
Phase 5 — Risk Manager
RISK-01 through RISK-06
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Literal

from loguru import logger

from config import settings
from backend.claude_analyst import SignalResult
import backend.model_manager as _mm

# ---------------------------------------------------------------------------
# RISK-01  Types
# ---------------------------------------------------------------------------

# Fallback contract-size table (used only when MT5 tick info is unavailable).
# For fixed-SL-amount mode the live tick_value / tick_size from MT5 is used instead.
_CONTRACT_SIZE: dict[str, float] = {
    "EURUSD": 100_000, "GBPUSD": 100_000, "AUDUSD": 100_000, "NZDUSD": 100_000,
    "USDCAD": 100_000, "USDCHF": 100_000, "USDJPY": 100_000, "EURJPY": 100_000,
    "GBPJPY": 100_000, "EURGBP": 100_000, "EURAUD": 100_000,
    "XAUUSD": 100, "XAGUSD": 5_000,
    "US30": 1, "NAS100": 1, "SPX500": 1, "US500": 1, "GER40": 1, "UK100": 1,
}
_DEFAULT_CONTRACT_SIZE = 100_000


@dataclass
class AccountState:
    """Snapshot of account state passed in from MT5 / server at call time."""
    equity: float
    open_trades: int
    daily_pnl: float
    balance: float = 0.0
    currency: str = "USD"

    @property
    def is_cent(self) -> bool:
        return self.currency.upper() == "USC"

    @property
    def daily_loss_pct(self) -> float:
        if self.equity <= 0:
            return 0.0
        return max(0.0, -self.daily_pnl / self.equity * 100)


@dataclass
class RiskDecision:
    """RISK-06 — Output returned to the caller."""
    approved: bool
    lot_size: float
    risk_amount: float
    sl_distance: float
    sizing_mode: str = "percent"          # "percent" | "fixed_sl"
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        parts = [f"[{status}] lot={self.lot_size:.2f} risk={self.risk_amount:.2f} ({self.sizing_mode})"]
        if self.reasons:
            parts.append("| " + "; ".join(self.reasons))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# RISK-02a  Percent-of-equity position sizing (legacy / fallback)
# ---------------------------------------------------------------------------

def calc_position_size(
    symbol: str,
    entry: float,
    sl: float,
    equity: float,
    risk_pct: float | None = None,
) -> tuple[float, float, float]:
    """
    Calculate lot size based on fixed-percentage equity risk.
    Returns (lot_size, risk_amount, sl_distance).
    """
    risk_pct = risk_pct if risk_pct is not None else settings.risk["account_risk_percent"]
    risk_amount = equity * (risk_pct / 100.0)

    sl_distance = abs(entry - sl)
    if sl_distance == 0:
        return 0.0, 0.0, 0.0

    base_symbol = symbol.rstrip("mc").upper()
    contract_size = _CONTRACT_SIZE.get(base_symbol, _DEFAULT_CONTRACT_SIZE)
    raw_lots = risk_amount / (sl_distance * contract_size)
    lot_size = max(0.01, int(raw_lots * 100) / 100)
    return lot_size, round(risk_amount, 2), round(sl_distance, 6)


# ---------------------------------------------------------------------------
# RISK-02b  Fixed-SL-amount position sizing (preferred)
# ---------------------------------------------------------------------------

def calc_position_size_fixed_sl(
    entry: float,
    sl: float,
    account_currency: str,
    tick_size: float,
    tick_value: float,
    min_lot: float,
    max_lot: float,
    lot_step: float = 0.01,
) -> tuple[float, float, float]:
    """
    Calculate lot size so the dollar value of the SL distance equals exactly
    the configured fixed SL amount (sl_amount_usd or sl_amount_usc).

    Formula:
        value_per_price_unit = tick_value / tick_size   (account-currency per lot per 1 price unit)
        lot_size = sl_amount / (sl_distance × value_per_price_unit)

    All monetary values are in account currency (USD for standard, USC for cent).
    tick_value from MT5 is already expressed in account currency, so the formula
    works correctly for both account types.

    Returns (lot_size, actual_risk_amount, sl_distance).
    """
    is_cent = account_currency.upper() == "USC"
    sl_amount = _mm.get_sl_amount_usc() if is_cent else _mm.get_sl_amount_usd()

    sl_distance = abs(entry - sl)
    if sl_distance < 1e-8 or tick_size < 1e-10 or tick_value < 1e-10:
        return min_lot, 0.0, 0.0

    # Monetary value gained/lost per lot per 1 unit of price movement
    value_per_unit = tick_value / tick_size

    # Raw lot size to risk exactly sl_amount
    raw_lots = sl_amount / (sl_distance * value_per_unit)

    # Floor to lot_step so we never exceed the intended risk amount
    if lot_step > 0:
        raw_lots = (int(raw_lots / lot_step)) * lot_step

    lot_size = max(min_lot, min(max_lot, raw_lots))
    lot_size = round(lot_size, 8)

    actual_risk = round(lot_size * sl_distance * value_per_unit, 2)

    logger.debug(
        f"Fixed-SL sizing: sl_amount={sl_amount} {account_currency} | "
        f"sl_dist={sl_distance:.5f} | value_per_unit={value_per_unit:.4f} | "
        f"raw_lots={raw_lots:.5f} → lot={lot_size} | actual_risk={actual_risk}"
    )

    return lot_size, actual_risk, round(sl_distance, 6)


# ---------------------------------------------------------------------------
# RISK-03  Max open trades check
# ---------------------------------------------------------------------------

def check_max_open_trades(open_trades: int) -> tuple[bool, str]:
    limit: int = settings.risk["max_open_trades"]
    if open_trades >= limit:
        return False, f"Max open trades reached ({open_trades}/{limit})"
    return True, ""


# ---------------------------------------------------------------------------
# RISK-04  Daily loss limit
# ---------------------------------------------------------------------------

def check_daily_loss(state: AccountState) -> tuple[bool, str]:
    limit_pct: float = settings.risk["max_daily_loss_percent"]
    loss_pct = state.daily_loss_pct
    if loss_pct >= limit_pct:
        return False, (
            f"Daily loss limit hit: {loss_pct:.2f}% >= {limit_pct:.2f}% "
            f"(lost {abs(state.daily_pnl):.2f} {state.currency})"
        )
    return True, ""


# ---------------------------------------------------------------------------
# RISK-05  SL distance validator (ATR-based)
# ---------------------------------------------------------------------------

def validate_sl_distance(sl_distance: float, atr: float) -> tuple[bool, str]:
    min_mult: float = settings.risk["min_sl_atr_multiplier"]
    max_mult: float = settings.risk["max_sl_atr_multiplier"]

    if atr <= 0:
        return True, ""

    min_allowed = atr * min_mult
    max_allowed = atr * max_mult

    if sl_distance < min_allowed:
        return False, (
            f"SL too tight: {sl_distance:.5f} < {min_allowed:.5f} "
            f"({min_mult}× ATR={atr:.5f})"
        )
    if sl_distance > max_allowed:
        return False, (
            f"SL too wide: {sl_distance:.5f} > {max_allowed:.5f} "
            f"({max_mult}× ATR={atr:.5f})"
        )
    return True, ""


# ---------------------------------------------------------------------------
# RISK-06  Main evaluate function — returns RiskDecision
# ---------------------------------------------------------------------------

def evaluate(
    signal: SignalResult,
    state: AccountState,
    atr: float,
    tick_info: dict | None = None,
) -> RiskDecision:
    """
    Full risk evaluation pipeline.

    tick_info (optional) — dict with keys: tick_size, tick_value, min_lot,
    max_lot, lot_step. When provided and use_fixed_sl_amount=True in settings,
    lot size is calculated to risk exactly sl_amount_usd / sl_amount_usc.
    Otherwise falls back to percent-of-equity sizing.
    """
    if not signal.is_actionable:
        return RiskDecision(
            approved=False, lot_size=0.0, risk_amount=0.0, sl_distance=0.0,
            reasons=["Signal is NO_TRADE — no risk evaluation needed"],
        )

    reasons: list[str] = []
    warnings: list[str] = []
    blocked = False

    # --- RISK-04: daily loss limit ---
    ok, reason = check_daily_loss(state)
    if not ok:
        reasons.append(reason)
        blocked = True
        logger.warning(f"Risk block (daily loss): {reason}")

    # --- RISK-03: max open trades ---
    ok, reason = check_max_open_trades(state.open_trades)
    if not ok:
        reasons.append(reason)
        blocked = True
        logger.warning(f"Risk block (max trades): {reason}")

    # --- Position size ---
    use_fixed: bool = bool(settings.risk.get("use_fixed_sl_amount", False))
    sizing_mode = "fixed_sl"

    if use_fixed and tick_info:
        lot_size, risk_amount, sl_distance = calc_position_size_fixed_sl(
            entry=signal.entry,          # type: ignore[arg-type]
            sl=signal.sl,                # type: ignore[arg-type]
            account_currency=state.currency,
            tick_size=float(tick_info.get("tick_size", 0.01)),
            tick_value=float(tick_info.get("tick_value", 1.0)),
            min_lot=float(tick_info.get("min_lot", 0.01)),
            max_lot=float(tick_info.get("max_lot", 1.0)),
            lot_step=float(tick_info.get("lot_step", 0.01)),
        )
        is_cent = state.currency.upper() == "USC"
        sl_amount_cfg = _mm.get_sl_amount_usc() if is_cent else _mm.get_sl_amount_usd()
        logger.info(
            f"Fixed-SL sizing: {signal.symbol} | "
            f"target_risk={sl_amount_cfg} {state.currency} | "
            f"actual_risk={risk_amount} {state.currency} | lot={lot_size}"
        )
    else:
        sizing_mode = "percent"
        lot_size, risk_amount, sl_distance = calc_position_size(
            symbol=signal.symbol,
            entry=signal.entry,          # type: ignore[arg-type]
            sl=signal.sl,                # type: ignore[arg-type]
            equity=state.equity,
        )
        if use_fixed and not tick_info:
            warnings.append("Fixed SL mode enabled but tick_info unavailable — using % equity fallback")

    if lot_size <= 0:
        reasons.append("Position size calculated as 0 — entry equals SL price")
        blocked = True

    # --- RISK-05: ATR SL validation ---
    ok, reason = validate_sl_distance(sl_distance, atr)
    if not ok:
        reasons.append(reason)
        blocked = True
        logger.warning(f"Risk block (SL distance): {reason}")

    # --- Soft warnings ---
    if state.daily_loss_pct >= settings.risk["max_daily_loss_percent"] * 0.75:
        warnings.append(
            f"Approaching daily loss limit ({state.daily_loss_pct:.1f}% of "
            f"{settings.risk['max_daily_loss_percent']}% limit)"
        )

    if state.open_trades >= settings.risk["max_open_trades"] - 1:
        warnings.append(
            f"One trade slot remaining ({state.open_trades}/{settings.risk['max_open_trades']})"
        )

    decision = RiskDecision(
        approved=not blocked,
        lot_size=lot_size if not blocked else 0.0,
        risk_amount=risk_amount,
        sl_distance=sl_distance,
        sizing_mode=sizing_mode,
        reasons=reasons,
        warnings=warnings,
    )

    logger.info(f"Risk decision for {signal.symbol}: {decision.summary}")
    return decision
