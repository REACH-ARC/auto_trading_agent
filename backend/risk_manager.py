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

# ---------------------------------------------------------------------------
# RISK-01  Types
# ---------------------------------------------------------------------------

# Contract size per 1 standard lot, in base-currency units.
# Position sizing formula: lot_size = risk_amount / (sl_distance * contract_size)
# Example EURUSD: 1 lot = 100,000 EUR; SL 30 pips (0.0030 USD) → risk per lot = $300
# Example XAUUSD: 1 lot = 100 oz;     SL $2/oz             → risk per lot = $200
_CONTRACT_SIZE: dict[str, float] = {
    # Standard forex majors / minors (1 lot = 100,000 units)
    "EURUSD": 100_000,
    "GBPUSD": 100_000,
    "AUDUSD": 100_000,
    "NZDUSD": 100_000,
    "USDCAD": 100_000,
    "USDCHF": 100_000,
    "USDJPY": 100_000,
    "EURJPY": 100_000,
    "GBPJPY": 100_000,
    "EURGBP": 100_000,
    "EURAUD": 100_000,
    # Metals
    "XAUUSD": 100,      # 100 troy oz per lot
    "XAGUSD": 5_000,    # 5,000 oz per lot
    # US indices (broker-dependent; 1 = conservative micro-contract approximation)
    "US30":   1,
    "NAS100": 1,
    "SPX500": 1,
    "US500":  1,
    # EU/UK indices
    "GER40":  1,
    "UK100":  1,
}
_DEFAULT_CONTRACT_SIZE = 100_000  # standard forex fallback


@dataclass
class AccountState:
    """Snapshot of account state passed in from MT5 / server at call time."""
    equity: float           # current account equity in account currency
    open_trades: int        # number of positions currently open
    daily_pnl: float        # today's realised P&L (negative = loss)
    balance: float = 0.0   # account balance (equity without floating P&L)

    @property
    def daily_loss_pct(self) -> float:
        """Daily loss as percentage of equity (positive number = loss)."""
        if self.equity <= 0:
            return 0.0
        return max(0.0, -self.daily_pnl / self.equity * 100)


@dataclass
class RiskDecision:
    """RISK-06 — Output returned to the caller."""
    approved: bool
    lot_size: float                     # 0.0 when not approved
    risk_amount: float                  # dollar risk on this trade
    sl_distance: float                  # absolute price distance entry→SL
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        status = "APPROVED" if self.approved else "REJECTED"
        parts = [f"[{status}] lot={self.lot_size:.2f} risk=${self.risk_amount:.2f}"]
        if self.reasons:
            parts.append("| " + "; ".join(self.reasons))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# RISK-02  Position size calculator
# ---------------------------------------------------------------------------

def calc_position_size(
    symbol: str,
    entry: float,
    sl: float,
    equity: float,
    risk_pct: float | None = None,
) -> tuple[float, float, float]:
    """
    Calculate lot size based on fixed-percentage risk.

    Returns (lot_size, risk_amount, sl_distance).
    lot_size is rounded down to 2 decimal places (0.01 = 1 micro lot).
    """
    risk_pct = risk_pct if risk_pct is not None else settings.risk["account_risk_percent"]
    risk_amount = equity * (risk_pct / 100.0)

    sl_distance = abs(entry - sl)
    if sl_distance == 0:
        return 0.0, 0.0, 0.0

    contract_size = _CONTRACT_SIZE.get(symbol.upper(), _DEFAULT_CONTRACT_SIZE)

    # lot_size = risk_amount / (sl_distance * contract_size)
    # EURUSD example: $100 / (0.003 USD/unit × 100,000 units/lot) = 0.33 lots
    raw_lots = risk_amount / (sl_distance * contract_size)

    # Clamp: minimum 0.01 lot (micro), round down to 0.01
    lot_size = max(0.01, int(raw_lots * 100) / 100)

    return lot_size, round(risk_amount, 2), round(sl_distance, 6)


# ---------------------------------------------------------------------------
# RISK-03  Max open trades check
# ---------------------------------------------------------------------------

def check_max_open_trades(open_trades: int) -> tuple[bool, str]:
    """Return (ok, reason). ok=False means trade should be blocked."""
    limit: int = settings.risk["max_open_trades"]
    if open_trades >= limit:
        return False, f"Max open trades reached ({open_trades}/{limit})"
    return True, ""


# ---------------------------------------------------------------------------
# RISK-04  Daily loss limit
# ---------------------------------------------------------------------------

def check_daily_loss(state: AccountState) -> tuple[bool, str]:
    """
    Return (ok, reason). ok=False means bot should pause — daily loss limit hit.
    Compares realised daily P&L against max_daily_loss_percent of equity.
    """
    limit_pct: float = settings.risk["max_daily_loss_percent"]
    loss_pct = state.daily_loss_pct

    if loss_pct >= limit_pct:
        return False, (
            f"Daily loss limit hit: {loss_pct:.2f}% >= {limit_pct:.2f}% "
            f"(lost ${abs(state.daily_pnl):.2f})"
        )
    return True, ""


# ---------------------------------------------------------------------------
# RISK-05  SL distance validator (ATR-based)
# ---------------------------------------------------------------------------

def validate_sl_distance(
    sl_distance: float,
    atr: float,
) -> tuple[bool, str]:
    """
    Ensure SL distance is within [min_atr * ATR, max_atr * ATR].
    Prevents trades with SL that is too tight (noise-prone) or too wide
    (risk/reward destroying).
    """
    min_mult: float = settings.risk["min_sl_atr_multiplier"]
    max_mult: float = settings.risk["max_sl_atr_multiplier"]

    if atr <= 0:
        return True, ""  # cannot validate without ATR — allow through

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
) -> RiskDecision:
    """
    Full risk evaluation pipeline:
      1. Daily loss limit check
      2. Max open trades check
      3. SL distance ATR validation
      4. Position size calculation
      5. Return RiskDecision with approved flag + lot size

    Always returns a RiskDecision — never raises.
    """
    if not signal.is_actionable:
        return RiskDecision(
            approved=False,
            lot_size=0.0,
            risk_amount=0.0,
            sl_distance=0.0,
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

    # --- RISK-02: position size ---
    lot_size, risk_amount, sl_distance = calc_position_size(
        symbol=signal.symbol,
        entry=signal.entry,  # type: ignore[arg-type]
        sl=signal.sl,        # type: ignore[arg-type]
        equity=state.equity,
    )

    if lot_size <= 0:
        reasons.append("Position size calculated as 0 — entry equals SL price")
        blocked = True

    # --- RISK-05: ATR SL validation ---
    ok, reason = validate_sl_distance(sl_distance, atr)
    if not ok:
        reasons.append(reason)
        blocked = True
        logger.warning(f"Risk block (SL distance): {reason}")

    # --- Soft warnings (don't block, just inform) ---
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
        reasons=reasons,
        warnings=warnings,
    )

    logger.info(f"Risk decision for {signal.symbol}: {decision.summary}")
    return decision
