"""
Phase 7 — MT5 ZeroMQ Bridge  (SRV-02, SRV-03)
Handles message serialisation between MT5 EA and the Python backend.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from loguru import logger

from backend.indicators import Bar, OHLCVData
from backend.risk_manager import AccountState

if TYPE_CHECKING:
    from backend.claude_analyst import SignalResult
    from backend.risk_manager import RiskDecision

# ---------------------------------------------------------------------------
# Message format (MT5 EA → Python)
# ---------------------------------------------------------------------------
# {
#   "symbol": "EURUSD",
#   "account": {"equity": 10000, "balance": 10000, "open_trades": 0, "daily_pnl": 0},
#   "timeframes": {
#     "M15": [{"time": "2024-01-15T10:00:00+00:00", "open":1.1,"high":1.105,"low":1.095,"close":1.1,"volume":1000},...],
#     "H1":  [...], "H4": [...], "D1": [...]
#   }
# }

# ---------------------------------------------------------------------------
# Message format (Python → MT5 EA)
# ---------------------------------------------------------------------------
# {
#   "symbol": "EURUSD",
#   "direction": "BUY",
#   "entry": 1.101, "sl": 1.098, "tp1": 1.1055, "tp2": 1.1085, "tp3": 1.113,
#   "confidence": 78, "lot_size": 0.33, "approved": true,
#   "reasoning": "...", "signal_id": 42, "timestamp": "..."
# }


def parse_mt5_message(raw: str | bytes) -> tuple[OHLCVData, AccountState]:
    """
    Deserialise a JSON message from the MT5 EA into typed Python objects.
    Raises ValueError on malformed input.
    """
    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from MT5: {e}") from e

    symbol: str = data.get("symbol", "")
    if not symbol:
        raise ValueError("MT5 message missing 'symbol' field")

    # --- account state ---
    acct = data.get("account", {})
    account = AccountState(
        equity=float(acct.get("equity", 0)),
        balance=float(acct.get("balance", 0)),
        open_trades=int(acct.get("open_trades", 0)),
        daily_pnl=float(acct.get("daily_pnl", 0)),
    )

    # --- OHLCV bars ---
    raw_tfs: dict[str, list[dict]] = data.get("timeframes", {})
    if not raw_tfs:
        raise ValueError("MT5 message missing 'timeframes' field")

    timeframes: dict = {}
    for tf, bars_raw in raw_tfs.items():
        bars: list[Bar] = []
        for b in bars_raw:
            raw_time = b.get("time", "")
            dt = datetime.fromisoformat(
                raw_time.replace("Z", "+00:00")
            ).astimezone(timezone.utc)
            bars.append(Bar(
                time=dt,
                open=float(b["open"]),
                high=float(b["high"]),
                low=float(b["low"]),
                close=float(b["close"]),
                volume=float(b.get("volume", 0)),
            ))
        if bars:
            timeframes[tf] = bars

    if not timeframes:
        raise ValueError("MT5 message contained no valid bars")

    ohlcv = OHLCVData(
        symbol=symbol,
        timeframes=timeframes,
        received_at=datetime.now(timezone.utc),
    )
    return ohlcv, account


def serialize_signal(
    signal: SignalResult,
    risk: RiskDecision,
    signal_id: int,
) -> str:
    """Serialise a processed signal into a JSON string for the MT5 EA."""
    payload: dict[str, Any] = {
        "symbol":     signal.symbol,
        "direction":  signal.direction,
        "entry":      signal.entry,
        "sl":         signal.sl,
        "tp1":        signal.tp1,
        "tp2":        signal.tp2,
        "tp3":        signal.tp3,
        "confidence": signal.confidence,
        "lot_size":   risk.lot_size,
        "approved":   risk.approved,
        "reasoning":  signal.reasoning,
        "invalidation": signal.invalidation,
        "risk_reward":  signal.risk_reward,
        "signal_id":  signal_id,
        "timestamp":  signal.created_at.isoformat(),
    }
    return json.dumps(payload, default=str)
