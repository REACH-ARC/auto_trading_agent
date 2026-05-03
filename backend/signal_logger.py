"""
Phase 6 — Signal Logger
LOG-01 through LOG-05
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import aiosqlite
from loguru import logger

from config import settings
from backend.claude_analyst import SignalResult
from backend.risk_manager import RiskDecision

# ---------------------------------------------------------------------------
# LOG-01  Types
# ---------------------------------------------------------------------------

Outcome = Literal["WIN", "LOSS", "BREAKEVEN", "PENDING", "SKIPPED"]

_DB_PATH = Path(settings.database["path"])


@dataclass
class SignalRecord:
    """Full persisted record — signal + risk decision + outcome."""
    id: int
    symbol: str
    direction: str
    confidence: int
    entry: float | None
    sl: float | None
    tp1: float | None
    tp2: float | None
    tp3: float | None
    lot_size: float
    risk_amount: float
    sl_distance: float
    reasoning: str
    invalidation: str
    confluence_factors: list[str]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_cost_usd: float
    outcome: Outcome
    close_price: float | None
    actual_pnl: float | None
    actual_rr: float | None
    created_at: datetime
    closed_at: datetime | None


@dataclass
class SignalStats:
    """LOG-05 — aggregated performance statistics."""
    total_signals: int
    actionable_signals: int          # BUY or SELL (not NO_TRADE)
    no_trade_signals: int
    pending_signals: int

    wins: int
    losses: int
    breakevens: int

    win_rate_pct: float              # wins / (wins + losses) × 100
    avg_rr: float                    # average actual R:R on closed trades
    total_pnl: float                 # sum of actual_pnl on closed trades
    total_api_cost_usd: float        # cumulative Claude API spend

    by_symbol: dict[str, dict]       # per-symbol breakdown
    by_direction: dict[str, int]     # BUY / SELL / NO_TRADE counts

    period_start: datetime | None    # earliest signal in DB
    period_end: datetime | None      # latest signal in DB


# ---------------------------------------------------------------------------
# LOG-02  Schema — create tables
# ---------------------------------------------------------------------------

_CREATE_SIGNALS_SQL = """
CREATE TABLE IF NOT EXISTS signals (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol              TEXT    NOT NULL,
    direction           TEXT    NOT NULL,
    confidence          INTEGER NOT NULL,
    entry               REAL,
    sl                  REAL,
    tp1                 REAL,
    tp2                 REAL,
    tp3                 REAL,
    lot_size            REAL    NOT NULL DEFAULT 0,
    risk_amount         REAL    NOT NULL DEFAULT 0,
    sl_distance         REAL    NOT NULL DEFAULT 0,
    reasoning           TEXT    NOT NULL DEFAULT '',
    invalidation        TEXT    NOT NULL DEFAULT '',
    confluence_factors  TEXT    NOT NULL DEFAULT '[]',
    input_tokens        INTEGER NOT NULL DEFAULT 0,
    output_tokens       INTEGER NOT NULL DEFAULT 0,
    cache_read_tokens   INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens  INTEGER NOT NULL DEFAULT 0,
    estimated_cost_usd  REAL    NOT NULL DEFAULT 0,
    outcome             TEXT    NOT NULL DEFAULT 'PENDING',
    close_price         REAL,
    actual_pnl          REAL,
    actual_rr           REAL,
    created_at          TEXT    NOT NULL,
    closed_at           TEXT
);
"""

_CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol    ON signals (symbol);",
    "CREATE INDEX IF NOT EXISTS idx_signals_outcome   ON signals (outcome);",
    "CREATE INDEX IF NOT EXISTS idx_signals_created   ON signals (created_at);",
    "CREATE INDEX IF NOT EXISTS idx_signals_direction ON signals (direction);",
]


async def init_db() -> None:
    """Create database directory and tables if they do not exist."""
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(_DB_PATH) as db:
        await db.execute(_CREATE_SIGNALS_SQL)
        for idx_sql in _CREATE_INDEXES_SQL:
            await db.execute(idx_sql)
        await db.commit()
    logger.info(f"Signal DB ready: {_DB_PATH}")


# ---------------------------------------------------------------------------
# LOG-03  Write signal
# ---------------------------------------------------------------------------

_INSERT_SQL = """
INSERT INTO signals (
    symbol, direction, confidence,
    entry, sl, tp1, tp2, tp3,
    lot_size, risk_amount, sl_distance,
    reasoning, invalidation, confluence_factors,
    input_tokens, output_tokens, cache_read_tokens, cache_write_tokens,
    estimated_cost_usd, outcome, created_at
) VALUES (
    :symbol, :direction, :confidence,
    :entry, :sl, :tp1, :tp2, :tp3,
    :lot_size, :risk_amount, :sl_distance,
    :reasoning, :invalidation, :confluence_factors,
    :input_tokens, :output_tokens, :cache_read_tokens, :cache_write_tokens,
    :estimated_cost_usd, :outcome, :created_at
)
"""


async def log_signal(
    signal: SignalResult,
    risk: RiskDecision,
) -> int:
    """
    LOG-03 — Persist a signal + risk decision to the DB.
    Returns the new row id.
    Outcome defaults to PENDING for actionable trades, SKIPPED for NO_TRADE / rejected.
    """
    if signal.direction == "NO_TRADE" or not risk.approved:
        outcome: Outcome = "SKIPPED"
    else:
        outcome = "PENDING"

    params = {
        "symbol":             signal.symbol,
        "direction":          signal.direction,
        "confidence":         signal.confidence,
        "entry":              signal.entry,
        "sl":                 signal.sl,
        "tp1":                signal.tp1,
        "tp2":                signal.tp2,
        "tp3":                signal.tp3,
        "lot_size":           risk.lot_size,
        "risk_amount":        risk.risk_amount,
        "sl_distance":        risk.sl_distance,
        "reasoning":          signal.reasoning,
        "invalidation":       signal.invalidation,
        "confluence_factors": json.dumps(signal.confluence_factors),
        "input_tokens":       signal.input_tokens,
        "output_tokens":      signal.output_tokens,
        "cache_read_tokens":  signal.cache_read_tokens,
        "cache_write_tokens": signal.cache_write_tokens,
        "estimated_cost_usd": signal.estimated_cost_usd,
        "outcome":            outcome,
        "created_at":         signal.created_at.isoformat(),
    }

    async with aiosqlite.connect(_DB_PATH) as db:
        cursor = await db.execute(_INSERT_SQL, params)
        await db.commit()
        row_id = cursor.lastrowid

    logger.info(f"Signal logged: id={row_id} {signal.symbol} {signal.direction} outcome={outcome}")
    return row_id  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# LOG-04  Update outcome
# ---------------------------------------------------------------------------

_UPDATE_OUTCOME_SQL = """
UPDATE signals
SET outcome     = :outcome,
    close_price = :close_price,
    actual_pnl  = :actual_pnl,
    actual_rr   = :actual_rr,
    closed_at   = :closed_at
WHERE id = :id
"""


async def update_outcome(
    signal_id: int,
    outcome: Outcome,
    close_price: float,
    actual_pnl: float | None = None,
) -> bool:
    """
    LOG-04 — Update a signal's outcome when the trade closes.
    actual_rr is auto-calculated from close_price vs entry & sl if not provided.
    Returns True if a row was updated.
    """
    # Fetch the original signal to compute actual_rr
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT entry, sl FROM signals WHERE id = ?", (signal_id,)
        ) as cursor:
            row = await cursor.fetchone()

    if row is None:
        logger.warning(f"update_outcome: signal id={signal_id} not found")
        return False

    entry = row["entry"]
    sl    = row["sl"]
    actual_rr: float | None = None

    if entry is not None and sl is not None and entry != sl:
        risk   = abs(entry - sl)
        reward = abs(close_price - entry)
        # Negative R:R means price moved against the trade
        direction_ok = (close_price > entry) if (entry > sl) else (close_price < entry)
        actual_rr = round((reward / risk) * (1 if direction_ok else -1), 2)

    async with aiosqlite.connect(_DB_PATH) as db:
        cursor = await db.execute(
            _UPDATE_OUTCOME_SQL,
            {
                "id":          signal_id,
                "outcome":     outcome,
                "close_price": close_price,
                "actual_pnl":  actual_pnl,
                "actual_rr":   actual_rr,
                "closed_at":   datetime.now(timezone.utc).isoformat(),
            },
        )
        await db.commit()
        updated = cursor.rowcount > 0

    logger.info(
        f"Outcome updated: id={signal_id} → {outcome} "
        f"close={close_price} rr={actual_rr} pnl={actual_pnl}"
    )
    return updated


# ---------------------------------------------------------------------------
# LOG-05  Stats query
# ---------------------------------------------------------------------------

async def get_stats(
    symbol: str | None = None,
    since: datetime | None = None,
) -> SignalStats:
    """
    LOG-05 — Compute aggregated performance stats.
    Optionally filter by symbol and/or a minimum created_at datetime.
    """
    conditions: list[str] = []
    params: list = []

    if symbol:
        conditions.append("symbol = ?")
        params.append(symbol)
    if since:
        conditions.append("created_at >= ?")
        params.append(since.isoformat())

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row

        # -- totals & direction counts --
        async with db.execute(f"""
            SELECT
                COUNT(*)                                        AS total,
                SUM(CASE WHEN direction != 'NO_TRADE' THEN 1 ELSE 0 END) AS actionable,
                SUM(CASE WHEN direction  = 'NO_TRADE' THEN 1 ELSE 0 END) AS no_trade,
                SUM(CASE WHEN outcome    = 'PENDING'  THEN 1 ELSE 0 END) AS pending,
                SUM(CASE WHEN outcome    = 'WIN'       THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN outcome    = 'LOSS'      THEN 1 ELSE 0 END) AS losses,
                SUM(CASE WHEN outcome    = 'BREAKEVEN' THEN 1 ELSE 0 END) AS breakevens,
                AVG(CASE WHEN outcome IN ('WIN','LOSS','BREAKEVEN')
                         THEN actual_rr END)                  AS avg_rr,
                SUM(COALESCE(actual_pnl, 0))                  AS total_pnl,
                SUM(estimated_cost_usd)                        AS total_cost,
                MIN(created_at)                                AS period_start,
                MAX(created_at)                                AS period_end
            FROM signals {where}
        """, params) as cur:
            row = await cur.fetchone()

        total      = row["total"]      or 0
        actionable = row["actionable"] or 0
        no_trade   = row["no_trade"]   or 0
        pending    = row["pending"]    or 0
        wins       = row["wins"]       or 0
        losses     = row["losses"]     or 0
        breakevens = row["breakevens"] or 0
        avg_rr     = row["avg_rr"]     or 0.0
        total_pnl  = row["total_pnl"]  or 0.0
        total_cost = row["total_cost"] or 0.0

        period_start = (
            datetime.fromisoformat(row["period_start"])
            if row["period_start"] else None
        )
        period_end = (
            datetime.fromisoformat(row["period_end"])
            if row["period_end"] else None
        )

        closed = wins + losses
        win_rate = round(wins / closed * 100, 1) if closed > 0 else 0.0

        # -- per-symbol breakdown --
        async with db.execute(f"""
            SELECT
                symbol,
                COUNT(*)                                        AS total,
                SUM(CASE WHEN outcome = 'WIN'  THEN 1 ELSE 0 END) AS wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) AS losses,
                AVG(CASE WHEN outcome IN ('WIN','LOSS','BREAKEVEN')
                         THEN actual_rr END)                   AS avg_rr,
                SUM(COALESCE(actual_pnl, 0))                   AS total_pnl
            FROM signals {where}
            GROUP BY symbol
        """, params) as cur:
            symbol_rows = await cur.fetchall()

        by_symbol: dict[str, dict] = {}
        for r in symbol_rows:
            sym = r["symbol"]
            sym_closed = (r["wins"] or 0) + (r["losses"] or 0)
            sym_wr = round((r["wins"] or 0) / sym_closed * 100, 1) if sym_closed > 0 else 0.0
            by_symbol[sym] = {
                "total":    r["total"],
                "wins":     r["wins"]    or 0,
                "losses":   r["losses"]  or 0,
                "win_rate": sym_wr,
                "avg_rr":   round(r["avg_rr"] or 0.0, 2),
                "total_pnl": round(r["total_pnl"] or 0.0, 2),
            }

        # -- direction counts --
        async with db.execute(f"""
            SELECT direction, COUNT(*) AS cnt
            FROM signals {where}
            GROUP BY direction
        """, params) as cur:
            dir_rows = await cur.fetchall()

        by_direction = {r["direction"]: r["cnt"] for r in dir_rows}

    return SignalStats(
        total_signals=total,
        actionable_signals=actionable,
        no_trade_signals=no_trade,
        pending_signals=pending,
        wins=wins,
        losses=losses,
        breakevens=breakevens,
        win_rate_pct=win_rate,
        avg_rr=round(avg_rr, 2),
        total_pnl=round(total_pnl, 2),
        total_api_cost_usd=round(total_cost, 5),
        by_symbol=by_symbol,
        by_direction=by_direction,
        period_start=period_start,
        period_end=period_end,
    )


# ---------------------------------------------------------------------------
# Helpers — fetch single record (used by server + tests)
# ---------------------------------------------------------------------------

async def get_signal(signal_id: int) -> SignalRecord | None:
    """Fetch a single signal record by id."""
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM signals WHERE id = ?", (signal_id,)
        ) as cursor:
            row = await cursor.fetchone()

    if row is None:
        return None

    return _row_to_record(row)


async def get_pending_signals() -> list[SignalRecord]:
    """Return all signals with outcome=PENDING (open trades awaiting close)."""
    async with aiosqlite.connect(_DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM signals WHERE outcome = 'PENDING' ORDER BY created_at DESC"
        ) as cursor:
            rows = await cursor.fetchall()

    return [_row_to_record(r) for r in rows]


def _row_to_record(row: aiosqlite.Row) -> SignalRecord:
    return SignalRecord(
        id=row["id"],
        symbol=row["symbol"],
        direction=row["direction"],
        confidence=row["confidence"],
        entry=row["entry"],
        sl=row["sl"],
        tp1=row["tp1"],
        tp2=row["tp2"],
        tp3=row["tp3"],
        lot_size=row["lot_size"],
        risk_amount=row["risk_amount"],
        sl_distance=row["sl_distance"],
        reasoning=row["reasoning"],
        invalidation=row["invalidation"],
        confluence_factors=json.loads(row["confluence_factors"]),
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cache_read_tokens=row["cache_read_tokens"],
        cache_write_tokens=row["cache_write_tokens"],
        estimated_cost_usd=row["estimated_cost_usd"],
        outcome=row["outcome"],
        close_price=row["close_price"],
        actual_pnl=row["actual_pnl"],
        actual_rr=row["actual_rr"],
        created_at=datetime.fromisoformat(row["created_at"]),
        closed_at=(
            datetime.fromisoformat(row["closed_at"]) if row["closed_at"] else None
        ),
    )
