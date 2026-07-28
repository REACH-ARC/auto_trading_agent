"""
Level Watcher — monitors price against key S/R levels between M5 bar closes.

Workflow (Option 3):
  1. update_levels()  called each M5 close with fresh S/R from indicators
  2. check_price()    called every ~30 s; records new hits without triggering entry
  3. pop_hits()       called at the NEXT M5 close; returns hits, clears the store

Entry only happens at candle close — level hits provide richer context to Claude,
not an immediate trigger. This avoids wick fakeouts while preserving level awareness.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from loguru import logger

from backend.indicators import SRLevel


@dataclass
class LevelHit:
    level_price: float
    level_type: str     # "support" | "resistance"
    strength: int       # historical touch count
    hit_at: datetime    # UTC timestamp of first touch this candle
    touch_price: float  # actual price at detection time


class LevelWatcher:
    """
    Tracks intra-candle S/R touches per symbol.
    All state is in-memory; no persistence needed between restarts.
    """

    def __init__(self, tolerance_pct: float = 0.001):
        # How close price must get to a level to count as a touch (default 0.1%)
        self._tolerance_pct = tolerance_pct
        self._levels: dict[str, list[SRLevel]] = {}
        self._hits: dict[str, list[LevelHit]] = {}

    def update_levels(self, symbol: str, levels: list[SRLevel]) -> None:
        """Replace watched levels for symbol. Called once per M5 close."""
        self._levels[symbol] = levels
        logger.debug(f"LevelWatcher [{symbol}]: watching {len(levels)} S/R levels")

    def check_price(self, symbol: str, current_price: float) -> list[LevelHit]:
        """
        Check current price against watched levels.
        Returns any newly triggered hits (each level fires at most once per candle).
        """
        levels = self._levels.get(symbol, [])
        if not levels:
            return []

        already_hit = {h.level_price for h in self._hits.get(symbol, [])}
        new_hits: list[LevelHit] = []

        for level in levels:
            if level.price in already_hit:
                continue
            if abs(current_price - level.price) / level.price <= self._tolerance_pct:
                hit = LevelHit(
                    level_price=level.price,
                    level_type=level.level_type,
                    strength=level.strength,
                    hit_at=datetime.now(timezone.utc),
                    touch_price=current_price,
                )
                new_hits.append(hit)
                already_hit.add(level.price)
                logger.info(
                    f"LevelWatcher [{symbol}]: touched {level.level_type} "
                    f"{level.price} (strength={level.strength}) @ {current_price:.5f}"
                )

        if new_hits:
            self._hits.setdefault(symbol, []).extend(new_hits)

        return new_hits

    def pop_hits(self, symbol: str) -> list[LevelHit]:
        """
        Return all hits recorded during the last candle and clear the store.
        Call this at M5 bar close before running the agent.
        """
        return self._hits.pop(symbol, [])

    def clear_symbol(self, symbol: str) -> None:
        """Discard all state for a symbol (used on symbol switch)."""
        self._levels.pop(symbol, None)
        self._hits.pop(symbol, None)
