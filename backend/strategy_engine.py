"""
Strategy Engine — dispatches to the active rule-based strategy.
Returns a SignalResult in the same shape as the AI analysts so the rest
of the pipeline (risk manager, Telegram, DB logger) needs no changes.
"""
from __future__ import annotations

from loguru import logger

from backend.claude_analyst import SignalResult
from backend.indicators import IndicatorResult, OHLCVData
from backend import model_manager


def run_strategy(ohlcv: OHLCVData, indicators: IndicatorResult) -> SignalResult:
    """Dispatch to the currently selected strategy and return a SignalResult."""
    strategy = model_manager.get_active_strategy()
    logger.info(
        f"Strategy engine: running '{strategy}' on {ohlcv.symbol}"
    )

    try:
        if strategy == "ema_pullback":
            from backend.strategies.ema_pullback import analyse
        elif strategy == "asian_breakout":
            from backend.strategies.asian_breakout import analyse  # type: ignore[no-redef]
        elif strategy == "sr_bounce":
            from backend.strategies.sr_bounce import analyse  # type: ignore[no-redef]
        else:
            logger.warning(f"Strategy '{strategy}' not implemented — returning NO_TRADE")
            return _no_trade(ohlcv.symbol, f"Strategy '{strategy}' not implemented")

        return analyse(ohlcv, indicators)

    except Exception as e:
        logger.error(f"Strategy engine error ({strategy}): {e}")
        return _no_trade(ohlcv.symbol, f"Strategy error: {e}")


def _no_trade(symbol: str, reason: str) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        direction="NO_TRADE",
        confidence=0,
        reasoning=reason,
        invalidation="",
        confluence_factors=[],
    )
