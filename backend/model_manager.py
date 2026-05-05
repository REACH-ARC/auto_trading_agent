"""
Runtime model selection — tracks which AI provider the agent loop should use.
Switched via Telegram /model command; survives hot-reload of the agent function.
"""
from __future__ import annotations

from config import settings as _settings

_active_model: str = "ollama"
_active_strategy: str = "ema_pullback"

_str_cfg = _settings._yaml.get("strategy", {})
_auto_trade: bool = bool(_str_cfg.get("auto_trade", True))
_scan_all_symbols: bool = bool(_str_cfg.get("scan_all_symbols", False))

AVAILABLE_MODELS: dict[str, str] = {
    "claude":    "External Model",
    "ollama":    "Ollama (local/free)",
    "strategy":  "Strategy (No AI)",
}

AVAILABLE_STRATEGIES: dict[str, str] = {
    "ema_pullback":   "EMA Pullback",
    "asian_breakout": "Asian Range Breakout",
    "sr_bounce":      "S/R Bounce + RSI",
}


def get_active_model() -> str:
    return _active_model


def set_active_model(model: str) -> None:
    global _active_model
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Model must be one of {list(AVAILABLE_MODELS)}")
    _active_model = model


def get_model_display_name() -> str:
    return AVAILABLE_MODELS.get(_active_model, _active_model)


def get_active_strategy() -> str:
    return _active_strategy


def set_active_strategy(strategy: str) -> None:
    global _active_strategy
    if strategy not in AVAILABLE_STRATEGIES:
        raise ValueError(f"Strategy must be one of {list(AVAILABLE_STRATEGIES)}")
    _active_strategy = strategy


def get_strategy_display_name() -> str:
    return AVAILABLE_STRATEGIES.get(_active_strategy, _active_strategy)


def get_auto_trade() -> bool:
    return _auto_trade


def set_auto_trade(value: bool) -> None:
    global _auto_trade
    _auto_trade = value


def get_scan_all_symbols() -> bool:
    return _scan_all_symbols


def set_scan_all_symbols(value: bool) -> None:
    global _scan_all_symbols
    _scan_all_symbols = value
