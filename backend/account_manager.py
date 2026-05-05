"""
Account type manager — module-level singleton for currency/suffix state.
Set once when MT5 connects; read by fetcher, tools, and Telegram watchlist.
"""
from __future__ import annotations

_currency: str = "USD"


def get_currency() -> str:
    return _currency


def set_currency(currency: str) -> None:
    global _currency
    _currency = currency.upper()


def is_cent() -> bool:
    return _currency == "USC"


def _get_suffix() -> str:
    from config import settings
    sym_cfg = settings.symbols
    if is_cent():
        return str(sym_cfg.get("cent_suffix", "c"))
    return str(sym_cfg.get("standard_suffix", "m"))


def apply_suffix(base_symbol: str) -> str:
    """Strip any known suffix then append the account-appropriate one."""
    return strip_suffix(base_symbol) + _get_suffix()


def strip_suffix(symbol: str) -> str:
    """Remove a known account suffix and return the bare base symbol."""
    from config import settings
    sym_cfg = settings.symbols
    for sfx in (
        str(sym_cfg.get("cent_suffix", "c")),
        str(sym_cfg.get("standard_suffix", "m")),
    ):
        if symbol.upper().endswith(sfx.upper()):
            return symbol[: -len(sfx)]
    return symbol


def get_watchlist() -> list[str]:
    """Return watchlist symbols with the correct suffix for the active account type."""
    from config import settings
    bases: list[str] = settings.symbols.get("watchlist_base", [])
    return [apply_suffix(b) for b in bases]
