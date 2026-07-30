"""
Claude Trading Agent — agentic loop with MT5 tool use.
Claude receives a market trigger, then autonomously calls tools to:
  - inspect account state and open positions
  - fetch and analyse market data
  - manage existing positions (trail SL, partial close)
  - place new orders when a high-quality setup exists
  - send a Telegram summary of every decision made
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

import anthropic
from loguru import logger

from config import settings
from backend.risk_manager import AccountState
import backend.mt5_tools as tools
import backend.model_manager as model_manager

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_TEMPLATE = """\
You are an autonomous trading agent with live access to MetaTrader 5.
You have tools to read market data, manage open positions, and execute trades.
You are triggered every time a new M5 bar closes.

## Decision loop — follow this exact order every trigger:

### Step 1 — Account check
Call get_account_info().
- If daily_loss_remaining <= 0: send a warning update and STOP. No new trades.
- If open_trades >= 3: skip Step 3 (no new entries), still do Step 2.

### Step 2 — Position management
Call get_open_positions(). MT5 handles TP automatically — do NOT manually close at TP1/TP2.
Your only job here is early exit protection:
- r_moved <= -0.9 (price within 10% of SL): check if the setup is still valid.
  If the key structure level that justified the entry is broken, call close_position() to exit early.
  If structure is still intact, leave it — let SL or TP handle the exit.
- r_moved >= 1.0 (trade is profitable by 1R): move SL to breakeven via modify_position() to lock in no-loss.
- Otherwise: leave the position alone. Do not interfere with winning trades.

### Step 3 — Market analysis
Call get_market_snapshot(symbol) for the primary symbol.
Analyse using this framework:

  0. MARKET REGIME — CLASSIFY FIRST, before anything else:
     - Is H4 ATR expanding or contracting vs its 20-period average?
     - Is price making new HH/HL (uptrend) or LH/LL (downtrend) on H4?
     - Or is price bouncing between the SAME support and resistance repeatedly? → RANGING
     If RANGING: Do NOT trade breakdowns or breakouts — they will fake out. Only trade bounces off the range edges.
     If TRENDING: Trade pullbacks to structure in the trend direction. Breakouts are valid with momentum.
  
  1. TREND  — H4 + D1 EMA alignment and price structure (HH/HL or LH/LL)
  2. STRUCTURE — where is price relative to key S/R levels?
  3. MOMENTUM — RSI and MACD confluence across M5, H1, H4
     CRITICAL: ALL four timeframes (M5, H1, H4, D1) must agree on direction. If ANY timeframe has active opposing momentum (RSI divergence, MACD opposing), reduce confidence by 15-20 points.
  4. ENTRY — prefer limit entries at S/R zones; breakout entries need strong momentum
  5. CONFIDENCE — score 0–100 counting confluence factors:
       +15 trend alignment (H4 + D1 agree)
       +15 price at key S/R level
       +10 RSI confluence (not overbought/oversold against direction)
       +10 MACD histogram supports direction on H1+H4
       +10 M5 momentum confirms direction (NOT opposing!)
       +10 no conflicting open position in same symbol
       +10 high liquidity session (for crypto: strong volume/volatility period)
       +20 additional strong factors (divergence, clean break, engulfing candle, etc.)
       -20 if market is RANGING and you're trading a breakdown/breakout
       -15 if ANY timeframe has active opposing momentum

### Step 4 — Trade decision
<<STEP4>>

### Step 5 — Telegram update (ALWAYS — even for no-trade)
Call send_update() to summarise the cycle. You MUST pass the structured JSON fields:
- message_type: Pick the most appropriate category (e.g., new_position, no_trade, info)
- symbol: The active symbol
- reason: Market bias, levels, and rationale for your decision
- account_balance and daily_pnl: From your account check
- Other fields as appropriate (entry, sl, tp, confidence, etc.)
- If there is an existing position with a 'time_remaining' field, YOU MUST pass the time left until exit into the 'time_remaining' JSON field.

## Crypto-specific rules (BTCUSDc, ETHUSDc, etc.)
- 24/7 market — do NOT penalise signals for being outside forex session hours
- Volatility is the session equivalent — trade during high-volume periods
- BTC often trends strongly; ride winners with trailing stops

## Trading philosophy — QUALITY OVER QUANTITY
- You should aim for 1-2 HIGH-QUALITY trades per day maximum. If no setup meets all criteria, it is perfectly fine to trade ZERO times. Patience is rewarded.
- BOTH directions are valid. Do NOT lock into one direction just because the daily chart is bearish. In a ranging market, buying at support is just as valid as selling at resistance.
- Never chase. If you missed the move, wait for the next setup.

## Hard rules (also enforced server-side)
- Risk exactly the fixed SL amount per trade: 1 000 USC (cent) or $50 USD (standard) — never % of balance
- Never exceed 3 open trades
- Never trade when daily loss limit is breached
- SL must be structure-based AND within the hard cap: 200 pips (forex) / $60 (XAUUSD) / 3% (crypto)
- Always provide tp1, tp2, and tp3 when calling place_order — never omit them
- If a tool returns an error, log it and adapt — do not retry blindly
- STRICT MOMENTUM ALIGNMENT: You CANNOT enter a trade if M5 or H1 have active opposing momentum. Example: if you want to SELL, M5 and H1 must NOT be bullish (no active FVG, no strong green candles, RSI must not be rising). Wait for ALL timeframes to align.
- DIRECTIONAL COOLDOWN (CIRCUIT BREAKER — enforced in code): After a Stop Loss hit, the system BLOCKS the same direction for 2 hours. If you try, the order will be rejected. After cooldown, you must have confidence >= 90 to re-enter that direction.
- FAKEOUT AWARENESS: If a key S/R level is broken, immediately reassess your bias. If the same level has been broken and reclaimed 2+ times, the market is RANGING — stop trading breakdowns.
- MAX 3 TRADES PER SYMBOL PER DAY (enforced in code). Orders beyond this limit will be rejected.
"""

_STEP4_CENT = """\
NOTE: This is a **cent account** (USC). 100 USC = 1 USD. Account balance displayed in USC.

- If confidence >= 85 AND no open position in the same direction for this symbol:
  a. Call get_symbol_info(symbol) to confirm tick value and lot constraints
  b. SL CONSTRAINT — place SL at the nearest structural invalidation (swing high/low).
  c. TP & VALIDITY CHECK:
       - Set TP1 = entry ± (SL_distance × 1.5)
       - Set TP2 = entry ± (SL_distance × 2.0)
       - Set TP3 = entry ± (SL_distance × 2.5)
       - For XAUUSD, the final TP3 gap MUST be between $15 and $25 from entry. 
         (This means your structural SL distance MUST be between $6 and $10).
       - If the required SL distance pushes TP3 outside this $15-$25 gap, or if structure exceeds hard caps for other pairs (200 pips forex / 3% crypto), skip this trade — return NO_TRADE.
  d. Calculate lot_size using the FIXED SL amount (1 000 USC per trade — do NOT use % of balance):
       lot_size = 1000 / (SL_distance_in_price × tick_value / tick_size)
     Then scale by confidence:
       - confidence 85–89 → use 85% of calculated lots (moderate)
       - confidence 90–94 → use 100% of calculated lots (confident)
       - confidence 95–100 → use 100% of calculated lots (high conviction)
     Clamp to broker min_lot and max_lot from get_symbol_info(). Round to broker lot_step.
  e. Call place_order() with order_type="MARKET", passing sl, tp1, tp2, and tp3.
     MT5 hard-closes the position at TP3; the trade manager steps SL to TP1 then TP2 as each is hit.
- If confidence < 85: NO_TRADE. Note what is missing."""

_STEP4_STANDARD = """\
NOTE: This is a **standard account** (USD). Account balance is in real USD.

- If confidence >= 85 AND no open position in the same direction for this symbol:
  a. Call get_symbol_info(symbol) to confirm tick value and lot constraints
  b. SL CONSTRAINT — place SL at the nearest structural invalidation (swing high/low).
  c. TP & VALIDITY CHECK:
       - Set TP1 = entry ± (SL_distance × 1.5)
       - Set TP2 = entry ± (SL_distance × 2.0)
       - Set TP3 = entry ± (SL_distance × 2.5)
       - For XAUUSD, the final TP3 gap MUST be between $15 and $25 from entry. 
         (This means your structural SL distance MUST be between $6 and $10).
       - If the required SL distance pushes TP3 outside this $15-$25 gap, or if structure exceeds hard caps for other pairs (200 pips forex / 3% crypto), skip this trade — return NO_TRADE.
  d. Calculate lot_size using the FIXED SL amount ($50 USD per trade — do NOT use % of balance):
       lot_size = 50 / (SL_distance_in_price × tick_value / tick_size)
     Then scale by confidence:
       - confidence 85–89 → use 85% of calculated lots (moderate)
       - confidence 90–94 → use 100% of calculated lots (full risk)
       - confidence 95–100 → use 100% of calculated lots (high conviction)
     Clamp to broker min_lot and max_lot from get_symbol_info(). Round to broker lot_step.
  e. Call place_order() with order_type="MARKET", passing sl, tp1, tp2, and tp3.
     MT5 hard-closes the position at TP3; the trade manager steps SL to TP1 then TP2 as each is hit.
- If confidence < 85: NO_TRADE. Note what is missing."""


_ALERT_ONLY_SUFFIX = """

## ALERT-ONLY MODE — auto_trade is OFF
place_order() is NOT available. Do not attempt to trade.
Analyse the market and manage existing positions as normal.
In your send_update(), include the trade you WOULD have placed:
direction, entry, SL, TP, lot size, and reason — so the trader can decide manually.
"""

_PRICE_ALERT_INSTRUCTIONS = """
## Price alerts — smarter market monitoring
Call set_price_alert() to register key price levels you want to watch. When price crosses one of these levels, 
a full agent cycle fires at the next bar close with your hint as context.
This replaces blind 5-min polling with targeted, event-driven analysis, saving API costs.

IF YOU HAVE AN OPEN POSITION:
  Set alerts near your SL (to check for early exit) or TP (to lock in profit).

IF YOU DO NOT HAVE AN OPEN POSITION (i.e. NO_TRADE):
  Set alerts at the key S/R levels you are waiting for before entering. 
  Example: If you want to SELL but price is in the middle of a range, set_price_alert(symbol, 4050, "above", "Resistance hit — look for bearish rejection to SELL").
  The system will sleep and ONLY wake you up when price hits your alert. 

Always call clear_price_alerts() before setting a new batch so stale levels don't pile up.
"""


def _make_system_prompt(is_cent: bool, auto_trade: bool = True, has_open_positions: bool = False, skip_analysis: bool = False) -> str:
    sl_usc = model_manager.get_sl_amount_usc()
    sl_usd = model_manager.get_sl_amount_usd()

    if skip_analysis:
        step4 = "NOTE: You are in position management mode. Review open positions and manage them. Do not place new trades."
    elif is_cent:
        step4 = (
            f"NOTE: This is a **cent account** (USC). 100 USC = 1 USD. Account balance displayed in USC.\n\n"
            f"- If confidence >= 65 AND no open position in the same direction for this symbol:\n"
            f"  a. Call get_symbol_info(symbol) to confirm tick value and lot constraints\n"
            f"  b. SL CONSTRAINT — place SL at the nearest structural invalidation (swing high/low).\n"
            f"     Target 100–200 pips for forex; $10–$60 for XAUUSD; 0.5–3% for crypto. Hard cap: 200 pips / $60 / 3%.\n"
            f"     If structure is beyond the cap, skip this trade — return NO_TRADE.\n"
            f"  c. Calculate lot_size using the FIXED SL amount ({sl_usc:,.0f} USC per trade"
            f" — do NOT use % of balance):\n"
            f"       lot_size = {sl_usc:,.0f} / (SL_distance_in_price x tick_value / tick_size)\n"
            f"     Then scale by confidence:\n"
            f"       - confidence 65-74  -> use 70% of calculated lots (cautious)\n"
            f"       - confidence 75-84  -> use 85% of calculated lots (moderate)\n"
            f"       - confidence 85-94  -> use 100% of calculated lots (confident)\n"
            f"       - confidence 95-100 -> use 100% of calculated lots (high conviction)\n"
            f"     Clamp to broker min_lot and max_lot from get_symbol_info(). Round to broker lot_step.\n"
            f"  d. Set all three TPs from entry:\n"
            f"       TP1 = entry +/- (SL_distance x 1.5)   — 150 pips at 100-pip SL\n"
            f"       TP2 = entry +/- (SL_distance x 2.0)   — 200 pips at 100-pip SL\n"
            f"       TP3 = entry +/- (SL_distance x 2.5)   — 250 pips at 100-pip SL\n"
            f"  e. Call place_order() with order_type=\"MARKET\", passing sl, tp1, tp2, and tp3.\n"
            f"     MT5 hard-closes at TP3; trade manager steps SL to TP1 then TP2 as each is hit.\n"
            f"- If confidence < 65: NO_TRADE. Note what is missing."
        )
    else:
        step4 = (
            f"NOTE: This is a **standard account** (USD). Account balance is in real USD.\n\n"
            f"- If confidence >= 65 AND no open position in the same direction for this symbol:\n"
            f"  a. Call get_symbol_info(symbol) to confirm tick value and lot constraints\n"
            f"  b. SL CONSTRAINT — place SL at the nearest structural invalidation (swing high/low).\n"
            f"     Target 100–200 pips for forex; $10–$60 for XAUUSD; 0.5–3% for crypto. Hard cap: 200 pips / $60 / 3%.\n"
            f"     If structure is beyond the cap, skip this trade — return NO_TRADE.\n"
            f"  c. Calculate lot_size using the FIXED SL amount (${sl_usd:,.2f} USD per trade"
            f" — do NOT use % of balance):\n"
            f"       lot_size = {sl_usd:,.2f} / (SL_distance_in_price x tick_value / tick_size)\n"
            f"     Then scale by confidence:\n"
            f"       - confidence 65-74  -> use 70% of calculated lots (cautious)\n"
            f"       - confidence 75-84  -> use 85% of calculated lots (moderate)\n"
            f"       - confidence 85-94  -> use 100% of calculated lots (full risk)\n"
            f"       - confidence 95-100 -> use 100% of calculated lots (high conviction)\n"
            f"     Clamp to broker min_lot and max_lot from get_symbol_info(). Round to broker lot_step.\n"
            f"  d. Set all three TPs from entry:\n"
            f"       TP1 = entry +/- (SL_distance x 1.5)   — 150 pips at 100-pip SL\n"
            f"       TP2 = entry +/- (SL_distance x 2.0)   — 200 pips at 100-pip SL\n"
            f"       TP3 = entry +/- (SL_distance x 2.5)   — 250 pips at 100-pip SL\n"
            f"  e. Call place_order() with order_type=\"MARKET\", passing sl, tp1, tp2, and tp3.\n"
            f"     MT5 hard-closes at TP3; trade manager steps SL to TP1 then TP2 as each is hit.\n"
            f"- If confidence < 65: NO_TRADE. Note what is missing."
        )

    prompt = _SYSTEM_PROMPT_TEMPLATE.replace("<<STEP4>>", step4)
    if not auto_trade:
        prompt += _ALERT_ONLY_SUFFIX
    
    # Always include alert instructions so AI can set entry alerts when NO_TRADE
    prompt += _PRICE_ALERT_INSTRUCTIONS
    return prompt


def _get_tools(auto_trade: bool = True, skip_analysis: bool = False) -> list[dict]:
    """Return the agent tool list. Removes place_order and get_symbol_info when managing positions or auto_trade is off."""
    tools = _TOOLS
    if not auto_trade or skip_analysis:
        tools = [t for t in tools if t["name"] not in ("place_order", "get_symbol_info")]
    return tools

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

_TOOLS = [
    {
        "name": "get_account_info",
        "description": (
            "Get live account state: equity, balance, free margin, open trade count, "
            "daily P&L, daily loss limit, and how much loss headroom remains."
        ),
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_open_positions",
        "description": (
            "List all open positions with ticket, direction, lot size, entry price, "
            "current price, SL, TP, profit in USD, and R moved (how many R the trade has gained)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Filter by symbol (optional — omit for all positions)",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_market_snapshot",
        "description": (
            "Fetch live OHLCV data and compute all indicators (RSI, MACD, ATR, EMA 20/50/200, "
            "S/R levels) across M5/H1/H4/D1. Returns a formatted analysis snapshot. "
            "Always call this before making any trade decision."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "e.g. BTCUSDc, XAUUSDc"}
            },
            "required": ["symbol"],
        },
    },
    {
        "name": "get_symbol_info",
        "description": (
            "Get broker specifications for a symbol: current bid/ask, spread, "
            "tick size, tick value, contract size, min/max lot, lot step. "
            "Call this before calculating lot size for a new order."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"symbol": {"type": "string"}},
            "required": ["symbol"],
        },
    },
    {
        "name": "place_order",
        "description": (
            "Place a BUY or SELL MARKET order on MT5 at the current price. "
            "Server-side guards enforce min/max lot size, max open trades, and daily loss limit. "
            "SL must be structure-based and within the hard pip cap. "
            "Always pass tp1, tp2, and tp3 — MT5 hard-closes at TP3, trade manager steps SL through TP1→TP2."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "direction": {"type": "string", "enum": ["BUY", "SELL"]},
                "lot_size": {"type": "number", "description": "Position size in lots — calculated from fixed SL amount, clamped to broker min/max from get_symbol_info()"},
                "sl": {"type": "number", "description": "Stop loss price (structure-based, within hard pip cap)"},
                "tp1": {"type": "number", "description": "Take profit 1 — 1.5R from entry (SL_distance × 1.5)"},
                "tp2": {"type": "number", "description": "Take profit 2 — 2.0R from entry (SL_distance × 2.0)"},
                "tp3": {"type": "number", "description": "Take profit 3 — 2.5R from entry (SL_distance × 2.5); MT5 closes here"},
                "time_stop_hours": {"type": "integer", "description": "Optional timeframe (hours) to hold position before auto-closing at market if target not reached. E.g. 4 for intraday, 24 for swing."},
                "comment": {"type": "string", "description": "Trade label (max 31 chars)"},
            },
            "required": ["symbol", "direction", "lot_size", "sl", "tp1", "tp2", "tp3"],
        },
    },
    {
        "name": "modify_position",
        "description": (
            "Modify the stop loss and/or take profit of an open position. "
            "Use this to move SL to breakeven after TP1, or trail SL after TP2."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "integer", "description": "Position ticket number"},
                "sl": {"type": "number", "description": "New stop loss price"},
                "tp": {"type": "number", "description": "New take profit price"},
            },
            "required": ["ticket"],
        },
    },
    {
        "name": "close_position",
        "description": (
            "Close a position fully or partially by ticket. "
            "Omit lot_size for a full close; provide lot_size for a partial close."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ticket": {"type": "integer", "description": "Position ticket number"},
                "lot_size": {
                    "type": "number",
                    "description": "Lots to close (omit = full close)",
                },
            },
            "required": ["ticket"],
        },
    },
    {
        "name": "send_update",
        "description": (
            "Send a Telegram message summarising this agent cycle's decisions. "
            "Always call this at the end — even for no-trade cycles. Use structured JSON fields."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message_type": {
                    "type": "string",
                    "enum": ["new_position", "manage_position", "news", "force_close", "no_trade", "info"],
                    "description": "The category of the update."
                },
                "symbol": {"type": "string", "description": "The symbol being analysed (e.g., XAUUSDm)"},
                "direction": {"type": "string", "enum": ["BUY", "SELL", "NONE"], "description": "Trade direction if applicable."},
                "entry_price": {"type": "number", "description": "Entry price for new trades, or current price for info"},
                "sl": {"type": "number", "description": "Stop Loss price"},
                "tp": {"type": "number", "description": "Take Profit price"},
                "r_moved": {"type": "number", "description": "Current R-multiple if managing a position"},
                "reason": {"type": "string", "description": "Short explanation of the decision (HTML supported)"},
                "confidence": {"type": "integer", "description": "Confidence score 0-100"},
                "account_balance": {"type": "number", "description": "Current account balance"},
                "daily_pnl": {"type": "number", "description": "Current daily P&L"},
                "time_remaining": {"type": "string", "description": "Time left until auto-close (e.g., '2.5h left') if applicable"}
            },
            "required": ["message_type", "symbol", "reason"],
        },
    },
    {
        "name": "set_price_alert",
        "description": (
            "Register a price level for the bot to monitor between bar closes. "
            "When price crosses this level, a full agent cycle fires at the next bar close "
            "with your hint as context — so you only do deep analysis when something you flagged happens. "
            "Use this after opening a position to watch for SL proximity, TP approach, or key S/R breaks. "
            "Set 2–4 alerts max. Always call clear_price_alerts() first to remove stale levels."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "Trading symbol, e.g. XAUUSDc"},
                "price": {"type": "number", "description": "The price level to watch"},
                "condition": {
                    "type": "string",
                    "enum": ["above", "below"],
                    "description": "Trigger when price moves above or below this level",
                },
                "hint": {
                    "type": "string",
                    "description": (
                        "What to analyse when this alert fires — e.g. "
                        "'SL proximity — check if structure broken for early exit'"
                    ),
                },
            },
            "required": ["symbol", "price", "condition", "hint"],
        },
    },
    {
        "name": "clear_price_alerts",
        "description": (
            "Remove all active price alerts for a symbol. "
            "Call this before setting a new batch of alerts so stale levels don't accumulate."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
            },
            "required": ["symbol"],
        },
    },
]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    name: str
    input: dict
    result: dict


@dataclass
class AgentResult:
    symbol: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    trade_placed: bool = False
    trade_ticket: int | None = None
    positions_managed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    error: str | None = None
    completed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def estimated_cost_usd(self) -> float:
        return round(
            (self.input_tokens / 1_000_000) * 3.00
            + (self.output_tokens / 1_000_000) * 15.00
            + (self.cache_read_tokens / 1_000_000) * 0.30
            + (self.cache_write_tokens / 1_000_000) * 3.75,
            6,
        )


# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


# ---------------------------------------------------------------------------
# OpenAI-compat client for Ollama
# ---------------------------------------------------------------------------

def _openai_tools(auto_trade: bool = True) -> list[dict]:
    """Convert Anthropic tool schemas to OpenAI function-call format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in _get_tools(auto_trade)
    ]


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def _dispatch_tool(name: str, tool_input: dict) -> dict:
    """Run a tool synchronously (called via asyncio.to_thread)."""
    if name == "get_account_info":
        return tools.get_account_info()
    if name == "get_open_positions":
        return tools.get_open_positions(tool_input.get("symbol"))
    if name == "get_market_snapshot":
        return tools.get_market_snapshot(tool_input["symbol"])
    if name == "get_symbol_info":
        return tools.get_symbol_info(tool_input["symbol"])
    if name == "place_order":
        return tools.place_order(**tool_input)
    if name == "modify_position":
        return tools.modify_position(**tool_input)
    if name == "close_position":
        return tools.close_position(**tool_input)
    if name == "set_price_alert":
        from backend.agent_alert_manager import get_alert_manager
        get_alert_manager().set_alert(
            tool_input["symbol"],
            float(tool_input["price"]),
            tool_input["condition"],
            tool_input.get("hint", ""),
        )
        return {"success": True, "alert": f"{tool_input['condition']} {tool_input['price']:.5f}"}
    if name == "clear_price_alerts":
        from backend.agent_alert_manager import get_alert_manager
        get_alert_manager().clear(tool_input["symbol"])
        return {"success": True}
    return {"error": f"Unknown tool: {name}"}


async def _execute_tool(
    name: str,
    tool_input: dict,
    telegram_notifier=None,
) -> dict:
    """Async dispatcher — Telegram is handled natively; MT5 calls run in a thread."""
    if name == "send_update":
        if telegram_notifier is not None:
            try:
                await telegram_notifier.send_agent_update(tool_input)
                return {"success": True}
            except Exception as e:
                logger.warning(f"Telegram send_update failed: {e}")
                return {"success": False, "error": str(e)}
        logger.info(f"[Telegram disabled] Agent update: {tool_input.get('reason', 'No reason provided')}")
        return {"success": True, "note": "Telegram disabled"}

    return await asyncio.to_thread(_dispatch_tool, name, tool_input)


# ---------------------------------------------------------------------------
# Ollama agent loop (OpenAI-compat API at localhost:11434)
# ---------------------------------------------------------------------------

async def _run_agent_ollama(
    symbol: str,
    account: AccountState,
    telegram_notifier=None,
    max_iterations: int = 15,
    level_hits=None,
    initial_prompt: str = "",
    auto_trade: bool = True,
) -> AgentResult:
    try:
        from openai import AsyncOpenAI
    except ImportError:
        raise RuntimeError("openai package required for Ollama — run: pip install openai>=1.0.0")

    result = AgentResult(symbol=symbol)
    ollama_cfg = settings.ollama
    base_url: str = ollama_cfg.get("base_url", "http://localhost:11434/v1")
    model_name: str = ollama_cfg.get("model", "gemini-3-flash-preview:latest")

    client = AsyncOpenAI(base_url=base_url, api_key="ollama")
    ot = _openai_tools(auto_trade)
    system_prompt = _make_system_prompt(account.is_cent, auto_trade)

    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_prompt},
    ]

    try:
        for iteration in range(max_iterations):
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=ot,
                tool_choice="auto",
            )

            choice = response.choices[0]
            finish = choice.finish_reason

            # Accumulate token usage
            if response.usage:
                result.input_tokens += response.usage.prompt_tokens or 0
                result.output_tokens += response.usage.completion_tokens or 0

            # Build assistant message for history
            assistant_msg: dict = {"role": "assistant", "content": choice.message.content or ""}
            if choice.message.tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in choice.message.tool_calls
                ]
            messages.append(assistant_msg)

            if finish == "stop" or not choice.message.tool_calls:
                logger.info(f"Ollama agent finished in {iteration + 1} iteration(s) for {symbol}")
                break

            # Execute tool calls
            for tc in choice.message.tool_calls:
                tool_input = json.loads(tc.function.arguments or "{}")
                logger.info(f"Ollama agent → {tc.function.name}({json.dumps(tool_input)[:150]})")
                tool_result = await _execute_tool(tc.function.name, tool_input, telegram_notifier)
                logger.info(f"Ollama agent ← {tc.function.name}: {json.dumps(tool_result)[:250]}")

                result.tool_calls.append(ToolCall(name=tc.function.name, input=tool_input, result=tool_result))

                if tc.function.name == "place_order" and tool_result.get("success"):
                    result.trade_placed = True
                    result.trade_ticket = tool_result.get("ticket")
                    
                    # Log the agent trade to the database
                    try:
                        from backend.signal_logger import log_signal
                        from backend.claude_analyst import SignalResult
                        from backend.risk_manager import RiskDecision
                        
                        entry_price = float(tool_result.get("entry_price") or tool_input.get("limit_price", 0.0))
                        
                        sig = SignalResult(
                            symbol=symbol,
                            direction=tool_input.get("direction", "BUY"),
                            confidence=tool_input.get("confidence", 80),
                            reasoning=tool_input.get("comment", "Agent Trade"),
                            invalidation="See agent history",
                            confluence_factors=[],
                            entry=entry_price,
                            sl=tool_input.get("sl"),
                            tp1=tool_input.get("tp1"),
                            tp2=tool_input.get("tp2"),
                            tp3=tool_input.get("tp3"),
                        )
                        risk = RiskDecision(
                            approved=True,
                            lot_size=tool_input.get("lot_size", 0.0),
                            risk_amount=0.0,
                            sl_distance=abs(entry_price - tool_input.get("sl", 0.0))
                        )
                        await log_signal(sig, risk)
                    except Exception as e:
                        logger.error(f"Failed to log Ollama agent trade to DB: {e}")

                elif tc.function.name in ("modify_position", "close_position") and tool_result.get("success"):
                    result.positions_managed += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })
        else:
            logger.warning(f"Ollama agent hit max_iterations ({max_iterations}) for {symbol}")

    except Exception as e:
        logger.error(f"Ollama agent loop error for {symbol}: {e}")
        result.error = str(e)

    logger.info(
        f"Ollama cycle done — {symbol} | "
        f"trade={result.trade_placed} ticket={result.trade_ticket} "
        f"managed={result.positions_managed} | "
        f"tokens in={result.input_tokens} out={result.output_tokens}"
    )
    return result


# ---------------------------------------------------------------------------
# Main agent entry point
# ---------------------------------------------------------------------------

def _build_initial_prompt(
    symbol: str,
    account: AccountState,
    level_hits=None,
    news_events=None,
    alert_hits=None,
    skip_analysis: bool = False,
) -> str:
    """Build the shared trigger prompt for both Claude and Ollama backends."""
    level_context = ""
    if level_hits:
        lines = []
        for hit in level_hits:
            side = "above" if hit.touch_price >= hit.level_price else "below"
            lines.append(
                f"  • {hit.level_type.upper()} {hit.level_price:.5f} "
                f"(strength={hit.strength}) — touched at {hit.hit_at.strftime('%H:%M UTC')}, "
                f"candle closed {side} this level"
            )
        level_context = (
            "\n\n⚡ LEVEL ALERT — price tested the following S/R zone(s) during this candle:\n"
            + "\n".join(lines)
            + "\nFactor this into your confidence scoring. A clean candle close beyond a "
            "key level is a high-quality confirmation signal."
        )

    news_context = ""
    if news_events:
        now = datetime.now(timezone.utc)
        lines = []
        for e in news_events:
            if e.is_blocking(now):
                mins_since = int((now - e.event_time).total_seconds() / 60)
                status = "ACTIVE — released" if mins_since >= 0 else "ACTIVE — imminent"
            else:
                mins_ahead = int((e.event_time - now).total_seconds() / 60)
                status = f"in {mins_ahead} min"
            lines.append(
                f"  • [{e.impact}] {e.currency} '{e.title}' "
                f"@ {e.event_time.strftime('%H:%M UTC')} ({status})"
            )
        news_context = (
            "\n\n📰 NEWS ALERT — high-impact event(s) detected for this symbol:\n"
            + "\n".join(lines)
            + "\n\nYou are in AI agent mode. News events are TRADING OPPORTUNITIES — do NOT skip analysis.\n"
            "- Analyse the price reaction: is there a clear directional move or reversal forming?\n"
            "- Check if the news candle direction aligns with the technical trend (H4/D1).\n"
            "- For news-driven entries, require confidence >= 75 (volatility demands higher conviction).\n"
            "- Use a wider SL beyond the spike high/low to avoid premature stop-outs.\n"
            "- If no clear directional bias yet, set NO_TRADE and explain what signal you are waiting for."
        )

    alert_context = ""
    if alert_hits:
        lines = []
        for a in alert_hits:
            lines.append(
                f"  • {a.condition.upper()} {a.price:.5f} — \"{a.hint}\""
            )
        alert_context = (
            "\n\n🔔 PRICE ALERT TRIGGERED — your registered alert(s) fired on this bar:\n"
            + "\n".join(lines)
            + "\nRun your full 5-step decision loop. Use your hint above to guide Step 3 analysis."
        )

    cycle_instruction = "Run your full decision loop now."
    if skip_analysis and not alert_hits and not level_hits and not news_events:
        cycle_instruction = (
            "⏭ POSITION MANAGEMENT ONLY — no level alerts or price alerts triggered.\n"
            "Skip Step 3 (market analysis) and Step 4 (new entries) — they are not needed this bar.\n"
            "Run Steps 1, 2, and 5 only.\n"
            "In Step 5, briefly note your open position status and what you are watching.\n"
            "Update your set_price_alert() levels if your plan has changed."
        )

    return (
        f"New M5 bar closed — {symbol}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        f"{level_context}"
        f"{news_context}"
        f"{alert_context}\n\n"
        f"Current account snapshot (cached — fetch fresh via get_account_info):\n"
        f"  Equity={account.equity:.2f}  Balance={account.balance:.2f}  "
        f"Open trades={account.open_trades}  Daily P&L={account.daily_pnl:+.2f}\n\n"
        f"{cycle_instruction}"
    )


async def _run_agent_claude(
    symbol: str,
    account: AccountState,
    telegram_notifier=None,
    max_iterations: int = 15,
    level_hits=None,
    auto_trade: bool = True,
    news_events=None,
    alert_hits=None,
    skip_analysis: bool = False,
) -> AgentResult:
    """Claude (Anthropic) backend — uses tool_use blocks and prompt caching."""
    result = AgentResult(symbol=symbol)
    client = _get_client()
    has_positions = account.open_trades > 0
    initial_prompt = _build_initial_prompt(
        symbol, account, level_hits, news_events, alert_hits, skip_analysis
    )
    system_prompt = _make_system_prompt(account.is_cent, auto_trade, has_positions, skip_analysis)
    active_tools = _get_tools(auto_trade, skip_analysis)

    messages: list[dict] = [{"role": "user", "content": initial_prompt}]
    total_input = total_output = total_cache_read = total_cache_write = 0

    try:
        for iteration in range(max_iterations):
            response = await asyncio.to_thread(
                client.messages.create,
                model=settings.claude["model"],
                max_tokens=4096,
                system=[
                    {
                        "type": "text",
                        "text": system_prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                tools=active_tools,
                messages=messages,
            )

            usage = response.usage
            total_input += getattr(usage, "input_tokens", 0)
            total_output += getattr(usage, "output_tokens", 0)
            total_cache_read += getattr(usage, "cache_read_input_tokens", 0)
            total_cache_write += getattr(usage, "cache_creation_input_tokens", 0)

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                logger.info(f"Agent finished in {iteration + 1} iteration(s) for {symbol}")
                break

            if response.stop_reason != "tool_use":
                logger.warning(f"Agent unexpected stop_reason={response.stop_reason}")
                break

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue

                logger.info(f"Agent → {block.name}({json.dumps(block.input)[:150]})")
                tool_result = await _execute_tool(block.name, block.input, telegram_notifier)
                logger.info(f"Agent ← {block.name}: {json.dumps(tool_result)[:250]}")

                result.tool_calls.append(ToolCall(name=block.name, input=block.input, result=tool_result))

                if block.name == "place_order" and tool_result.get("success"):
                    result.trade_placed = True
                    result.trade_ticket = tool_result.get("ticket")
                    
                    # Log the agent trade to the database
                    try:
                        from backend.signal_logger import log_signal
                        from backend.claude_analyst import SignalResult
                        from backend.risk_manager import RiskDecision
                        
                        entry_price = float(tool_result.get("entry_price") or block.input.get("limit_price", 0.0))
                        
                        sig = SignalResult(
                            symbol=symbol,
                            direction=block.input.get("direction", "BUY"),
                            confidence=block.input.get("confidence", 80), # Default to high confidence for agent trades
                            reasoning=block.input.get("comment", "Agent Trade"),
                            invalidation="See agent history",
                            confluence_factors=[],
                            entry=entry_price,
                            sl=block.input.get("sl"),
                            tp1=block.input.get("tp1"),
                            tp2=block.input.get("tp2"),
                            tp3=block.input.get("tp3"),
                        )
                        risk = RiskDecision(
                            approved=True,
                            lot_size=block.input.get("lot_size", 0.0),
                            risk_amount=0.0,
                            sl_distance=abs(entry_price - block.input.get("sl", 0.0))
                        )
                        await log_signal(sig, risk)
                    except Exception as e:
                        logger.error(f"Failed to log agent trade to DB: {e}")

                elif block.name in ("modify_position", "close_position") and tool_result.get("success"):
                    result.positions_managed += 1

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(tool_result),
                })

            messages.append({"role": "user", "content": tool_results})

        else:
            logger.warning(f"Agent hit max_iterations ({max_iterations}) for {symbol}")

    except Exception as e:
        logger.error(f"Agent loop error for {symbol}: {e}")
        result.error = str(e)

    result.input_tokens = total_input
    result.output_tokens = total_output
    result.cache_read_tokens = total_cache_read
    result.cache_write_tokens = total_cache_write

    logger.info(
        f"Claude cycle done — {symbol} | "
        f"trade={result.trade_placed} ticket={result.trade_ticket} "
        f"managed={result.positions_managed} | "
        f"tokens in={total_input} out={total_output} "
        f"cache_read={total_cache_read} cost=${result.estimated_cost_usd:.5f}"
    )
    return result


async def run_agent(
    symbol: str,
    account: AccountState,
    telegram_notifier=None,
    max_iterations: int = 15,
    level_hits=None,
    news_events=None,
    alert_hits=None,
    skip_analysis: bool = False,
) -> AgentResult:
    """
    Run one full agent cycle triggered by a new M5 bar.
    Dispatches to Claude or Ollama backend based on the active model selection.
    Switch models at runtime via Telegram /model command.
    """
    active     = model_manager.get_active_model()
    auto_trade = model_manager.get_auto_trade()
    logger.info(
        f"run_agent — {symbol} model={active} auto_trade={auto_trade} "
        f"news={len(news_events) if news_events else 0} "
        f"alerts={len(alert_hits) if alert_hits else 0} "
        f"skip_analysis={skip_analysis}"
    )

    if active == "multi_agent":
        from backend.multi_agent import run_agent_multi_model
        return await run_agent_multi_model(
            symbol=symbol,
            account=account,
            telegram_notifier=telegram_notifier,
            max_iterations=max_iterations,
            level_hits=level_hits,
            news_events=news_events,
            alert_hits=alert_hits,
            skip_analysis=skip_analysis,
            auto_trade=auto_trade,
        )

    if active == "ollama":
        initial_prompt = _build_initial_prompt(
            symbol, account, level_hits, news_events, alert_hits, skip_analysis
        )
        return await _run_agent_ollama(
            symbol=symbol,
            account=account,
            telegram_notifier=telegram_notifier,
            max_iterations=max_iterations,
            level_hits=level_hits,
            initial_prompt=initial_prompt,
            auto_trade=auto_trade,
        )

    return await _run_agent_claude(
        symbol=symbol,
        account=account,
        telegram_notifier=telegram_notifier,
        max_iterations=max_iterations,
        level_hits=level_hits,
        auto_trade=auto_trade,
        news_events=news_events,
        alert_hits=alert_hits,
        skip_analysis=skip_analysis,
    )
