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
You are triggered every time a new M15 bar closes.

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
  1. TREND  — H4 + D1 EMA alignment and price structure (HH/HL or LH/LL)
  2. STRUCTURE — where is price relative to key S/R levels?
  3. MOMENTUM — RSI and MACD confluence across M15, H1, H4
  4. ENTRY — prefer limit entries at S/R zones; breakout entries need strong momentum
  5. CONFIDENCE — score 0–100 counting confluence factors:
       +15 trend alignment (H4 + D1 agree)
       +15 price at key S/R level
       +10 RSI confluence (not overbought/oversold against direction)
       +10 MACD histogram supports direction on H1+H4
       +10 M15 momentum confirms direction
       +10 no conflicting open position in same symbol
       +10 high liquidity session (for crypto: strong volume/volatility period)
       +20 additional strong factors (divergence, clean break, engulfing candle, etc.)

### Step 4 — Trade decision
<<STEP4>>

### Step 5 — Telegram update (ALWAYS — even for no-trade)
Call send_update() with a summary covering:
- Market bias and key levels observed
- Any position management actions taken
- Trade placed (entry, SL, TP, lot, reason) OR why no trade was taken
- Current account: equity, daily P&L, open trades

## Crypto-specific rules (BTCUSDc, ETHUSDc, etc.)
- 24/7 market — do NOT penalise signals for being outside forex session hours
- Volatility is the session equivalent — trade during high-volume periods
- BTC often trends strongly; ride winners with trailing stops

## Hard rules (also enforced server-side)
- Risk exactly the fixed SL amount per trade: 1 000 USC (cent) or $50 USD (standard) — never % of balance
- Never exceed 3 open trades
- Never trade when daily loss limit is breached
- SL must always be structure-based, never a fixed pip value
- If a tool returns an error, log it and adapt — do not retry blindly
"""

_STEP4_CENT = """\
NOTE: This is a **cent account** (USC). 100 USC = 1 USD. Account balance displayed in USC.

- If confidence >= 65 AND no open position in the same direction for this symbol:
  a. Call get_symbol_info(symbol) to confirm tick value and lot constraints
  b. Calculate lot_size using the FIXED SL amount (1 000 USC per trade — do NOT use % of balance):
       lot_size = 1000 / (SL_distance_in_price × tick_value / tick_size)
     Then scale by confidence:
       - confidence 65–74 → use 70% of calculated lots (cautious)
       - confidence 75–84 → use 85% of calculated lots (moderate)
       - confidence 85–94 → use 100% of calculated lots (confident)
       - confidence 95–100 → use 100% of calculated lots (high conviction)
     Clamp to broker min_lot and max_lot from get_symbol_info(). Round to broker lot_step.
  c. Set SL at the nearest structural invalidation level (swing high/low)
  d. Set TP1 = entry ± (SL_distance × 1.5) — this is the only TP; MT5 closes the full position here automatically
  e. Call place_order() with order_type="MARKET" only — always enter at current price
- If confidence < 65: NO_TRADE. Note what is missing."""

_STEP4_STANDARD = """\
NOTE: This is a **standard account** (USD). Account balance is in real USD.

- If confidence >= 65 AND no open position in the same direction for this symbol:
  a. Call get_symbol_info(symbol) to confirm tick value and lot constraints
  b. Calculate lot_size using the FIXED SL amount ($50 USD per trade — do NOT use % of balance):
       lot_size = 50 / (SL_distance_in_price × tick_value / tick_size)
     Then scale by confidence:
       - confidence 65–74 → use 70% of calculated lots (cautious)
       - confidence 75–84 → use 85% of calculated lots (moderate)
       - confidence 85–94 → use 100% of calculated lots (full risk)
       - confidence 95–100 → use 100% of calculated lots (high conviction)
     Clamp to broker min_lot and max_lot from get_symbol_info(). Round to broker lot_step.
  c. Set SL at the nearest structural invalidation level (swing high/low)
  d. Set TP1 = entry ± (SL_distance × 1.5) — this is the only TP; MT5 closes the full position here automatically
  e. Call place_order() with order_type="MARKET" only — always enter at current price
- If confidence < 65: NO_TRADE. Note what is missing."""


_ALERT_ONLY_SUFFIX = """

## ALERT-ONLY MODE — auto_trade is OFF
place_order() is NOT available. Do not attempt to trade.
Analyse the market and manage existing positions as normal.
In your send_update(), include the trade you WOULD have placed:
direction, entry, SL, TP, lot size, and reason — so the trader can decide manually.
"""


def _make_system_prompt(is_cent: bool, auto_trade: bool = True) -> str:
    step4 = _STEP4_CENT if is_cent else _STEP4_STANDARD
    prompt = _SYSTEM_PROMPT_TEMPLATE.replace("<<STEP4>>", step4)
    if not auto_trade:
        prompt += _ALERT_ONLY_SUFFIX
    return prompt


def _get_tools(auto_trade: bool = True) -> list[dict]:
    """Return the agent tool list. Removes place_order when auto_trade is disabled."""
    if auto_trade:
        return _TOOLS
    return [t for t in _TOOLS if t["name"] != "place_order"]

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
            "S/R levels) across M15/H1/H4/D1. Returns a formatted analysis snapshot. "
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
            "SL must be structure-based. TP at 1.5R."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string"},
                "direction": {"type": "string", "enum": ["BUY", "SELL"]},
                "lot_size": {"type": "number", "description": "Position size in lots — calculated from fixed SL amount, clamped to broker min/max from get_symbol_info()"},
                "sl": {"type": "number", "description": "Stop loss price (structure-based)"},
                "tp1": {"type": "number", "description": "Take profit — MT5 closes full position here (1.5R)"},
                "comment": {"type": "string", "description": "Trade label (max 31 chars)"},
            },
            "required": ["symbol", "direction", "lot_size", "sl", "tp1"],
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
            "Always call this at the end — even for no-trade cycles."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Summary message (HTML supported for Telegram)",
                },
                "message_type": {
                    "type": "string",
                    "enum": ["info", "trade_open", "trade_close", "warning", "no_trade"],
                },
            },
            "required": ["message", "message_type"],
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
    return {"error": f"Unknown tool: {name}"}


async def _execute_tool(
    name: str,
    tool_input: dict,
    telegram_notifier=None,
) -> dict:
    """Async dispatcher — Telegram is handled natively; MT5 calls run in a thread."""
    if name == "send_update":
        msg = tool_input.get("message", "")
        if telegram_notifier is not None:
            try:
                await telegram_notifier.send_agent_update(msg)
                return {"success": True}
            except Exception as e:
                logger.warning(f"Telegram send_update failed: {e}")
                return {"success": False, "error": str(e)}
        logger.info(f"[Telegram disabled] Agent update: {msg[:200]}")
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

    return (
        f"New M15 bar closed — {symbol}\n"
        f"Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
        f"{level_context}\n\n"
        f"Current account snapshot (cached — fetch fresh via get_account_info):\n"
        f"  Equity={account.equity:.2f}  Balance={account.balance:.2f}  "
        f"Open trades={account.open_trades}  Daily P&L={account.daily_pnl:+.2f}\n\n"
        f"Run your full decision loop now."
    )


async def _run_agent_claude(
    symbol: str,
    account: AccountState,
    telegram_notifier=None,
    max_iterations: int = 15,
    level_hits=None,
    auto_trade: bool = True,
) -> AgentResult:
    """Claude (Anthropic) backend — uses tool_use blocks and prompt caching."""
    result = AgentResult(symbol=symbol)
    client = _get_client()
    initial_prompt = _build_initial_prompt(symbol, account, level_hits)
    system_prompt = _make_system_prompt(account.is_cent, auto_trade)
    active_tools = _get_tools(auto_trade)

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
) -> AgentResult:
    """
    Run one full agent cycle triggered by a new M15 bar.
    Dispatches to Claude or Ollama backend based on the active model selection.
    Switch models at runtime via Telegram /model command.
    """
    active     = model_manager.get_active_model()
    auto_trade = model_manager.get_auto_trade()
    logger.info(f"run_agent — {symbol} model={active} auto_trade={auto_trade}")

    if active == "ollama":
        initial_prompt = _build_initial_prompt(symbol, account, level_hits)
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
    )
