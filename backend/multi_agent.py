"""
Multi-Model Agent Architecture
Uses Ollama (GPT-OSS-120B) for async timeframe scouts, and DeepSeek-V4-Pro for the main brain.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from loguru import logger
try:
    from openai import AsyncOpenAI
except ImportError:
    raise RuntimeError("openai package required — run: pip install openai>=1.0.0")

from config import settings
from backend.risk_manager import AccountState
from backend.claude_agent import (
    AgentResult, ToolCall, _execute_tool, _build_initial_prompt, _get_tools
)

_DEEPSEEK_SYSTEM_PROMPT = """\
You are an autonomous trading agent with live access to MetaTrader 5.
You are powered by DeepSeek-V4-Pro. You have tools to manage open positions and execute trades.
You are triggered every time a new M5 bar closes.

## Decision loop — follow this exact order every trigger:

### Step 1 — Account check
Call get_account_info().
- If daily_loss_remaining <= 0: send a warning update and STOP. No new trades.
- If open_trades >= 3: skip Step 3 (no new entries), still do Step 2.

### Step 2 — Position management
Call get_open_positions(). MT5 handles TP automatically — do NOT manually close at TP1/TP2.
Your only job here is early exit protection:
- r_moved <= -0.9: check if the setup is still valid. Call close_position() if structure is broken.
- r_moved >= 1.0: move SL to breakeven via modify_position().
- Otherwise: leave the position alone.

### Step 3 — Market analysis (Pre-computed)
You do NOT need to call get_market_snapshot. The technical analysis for all timeframes has already been 
completed by the Scout models (GPT-OSS-120B) and is provided in the user prompt below.
Read the Scout Summaries to evaluate:

  0. MARKET REGIME — CLASSIFY FIRST:
     - TRENDING: Price making new HH/HL or LH/LL. Trade pullbacks in trend direction.
     - RANGING: Price bouncing between same S/R. Do NOT trade breakdowns — only range bounces.
  
  1. TREND (H4/D1)
  2. STRUCTURE (S/R levels)
  3. MOMENTUM (RSI/MACD) — ALL 4 timeframes must agree. If ANY opposes, reduce confidence by 15-20.
  4. ENTRY & CONFIDENCE (Score 0-100)

## Trading philosophy — QUALITY OVER QUANTITY
- Aim for 1-2 HIGH-QUALITY trades per day maximum. Zero trades is perfectly fine.
- BOTH directions are valid. Do NOT lock into one direction because the daily is bearish.
- Never chase. Wait for the next setup.

### Step 4 — Trade decision
<<STEP4>>

### Step 4.5 — Price Alerts (Smarter Monitoring)
If you decided NO_TRADE, call set_price_alert() to register the key S/R levels you are waiting for before entering (e.g. "Wait for price to hit 4050 resistance"). 
The system will sleep and ONLY wake you up when price hits your alert, saving API costs.
Always call clear_price_alerts() before setting a new batch.

### Step 5 — Telegram update (ALWAYS)
Call send_update() to summarise the cycle. You MUST pass the structured JSON fields:
- message_type: Pick the most appropriate category (e.g., new_position, no_trade, info)
- symbol: The active symbol
- reason: Market bias, levels, and rationale for your decision
- account_balance and daily_pnl: From your account check
- Other fields as appropriate (entry, sl, tp, confidence, etc.)
- If there is an existing position with a 'time_remaining' field, YOU MUST pass the time left until exit into the 'time_remaining' JSON field.

## Hard rules (enforced in code — orders will be REJECTED if violated)
- MAX 3 TRADES PER SYMBOL PER DAY
- 2-HOUR DIRECTIONAL COOLDOWN after Stop Loss hit (same direction blocked)
- Minimum SL distance for XAUUSD: $6
- STRICT MOMENTUM ALIGNMENT: ALL timeframes must agree. Do NOT sell if M5/H1 are bullish.
- FAKEOUT AWARENESS: If S/R broken and reclaimed 2+ times → market is RANGING. Stop trading breakdowns.
"""


async def _run_scout(symbol: str, timeframe: str) -> str:
    """Run an async scout on Ollama Cloud for a specific timeframe."""
    ollama_cfg = settings.ollama
    base_url = ollama_cfg.get("base_url", "http://localhost:11434/v1")
    model_name = ollama_cfg.get("model", "gpt-oss:120b-cloud")
    client = AsyncOpenAI(base_url=base_url, api_key=settings.ollama_api_key)
    
    from backend.mt5_tools import get_market_snapshot
    snapshot_result = await asyncio.to_thread(get_market_snapshot, symbol)
    snapshot = snapshot_result.get("snapshot", "") if isinstance(snapshot_result, dict) else str(snapshot_result)

    prompt = (
        f"You are a technical analyst scout. Review this market snapshot for {symbol}. "
        f"Focus specifically on the {timeframe} timeframe. "
        f"Identify the trend, key S/R levels, and momentum. Keep your response under 100 words.\n\n"
        f"{snapshot}"
    )

    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        content = response.choices[0].message.content or ""
        # Some reasoning models might put everything in 'reasoning' if interrupted or depending on API
        reasoning = getattr(response.choices[0].message, "reasoning", None) or ""
        if not content and reasoning:
            content = f"<reasoning>\n{reasoning}\n</reasoning>"
        return f"[{timeframe} Scout]:\n" + content.strip()
    except Exception as e:
        logger.warning(f"Scout failed for {timeframe}: {e}")
        return f"[{timeframe} Scout]: Analysis failed ({e})"


async def get_scout_summaries(symbol: str) -> str:
    """Fire parallel scout tasks for multiple timeframes."""
    timeframes = ["M5", "H1", "H4", "D1"]
    logger.info(f"Firing {len(timeframes)} async scouts on Ollama Cloud for {symbol}...")
    tasks = [_run_scout(symbol, tf) for tf in timeframes]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)


def _deepseek_tools(auto_trade: bool = True, skip_analysis: bool = False) -> list[dict]:
    """Return tools for DeepSeek, removing get_market_snapshot."""
    tools = _get_tools(auto_trade, skip_analysis)
    tools = [t for t in tools if t["name"] != "get_market_snapshot"]
    
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["input_schema"],
            },
        }
        for t in tools
    ]


async def run_agent_multi_model(
    symbol: str,
    account: AccountState,
    telegram_notifier=None,
    max_iterations: int = 15,
    level_hits=None,
    news_events=None,
    alert_hits=None,
    skip_analysis: bool = False,
    auto_trade: bool = True,
) -> AgentResult:
    """The Main Brain loop using DeepSeek-V4-Pro."""
    result = AgentResult(symbol=symbol)
    deepseek_cfg = settings.deepseek
    base_url = deepseek_cfg.get("base_url", "https://api.deepseek.com")
    model_name = deepseek_cfg.get("model", "deepseek-v4-pro")
    
    api_key = settings.deepseek_api_key or "sk-no-key-provided"
    client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    scout_text = ""
    if not skip_analysis:
        scout_text = await get_scout_summaries(symbol)
        scout_text = f"\n\n### SCOUT SUMMARIES (GPT-OSS-120B)\n{scout_text}\n"

    initial_prompt = _build_initial_prompt(
        symbol, account, level_hits, news_events, alert_hits, skip_analysis
    )
    initial_prompt += scout_text

    from backend.claude_agent import get_step4_cent, get_step4_standard, _ALERT_ONLY_SUFFIX, _PRICE_ALERT_INSTRUCTIONS
    if skip_analysis:
        step4 = "NOTE: You are in position management mode. Review open positions and manage them. Do not place new trades."
    else:
        step4 = get_step4_cent() if account.is_cent else get_step4_standard()
    sys_prompt = _DEEPSEEK_SYSTEM_PROMPT.replace("<<STEP4>>", step4)
    if not auto_trade:
        sys_prompt += _ALERT_ONLY_SUFFIX
    if account.open_trades > 0:
        sys_prompt += _PRICE_ALERT_INSTRUCTIONS

    ot = _deepseek_tools(auto_trade, skip_analysis)
    messages: list[dict] = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": initial_prompt},
    ]

    try:
        for iteration in range(max_iterations):
            response = await client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=ot if ot else None,
                tool_choice="auto" if ot else "none",
            )

            choice = response.choices[0]
            finish = choice.finish_reason

            if response.usage:
                result.input_tokens += response.usage.prompt_tokens or 0
                result.output_tokens += response.usage.completion_tokens or 0

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
                logger.info(f"DeepSeek agent finished in {iteration + 1} iteration(s) for {symbol}")
                break

            for tc in choice.message.tool_calls:
                tool_input = json.loads(tc.function.arguments or "{}")
                logger.info(f"DeepSeek agent → {tc.function.name}({json.dumps(tool_input)[:150]})")
                tool_result = await _execute_tool(tc.function.name, tool_input, telegram_notifier)
                logger.info(f"DeepSeek agent ← {tc.function.name}: {json.dumps(tool_result)[:250]}")

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
                            reasoning=tool_input.get("comment", "Multi-Agent Trade"),
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
                        logger.error(f"Failed to log Multi-Agent trade to DB: {e}")

                elif tc.function.name in ("modify_position", "close_position") and tool_result.get("success"):
                    result.positions_managed += 1

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(tool_result),
                })
        else:
            logger.warning(f"DeepSeek agent hit max_iterations ({max_iterations}) for {symbol}")

    except Exception as e:
        logger.error(f"DeepSeek agent loop error for {symbol}: {e}")
        result.error = str(e)

    logger.info(
        f"Multi-Model cycle done — {symbol} | "
        f"trade={result.trade_placed} ticket={result.trade_ticket} "
        f"managed={result.positions_managed} | "
        f"tokens in={result.input_tokens} out={result.output_tokens}"
    )
    return result
