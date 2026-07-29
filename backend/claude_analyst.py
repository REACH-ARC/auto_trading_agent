"""
Phase 3 — Claude AI Analyst Module
AI-01 through AI-08
"""
from __future__ import annotations

import json
import re
import time as time_mod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

import anthropic
from loguru import logger

from config import settings
from backend.indicators import IndicatorResult, OHLCVData, SRLevel
from backend.pattern_analyst import PatternResult, detect_patterns
from backend.news_filter import NewsEvent

# ---------------------------------------------------------------------------
# AI-01  Base structure — types & constants
# ---------------------------------------------------------------------------

TradeDirection = Literal["BUY", "SELL", "NO_TRADE"]

# Minimum system-prompt size to qualify for prompt caching (~1024 tokens).
# The detailed persona below comfortably exceeds that threshold.
_SYSTEM_PROMPT = """\
You are a senior professional forex and CFD trader with 15+ years of experience
in technical analysis. You specialise in price action, multi-timeframe confluence,
and risk-managed trade execution on MetaTrader 5.

Your job is to analyse the market snapshot provided and output a precise trading
signal in valid JSON. You must be disciplined: if the setup is not high-quality,
return NO_TRADE. Never force a trade.

## Analysis framework (follow in order)
1. TREND — identify the dominant trend on H4 and D1 using EMA alignment and
   price structure (higher highs / higher lows for bull; opposite for bear).
2. STRUCTURE — identify key support and resistance zones from the provided S/R
   levels. Note which zones price is currently approaching or respecting.
3. MOMENTUM — assess RSI and MACD across all timeframes for confluence.
   Divergence between price and RSI is a high-value signal.
4. ENTRY — identify the optimal entry price. Prefer limit entries at S/R zones,
   or breakout entries on strong momentum confirmation.
5. RISK — place SL at the nearest structural invalidation level (swing point,
   S/R zone). Target SL range: 100–200 pips for forex pairs; for XAUUSD
   target $10–$60; for crypto 0.5–3% of entry. Hard cap at 200 pips (forex) /
   $60 (XAUUSD) / 3% (crypto) — if structure requires wider, return NO_TRADE.
6. REWARD — set TP1 at 1.5R, TP2 at 2.0R, TP3 at 2.5R from the entry.
7. CONFLUENCE — count how many factors align (trend + structure + momentum +
   session). More confluence = higher confidence score.
8. CONFIDENCE — score 0–100. Only return BUY or SELL if confidence >= 80.
   Below 80, return NO_TRADE with reasoning explaining what is missing.

## Session guidance
- For forex and gold (XAUUSD): prefer signals during London (07:00–16:00 UTC) and
  New York (12:00–21:00 UTC). London/NY overlap (12:00–16:00 UTC) is highest-liquidity.
  Avoid new entries in the last 30 minutes of a session close.
- For crypto (BTC, ETH, and any symbol containing "BTC" or "USD"c suffix):
  crypto trades 24/7 — do NOT penalise signals based on forex session hours.
  Instead weight by volatility and volume: highest BTC volume typically occurs
  12:00–20:00 UTC but valid setups exist at any hour. Judge by price action and
  indicator confluence alone, not by session.

## Output format
Respond ONLY with a single valid JSON object — no markdown fences, no extra text.

{
  "direction": "BUY" | "SELL" | "NO_TRADE",
  "entry": <float or null if NO_TRADE>,
  "sl": <float or null if NO_TRADE>,
  "tp1": <float or null if NO_TRADE>,
  "tp2": <float or null if NO_TRADE>,
  "tp3": <float or null if NO_TRADE>,
  "confidence": <integer 0–100>,
  "reasoning": "<concise 2–4 sentence explanation of the setup>",
  "invalidation": "<what price action would invalidate this trade>",
  "confluence_factors": ["<factor1>", "<factor2>", ...]
}
"""


@dataclass
class SignalResult:
    """Parsed and validated output from Claude."""
    symbol: str
    direction: TradeDirection
    confidence: int
    reasoning: str
    invalidation: str
    confluence_factors: list[str]
    entry: float | None = None
    sl: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_actionable(self) -> bool:
        return self.direction != "NO_TRADE" and self.entry is not None

    @property
    def risk_reward(self) -> float | None:
        if not self.is_actionable or self.entry is None or self.sl is None or self.tp1 is None:
            return None
        risk = abs(self.entry - self.sl)
        reward = abs(self.tp1 - self.entry)
        return round(reward / risk, 2) if risk > 0 else None

    @property
    def estimated_cost_usd(self) -> float:
        # claude-sonnet-4-6 pricing (per million tokens)
        input_cost = (self.input_tokens / 1_000_000) * 3.00
        output_cost = (self.output_tokens / 1_000_000) * 15.00
        cache_read_cost = (self.cache_read_tokens / 1_000_000) * 0.30
        cache_write_cost = (self.cache_write_tokens / 1_000_000) * 3.75
        return round(input_cost + output_cost + cache_read_cost + cache_write_cost, 6)



# ---------------------------------------------------------------------------
# AI-03  Market snapshot formatter
# ---------------------------------------------------------------------------

def _fmt_sr(levels: list[SRLevel], current_price: float) -> str:
    if not levels:
        return "  None detected"
    lines = []
    for lvl in levels[:6]:  # top 6 by strength
        distance_pct = ((lvl.price - current_price) / current_price) * 100
        direction = "above" if lvl.price > current_price else "below"
        lines.append(
            f"  {lvl.level_type.upper():12s} {lvl.price:.5f}  "
            f"({abs(distance_pct):.2f}% {direction}, strength={lvl.strength})"
        )
    return "\n".join(lines)


def _fmt_rsi(value: float) -> str:
    if value >= 70:
        return f"{value:.1f} (OVERBOUGHT)"
    if value <= 30:
        return f"{value:.1f} (OVERSOLD)"
    return f"{value:.1f}"


def _fmt_patterns(patterns: PatternResult) -> list[str]:
    """Format detected patterns as snapshot lines for Claude."""
    lines: list[str] = ["", "--- Pattern Analysis ---"]

    if patterns.candle_patterns:
        lines.append("  Candlestick:")
        for p in patterns.candle_patterns:
            lines.append(f"    [{p.timeframe}] {p.name} — {p.direction} ({p.strength})")
    else:
        lines.append("  Candlestick: None detected on M5/H1/H4")

    if patterns.chart_patterns:
        lines.append("  Chart:")
        for p in patterns.chart_patterns:
            lines.append(
                f"    [{p.timeframe}] {p.name} — {p.direction} "
                f"({p.confidence} confidence)  key level: {p.key_level}"
            )

    if patterns.fvg_patterns:
        active = [p for p in patterns.fvg_patterns if not p.mitigated]
        mitig  = [p for p in patterns.fvg_patterns if p.mitigated]
        lines.append("  Fair Value Gaps (FVG):")
        for p in active:
            lines.append(
                f"    [{p.timeframe}] {p.direction} FVG "
                f"zone {p.fvg_low}–{p.fvg_high}  mid={p.midpoint}  [ACTIVE]"
            )
        for p in mitig:
            lines.append(
                f"    [{p.timeframe}] {p.direction} FVG "
                f"zone {p.fvg_low}–{p.fvg_high}  mid={p.midpoint}  [mitigated]"
            )
    else:
        lines.append("  FVG: None detected")

    return lines


def format_market_snapshot(
    data: OHLCVData,
    indicators: IndicatorResult,
    patterns: PatternResult | None = None,
    news_events: list[NewsEvent] | None = None,
    dxy_trend: str | None = None,
) -> str:
    """Convert raw OHLCV + indicators into a structured text prompt for Claude."""
    tfs = list(data.timeframes.keys())
    current_price = data.timeframes[tfs[0]][-1].close

    lines: list[str] = [
        f"=== MARKET SNAPSHOT ===",
        f"Symbol       : {data.symbol}",
        f"Current Price: {current_price:.5f}",
        f"Timestamp    : {data.received_at.strftime('%Y-%m-%d %H:%M UTC')}",
        f"Trend Bias   : {indicators.trend_bias} (from H4 EMA alignment)",
    ]

    if dxy_trend is not None:
        lines += [
            f"",
            f"--- Multi-Asset Correlation ---",
            f"  DXY Trend: {dxy_trend}",
        ]

    lines += [
        f"",
        f"--- Active Sessions ---",
    ]

    if indicators.session.active_sessions:
        lines.append(f"  {', '.join(indicators.session.active_sessions)}")
    else:
        lines.append("  No major session active (off-hours)")

    lines += [
        f"  High Liquidity : {indicators.session.is_high_liquidity}",
        f"  London/NY Overlap: {indicators.session.is_overlap}",
        f"",
        f"--- Multi-Timeframe Indicators ---",
    ]

    for tf in tfs:
        rsi = indicators.rsi.get(tf, 50.0)
        macd = indicators.macd.get(tf, {})
        atr = indicators.atr.get(tf, 0.0)
        emas = indicators.ema.get(tf, {})

        macd_bias = "BULLISH" if macd.get("is_bullish") else "BEARISH"
        e20 = emas.get(20, 0)
        e50 = emas.get(50, 0)
        e200 = emas.get(200, 0)
        ema_order = f"EMA20={e20:.5f}  EMA50={e50:.5f}  EMA200={e200:.5f}"
        vol_ratio = indicators.rvol.get(tf, 1.0)
        vol_label = "HIGH VOLUME" if vol_ratio > 1.5 else "Low Volume" if vol_ratio < 0.7 else "Average"

        lines += [
            f"",
            f"  [{tf}]",
            f"    RSI(14)  : {_fmt_rsi(rsi)}",
            f"    MACD     : value={macd.get('value', 0):.6f}  "
            f"signal={macd.get('signal', 0):.6f}  "
            f"hist={macd.get('histogram', 0):.6f}  bias={macd_bias}",
            f"    ATR(14)  : {atr:.6f}",
            f"    RVOL(20) : {vol_ratio}x ({vol_label})",
            f"    EMAs     : {ema_order}",
        ]

    lines += [
        f"",
        f"--- Support & Resistance (H4, top 6 by strength) ---",
        _fmt_sr(indicators.support_resistance, current_price),
    ]

    if patterns is not None:
        lines += _fmt_patterns(patterns) if patterns else ["", "  Pattern analysis skipped."]

    lines += ["", "--- Overextension / Mean Reversion ---"]
    for tf in tfs:
        dist = indicators.ema_distance_atr.get(tf, 0.0)
        label = "Neutral"
        if dist >= 1.5:
            label = "Overextended Bullish"
        elif dist <= -1.5:
            label = "Overextended Bearish"
        elif dist > 0.5:
            label = "Bullish Momentum"
        elif dist < -0.5:
            label = "Bearish Momentum"
        
        sign = "+" if dist > 0 else ""
        lines.append(f"  [{tf}] Price vs EMA20: {sign}{dist:.1f} ATR ({label})")

    lines += ["", "--- Session Liquidity Pools ---"]
    highs = indicators.session_highs
    lows = indicators.session_lows
    if "Daily" in highs:
        lines.append(f"  Daily High : {highs['Daily']:.5f}")
        lines.append(f"  Daily Low  : {lows['Daily']:.5f}")
    if "Asian" in highs:
        lines.append(f"  Asian High : {highs['Asian']:.5f}")
        lines.append(f"  Asian Low  : {lows['Asian']:.5f}")

    if news_events is not None:
        lines += ["", "--- Upcoming Macro News (Next 12h) ---"]
        if not news_events:
            lines.append("  No high-impact news in the next 12 hours.")
        else:
            now = datetime.now(timezone.utc)
            for event in news_events:
                hrs = (event.event_time - now).total_seconds() / 3600.0
                lines.append(f"  [{event.impact}] {event.title} (in {hrs:.1f} hours)")

    lines.append("")
    lines.append("=== END SNAPSHOT ===")
    lines.append("")
    lines.append("Analyse the above snapshot and provide your trading signal as JSON.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# AI-04  Claude API call with prompt caching
# AI-07  Retry + error handling
# AI-08  Token usage logger
# ---------------------------------------------------------------------------

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _call_claude(snapshot_text: str, retries: int = 3, backoff: float = 2.0) -> tuple[str, dict]:
    """
    Call Claude API with prompt caching on the system prompt.
    Returns (raw_response_text, usage_dict).
    Retries on transient API errors with exponential backoff.
    """
    client = _get_client()
    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.messages.create(
                model=settings.claude["model"],
                max_tokens=settings.claude["max_tokens"],
                temperature=settings.claude["temperature"],
                system=[
                    {
                        "type": "text",
                        "text": _SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},  # AI-04: cache system prompt
                    }
                ],
                messages=[
                    {"role": "user", "content": snapshot_text}
                ],
            )

            usage = response.usage
            usage_dict = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            }

            _log_token_usage(usage_dict)  # AI-08
            return response.content[0].text, usage_dict

        except anthropic.RateLimitError as e:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(f"Claude rate limit (attempt {attempt}/{retries}), retrying in {wait}s: {e}")
            last_error = e
            time_mod.sleep(wait)

        except anthropic.APIStatusError as e:
            if e.status_code >= 500:
                wait = backoff * (2 ** (attempt - 1))
                logger.warning(f"Claude server error {e.status_code} (attempt {attempt}/{retries}), retrying in {wait}s")
                last_error = e
                time_mod.sleep(wait)
            else:
                raise  # 4xx errors are caller mistakes — don't retry

        except anthropic.APIConnectionError as e:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(f"Claude connection error (attempt {attempt}/{retries}), retrying in {wait}s: {e}")
            last_error = e
            time_mod.sleep(wait)

    raise RuntimeError(f"Claude API failed after {retries} attempts") from last_error


def _log_token_usage(usage: dict) -> None:
    """AI-08 — log token counts and estimated cost per call."""
    inp = usage.get("input_tokens", 0)
    out = usage.get("output_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cache_write = usage.get("cache_creation_input_tokens", 0)

    cost_usd = (
        (inp / 1_000_000) * 3.00
        + (out / 1_000_000) * 15.00
        + (cache_read / 1_000_000) * 0.30
        + (cache_write / 1_000_000) * 3.75
    )
    cache_saving_pct = (cache_read / inp * 100) if inp > 0 else 0

    logger.info(
        f"Claude tokens — input={inp} output={out} "
        f"cache_read={cache_read} cache_write={cache_write} "
        f"cost=${cost_usd:.5f} cache_saving={cache_saving_pct:.0f}%"
    )


# ---------------------------------------------------------------------------
# AI-05  Response parser
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict:
    """Extract JSON from Claude's response, tolerating minor formatting issues."""
    text = text.strip()

    # Strip markdown fences if Claude ignored the instruction
    fenced = re.search(r"```(?:json)?\s*([\s\S]+?)```", text)
    if fenced:
        text = fenced.group(1).strip()

    # Find outermost { ... }
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in Claude response: {text[:200]}")

    return json.loads(text[start:end])


def _parse_signal(raw: dict, symbol: str, usage: dict) -> SignalResult:
    """Validate and convert Claude's raw dict into a typed SignalResult."""
    direction: TradeDirection = raw.get("direction", "NO_TRADE")
    if direction not in ("BUY", "SELL", "NO_TRADE"):
        logger.warning(f"Unexpected direction '{direction}' — treating as NO_TRADE")
        direction = "NO_TRADE"

    confidence = int(raw.get("confidence", 0))
    confidence = max(0, min(100, confidence))  # clamp to 0–100

    return SignalResult(
        symbol=symbol,
        direction=direction,
        confidence=confidence,
        reasoning=str(raw.get("reasoning", "")),
        invalidation=str(raw.get("invalidation", "")),
        confluence_factors=list(raw.get("confluence_factors", [])),
        entry=float(raw["entry"]) if raw.get("entry") is not None else None,
        sl=float(raw["sl"]) if raw.get("sl") is not None else None,
        tp1=float(raw["tp1"]) if raw.get("tp1") is not None else None,
        tp2=float(raw["tp2"]) if raw.get("tp2") is not None else None,
        tp3=float(raw["tp3"]) if raw.get("tp3") is not None else None,
        input_tokens=usage.get("input_tokens", 0),
        output_tokens=usage.get("output_tokens", 0),
        cache_read_tokens=usage.get("cache_read_input_tokens", 0),
        cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
    )


# ---------------------------------------------------------------------------
# AI-06  Confidence threshold filter
# ---------------------------------------------------------------------------

def _check_confidence(signal: SignalResult) -> SignalResult:
    """Downgrade to NO_TRADE if confidence is below the configured threshold."""
    threshold: int = settings.claude["confidence_threshold"]
    if signal.direction != "NO_TRADE" and signal.confidence < threshold:
        logger.info(
            f"Signal filtered: confidence {signal.confidence}% < threshold {threshold}% "
            f"({signal.symbol} {signal.direction})"
        )
        signal.direction = "NO_TRADE"
        signal.entry = signal.sl = signal.tp1 = signal.tp2 = signal.tp3 = None
        signal.reasoning = (
            f"[Filtered: confidence {signal.confidence}% below threshold {threshold}%] "
            + signal.reasoning
        )
    return signal


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyse(data: OHLCVData, indicators: IndicatorResult) -> SignalResult:
    """
    Full pipeline: format snapshot → call Claude → parse → filter → return signal.
    This is the single entry point called by the backend server.
    """
    patterns = detect_patterns(data)
    snapshot = format_market_snapshot(data, indicators, patterns)

    logger.info(f"Requesting Claude analysis for {data.symbol}…")
    raw_text, usage = _call_claude(snapshot)

    try:
        raw_dict = _extract_json(raw_text)
    except (ValueError, json.JSONDecodeError) as e:
        logger.error(f"Failed to parse Claude response: {e}\nRaw: {raw_text[:500]}")
        return SignalResult(
            symbol=data.symbol,
            direction="NO_TRADE",
            confidence=0,
            reasoning=f"Parse error: {e}",
            invalidation="",
            confluence_factors=[],
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cache_read_tokens=usage.get("cache_read_input_tokens", 0),
            cache_write_tokens=usage.get("cache_creation_input_tokens", 0),
        )

    signal = _parse_signal(raw_dict, data.symbol, usage)
    signal = _check_confidence(signal)

    logger.info(
        f"Signal: {signal.symbol} {signal.direction} "
        f"confidence={signal.confidence}% "
        f"entry={signal.entry} sl={signal.sl} "
        f"tp1={signal.tp1} rr={signal.risk_reward}"
    )
    return signal
