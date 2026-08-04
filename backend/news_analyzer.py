"""
AI Analysis of News Articles
"""
import json
import asyncio
from loguru import logger
from config import settings

_NEWS_PROMPT = """\
You are an expert financial analyst. Read the following news article headline and summary about gold.
Determine the fundamental impact on the price of Gold (XAUUSD).

Return ONLY a strictly valid JSON object with these exact keys:
{{
  "impact": "HIGH" | "MEDIUM" | "LOW",
  "bias": "BULLISH" | "BEARISH" | "NEUTRAL",
  "confidence": <integer 0-100>,
  "summary": "2-3 sentence summary of the news",
  "key_points": ["point 1", "point 2"]
}}

Article Headline: {title}
Summary/Description: {summary}
"""

async def analyze_news_article(title: str, summary: str) -> dict | None:
    model_key = settings._yaml.get("kitco_news", {}).get("model", "deepseek")
    prompt = _NEWS_PROMPT.format(title=title, summary=summary)
    
    try:
        if model_key == "claude":
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            resp = await client.messages.create(
                model=settings.claude.get("model", "claude-3-5-sonnet-20241022"),
                max_tokens=500,
                temperature=0.1,
                system="You are an expert financial analyst.",
                messages=[{"role": "user", "content": prompt}]
            )
            raw = resp.content[0].text
        else:
            # deepseek or ollama
            from openai import AsyncOpenAI
            if model_key == "deepseek":
                base_url = settings._yaml.get("deepseek", {}).get("base_url", "https://api.deepseek.com")
                model_name = settings._yaml.get("deepseek", {}).get("model", "deepseek-chat")
                api_key = settings.deepseek_api_key
            else: # ollama
                base_url = settings.ollama.get("base_url", "http://localhost:11434/v1")
                model_name = settings.ollama.get("model", "gpt-oss:120b-cloud")
                api_key = settings.ollama_api_key or "ollama"
                
            client = AsyncOpenAI(base_url=base_url, api_key=api_key)
            kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "You are an expert financial analyst. Return ONLY valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
            }
            if model_key == "deepseek":
                kwargs["response_format"] = {"type": "json_object"}
                
            resp = await client.chat.completions.create(**kwargs)
            raw = resp.choices[0].message.content
            
        # Parse JSON
        start = raw.find('{')
        end = raw.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(raw[start:end])
        return json.loads(raw)
    except Exception as e:
        logger.error(f"Failed to analyze news '{title}' with model {model_key}: {e}")
        return None
