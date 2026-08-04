"""
Kitco News Scraper
Polls Kitco's gold news page, parses the Next.js state for new articles, 
and maintains a cache of seen articles.
"""
import asyncio
import json
import re
from datetime import datetime, timezone
import httpx
from loguru import logger
from config import settings

_seen_urls = set()
_recent_analyses = []  # Stores dicts of analyzed news

async def fetch_latest_gold_news() -> list[dict]:
    url = "https://www.kitco.com/news/category/commodities/gold"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(url, headers=headers, timeout=15.0)
            res.raise_for_status()
            
            match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', res.text)
            if not match:
                logger.warning("Kitco scraper: __NEXT_DATA__ not found in HTML")
                return []
                
            data = json.loads(match.group(1))
            state = data.get('props', {}).get('pageProps', {}).get('dehydratedState', {})
            queries = state.get('queries', [])
            
            articles = []
            for q in queries:
                q_key = q.get('queryKey', [])
                if q_key and q_key[0] == 'newsByCategoryGeneric':
                    q_data = q.get('state', {}).get('data', {})
                    if 'nodeListByCategory' in q_data:
                        node_list = q_data['nodeListByCategory']
                        items = node_list.get('items', [])
                        for item in items:
                            if 'title' in item and 'urlAlias' in item:
                                articles.append({
                                    "title": item['title'],
                                    "url": f"https://www.kitco.com{item['urlAlias']}",
                                    "summary": item.get('teaser', '') or item.get('summary', '') or item.get('description', '')
                                })
                        break
            return articles
    except Exception as e:
        logger.error(f"Kitco scraper failed: {e}")
        return []

async def get_new_articles() -> list[dict]:
    """Fetch and return only articles we haven't seen before."""
    global _seen_urls
    
    kitco_cfg = settings._yaml.get("kitco_news", {})
    if not kitco_cfg.get("enabled", True):
        return []
        
    articles = await fetch_latest_gold_news()
    if not articles:
        return []
        
    # If this is the first run, just populate the cache and don't alert on everything
    if not _seen_urls:
        logger.info(f"Kitco scraper initialized: {len(articles)} articles found.")
        for a in articles:
            _seen_urls.add(a['url'])
        return []
        
    new_articles = []
    for a in articles:
        if a['url'] not in _seen_urls:
            new_articles.append(a)
            _seen_urls.add(a['url'])
            
    # Keep set size bounded
    if len(_seen_urls) > 500:
        _seen_urls = set(list(_seen_urls)[-100:])
        
    return new_articles

def add_analyzed_news(analysis: dict) -> None:
    """Store the analysis and keep only the latest 5 to avoid context bloat."""
    global _recent_analyses
    analysis['timestamp'] = datetime.now(timezone.utc).isoformat()
    _recent_analyses.insert(0, analysis)
    _recent_analyses = _recent_analyses[:5]

def get_recent_kitco_news() -> list[dict]:
    """Return the recent Kitco analyses to be injected into the AI prompt."""
    return _recent_analyses

