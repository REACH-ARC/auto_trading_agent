import httpx
import asyncio
import json
import re

async def fetch_kitco():
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    async with httpx.AsyncClient() as client:
        res = await client.get("https://www.kitco.com/news/category/commodities/gold", headers=headers)
        html = res.text
        
        match = re.search(r'<script id="__NEXT_DATA__" type="application/json">(.+?)</script>', html)
        if match:
            data = json.loads(match.group(1))
            state = data['props']['pageProps']['dehydratedState']
            queries = state.get('queries', [])
            
            for q in queries:
                qkey = q.get('queryKey', [])
                if qkey and qkey[0] == 'newsByCategoryGeneric':
                    print("Found query")
                    state_data = q.get('state', {}).get('data', {})
                    print("Keys in data:", state_data.keys())
        else:
            print("No __NEXT_DATA__ found")

if __name__ == "__main__":
    asyncio.run(fetch_kitco())
