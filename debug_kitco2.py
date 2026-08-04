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
                    state_data = q.get('state', {}).get('data', {})
                    if 'nodeListByCategory' in state_data:
                        items = state_data['nodeListByCategory']
                        if items and 'items' in items:
                            # wait, is it items['items'] ?
                            print("Is items a dict or list?", type(items))
                            if isinstance(items, dict):
                                print("Keys of items:", items.keys())
                                if 'items' in items:
                                    print("First item keys:", items['items'][0].keys())
                        else:
                            print("Items is:", type(items))
                            if isinstance(items, list) and items:
                                print("First item keys:", items[0].keys())
        else:
            print("No __NEXT_DATA__ found")

if __name__ == "__main__":
    asyncio.run(fetch_kitco())
