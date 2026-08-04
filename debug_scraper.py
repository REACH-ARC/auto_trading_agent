import asyncio
from backend.kitco_scraper import fetch_latest_gold_news

async def main():
    articles = await fetch_latest_gold_news()
    print("Articles found:", len(articles))

if __name__ == "__main__":
    asyncio.run(main())
