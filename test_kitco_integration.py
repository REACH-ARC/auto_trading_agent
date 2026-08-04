import asyncio
import logging
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("kitco_test")

async def mock_send_raw(x):
    print(f"[TELEGRAM SEND RAW] {x}")

async def mock_send_to(x, y):
    print(f"[TELEGRAM SEND TO {x}] {y}")

async def test_kitco():
    from backend.main import _check_kitco_news
    from backend import kitco_scraper
    
    logger.info("Starting manual kitco news check")
    
    # disable actually sending telegrams in tests
    from notifications.telegram_bot import notifier
    notifier._send_raw = mock_send_raw
    notifier._send_to = mock_send_to
    notifier._enabled = True
    
    # Pre-populate _seen_urls but leave out the first article so it triggers as new
    articles = await kitco_scraper.fetch_latest_gold_news()
    if articles:
        for a in articles[1:]:
            kitco_scraper._seen_urls.add(a['url'])
        logger.info(f"Pre-populated seen_urls with {len(kitco_scraper._seen_urls)} articles.")
    else:
        logger.error("No articles found on kitco!")
        return

    await _check_kitco_news()
    logger.info("Test complete")

if __name__ == "__main__":
    asyncio.run(test_kitco())
