import asyncio
from unittest.mock import MagicMock, AsyncMock
import logging
logging.basicConfig(level=logging.INFO)

from backend.model_manager import set_sl_amount_usc, get_sl_amount_usc
from notifications.telegram_bot import TelegramNotifier

async def main():
    notifier = TelegramNotifier()
    # mock update
    update = MagicMock()
    update.callback_query = MagicMock()
    update.callback_query.data = "set_sl:usc:500.0"
    update.callback_query.answer = AsyncMock()
    update.callback_query.edit_message_text = AsyncMock()
    update.effective_chat.id = "-1003939376911"
    
    # mock context
    context = MagicMock()
    
    # bypass auth for test
    notifier._is_authorized = MagicMock(return_value=True)
    
    print("Testing _cb_sl")
    await notifier._cb_sl(update, context)
    print("USC SL:", get_sl_amount_usc())

asyncio.run(main())
