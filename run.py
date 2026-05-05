import sys
import asyncio

# ZeroMQ (via tornado) requires SelectorEventLoop on Windows.
# Must be set before uvicorn creates the event loop.
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="127.0.0.1", port=8000, reload=False)
