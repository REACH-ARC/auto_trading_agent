# MT5 AI Analyst Bot

A real-time trading signal generator that connects MetaTrader 5 to Claude AI.
The EA streams multi-timeframe OHLCV data to a Python backend, which runs technical analysis
and calls Claude to produce entry, stop-loss, and take-profit signals.
Signals are drawn on the MT5 chart and pushed to Telegram.

---

## Architecture

```
MT5 Terminal
  └─ AI_Analyst.mq5 (Expert Advisor)
       │  ZeroMQ PUSH  (port 5555)  ──► Python Backend
       │                                  ├─ Indicator Engine  (RSI, MACD, ATR, EMA, S/R)
       │                                  ├─ News Filter       (Forex Factory calendar)
       │                                  ├─ Claude AI Analyst (claude-sonnet-4-6)
       │                                  ├─ Risk Manager      (position sizing, daily loss)
       │                                  ├─ Signal Logger     (SQLite)
       │                                  └─ Multi-Symbol Scanner (APScheduler)
       └─ ZeroMQ SUB   (port 5556)  ◄── Signal JSON
                                          └─ Telegram Bot      (alerts + daily summary)
```

---

## Requirements

| Component | Version |
|-----------|---------|
| Python | 3.11 or newer |
| MetaTrader 5 | Build 4000+ (any broker) |
| mql-zmq library | Latest (see EA installation) |
| Anthropic API key | claude-sonnet-4-6 access |
| Telegram Bot | Optional — for alerts |

---

## Quick Start

### 1 — Clone and install

```bash
git clone <repo-url>
cd RealTimeBotAnalyst

# Create and activate virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS

source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2 — Configure secrets

```bash
cp .env.example .env
```

Edit `.env` and fill in:

```env
ANTHROPIC_API_KEY=sk-ant-your-key-here
TELEGRAM_BOT_TOKEN=your-bot-token-here   # optional
TELEGRAM_CHAT_ID=your-chat-id-here       # optional
```

### 3 — Start the server

**Windows:**
```
start.bat
```

**Linux / macOS:**
```bash
chmod +x start.sh
./start.sh
```

The server starts at `http://127.0.0.1:8000`.
API docs are available at `http://127.0.0.1:8000/docs`.

### 4 — Install and run the MT5 EA

See the [MT5 EA Installation](#mt5-ea-installation) section below.

---

## MT5 EA Installation

### Step 1 — Install the mql-zmq library

The EA uses ZeroMQ to communicate with the Python backend.
Download the prebuilt library from [github.com/dingmaotu/mql-zmq](https://github.com/dingmaotu/mql-zmq/releases).

1. Open MetaTrader 5 → **File → Open Data Folder**
2. Copy the following into your MT5 data folder:
   - `Include/Zmq/` → `MQL5/Include/Zmq/`
   - `Libraries/libzmq.dll` (64-bit) → `MQL5/Libraries/libzmq.dll`
3. Restart MetaTrader 5 so it loads the new library.

### Step 2 — Copy the EA file

1. In MT5, go to **File → Open Data Folder → MQL5 → Experts**
2. Copy `mt5_ea/AI_Analyst.mq5` into that folder.

### Step 3 — Compile

1. Open **MetaEditor** (press F4 in MT5 or Tools → MetaEditor)
2. In the Navigator panel, find `AI_Analyst.mq5` under **Experts**
3. Press **F7** (or Compile button) — you should see 0 errors

### Step 4 — Attach to chart

1. Open a chart for any watchlist symbol (e.g. EURUSD, XAUUSD)
2. Drag `AI_Analyst` from the Navigator → Expert Advisors onto the chart
3. In the EA settings dialog, configure:

| Input Parameter | Default | Description |
|-----------------|---------|-------------|
| Python Backend Host | `127.0.0.1` | Must match `server.host` in settings.yaml |
| ZeroMQ PUSH Port | `5555` | Must match `server.zmq_pull_port` |
| ZeroMQ SUB Port | `5556` | Must match `server.zmq_pub_port` |
| Bars Per Timeframe | `100` | OHLCV bars sent for each timeframe |
| Send Only On New M15 Bar | `true` | Throttles data to one send per M15 close |
| Enable Auto-Trade | `false` | Set `true` to allow the EA to place orders |
| Lot Size Override | `0.0` | 0 = use risk-manager sizing; >0 = fixed lots |

4. Enable **Allow Algo Trading** (the green robot icon in the toolbar)
5. Click **OK** — the EA will connect and begin sending data

### Step 5 — Verify connection

Check the MT5 **Experts** tab (bottom panel) for messages like:

```
AI_Analyst: ZeroMQ connected — PUSH 5555 / SUB 5556
AI_Analyst: Sent M15 bar data for EURUSD
AI_Analyst: Signal received — BUY 1.10100 SL 1.09800 TP1 1.10550
```

On the Python side, the server log (`logs/bot.log`) will show:

```
INFO | Pipeline complete: EURUSD BUY conf=78% approved=True id=1
```

---

## API Reference

All endpoints available at `http://127.0.0.1:8000/docs` (Swagger UI).

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server status, uptime, scanner stats |
| POST | `/signal` | Manual signal request (for testing without MT5) |
| GET | `/stats` | Win rate, signal counts, P&L by symbol |
| POST | `/outcome/{id}` | Record trade result (WIN/LOSS/BREAKEVEN) |
| GET | `/signal/{id}` | Fetch a single signal record |
| GET | `/scanner/status` | Scanner config and symbol data cache ages |
| POST | `/scanner/trigger` | Fire one scanner cycle immediately |

### Example — manual signal request

```bash
curl -X POST http://127.0.0.1:8000/signal \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "EURUSD",
    "equity": 10000,
    "open_trades": 0,
    "daily_pnl": 0,
    "bars_h4": [
      {"time": "2024-01-15T10:00:00+00:00",
       "open": 1.1000, "high": 1.1050, "low": 1.0980, "close": 1.1030, "volume": 5000}
    ]
  }'
```

---

## settings.yaml Reference

All options live in `config/settings.yaml`. Secrets go in `.env` — never in this file.

### `claude`

| Key | Default | Description |
|-----|---------|-------------|
| `model` | `claude-sonnet-4-6` | Anthropic model ID |
| `max_tokens` | `1024` | Max tokens in Claude's response |
| `temperature` | `0.2` | Lower = more deterministic analysis |
| `confidence_threshold` | `70` | Signals below this % are downgraded to NO_TRADE |
| `cache_system_prompt` | `true` | Enable prompt caching (reduces cost ~90%) |

### `server`

| Key | Default | Description |
|-----|---------|-------------|
| `host` | `127.0.0.1` | FastAPI bind address |
| `port` | `8000` | FastAPI port |
| `zmq_pull_port` | `5555` | ZeroMQ PULL socket — receives data from MT5 |
| `zmq_pub_port` | `5556` | ZeroMQ PUB socket — broadcasts signals to MT5 |

### `symbols`

| Key | Default | Description |
|-----|---------|-------------|
| `watchlist` | `[EURUSD, XAUUSD, ...]` | Symbols the scanner monitors each cycle |
| `default_timeframes` | `[M15, H1, H4, D1]` | Timeframes requested from MT5 |
| `bars_per_timeframe` | `100` | OHLCV bars per timeframe per request |

### `risk`

| Key | Default | Description |
|-----|---------|-------------|
| `account_risk_percent` | `1.0` | % of equity risked per trade |
| `max_open_trades` | `3` | Block new signals when this many trades are open |
| `max_daily_loss_percent` | `3.0` | Pause the bot when daily loss reaches this % |
| `min_sl_atr_multiplier` | `0.5` | SL closer than 0.5× ATR is rejected as too tight |
| `max_sl_atr_multiplier` | `3.0` | SL wider than 3× ATR is rejected as too wide |
| `tp1_rr_ratio` | `1.5` | TP1 target in R multiples |
| `tp2_rr_ratio` | `2.5` | TP2 target in R multiples |
| `tp3_rr_ratio` | `4.0` | TP3 target in R multiples |

### `news_filter`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Toggle the news blackout window |
| `block_minutes_before` | `30` | Block signals this many minutes before a high-impact event |
| `block_minutes_after` | `30` | Block signals this many minutes after a high-impact event |
| `impact_levels` | `[HIGH]` | Which impact levels trigger a block (`HIGH`, `MEDIUM`) |
| `calendar_refresh_hours` | `4` | How often to re-fetch the economic calendar |

### `scanner`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable the multi-symbol scanner loop |
| `interval_minutes` | `15` | Run a full scan cycle every N minutes |
| `top_signals_per_cycle` | `1` | Number of top-ranked signals published per cycle |

### `sessions`

Defines UTC open/close times for each trading session.
Used to weight signals: London/NY overlap gets a score bonus.

| Session | Open (UTC) | Close (UTC) |
|---------|-----------|------------|
| Sydney | 21:00 | 06:00 |
| Tokyo | 00:00 | 09:00 |
| London | 07:00 | 16:00 |
| New York | 12:00 | 21:00 |

### `indicators`

| Key | Default | Description |
|-----|---------|-------------|
| `rsi_period` | `14` | RSI lookback period |
| `macd_fast` | `12` | MACD fast EMA |
| `macd_slow` | `26` | MACD slow EMA |
| `macd_signal` | `9` | MACD signal line EMA |
| `atr_period` | `14` | ATR lookback period |
| `ema_periods` | `[20, 50, 200]` | EMA periods calculated per timeframe |
| `sr_lookback_bars` | `50` | Bars used for swing high/low S/R detection |

### `telegram`

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable Telegram alerts (requires `TELEGRAM_BOT_TOKEN` in `.env`) |
| `send_signal_alerts` | `true` | Push an alert for every actionable signal |
| `send_daily_summary` | `true` | Push a daily P&L / win-rate summary |
| `daily_summary_time` | `00:00` | UTC time to send the daily summary (HH:MM) |

### `logging`

| Key | Default | Description |
|-----|---------|-------------|
| `level` | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `file` | `logs/bot.log` | Log file path (rotated automatically) |
| `rotation` | `10 MB` | Rotate when file reaches this size |
| `retention` | `30 days` | Delete rotated files older than this |

### `database`

| Key | Default | Description |
|-----|---------|-------------|
| `path` | `data/signals.db` | SQLite database path for signal history |

---

## Running Tests

```bash
pytest tests/ -v
```

All 274 tests should pass with no external dependencies (Claude API and Telegram are mocked).

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'zmq'`**
Run `pip install pyzmq`.

**EA connects but no signals arrive**
- Confirm the Python server is running (`curl http://127.0.0.1:8000/health`)
- Check that **Allow Algo Trading** is enabled in MT5
- Check the Experts tab for ZeroMQ connection errors

**`ANTHROPIC_API_KEY` error on startup**
Ensure `.env` exists and contains a valid key. Run `cp .env.example .env` if missing.

**Telegram bot not sending messages**
Set `telegram.enabled: true` in `settings.yaml` and provide `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID` in `.env`.

**High API costs**
Prompt caching is enabled by default (`claude.cache_system_prompt: true`), saving ~90% on repeated calls.
Check `GET /stats` → `total_api_cost_usd` to monitor spend.

**Daily loss limit hit — bot paused**
The risk manager stops approving new trades when `daily_pnl` reaches `risk.max_daily_loss_percent`.
It resets automatically the next trading day when MT5 sends updated account data.
