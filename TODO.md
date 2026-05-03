# MT5 AI Analyst Bot — Task List

## Legend
- [ ] Not started
- [x] Done
- [~] In progress

---

## Phase 1 — Project Setup

- [x] **SETUP-01** Create project folder structure
- [x] **SETUP-02** Create `requirements.txt` with all dependencies
- [x] **SETUP-03** Create `config/settings.yaml` (API keys, symbols, risk params)
- [x] **SETUP-04** Create `.env` file template for secrets
- [x] **SETUP-05** Create `config/loader.py` to load settings + env vars

---

## Phase 2 — Indicator Engine

- [x] **IND-01** Create `backend/indicators.py` — OHLCV data class / types
- [x] **IND-02** Add RSI calculation (multi-timeframe)
- [x] **IND-03** Add MACD calculation (multi-timeframe)
- [x] **IND-04** Add ATR calculation (used for adaptive SL/TP)
- [x] **IND-05** Add EMA 20 / EMA 50 / EMA 200 calculation
- [x] **IND-06** Add support & resistance level detection (swing highs/lows)
- [x] **IND-07** Add market session detector (London / NY / Tokyo / Sydney)
- [x] **IND-08** Write unit tests for each indicator function

---

## Phase 3 — Claude AI Analyst Module

- [x] **AI-01** Create `backend/claude_analyst.py` — base structure
- [x] **AI-02** Write system prompt (professional analyst persona + output format)
- [x] **AI-03** Implement market snapshot formatter (OHLCV + indicators → structured text)
- [x] **AI-04** Implement Claude API call with prompt caching (cache system prompt)
- [x] **AI-05** Parse Claude JSON response → `SignalResult` dataclass
- [x] **AI-06** Add confidence threshold filter (skip signals below 70%)
- [x] **AI-07** Add retry + error handling for API failures
- [x] **AI-08** Add token usage logger (track cost per call)

---

## Phase 4 — News Filter

- [x] **NEWS-01** Create `backend/news_filter.py`
- [x] **NEWS-02** Fetch economic calendar from Forex Factory or investing.com
- [x] **NEWS-03** Parse upcoming high-impact news events (next 2 hours)
- [x] **NEWS-04** Block signal generation during high-impact windows (±30 min)
- [x] **NEWS-05** Cache calendar data (refresh every 4 hours)

---

## Phase 5 — Risk Manager

- [x] **RISK-01** Create `backend/risk_manager.py`
- [x] **RISK-02** Calculate position size by account % risk (e.g. 1% per trade)
- [x] **RISK-03** Enforce max open trades limit (configurable)
- [x] **RISK-04** Enforce max daily loss limit — pause bot if hit
- [x] **RISK-05** Validate SL distance is within ATR-based range (not too tight / wide)
- [x] **RISK-06** Return risk-adjusted lot size + go/no-go flag to caller

---

## Phase 6 — Signal Logger

- [x] **LOG-01** Create `backend/signal_logger.py`
- [x] **LOG-02** Create SQLite schema: `signals` table (symbol, direction, entry, sl, tp, confidence, timestamp)
- [x] **LOG-03** Write signal to DB on every Claude response
- [x] **LOG-04** Update signal outcome (win/loss/breakeven) when trade closes
- [x] **LOG-05** Query win rate, avg RR, total signals — expose as stats endpoint

---

## Phase 7 — Python Backend Server

- [x] **SRV-01** Create `backend/main.py` — FastAPI app skeleton
- [x] **SRV-02** Create `backend/mt5_bridge.py` — ZeroMQ PULL socket listener
- [x] **SRV-03** Wire ZeroMQ receiver → indicator engine → Claude analyst pipeline
- [x] **SRV-04** Wire news filter check before calling Claude
- [x] **SRV-05** Wire risk manager check before returning signal
- [x] **SRV-06** POST `/signal` endpoint — manual trigger for testing
- [x] **SRV-07** GET `/stats` endpoint — return win rate + signal count
- [x] **SRV-08** GET `/health` endpoint — server status check
- [x] **SRV-09** Add structured logging (loguru) — log every signal decision

---

## Phase 8 — MT5 Expert Advisor (MQL5)

- [x] **EA-01** Create `mt5_ea/AI_Analyst.mq5` — EA skeleton with OnInit/OnTick/OnDeinit
- [x] **EA-02** Implement ZeroMQ PUSH socket — send OHLCV data to Python
- [x] **EA-03** Collect M15, H1, H4, D1 OHLCV bars (last 100 bars each)
- [x] **EA-04** Implement ZeroMQ SUB socket — receive signal JSON from Python
- [x] **EA-05** Parse signal JSON in MQL5
- [x] **EA-06** Draw entry arrow (buy/sell) on chart at signal price
- [x] **EA-07** Draw SL horizontal line (red dashed)
- [x] **EA-08** Draw TP1, TP2, TP3 horizontal lines (green dashed)
- [x] **EA-09** Display signal label: symbol, direction, confidence %, reasoning snippet
- [x] **EA-10** Add input parameters (lot size override, enable/disable auto-trade)
- [x] **EA-11** Optional: auto place market order with SL/TP when signal fires

---

## Phase 9 — Telegram Notifications

- [x] **TG-01** Create `notifications/telegram_bot.py`
- [x] **TG-02** Format signal message (symbol, direction, entry, SL, TP, confidence, reasoning)
- [x] **TG-03** Send signal alert to configured Telegram chat/channel
- [x] **TG-04** Send daily summary (win rate, signals today, P&L estimate)
- [x] **TG-05** Add `/status` command — bot health check via Telegram

---

## Phase 10 — Multi-Symbol Scanner

- [x] **SCAN-01** Create `backend/scanner.py` — async scanner loop
- [x] **SCAN-02** Define symbol watchlist in `settings.yaml` (e.g. EURUSD, XAUUSD, US30)
- [x] **SCAN-03** Fetch latest data for all symbols every N minutes (configurable)
- [x] **SCAN-04** Score each symbol by confluence strength
- [x] **SCAN-05** Send signal only for top-ranked symbol per scan cycle
- [x] **SCAN-06** Schedule scanner with APScheduler (cron-style)

---

## Phase 11 — Testing & QA

- [x] **TEST-01** Unit test indicator calculations (compare vs known values)
- [x] **TEST-02** Unit test Claude prompt formatter output
- [x] **TEST-03** Unit test signal JSON parser (valid + malformed responses)
- [x] **TEST-04** Unit test risk manager (edge cases: 0 balance, max trades hit)
- [x] **TEST-05** Integration test: Python server → Claude API → signal response
- [x] **TEST-06** Manual end-to-end test: MT5 EA → Python → Claude → chart drawing (manual — run EA on live chart, confirm arrows/labels draw and signals appear in Telegram)

---

## Phase 12 — Deployment & Docs

- [x] **DEP-01** Create `start.bat` / `start.sh` — one-click server launcher
- [x] **DEP-02** Add `README.md` — setup guide, config instructions
- [x] **DEP-03** Add MT5 EA installation guide (copy .mq5, compile, attach to chart)
- [x] **DEP-04** Document all `settings.yaml` options
- [x] **DEP-05** Add `.gitignore` (exclude `.env`, `*.db`, `__pycache__`)

---

## Task Count Summary

| Phase | Tasks | Status |
|---|---|---|
| Setup | 5 | 5 / 5 ✅ |
| Indicators | 8 | 8 / 8 ✅ |
| Claude AI Module | 8 | 8 / 8 ✅ |
| News Filter | 5 | 5 / 5 ✅ |
| Risk Manager | 6 | 6 / 6 ✅ |
| Signal Logger | 5 | 5 / 5 ✅ |
| Backend Server | 9 | 9 / 9 ✅ |
| MT5 EA | 11 | 11 / 11 ✅ |
| Telegram | 5 | 5 / 5 ✅ |
| Multi-Symbol Scanner | 6 | 6 / 6 ✅ |
| Testing | 6 | 6 / 6 ✅ |
| Deployment | 5 | 5 / 5 ✅ |
| **TOTAL** | **79** | **79 / 79** ✅ |
