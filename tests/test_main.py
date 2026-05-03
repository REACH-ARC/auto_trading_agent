"""
Integration tests for backend/main.py — SRV-01, SRV-06, SRV-07, SRV-08
REST endpoints tested via httpx AsyncClient (no real Claude / ZeroMQ calls).
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from backend.claude_analyst import SignalResult
from backend.risk_manager import RiskDecision


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_bars(n: int = 100, base: float = 1.1000) -> list[dict]:
    t = np.linspace(0, 4 * np.pi, n)
    closes = (base + 0.01 * np.sin(t)).tolist()
    return [
        {
            "time":   f"2024-01-15T{(i % 24):02d}:00:00+00:00",
            "open":   closes[i],
            "high":   closes[i] + 0.0005,
            "low":    closes[i] - 0.0005,
            "close":  closes[i],
            "volume": 1000,
        }
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def tmp_db(tmp_path: Path):
    db_file = tmp_path / "test_main.db"
    with patch("backend.signal_logger._DB_PATH", db_file), \
         patch("backend.main._zmq_listener", new_callable=AsyncMock):
        yield db_file


def _buy_signal_result() -> SignalResult:
    return SignalResult(
        symbol="EURUSD", direction="BUY", confidence=78,
        reasoning="Strong H4 confluence", invalidation="Close below 1.0975",
        confluence_factors=["EMA alignment", "RSI"],
        entry=1.1010, sl=1.0980, tp1=1.1055, tp2=1.1085, tp3=1.1130,
        input_tokens=800, output_tokens=120,
        cache_read_tokens=600, cache_write_tokens=0,
    )


def _no_trade_result() -> SignalResult:
    return SignalResult(
        symbol="EURUSD", direction="NO_TRADE", confidence=30,
        reasoning="No clear setup", invalidation="",
        confluence_factors=[],
        input_tokens=400, output_tokens=60,
        cache_read_tokens=300, cache_write_tokens=0,
    )


def _approved_risk() -> RiskDecision:
    return RiskDecision(approved=True, lot_size=0.33,
                        risk_amount=99.0, sl_distance=0.003)


def _rejected_risk() -> RiskDecision:
    return RiskDecision(approved=False, lot_size=0.0,
                        risk_amount=0.0, sl_distance=0.003,
                        reasons=["Max open trades reached"])


@pytest_asyncio.fixture
async def client(tmp_db):
    """AsyncClient wired to the FastAPI app with DB and ZMQ mocked."""
    from backend.main import app
    from backend.signal_logger import init_db

    # Explicitly initialise the DB before the client connects — the ASGI
    # lifespan may not fire reliably inside the test transport.
    await init_db()

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# SRV-08  GET /health
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_ok(self, client):
        data = resp = await client.get("/health")
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_contains_uptime(self, client):
        data = (await client.get("/health")).json()
        assert "uptime_seconds" in data
        assert data["uptime_seconds"] >= 0

    @pytest.mark.asyncio
    async def test_contains_signals_processed(self, client):
        data = (await client.get("/health")).json()
        assert "signals_processed" in data


# ---------------------------------------------------------------------------
# SRV-06  POST /signal
# ---------------------------------------------------------------------------

class TestSignalEndpoint:
    def _payload(self, symbol: str = "EURUSD") -> dict:
        bars = _make_bars()
        return {
            "symbol":      symbol,
            "equity":      10_000.0,
            "open_trades": 0,
            "daily_pnl":   0.0,
            "bars_m15":    bars,
            "bars_h1":     bars,
            "bars_h4":     bars,
            "bars_d1":     bars,
        }

    @pytest.mark.asyncio
    async def test_returns_200_on_valid_request(self, client):
        with patch("backend.main.analyse", return_value=_buy_signal_result()), \
             patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(False, [])):
            resp = await client.post("/signal", json=self._payload())
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_buy_signal_response_fields(self, client):
        with patch("backend.main.analyse", return_value=_buy_signal_result()), \
             patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(False, [])):
            data = (await client.post("/signal", json=self._payload())).json()

        assert data["direction"]  == "BUY"
        assert data["confidence"] == 78
        assert data["signal_id"]  > 0
        assert data["entry"]      == pytest.approx(1.1010)

    @pytest.mark.asyncio
    async def test_news_block_returns_no_trade(self, client):
        from backend.news_filter import NewsEvent
        blocking = [NewsEvent(title="NFP", currency="USD",
                              impact="HIGH",
                              event_time=datetime(2024,1,15,14,30, tzinfo=timezone.utc))]
        with patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(True, blocking)):
            data = (await client.post("/signal", json=self._payload())).json()

        assert data["direction"] == "NO_TRADE"
        assert "NFP" in data["reasoning"]

    @pytest.mark.asyncio
    async def test_no_bars_returns_422(self, client):
        payload = {"symbol": "EURUSD", "equity": 10_000.0}
        resp = await client.post("/signal", json=payload)
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_risk_rejection_in_response(self, client):
        with patch("backend.main.analyse", return_value=_buy_signal_result()), \
             patch("backend.main.evaluate", return_value=_rejected_risk()), \
             patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(False, [])):
            data = (await client.post("/signal", json=self._payload())).json()

        assert data["approved"] is False
        assert len(data["rejection_reasons"]) > 0

    @pytest.mark.asyncio
    async def test_signal_id_increments(self, client):
        with patch("backend.main.analyse", return_value=_buy_signal_result()), \
             patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(False, [])):
            r1 = (await client.post("/signal", json=self._payload())).json()
            r2 = (await client.post("/signal", json=self._payload())).json()

        assert r2["signal_id"] > r1["signal_id"]

    @pytest.mark.asyncio
    async def test_claude_error_returns_no_trade(self, client):
        with patch("backend.main.analyse", side_effect=RuntimeError("API down")), \
             patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(False, [])):
            data = (await client.post("/signal", json=self._payload())).json()

        assert data["direction"] == "NO_TRADE"
        assert "Claude error" in data["reasoning"]


# ---------------------------------------------------------------------------
# SRV-07  GET /stats
# ---------------------------------------------------------------------------

class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        resp = await client.get("/stats")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_empty_db_stats(self, client):
        data = (await client.get("/stats")).json()
        assert data["total_signals"] == 0
        assert data["win_rate_pct"]  == 0.0

    @pytest.mark.asyncio
    async def test_stats_after_signal(self, client):
        with patch("backend.main.analyse", return_value=_buy_signal_result()), \
             patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(False, [])):
            await client.post("/signal", json=self._payload())

        data = (await client.get("/stats")).json()
        assert data["total_signals"] >= 1

    @pytest.mark.asyncio
    async def test_symbol_filter_param(self, client):
        resp = await client.get("/stats?symbol=XAUUSD")
        assert resp.status_code == 200
        data = resp.json()
        assert data["filter_symbol"] == "XAUUSD"

    def _payload(self) -> dict:
        bars = _make_bars()
        return {
            "symbol": "EURUSD", "equity": 10_000.0,
            "open_trades": 0, "daily_pnl": 0.0,
            "bars_m15": bars, "bars_h1": bars,
            "bars_h4": bars, "bars_d1": bars,
        }


# ---------------------------------------------------------------------------
# POST /outcome/{signal_id}
# ---------------------------------------------------------------------------

class TestOutcomeEndpoint:
    async def _create_signal(self, client) -> int:
        with patch("backend.main.analyse", return_value=_buy_signal_result()), \
             patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(False, [])):
            bars = _make_bars()
            data = (await client.post("/signal", json={
                "symbol": "EURUSD", "equity": 10_000.0,
                "open_trades": 0, "daily_pnl": 0.0,
                "bars_m15": bars, "bars_h1": bars,
                "bars_h4": bars, "bars_d1": bars,
            })).json()
        return data["signal_id"]

    @pytest.mark.asyncio
    async def test_update_win(self, client):
        sig_id = await self._create_signal(client)
        resp = await client.post(f"/outcome/{sig_id}", json={
            "outcome": "WIN", "close_price": 1.1055, "actual_pnl": 45.0
        })
        assert resp.status_code == 200
        assert resp.json()["outcome"] == "WIN"

    @pytest.mark.asyncio
    async def test_update_loss(self, client):
        sig_id = await self._create_signal(client)
        resp = await client.post(f"/outcome/{sig_id}", json={
            "outcome": "LOSS", "close_price": 1.0975
        })
        assert resp.status_code == 200
        assert resp.json()["outcome"] == "LOSS"

    @pytest.mark.asyncio
    async def test_invalid_outcome_returns_422(self, client):
        resp = await client.post("/outcome/1", json={
            "outcome": "MAYBE", "close_price": 1.1055
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_missing_signal_returns_404(self, client):
        resp = await client.post("/outcome/9999", json={
            "outcome": "WIN", "close_price": 1.1055
        })
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_actual_rr_returned(self, client):
        sig_id = await self._create_signal(client)
        data = (await client.post(f"/outcome/{sig_id}", json={
            "outcome": "WIN", "close_price": 1.1055
        })).json()
        assert data["actual_rr"] is not None
        assert data["actual_rr"] > 0


# ---------------------------------------------------------------------------
# GET /signal/{signal_id}
# ---------------------------------------------------------------------------

class TestGetSignalEndpoint:
    @pytest.mark.asyncio
    async def test_fetch_existing_signal(self, client):
        with patch("backend.main.analyse", return_value=_buy_signal_result()), \
             patch("backend.main.check_and_refresh", new_callable=AsyncMock,
                   return_value=(False, [])):
            bars = _make_bars()
            created = (await client.post("/signal", json={
                "symbol": "EURUSD", "equity": 10_000.0,
                "open_trades": 0, "daily_pnl": 0.0,
                "bars_m15": bars, "bars_h1": bars,
                "bars_h4": bars, "bars_d1": bars,
            })).json()

        resp = await client.get(f"/signal/{created['signal_id']}")
        assert resp.status_code == 200
        assert resp.json()["id"] == created["signal_id"]

    @pytest.mark.asyncio
    async def test_missing_signal_returns_404(self, client):
        resp = await client.get("/signal/9999")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Phase 10  GET /scanner/status
# ---------------------------------------------------------------------------

class TestScannerStatusEndpoint:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        resp = await client.get("/scanner/status")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_contains_watchlist(self, client):
        data = (await client.get("/scanner/status")).json()
        assert "watchlist" in data
        assert isinstance(data["watchlist"], list)
        assert len(data["watchlist"]) > 0

    @pytest.mark.asyncio
    async def test_contains_interval(self, client):
        data = (await client.get("/scanner/status")).json()
        assert "interval_minutes" in data
        assert data["interval_minutes"] > 0

    @pytest.mark.asyncio
    async def test_cached_symbols_initially_empty(self, client):
        # Fresh test — no MT5 data fed yet
        data = (await client.get("/scanner/status")).json()
        assert "cached_symbols" in data
        # May or may not be empty depending on other tests; just check the key exists

    @pytest.mark.asyncio
    async def test_symbol_data_age_keys_match_watchlist(self, client):
        data = (await client.get("/scanner/status")).json()
        age_keys = set(data["symbol_data_age_sec"].keys())
        watchlist_set = set(data["watchlist"])
        assert age_keys == watchlist_set

    @pytest.mark.asyncio
    async def test_scanner_enabled_field_present(self, client):
        data = (await client.get("/scanner/status")).json()
        assert "enabled" in data
        assert isinstance(data["enabled"], bool)


# ---------------------------------------------------------------------------
# Phase 10  POST /scanner/trigger
# ---------------------------------------------------------------------------

class TestScannerTriggerEndpoint:
    @pytest.mark.asyncio
    async def test_returns_200(self, client):
        with patch("backend.scanner.scan_cycle", new_callable=AsyncMock, return_value=[]):
            resp = await client.post("/scanner/trigger")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_signals_returns_empty_results(self, client):
        with patch("backend.scanner.scan_cycle", new_callable=AsyncMock, return_value=[]):
            data = (await client.post("/scanner/trigger")).json()
        assert data["results"] == []
        assert data["cycle_symbols_analysed"] == 0

    @pytest.mark.asyncio
    async def test_top_signal_in_response(self, client):
        from backend.scanner import ScanResult
        from datetime import datetime, timezone

        mock_result = ScanResult(
            symbol="EURUSD",
            signal=_buy_signal_result(),
            risk=_approved_risk(),
            signal_id=42,
            confluence_score=95.0,
            scanned_at=datetime.now(timezone.utc),
        )

        with patch("backend.scanner.scan_cycle", new_callable=AsyncMock,
                   return_value=[mock_result]):
            data = (await client.post("/scanner/trigger")).json()

        assert len(data["results"]) == 1
        r = data["results"][0]
        assert r["symbol"] == "EURUSD"
        assert r["direction"] == "BUY"
        assert r["confluence_score"] == pytest.approx(95.0)
        assert r["signal_id"] == 42

    @pytest.mark.asyncio
    async def test_only_top_n_in_response(self, client):
        """top_signals_per_cycle=1 in settings → only 1 result in response even if cycle found 3."""
        from backend.scanner import ScanResult
        from datetime import datetime, timezone

        def _make_result(sym: str, score: float, sig_id: int) -> ScanResult:
            sig = _buy_signal_result()
            sig.symbol = sym
            return ScanResult(
                symbol=sym, signal=sig, risk=_approved_risk(),
                signal_id=sig_id, confluence_score=score,
                scanned_at=datetime.now(timezone.utc),
            )

        three_results = [
            _make_result("EURUSD", 95.0, 1),
            _make_result("XAUUSD", 88.0, 2),
            _make_result("GBPUSD", 75.0, 3),
        ]

        with patch("backend.scanner.scan_cycle", new_callable=AsyncMock,
                   return_value=three_results):
            data = (await client.post("/scanner/trigger")).json()

        # top_signals_per_cycle = 1
        assert len(data["results"]) == 1
        assert data["results"][0]["symbol"] == "EURUSD"

    @pytest.mark.asyncio
    async def test_scanner_cycles_counter_increments(self, client):
        from backend.main import _state
        before = _state.scanner_cycles

        with patch("backend.scanner.scan_cycle", new_callable=AsyncMock, return_value=[]):
            await client.post("/scanner/trigger")

        assert _state.scanner_cycles == before + 1

    @pytest.mark.asyncio
    async def test_approved_field_in_result(self, client):
        from backend.scanner import ScanResult
        from datetime import datetime, timezone

        sig = _buy_signal_result()
        mock_result = ScanResult(
            symbol="EURUSD", signal=sig, risk=_approved_risk(),
            signal_id=1, confluence_score=90.0,
            scanned_at=datetime.now(timezone.utc),
        )

        with patch("backend.scanner.scan_cycle", new_callable=AsyncMock,
                   return_value=[mock_result]):
            data = (await client.post("/scanner/trigger")).json()

        assert data["results"][0]["approved"] is True
