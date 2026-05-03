"""
MT5 Connection Test Script
Simulates what the MT5 EA sends to the backend.
Run with: python test_mt5_connection.py
Backend must be running: python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
"""
import json
import sys
from datetime import datetime, timedelta, timezone

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

BASE_URL = "http://127.0.0.1:8000"


def _make_bars(count: int = 100, base_price: float = 1.08500, spread: float = 0.0010) -> list[dict]:
    """Generate realistic-looking OHLCV bars (newest last)."""
    bars = []
    now = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    price = base_price
    for i in range(count - 1, -1, -1):
        ts = now - timedelta(minutes=15 * i)
        change = (hash(str(ts)) % 100 - 50) * 0.00002
        open_ = price
        close = round(price + change, 5)
        high  = round(max(open_, close) + abs(change) * 0.5 + spread * 0.3, 5)
        low   = round(min(open_, close) - abs(change) * 0.5, 5)
        bars.append({
            "time":   ts.strftime("%Y-%m-%dT%H:%M:%S+00:00"),
            "open":   open_,
            "high":   high,
            "low":    low,
            "close":  close,
            "volume": 1000 + abs(hash(str(ts))) % 5000,
        })
        price = close
    return bars


def test_health():
    print("=" * 55)
    print("1. GET /health")
    print("=" * 55)
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=5)
        data = r.json()
        print(f"   Status     : {r.status_code} {r.reason}")
        print(f"   Bot status : {data.get('status')}")
        print(f"   Uptime     : {data.get('uptime_seconds')}s")
        print(f"   ZMQ alive  : {data.get('zmq_listener_alive')}")
        print(f"   Signals    : {data.get('signals_processed')}")
        print(f"   Scans done : {data.get('scanner_cycles')}")
        print(f"   Last scan  : {data.get('last_scan_at') or 'never'}")
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        print("   FAILED — backend is not running on http://127.0.0.1:8000")
        print("   Start it with: uvicorn backend.main:app --host 127.0.0.1 --port 8000")
        return False


def test_scanner_status():
    print()
    print("=" * 55)
    print("2. GET /scanner/status")
    print("=" * 55)
    try:
        r = requests.get(f"{BASE_URL}/scanner/status", timeout=5)
        data = r.json()
        print(f"   Scanner enabled  : {data.get('enabled')}")
        print(f"   Interval         : {data.get('interval_minutes')} min")
        print(f"   Watchlist        : {data.get('watchlist')}")
        print(f"   Cached symbols   : {data.get('cached_symbols') or 'none (no MT5 data yet)'}")
        ages = data.get("symbol_data_age_sec", {})
        for sym, age in ages.items():
            status = f"{age}s old" if age is not None else "NO DATA"
            print(f"     {sym:<12} {status}")
    except requests.exceptions.ConnectionError:
        print("   FAILED — backend not reachable")


def test_signal(symbol: str = "EURUSD", base_price: float = 1.08500):
    print()
    print("=" * 55)
    print(f"3. POST /signal  (simulating MT5 EA for {symbol})")
    print("=" * 55)

    payload = {
        "symbol":      symbol,
        "equity":      10000.0,
        "balance":     10000.0,
        "open_trades": 0,
        "daily_pnl":   0.0,
        "bars_m15":    _make_bars(100, base_price, 0.0002),
        "bars_h1":     _make_bars(100, base_price, 0.0002),
        "bars_h4":     _make_bars(100, base_price, 0.0002),
        "bars_d1":     _make_bars(100, base_price, 0.0002),
    }

    print(f"   Sending {len(payload['bars_m15'])} bars × 4 timeframes ...")
    try:
        r = requests.post(f"{BASE_URL}/signal", json=payload, timeout=60)
        if r.status_code == 200:
            data = r.json()
            print(f"   HTTP            : {r.status_code} OK")
            print(f"   Signal ID       : {data.get('signal_id')}")
            print(f"   Direction       : {data.get('direction')}")
            print(f"   Confidence      : {data.get('confidence')}%")
            print(f"   Approved        : {data.get('approved')}")
            print(f"   Entry           : {data.get('entry')}")
            print(f"   SL              : {data.get('sl')}")
            print(f"   TP1             : {data.get('tp1')}")
            print(f"   Risk/Reward     : {data.get('risk_reward')}")
            print(f"   Est. Cost (USD) : ${data.get('estimated_cost_usd')}")
            print(f"   Reasoning       : {data.get('reasoning', '')[:120]}")
            if data.get("rejection_reasons"):
                print(f"   Rejected because: {data['rejection_reasons']}")
        else:
            print(f"   HTTP {r.status_code}: {r.text[:300]}")
    except requests.exceptions.ConnectionError:
        print("   FAILED — backend not reachable")
    except requests.exceptions.Timeout:
        print("   TIMEOUT — Claude took >60s (check API key and network)")


def test_stats():
    print()
    print("=" * 55)
    print("4. GET /stats")
    print("=" * 55)
    try:
        r = requests.get(f"{BASE_URL}/stats", timeout=5)
        data = r.json()
        print(f"   Total signals     : {data.get('total_signals')}")
        print(f"   Actionable        : {data.get('actionable_signals')}")
        print(f"   No-trade          : {data.get('no_trade_signals')}")
        print(f"   Win rate          : {data.get('win_rate_pct')}%")
        print(f"   Total API cost    : ${data.get('total_api_cost_usd')}")
    except requests.exceptions.ConnectionError:
        print("   FAILED — backend not reachable")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Test MT5 backend connection")
    parser.add_argument("--symbol", default="EURUSD", help="Symbol to test (default: EURUSD)")
    parser.add_argument("--price",  default=1.08500, type=float, help="Base price for test bars")
    parser.add_argument("--skip-signal", action="store_true",
                        help="Skip the Claude /signal call (health check only)")
    args = parser.parse_args()

    print()
    print("MT5 Backend Connection Test")
    print(f"Target: {BASE_URL}")
    print()

    ok = test_health()
    if not ok:
        sys.exit(1)

    test_scanner_status()

    if not args.skip_signal:
        test_signal(args.symbol, args.price)

    test_stats()

    print()
    print("Done.")
