"""
Unit tests for backend/mt5_tools.py
All MT5 calls are mocked — no live terminal required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

import backend.mt5_tools as tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mt5(
    *,
    equity: float = 10_000.0,
    balance: float = 10_000.0,
    positions: list | None = None,
    order_retcode: int = None,  # mt5.TRADE_RETCODE_DONE = 10009
) -> MagicMock:
    """Build a fully configured mock MT5 module."""
    mt5 = MagicMock()

    # Constants
    mt5.TRADE_RETCODE_DONE = 10009
    mt5.TRADE_ACTION_DEAL = 1
    mt5.TRADE_ACTION_PENDING = 5
    mt5.TRADE_ACTION_SLTP = 6
    mt5.ORDER_TYPE_BUY = 0
    mt5.ORDER_TYPE_SELL = 1
    mt5.ORDER_TYPE_BUY_LIMIT = 2
    mt5.ORDER_TYPE_SELL_LIMIT = 3
    mt5.ORDER_TYPE_BUY_STOP = 4
    mt5.ORDER_TYPE_SELL_STOP = 5
    mt5.ORDER_TIME_GTC = 1
    mt5.ORDER_FILLING_IOC = 1

    # account_info
    account = MagicMock()
    account.login = 12345
    account.server = "Exness-MT5Real"
    account.equity = equity
    account.balance = balance
    account.margin_free = balance * 0.9
    account.margin = balance * 0.1
    account.currency = "USD"
    account.leverage = 200
    mt5.account_info.return_value = account

    # positions_get
    positions = positions or []
    mt5.positions_get.return_value = positions

    # order_send — default success
    if order_retcode is None:
        order_retcode = mt5.TRADE_RETCODE_DONE
    send_result = MagicMock()
    send_result.retcode = order_retcode
    send_result.order = 99001
    send_result.price = 95000.0
    send_result.volume = 0.01
    send_result.comment = "Request executed"
    mt5.order_send.return_value = send_result

    # symbol_info
    sym = MagicMock()
    sym.digits = 2
    sym.point = 0.01
    sym.trade_tick_size = 0.01
    sym.trade_tick_value = 1.0
    sym.trade_contract_size = 1.0
    sym.volume_min = 0.01
    sym.volume_max = 100.0
    sym.volume_step = 0.01
    sym.currency_profit = "USD"
    sym.swap_long = -5.0
    sym.swap_short = -5.0
    sym.spread = 30
    mt5.symbol_info.return_value = sym

    # symbol_info_tick
    tick = MagicMock()
    tick.bid = 94990.0
    tick.ask = 95000.0
    mt5.symbol_info_tick.return_value = tick

    return mt5


def _make_position(
    ticket=99001, symbol="BTCUSDc", type_=0,
    volume=0.01, price_open=94000.0, price_current=95000.0,
    sl=93000.0, tp=95500.0, profit=100.0, swap=-1.0,
    time=1_700_000_000, comment="ClaudeAgent",
) -> MagicMock:
    pos = MagicMock()
    pos.ticket = ticket
    pos.symbol = symbol
    pos.type = type_          # 0=BUY, 1=SELL
    pos.volume = volume
    pos.price_open = price_open
    pos.price_current = price_current
    pos.sl = sl
    pos.tp = tp
    pos.profit = profit
    pos.swap = swap
    pos.time = time
    pos.comment = comment
    return pos


# ---------------------------------------------------------------------------
# get_account_info
# ---------------------------------------------------------------------------

class TestGetAccountInfo:
    def test_returns_all_fields(self):
        mt5 = _make_mt5(equity=10_500.0, balance=10_000.0)
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.get_account_info()

        assert result["equity"] == 10_500.0
        assert result["balance"] == 10_000.0
        assert result["daily_pnl"] == pytest.approx(500.0)
        assert result["daily_loss_limit"] == pytest.approx(300.0)  # 3% of 10_000
        assert result["daily_loss_remaining"] == pytest.approx(800.0)
        assert result["login"] == 12345
        assert result["currency"] == "USD"

    def test_returns_error_when_account_none(self):
        mt5 = _make_mt5()
        mt5.account_info.return_value = None
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.get_account_info()
        assert "error" in result

    def test_open_trades_counted(self):
        pos = _make_position()
        mt5 = _make_mt5(positions=[pos, pos])
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.get_account_info()
        assert result["open_trades"] == 2


# ---------------------------------------------------------------------------
# get_open_positions
# ---------------------------------------------------------------------------

class TestGetOpenPositions:
    def test_no_positions(self):
        mt5 = _make_mt5()
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.get_open_positions()
        assert result["count"] == 0
        assert result["positions"] == []

    def test_buy_position_formatted(self):
        pos = _make_position(type_=0, price_open=94000.0, price_current=95000.0, sl=93000.0)
        mt5 = _make_mt5(positions=[pos])
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.get_open_positions()

        assert result["count"] == 1
        p = result["positions"][0]
        assert p["direction"] == "BUY"
        assert p["ticket"] == 99001
        assert p["entry_price"] == 94000.0
        assert p["sl"] == 93000.0
        # r_moved = (95000 - 94000) / (94000 - 93000) = 1.0
        assert p["r_moved"] == pytest.approx(1.0)

    def test_sell_position_formatted(self):
        pos = _make_position(type_=1, price_open=95000.0, price_current=94000.0, sl=96000.0)
        mt5 = _make_mt5(positions=[pos])
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.get_open_positions()
        assert result["positions"][0]["direction"] == "SELL"

    def test_filters_by_symbol(self):
        mt5 = _make_mt5()
        mt5.positions_get.return_value = []
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            tools.get_open_positions(symbol="BTCUSDc")
        mt5.positions_get.assert_called_once_with(symbol="BTCUSDc")

    def test_no_symbol_fetches_all(self):
        mt5 = _make_mt5()
        mt5.positions_get.return_value = []
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            tools.get_open_positions()
        mt5.positions_get.assert_called_once_with()


# ---------------------------------------------------------------------------
# get_symbol_info
# ---------------------------------------------------------------------------

class TestGetSymbolInfo:
    def test_returns_specs(self):
        mt5 = _make_mt5()
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.get_symbol_info("BTCUSDc")

        assert result["symbol"] == "BTCUSDc"
        assert result["bid"] == pytest.approx(94990.0)
        assert result["ask"] == pytest.approx(95000.0)
        assert result["min_lot"] == 0.01

    def test_symbol_not_found(self):
        mt5 = _make_mt5()
        mt5.symbol_info.return_value = None
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.get_symbol_info("INVALID")
        assert "error" in result


# ---------------------------------------------------------------------------
# place_order — risk guards
# ---------------------------------------------------------------------------

class TestPlaceOrderGuards:
    def test_lot_exceeds_max_rejected(self):
        mt5 = _make_mt5()
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="BTCUSDc", direction="BUY",
                lot_size=999.0, sl=93000.0, tp1=96000.0,
            )
        assert "error" in result
        assert "max_lot_size" in result["error"]
        mt5.order_send.assert_not_called()

    def test_max_open_trades_rejected(self):
        positions = [_make_position(ticket=i) for i in range(3)]
        mt5 = _make_mt5(positions=positions)
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="BTCUSDc", direction="BUY",
                lot_size=0.01, sl=93000.0, tp1=96000.0,
            )
        assert "error" in result
        assert "Max open trades" in result["error"]
        mt5.order_send.assert_not_called()

    def test_daily_loss_limit_rejected(self):
        # equity < balance means daily loss
        mt5 = _make_mt5(equity=9_600.0, balance=10_000.0)  # -400 loss, limit is -300
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="BTCUSDc", direction="BUY",
                lot_size=0.01, sl=93000.0, tp1=96000.0,
            )
        assert "error" in result
        assert "Daily loss limit" in result["error"]
        mt5.order_send.assert_not_called()

    def test_sl_too_close_rejected(self):
        mt5 = _make_mt5()
        # ask=95000, sl=94999.99 → distance < 10 points (0.10)
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="BTCUSDc", direction="BUY",
                lot_size=0.01, sl=94999.95, tp1=96000.0,
            )
        assert "error" in result
        assert "too close" in result["error"]
        mt5.order_send.assert_not_called()

    def test_symbol_not_found_rejected(self):
        mt5 = _make_mt5()
        mt5.symbol_info.return_value = None
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="INVALID", direction="BUY",
                lot_size=0.01, sl=93000.0, tp1=96000.0,
            )
        assert "error" in result


class TestPlaceOrderSuccess:
    def test_market_buy_placed(self):
        mt5 = _make_mt5()
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="BTCUSDc", direction="BUY",
                lot_size=0.01, sl=93000.0, tp1=96500.0,
                tp2=98500.0, tp3=102000.0,
                comment="TestOrder",
            )

        assert result.get("success") is True
        assert result["ticket"] == 99001
        assert result["direction"] == "BUY"
        assert result["sl"] == 93000.0
        assert result["tp1"] == 96500.0
        mt5.order_send.assert_called_once()

    def test_order_type_determines_action(self):
        mt5 = _make_mt5()
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="BTCUSDc", direction="BUY",
                lot_size=0.01, sl=93000.0, tp1=96500.0,
                order_type="LIMIT", limit_price=94500.0,
            )
        assert result.get("success") is True
        sent_request = mt5.order_send.call_args[0][0]
        assert sent_request["action"] == mt5.TRADE_ACTION_PENDING

    def test_limit_order_requires_price(self):
        mt5 = _make_mt5()
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="BTCUSDc", direction="BUY",
                lot_size=0.01, sl=93000.0, tp1=96500.0,
                order_type="LIMIT",  # no limit_price
            )
        assert "error" in result

    def test_mt5_rejection_returned_as_error(self):
        mt5 = _make_mt5()
        reject = MagicMock()
        reject.retcode = 10006  # rejected
        reject.comment = "Invalid price"
        mt5.order_send.return_value = reject
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.place_order(
                symbol="BTCUSDc", direction="BUY",
                lot_size=0.01, sl=93000.0, tp1=96500.0,
            )
        assert "error" in result
        assert "Invalid price" in result["error"]


# ---------------------------------------------------------------------------
# modify_position
# ---------------------------------------------------------------------------

class TestModifyPosition:
    def test_success(self):
        pos = _make_position(sl=93000.0, tp=96500.0)
        mt5 = _make_mt5()
        mt5.positions_get.return_value = [pos]
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.modify_position(ticket=99001, sl=94000.0)

        assert result.get("success") is True
        assert result["new_sl"] == 94000.0
        assert result["new_tp"] == 96500.0  # unchanged

    def test_position_not_found(self):
        mt5 = _make_mt5()
        mt5.positions_get.return_value = []
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.modify_position(ticket=99001, sl=94000.0)
        assert "error" in result

    def test_mt5_rejection_returned(self):
        pos = _make_position()
        mt5 = _make_mt5()
        mt5.positions_get.return_value = [pos]
        reject = MagicMock()
        reject.retcode = 10006
        reject.comment = "Modify denied"
        mt5.order_send.return_value = reject
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.modify_position(ticket=99001, sl=94000.0)
        assert "error" in result


# ---------------------------------------------------------------------------
# close_position
# ---------------------------------------------------------------------------

class TestClosePosition:
    def test_full_close(self):
        pos = _make_position(volume=0.02, type_=0)
        mt5 = _make_mt5()
        mt5.positions_get.return_value = [pos]
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.close_position(ticket=99001)

        assert result.get("success") is True
        assert result["closed_volume"] == 0.02
        assert result["remaining_volume"] == pytest.approx(0.0)

    def test_partial_close(self):
        pos = _make_position(volume=0.04, type_=0)
        mt5 = _make_mt5()
        mt5.positions_get.return_value = [pos]
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.close_position(ticket=99001, lot_size=0.02)

        assert result["closed_volume"] == 0.02
        assert result["remaining_volume"] == pytest.approx(0.02)

    def test_cannot_close_more_than_open(self):
        pos = _make_position(volume=0.01, type_=0)
        mt5 = _make_mt5()
        mt5.positions_get.return_value = [pos]
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.close_position(ticket=99001, lot_size=999.0)

        # Should cap at actual volume
        assert result["closed_volume"] == pytest.approx(0.01)

    def test_position_not_found(self):
        mt5 = _make_mt5()
        mt5.positions_get.return_value = []
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            result = tools.close_position(ticket=99001)
        assert "error" in result

    def test_sell_closes_with_buy_order(self):
        pos = _make_position(type_=1, volume=0.01)  # SELL position
        mt5 = _make_mt5()
        mt5.positions_get.return_value = [pos]
        with patch("backend.mt5_tools._mt5", return_value=mt5):
            tools.close_position(ticket=99001)

        sent = mt5.order_send.call_args[0][0]
        assert sent["type"] == mt5.ORDER_TYPE_BUY  # closing SELL = BUY
