"""
Unit tests for backend/claude_agent.py
Claude API and MT5 tools are fully mocked — no live connections required.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.claude_agent import (
    AgentResult,
    ToolCall,
    _dispatch_tool,
    run_agent,
)
from backend.risk_manager import AccountState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_account(equity=10_000.0, balance=10_000.0, open_trades=0, daily_pnl=0.0):
    return AccountState(
        equity=equity, balance=balance,
        open_trades=open_trades, daily_pnl=daily_pnl,
    )


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(name: str, input_: dict, block_id: str = "tu_001") -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.name = name
    b.input = input_
    b.id = block_id
    return b


def _response(content: list, stop_reason: str = "end_turn") -> MagicMock:
    usage = MagicMock()
    usage.input_tokens = 500
    usage.output_tokens = 200
    usage.cache_read_input_tokens = 0
    usage.cache_creation_input_tokens = 0

    resp = MagicMock()
    resp.content = content
    resp.stop_reason = stop_reason
    resp.usage = usage
    return resp


# ---------------------------------------------------------------------------
# AgentResult
# ---------------------------------------------------------------------------

class TestAgentResult:
    def test_estimated_cost_zero_when_no_tokens(self):
        r = AgentResult(symbol="BTCUSDc")
        assert r.estimated_cost_usd == 0.0

    def test_estimated_cost_positive(self):
        r = AgentResult(
            symbol="BTCUSDc",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=0,
            cache_write_tokens=0,
        )
        assert r.estimated_cost_usd > 0

    def test_cache_read_cheaper_than_full_input(self):
        r_cached = AgentResult(
            symbol="BTCUSDc",
            input_tokens=100,
            output_tokens=100,
            cache_read_tokens=900,
        )
        r_uncached = AgentResult(
            symbol="BTCUSDc",
            input_tokens=1000,
            output_tokens=100,
        )
        assert r_cached.estimated_cost_usd < r_uncached.estimated_cost_usd

    def test_trade_placed_defaults_false(self):
        r = AgentResult(symbol="BTCUSDc")
        assert r.trade_placed is False
        assert r.trade_ticket is None

    def test_positions_managed_defaults_zero(self):
        r = AgentResult(symbol="BTCUSDc")
        assert r.positions_managed == 0


# ---------------------------------------------------------------------------
# _dispatch_tool
# ---------------------------------------------------------------------------

class TestDispatchTool:
    def test_routes_get_account_info(self):
        with patch("backend.claude_agent.tools.get_account_info", return_value={"equity": 10000}) as m:
            result = _dispatch_tool("get_account_info", {})
        m.assert_called_once_with()
        assert result["equity"] == 10000

    def test_routes_get_open_positions_no_symbol(self):
        with patch("backend.claude_agent.tools.get_open_positions", return_value={"count": 0}) as m:
            _dispatch_tool("get_open_positions", {})
        m.assert_called_once_with(None)

    def test_routes_get_open_positions_with_symbol(self):
        with patch("backend.claude_agent.tools.get_open_positions", return_value={"count": 0}) as m:
            _dispatch_tool("get_open_positions", {"symbol": "BTCUSDc"})
        m.assert_called_once_with("BTCUSDc")

    def test_routes_get_market_snapshot(self):
        with patch("backend.claude_agent.tools.get_market_snapshot", return_value={"snapshot": "..."}) as m:
            _dispatch_tool("get_market_snapshot", {"symbol": "BTCUSDc"})
        m.assert_called_once_with("BTCUSDc")

    def test_routes_get_symbol_info(self):
        with patch("backend.claude_agent.tools.get_symbol_info", return_value={}) as m:
            _dispatch_tool("get_symbol_info", {"symbol": "BTCUSDc"})
        m.assert_called_once_with("BTCUSDc")

    def test_routes_place_order(self):
        order_args = {
            "symbol": "BTCUSDc", "direction": "BUY",
            "lot_size": 0.01, "sl": 93000.0, "tp1": 96500.0,
        }
        with patch("backend.claude_agent.tools.place_order", return_value={"success": True}) as m:
            _dispatch_tool("place_order", order_args)
        m.assert_called_once_with(**order_args)

    def test_routes_modify_position(self):
        with patch("backend.claude_agent.tools.modify_position", return_value={"success": True}) as m:
            _dispatch_tool("modify_position", {"ticket": 99001, "sl": 94000.0})
        m.assert_called_once_with(ticket=99001, sl=94000.0)

    def test_routes_close_position(self):
        with patch("backend.claude_agent.tools.close_position", return_value={"success": True}) as m:
            _dispatch_tool("close_position", {"ticket": 99001})
        m.assert_called_once_with(ticket=99001)

    def test_unknown_tool_returns_error(self):
        result = _dispatch_tool("nonexistent_tool", {})
        assert "error" in result


# ---------------------------------------------------------------------------
# run_agent — async integration tests
# ---------------------------------------------------------------------------

class TestRunAgent:
    @patch("backend.claude_agent._get_client")
    async def test_end_turn_immediately(self, mock_get_client):
        """Claude returns end_turn on first call — agent completes cleanly."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _response(
            [_text_block("No trade this cycle.")],
            stop_reason="end_turn",
        )

        result = await run_agent("BTCUSDc", _make_account())

        assert isinstance(result, AgentResult)
        assert result.error is None
        assert result.trade_placed is False
        assert result.input_tokens == 500
        assert result.output_tokens == 200
        mock_client.messages.create.assert_called_once()

    @patch("backend.claude_agent._get_client")
    async def test_tool_use_then_end_turn(self, mock_get_client):
        """Claude calls get_account_info, then ends turn."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_block = _tool_use_block("get_account_info", {}, "tu_001")

        mock_client.messages.create.side_effect = [
            _response([tool_block], stop_reason="tool_use"),
            _response([_text_block("Done.")], stop_reason="end_turn"),
        ]

        with patch("backend.claude_agent.tools.get_account_info", return_value={"equity": 10000.0}):
            result = await run_agent("BTCUSDc", _make_account())

        assert result.error is None
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_account_info"
        assert result.tool_calls[0].result == {"equity": 10000.0}
        assert mock_client.messages.create.call_count == 2

    @patch("backend.claude_agent._get_client")
    async def test_place_order_sets_trade_placed(self, mock_get_client):
        """Claude calls place_order successfully → trade_placed=True, ticket set."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        order_input = {
            "symbol": "BTCUSDc", "direction": "BUY",
            "lot_size": 0.01, "sl": 93000.0, "tp1": 96500.0,
        }
        tool_block = _tool_use_block("place_order", order_input, "tu_002")

        mock_client.messages.create.side_effect = [
            _response([tool_block], stop_reason="tool_use"),
            _response([_text_block("Trade placed.")], stop_reason="end_turn"),
        ]

        order_result = {"success": True, "ticket": 99001, "entry_price": 95000.0}
        with patch("backend.claude_agent.tools.place_order", return_value=order_result):
            result = await run_agent("BTCUSDc", _make_account())

        assert result.trade_placed is True
        assert result.trade_ticket == 99001

    @patch("backend.claude_agent._get_client")
    async def test_modify_position_increments_managed(self, mock_get_client):
        """Claude calls modify_position → positions_managed incremented."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_block = _tool_use_block(
            "modify_position", {"ticket": 99001, "sl": 94000.0}, "tu_003"
        )
        mock_client.messages.create.side_effect = [
            _response([tool_block], stop_reason="tool_use"),
            _response([_text_block("Done.")], stop_reason="end_turn"),
        ]

        with patch("backend.claude_agent.tools.modify_position", return_value={"success": True}):
            result = await run_agent("BTCUSDc", _make_account())

        assert result.positions_managed == 1

    @patch("backend.claude_agent._get_client")
    async def test_close_position_increments_managed(self, mock_get_client):
        """Claude calls close_position → positions_managed incremented."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_block = _tool_use_block("close_position", {"ticket": 99001}, "tu_004")
        mock_client.messages.create.side_effect = [
            _response([tool_block], stop_reason="tool_use"),
            _response([_text_block("Closed.")], stop_reason="end_turn"),
        ]

        with patch("backend.claude_agent.tools.close_position", return_value={"success": True}):
            result = await run_agent("BTCUSDc", _make_account())

        assert result.positions_managed == 1

    @patch("backend.claude_agent._get_client")
    async def test_failed_order_does_not_set_trade_placed(self, mock_get_client):
        """place_order returns error → trade_placed stays False."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_block = _tool_use_block(
            "place_order",
            {"symbol": "BTCUSDc", "direction": "BUY", "lot_size": 0.01, "sl": 93000.0, "tp1": 96500.0},
        )
        mock_client.messages.create.side_effect = [
            _response([tool_block], stop_reason="tool_use"),
            _response([_text_block("Order failed.")], stop_reason="end_turn"),
        ]

        with patch("backend.claude_agent.tools.place_order", return_value={"error": "Max open trades reached"}):
            result = await run_agent("BTCUSDc", _make_account())

        assert result.trade_placed is False
        assert result.trade_ticket is None

    @patch("backend.claude_agent._get_client")
    async def test_send_update_calls_telegram(self, mock_get_client):
        """send_update tool calls telegram_notifier.send_agent_update."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_block = _tool_use_block(
            "send_update",
            {"message": "No trade.", "message_type": "no_trade"},
        )
        mock_client.messages.create.side_effect = [
            _response([tool_block], stop_reason="tool_use"),
            _response([_text_block("Done.")], stop_reason="end_turn"),
        ]

        notifier = MagicMock()
        notifier.send_agent_update = AsyncMock()

        result = await run_agent("BTCUSDc", _make_account(), telegram_notifier=notifier)

        notifier.send_agent_update.assert_called_once_with("No trade.")

    @patch("backend.claude_agent._get_client")
    async def test_send_update_without_telegram_succeeds(self, mock_get_client):
        """send_update with no notifier returns success without crashing."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_block = _tool_use_block(
            "send_update",
            {"message": "No trade.", "message_type": "no_trade"},
        )
        mock_client.messages.create.side_effect = [
            _response([tool_block], stop_reason="tool_use"),
            _response([_text_block("Done.")], stop_reason="end_turn"),
        ]

        result = await run_agent("BTCUSDc", _make_account(), telegram_notifier=None)
        assert result.error is None

    @patch("backend.claude_agent._get_client")
    async def test_max_iterations_reached(self, mock_get_client):
        """Agent stops and records no error when max_iterations hit."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        # Always return tool_use so the loop never naturally ends
        tool_block = _tool_use_block("get_account_info", {})
        mock_client.messages.create.return_value = _response([tool_block], stop_reason="tool_use")

        with patch("backend.claude_agent.tools.get_account_info", return_value={"equity": 10000.0}):
            result = await run_agent("BTCUSDc", _make_account(), max_iterations=3)

        assert mock_client.messages.create.call_count == 3
        assert result.error is None  # hitting max is not an error

    @patch("backend.claude_agent._get_client")
    async def test_api_exception_recorded(self, mock_get_client):
        """Exception from Claude API is caught and stored in result.error."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("API down")

        result = await run_agent("BTCUSDc", _make_account())

        assert result.error is not None
        assert "API down" in result.error
        assert result.trade_placed is False

    @patch("backend.claude_agent._get_client")
    async def test_token_usage_accumulates_across_iterations(self, mock_get_client):
        """Token counts from multiple API calls are summed in the result."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        tool_block = _tool_use_block("get_account_info", {})
        # 2 iterations: tool_use then end_turn
        mock_client.messages.create.side_effect = [
            _response([tool_block], stop_reason="tool_use"),   # 500 in, 200 out
            _response([_text_block("Done.")], stop_reason="end_turn"),  # 500 in, 200 out
        ]

        with patch("backend.claude_agent.tools.get_account_info", return_value={"equity": 10000.0}):
            result = await run_agent("BTCUSDc", _make_account())

        assert result.input_tokens == 1000   # 500 × 2
        assert result.output_tokens == 400   # 200 × 2

    @patch("backend.claude_agent._get_client")
    async def test_prompt_caching_param_sent(self, mock_get_client):
        """Verifies ephemeral cache_control is sent in system prompt."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        mock_client.messages.create.return_value = _response(
            [_text_block("Done.")], stop_reason="end_turn"
        )

        await run_agent("BTCUSDc", _make_account())

        call_kwargs = mock_client.messages.create.call_args.kwargs
        system = call_kwargs["system"]
        assert isinstance(system, list)
        assert system[0]["cache_control"] == {"type": "ephemeral"}

    @patch("backend.claude_agent._get_client")
    async def test_multiple_tool_calls_in_one_response(self, mock_get_client):
        """Claude can call multiple tools in a single response turn."""
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client

        block1 = _tool_use_block("get_account_info", {}, "tu_001")
        block2 = _tool_use_block("get_open_positions", {}, "tu_002")

        mock_client.messages.create.side_effect = [
            _response([block1, block2], stop_reason="tool_use"),
            _response([_text_block("Done.")], stop_reason="end_turn"),
        ]

        with (
            patch("backend.claude_agent.tools.get_account_info", return_value={"equity": 10000.0}),
            patch("backend.claude_agent.tools.get_open_positions", return_value={"count": 0}),
        ):
            result = await run_agent("BTCUSDc", _make_account())

        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "get_account_info"
        assert result.tool_calls[1].name == "get_open_positions"
