"""MCP tool modules. Each module exposes ``register(mcp)``."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer


def register_all(mcp: MCPServer) -> None:
    from trading_mcp.tools import analysis, backtest, market, ml, technical

    for module in (market, technical, analysis, backtest, ml):
        module.register(mcp)
