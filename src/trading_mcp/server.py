"""MCP server assembly: tool registration and the HTTP (ASGI) application.

Exposed HTTP surface:
    POST/GET/DELETE /mcp   Streamable HTTP MCP endpoint (token protected)
    GET /health            liveness probe (public, no sensitive data)
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from trading_mcp import __version__
from trading_mcp.auth import TokenAuthMiddleware
from trading_mcp.config import Settings, get_settings

logger = logging.getLogger(__name__)

INSTRUCTIONS = """\
Financial research server: market data, technical indicators, multi-timeframe analysis, screener,
backtesting (incl. walk-forward), machine learning, risk and portfolio analytics.
Outputs are descriptive and probabilistic. Never present them as investment advice or certain
predictions. Always mention data source limitations, fees/slippage and overfitting risk when
discussing backtests or model results."""


def create_mcp() -> MCPServer:
    """Build the MCP server and register every tool module."""
    from trading_mcp.tools import register_all

    mcp = MCPServer(name="trading-mcp", title="Trading MCP", version=__version__, instructions=INSTRUCTIONS)
    register_all(mcp)

    @mcp.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> JSONResponse:
        settings = get_settings()
        return JSONResponse(
            {
                "status": "ok",
                "service": "trading-mcp",
                "version": __version__,
                "data_provider": settings.market_data_provider,
                "auth_required": bool(settings.mcp_auth_token),
            }
        )

    return mcp


def create_app(settings: Settings | None = None) -> Starlette:
    """Return the production ASGI app (MCP + /health + auth + CORS)."""
    settings = settings or get_settings()
    mcp = create_mcp()
    security = TransportSecuritySettings(
        enable_dns_rebinding_protection=bool(settings.allowed_hosts),
        allowed_hosts=settings.allowed_hosts,
        allowed_origins=settings.cors_origins,
    )
    app = mcp.streamable_http_app(
        streamable_http_path="/mcp",
        stateless_http=True,  # no server-side session state: survives restarts / scales horizontally
        json_response=True,
        transport_security=security,
        host=settings.host,
    )
    if settings.mcp_auth_token:
        app.add_middleware(TokenAuthMiddleware, token=settings.mcp_auth_token, protected_prefix="/mcp")
    else:
        logger.warning("MCP_AUTH_TOKEN is empty: /mcp is UNAUTHENTICATED. Only acceptable for local development.")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "Mcp-Session-Id", "Mcp-Protocol-Version", "Last-Event-ID"],
        expose_headers=["Mcp-Session-Id"],
    )
    return app
