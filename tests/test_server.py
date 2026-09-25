"""HTTP surface and MCP protocol tests (in-process, no network)."""

from __future__ import annotations

import pytest
from mcp import Client
from starlette.testclient import TestClient

from trading_mcp.server import create_app, create_mcp

INIT = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}},
}
TOKEN = "test-token-0123456789abcdef"
HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}


@pytest.fixture
def http():
    with TestClient(create_app()) as client:
        yield client


def test_health_is_public(http):
    r = http.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["auth_required"] is True
    assert "test-token-0123456789abcdef" not in r.text


def test_mcp_requires_token(http):
    assert http.post("/mcp", json=INIT, headers=HEADERS).status_code == 401
    assert http.post("/mcp?key=wrong", json=INIT, headers=HEADERS).status_code == 401


@pytest.mark.parametrize(
    "auth", [{"params": {"key": TOKEN}}, {"headers": {"Authorization": f"Bearer {TOKEN}"}}]
)
def test_mcp_accepts_token_in_query_or_header(http, auth):
    headers = {**HEADERS, **auth.get("headers", {})}
    r = http.post("/mcp", json=INIT, headers=headers, params=auth.get("params"))
    assert r.status_code == 200
    assert r.json()["result"]["serverInfo"]["name"] == "trading-mcp"


def test_unknown_paths_not_exposed(http):
    assert http.get("/docs").status_code == 404
    assert http.get("/").status_code == 404


async def test_market_tools_via_mcp_client():
    async with Client(create_mcp()) as client:
        names = {t.name for t in (await client.list_tools()).tools}
        assert {"get_quote", "get_historical_data", "search_symbol"} <= names

        quote = await client.call_tool("get_quote", {"ticker": "NVDA"})
        assert not quote.is_error and quote.structured_content["price"] > 0

        hist = await client.call_tool("get_historical_data", {"ticker": "NVDA", "timeframe": "1h", "limit": 10})
        assert not hist.is_error
        assert len(hist.structured_content["bars"]) == 10

        bad = await client.call_tool("get_historical_data", {"ticker": "NVDA", "timeframe": "3h"})
        assert bad.is_error and "Unsupported timeframe" in bad.content[0].text


def test_security_headers(http):
    r = http.get("/health")
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["cache-control"] == "no-store"


def test_refuses_to_start_without_token(monkeypatch):
    from trading_mcp.config import Settings

    with pytest.raises(RuntimeError, match="MCP_AUTH_TOKEN is not set"):
        create_app(Settings(mcp_auth_token=""))
    with pytest.raises(RuntimeError, match="at least"):
        create_app(Settings(mcp_auth_token="short"))
    assert create_app(Settings(mcp_auth_token="", allow_unauthenticated=True)) is not None


def test_rate_limit():
    from trading_mcp.config import Settings

    app = create_app(Settings(mcp_auth_token="x" * 30, rate_limit_per_minute=3))
    with TestClient(app) as c:
        codes = [c.post("/mcp", json=INIT, headers=HEADERS).status_code for _ in range(5)]
        assert [c.get("/health").status_code for _ in range(5)] == [200] * 5  # /health is never rate limited
    assert codes[:3] == [401, 401, 401] and codes[3:] == [429, 429]


def test_access_log_redacts_token(http, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger="trading_mcp.access"):
        http.post("/mcp?key=test-token-0123456789abcdef", json=INIT, headers=HEADERS)
    assert "POST /mcp" in caplog.text
    assert "test-token-0123456789abcdef" not in caplog.text
