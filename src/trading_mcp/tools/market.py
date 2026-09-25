"""Market data MCP tools."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from trading_mcp.data.factory import get_provider
from trading_mcp.tools.common import clean, frame_to_records, load_ohlcv, safe_tool, source_info

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


@safe_tool
def get_quote(ticker: str, exchange: str | None = None) -> dict[str, Any]:
    """Latest price, change vs previous close, volume, timestamp and data source for a symbol.

    Accepts stocks (NVDA), ETFs (SPY), indices (SPX or ^GSPC), forex (EURUSD), crypto (BTCUSD or
    BTC-USD), futures (ES1! or ES=F) and TradingView-style prefixes (NASDAQ:NVDA). ``exchange`` adds
    a market suffix for non-US listings (e.g. ticker=MC exchange=PA for LVMH in Paris).
    Free data is usually delayed.
    """
    return clean(get_provider().get_quote(ticker, exchange).model_dump())


@safe_tool
def get_historical_data(
    ticker: str,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    limit: int = 300,
) -> dict[str, Any]:
    """OHLCV bars for a symbol.

    timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w. start/end: ISO dates (UTC); defaults depend on the
    timeframe. Yahoo limits intraday history (1m: ~7 days, 5m-30m: ~60 days, 1h/4h: ~2 years).
    ``limit`` caps the number of most recent bars returned (max 5000) to keep responses small.
    Prices are split/dividend adjusted.
    """
    limit = max(1, min(int(limit), 5000))
    df = load_ohlcv(ticker, timeframe, start, end)
    return {"meta": source_info(df), "returned_bars": min(limit, len(df)), "bars": frame_to_records(df, limit)}


@safe_tool
def search_symbol(query: str, limit: int = 10) -> dict[str, Any]:
    """Search tickers by company name or partial symbol (e.g. 'nvidia', 'apple', 'bitcoin')."""
    matches = get_provider().search_symbol(query, max(1, min(int(limit), 25)))
    return {"query": query, "results": [m.model_dump() for m in matches], "source": get_provider().name}


def register(mcp: MCPServer) -> None:
    for fn in (get_quote, get_historical_data, search_symbol):
        mcp.add_tool(fn, annotations=READ_ONLY)
