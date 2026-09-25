"""Multi-timeframe analysis and screener MCP tools."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from trading_mcp.analysis.snapshot import compute_snapshot
from trading_mcp.data.factory import get_provider
from trading_mcp.screener.screener import UNIVERSES, ScreenerFilters, run_screen
from trading_mcp.tools.common import RESEARCH_DISCLAIMER, clean, load_ohlcv, parse_timeframe, safe_tool

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
SORTABLE = {"change_percent", "relative_volume", "rsi_14", "volatility_20_annualized", "volume",
            "distance_vwap_percent", "distance_sma50_percent", "distance_sma200_percent", "market_cap", "price"}


@safe_tool
def multi_timeframe_analysis(
    ticker: str, timeframes: list[str] | None = None
) -> dict[str, Any]:
    """Per-timeframe technical picture: trend, RSI, MACD, EMAs/SMAs, VWAP (intraday), volatility, volume
    and descriptive signals. Default timeframes: 5m, 15m, 1h, 4h, 1d.

    Returns observed conditions and an alignment count across timeframes - NOT a buy/sell recommendation.
    """
    tfs = [parse_timeframe(t) for t in (timeframes or ["5m", "15m", "1h", "4h", "1d"])]
    if not 1 <= len(tfs) <= 8:
        raise ValueError("Provide between 1 and 8 timeframes.")

    def analyse(tf):  # noqa: ANN001, ANN202
        try:
            df = load_ohlcv(ticker, tf.value)
            snap = compute_snapshot(df, tf)
            snap["bars_used"] = len(df)
            snap["source"] = df.attrs.get("source")
            return tf.value, snap
        except Exception as exc:  # noqa: BLE001 - report per-timeframe failures
            return tf.value, {"error": str(exc)}

    with ThreadPoolExecutor(max_workers=len(tfs)) as pool:
        results = dict(pool.map(analyse, tfs))

    trends = [r.get("trend") for r in results.values() if "trend" in r]
    return clean({
        "ticker": ticker,
        "timeframes": results,
        "alignment": {
            "up": trends.count("up"), "down": trends.count("down"), "sideways": trends.count("sideways"),
            "description": _alignment_text(trends),
        },
        "trend_definition": "up: close > EMA50, EMA20 > EMA50 and EMA50 rising over 5 bars; down: the mirror; "
                            "otherwise sideways.",
        "disclaimer": RESEARCH_DISCLAIMER,
    })


def _alignment_text(trends: list[str | None]) -> str:
    if not trends:
        return "No timeframe could be analysed."
    if all(t == "up" for t in trends):
        return "All analysed timeframes show an up-trend structure."
    if all(t == "down" for t in trends):
        return "All analysed timeframes show a down-trend structure."
    return "Timeframes disagree (mixed structure)."


@safe_tool
def market_screener(
    universe: str = "us_mega_caps",
    tickers: list[str] | None = None,
    timeframe: str = "1d",
    filters: ScreenerFilters | None = None,
    sort_by: str = "change_percent",
    descending: bool = True,
    limit: int = 25,
) -> dict[str, Any]:
    """Scan a universe of symbols and keep those matching ALL filters.

    universe: one of us_mega_caps, us_etfs, crypto, forex_majors, cac40 - or pass your own ``tickers`` (max 100).
    filters (all optional): min/max_price, min_volume, min/max_change_percent, min/max_rsi,
    min/max_volatility (annualised fraction), min/max_market_cap, trend (up/down/sideways),
    min_relative_volume, breakout (up/down/any, vs prior 20-bar high/low), min/max_distance_vwap_percent,
    min/max_distance_sma50_percent, min/max_distance_sma200_percent, min/max_distance_ema20_percent.
    Example: high relative volume, up > 3%, RSI 50-70 ->
    {"min_relative_volume": 1.5, "min_change_percent": 3, "min_rsi": 50, "max_rsi": 70}.
    The universe is a fixed list of current constituents (survivorship bias for historical use).
    """
    if tickers:
        universe_list, universe_name = tickers, "custom"
    elif universe in UNIVERSES:
        universe_list, universe_name = UNIVERSES[universe], universe
    else:
        raise ValueError(f"Unknown universe '{universe}'. Choose from {sorted(UNIVERSES)} or pass tickers.")
    if sort_by not in SORTABLE:
        raise ValueError(f"sort_by must be one of {sorted(SORTABLE)}")
    filters = filters or ScreenerFilters()
    tf = parse_timeframe(timeframe)
    result = run_screen(get_provider(), universe_list, tf, filters)
    matches = sorted(
        result["matches"],
        key=lambda s: (s.get(sort_by) is None, -(s.get(sort_by) or 0) if descending else (s.get(sort_by) or 0)),
    )[: max(1, min(int(limit), 100))]
    return clean({
        "universe": universe_name, "timeframe": tf.value, "scanned": len(universe_list),
        "filters": filters.model_dump(exclude_none=True),
        "match_count": len(result["matches"]),
        "matches": [_compact(s) for s in matches],
        "errors": result["errors"],
        "rejected_sample": result["rejected"][:10],
        "disclaimer": RESEARCH_DISCLAIMER,
    })


_COMPACT_KEYS = ("ticker", "timestamp", "price", "change_percent", "volume", "relative_volume", "rsi_14", "trend",
                 "breakout_20", "volatility_20_annualized", "distance_vwap_percent", "distance_sma50_percent",
                 "distance_sma200_percent", "market_cap", "signals")


def _compact(snap: dict[str, Any]) -> dict[str, Any]:
    return {k: snap.get(k) for k in _COMPACT_KEYS if k in snap}


def register(mcp: MCPServer) -> None:
    for fn in (multi_timeframe_analysis, market_screener):
        mcp.add_tool(fn, annotations=READ_ONLY)
