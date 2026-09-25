import numpy as np
import pandas as pd
import pytest
from mcp import Client

from trading_mcp.analysis.snapshot import classify_trend, compute_snapshot
from trading_mcp.data.base import Timeframe
from trading_mcp.data.synthetic import SyntheticProvider
from trading_mcp.screener.screener import ScreenerFilters, passes, run_screen
from trading_mcp.server import create_mcp


def _trend_frame(slope: float, n: int = 120) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    close = pd.Series(100 + slope * np.arange(n), index=idx, dtype=float)
    return pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1e6})


def test_trend_classification():
    assert classify_trend(_trend_frame(1)["close"]) == "up"
    assert classify_trend(_trend_frame(-0.5)["close"]) == "down"
    assert classify_trend(_trend_frame(0.5, 10)["close"]) == "insufficient_data"


def test_snapshot_fields_and_breakout():
    df = _trend_frame(1)
    df.iloc[-1, df.columns.get_loc("close")] = 1000  # jump far above the prior 20-bar high
    df.iloc[-1, df.columns.get_loc("volume")] = 5e6
    snap = compute_snapshot(df, Timeframe.D1)
    assert snap["breakout_20"] == "up"
    assert snap["relative_volume"] == pytest.approx(5.0)
    assert snap["rsi_14"] > 70
    assert any("overbought" in s for s in snap["signals"])
    assert snap["vwap_type"].startswith("rolling")


def test_filters_logic():
    snap = {"price": 50, "change_percent": 4, "rsi_14": 60, "relative_volume": 2, "trend": "up", "breakout_20": "none",
            "volume": 1e6}
    ok, _ = passes(snap, ScreenerFilters(min_change_percent=3, min_rsi=50, max_rsi=70, min_relative_volume=1.5))
    assert ok
    ok, why = passes(snap, ScreenerFilters(max_rsi=55, trend="down", breakout="any"))
    assert not ok and len(why) == 3
    ok, why = passes(snap, ScreenerFilters(min_distance_sma200_percent=0))
    assert not ok and "unavailable" in why[0]


def test_run_screen_handles_bad_symbols():
    class Flaky(SyntheticProvider):
        def get_history(self, ticker, *a, **k):
            if ticker == "BAD":
                raise ValueError("boom")
            return super().get_history(ticker, *a, **k)

    res = run_screen(Flaky(), ["AAPL", "BAD", "MSFT"], Timeframe.D1, ScreenerFilters())
    assert {m["ticker"] for m in res["matches"]} == {"AAPL", "MSFT"}
    assert res["errors"][0]["ticker"] == "BAD"


async def test_mtf_and_screener_tools():
    async with Client(create_mcp()) as client:
        r = await client.call_tool("multi_timeframe_analysis", {"ticker": "NVDA", "timeframes": ["5m", "1h", "1d"]})
        assert not r.is_error, r.content
        data = r.structured_content
        assert set(data["timeframes"]) == {"5m", "1h", "1d"}
        assert data["timeframes"]["5m"]["vwap_type"].startswith("session")
        assert "rsi_14" in data["timeframes"]["1d"] and data["timeframes"]["1d"]["signals"]

        r = await client.call_tool("market_screener", {"universe": "us_mega_caps", "filters": {"min_rsi": 0}})
        assert not r.is_error, r.content
        assert r.structured_content["match_count"] == 40

        r = await client.call_tool("market_screener", {
            "tickers": ["AAPL", "MSFT", "NVDA"],
            "filters": {"min_rsi": 101},
        })
        assert r.structured_content["match_count"] == 0

        r = await client.call_tool("market_screener", {"tickers": ["AAPL"], "filters": {"min_market_cap": 1}})
        assert r.structured_content["matches"][0]["market_cap"] > 0

        bad = await client.call_tool("market_screener", {"universe": "nope"})
        assert bad.is_error
