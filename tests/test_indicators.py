import numpy as np
import pandas as pd
import pytest
from mcp import Client

from trading_mcp.indicators import technical as ta
from trading_mcp.server import create_mcp

# StockCharts' RSI(14) example data. Their sheet shows 70.53 / 66.32 because it rounds the average
# gain/loss to 2 decimals; the exact Wilder values (computed by hand: avg gain 0.238571, avg loss 0.1)
# are 70.464 and 66.250.
RSI_CLOSES = [44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28,
              46.28, 46.00, 46.03]


def _series(values):
    return pd.Series(values, index=pd.date_range("2024-01-01", periods=len(values), freq="D", tz="UTC"), dtype=float)


def test_rsi_matches_stockcharts_reference():
    r = ta.rsi(_series(RSI_CLOSES), 14)
    assert r.iloc[:14].isna().all()
    assert r.iloc[14] == pytest.approx(70.4641, abs=1e-3)
    assert r.iloc[15] == pytest.approx(66.2496, abs=1e-3)
    assert r.iloc[14] == pytest.approx(70.53, abs=0.1)  # still within rounding of the published figure


def test_rsi_extremes():
    assert ta.rsi(_series(range(1, 40)), 14).iloc[-1] == 100
    assert ta.rsi(_series(range(40, 1, -1)), 14).iloc[-1] == pytest.approx(0)


def test_sma_ema():
    s = _series([1, 2, 3, 4, 5])
    assert ta.sma(s, 3).tolist()[2:] == [2, 3, 4]
    e = ta.ema(s, 3)
    assert np.isnan(e.iloc[1]) and e.iloc[-1] == pytest.approx(4.0625)  # alpha=0.5 recursion from 1


def test_macd_consistency(ohlcv):
    m = ta.macd(ohlcv["close"])
    expected = ta.ema(ohlcv["close"], 12) - ta.ema(ohlcv["close"], 26)
    pd.testing.assert_series_equal(m["macd"], expected, check_names=False)
    assert np.allclose((m["macd"] - m["signal"]).dropna(), m["histogram"].dropna())
    with pytest.raises(ValueError):
        ta.macd(ohlcv["close"], fast=26, slow=12)


def test_bollinger_contains_price_mostly(ohlcv):
    bb = ta.bollinger_bands(ohlcv["close"], 20, 2).dropna()
    close = ohlcv["close"].loc[bb.index]
    inside = ((close <= bb["upper"]) & (close >= bb["lower"])).mean()
    assert inside > 0.85
    assert (bb["upper"] > bb["lower"]).all()


def test_atr_constant_range():
    idx = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
    df = pd.DataFrame({"open": 10.0, "high": 11.0, "low": 9.0, "close": 10.0, "volume": 1.0}, index=idx)
    assert ta.atr(df, 14).iloc[-1] == pytest.approx(2.0)


def test_adx_trend_detection():
    idx = pd.date_range("2024-01-01", periods=100, freq="D", tz="UTC")
    close = pd.Series(np.linspace(100, 200, 100), index=idx)
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close, "volume": 1.0})
    out = ta.adx(df, 14).iloc[-1]
    assert out["adx"] > 50 and out["plus_di"] > out["minus_di"]


def test_stochastic_bounds(ohlcv):
    st = ta.stochastic(ohlcv).dropna()
    assert st["k"].between(0, 100).all() and st["d"].between(0, 100).all()


def test_vwap_session_reset():
    idx = pd.date_range("2024-01-01 14:00", periods=4, freq="12h", tz="UTC")
    df = pd.DataFrame({"high": [10, 20, 30, 40], "low": [10, 20, 30, 40], "close": [10, 20, 30, 40],
                       "volume": [1, 1, 1, 3], "open": 0}, index=idx)
    v = ta.vwap(df, "session")
    # bars: day1 14:00 (10), day2 02:00 (20), day2 14:00 (30), day3 02:00 (40)
    assert v.tolist() == [10, 20, 25, 40]
    assert ta.vwap(df, "none").iloc[-1] == pytest.approx((10 + 20 + 30 + 120) / 6)


def test_returns_volatility_drawdown():
    s = _series([100, 110, 99, 120])
    assert ta.returns(s).iloc[1] == pytest.approx(0.10)
    assert ta.returns(s, "log").iloc[1] == pytest.approx(np.log(1.1))
    assert ta.max_drawdown(s) == pytest.approx(-0.10)
    vol = ta.volatility(_series(100 * np.exp(np.cumsum(np.full(50, 0.01)))), 10)
    assert vol.iloc[-1] == pytest.approx(0, abs=1e-9)  # constant returns -> zero volatility


def test_relative_volume_uses_only_past():
    v = _series([10] * 20 + [30])
    assert ta.relative_volume(v, 20).iloc[-1] == pytest.approx(3.0)


def test_correlation_matrix():
    a = _series(100 * np.exp(np.cumsum(np.random.default_rng(0).normal(0, 0.01, 200))))
    corr = ta.correlation_matrix(pd.DataFrame({"a": a, "b": a * 2, "c": 1 / a}))
    assert corr.loc["a", "b"] == pytest.approx(1.0)
    assert corr.loc["a", "c"] < -0.9


def test_invalid_period():
    with pytest.raises(ValueError):
        ta.sma(_series([1, 2, 3]), 0)


TOOL_CALLS = [
    ("calculate_rsi", {}), ("calculate_macd", {}), ("calculate_sma", {"period": 50}), ("calculate_ema", {}),
    ("calculate_bollinger_bands", {}), ("calculate_atr", {}), ("calculate_adx", {}), ("calculate_stochastic", {}),
    ("calculate_vwap", {"timeframe": "15m"}), ("calculate_volatility", {}), ("calculate_returns", {"kind": "log"}),
    ("calculate_drawdown", {}),
]


async def test_indicator_tools_via_mcp():
    async with Client(create_mcp()) as client:
        for name, extra in TOOL_CALLS:
            res = await client.call_tool(name, {"ticker": "NVDA", "tail": 5, **extra})
            assert not res.is_error, (name, res.content)
            data = res.structured_content
            assert data["latest"]["timestamp"] is not None, name
            assert len(data["series"]) == 5, name
        corr = await client.call_tool("calculate_correlation", {"tickers": ["NVDA", "AAPL", "SPY"]})
        assert not corr.is_error and corr.structured_content["matrix"]["NVDA"]["NVDA"] == pytest.approx(1)
        bad = await client.call_tool("calculate_rsi", {"ticker": "NVDA", "period": 0})
        assert bad.is_error
