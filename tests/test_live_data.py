"""Live checks against the real Yahoo provider. Run with ``pytest -m live`` (needs internet)."""

from __future__ import annotations

import pytest

from trading_mcp.data.base import Timeframe
from trading_mcp.data.yahoo import YahooProvider

pytestmark = pytest.mark.live


@pytest.mark.parametrize("ticker", ["NVDA", "SPY", "BTCUSD", "EURUSD", "SPX", "ES1!"])
def test_live_quote(ticker):
    q = YahooProvider().get_quote(ticker)
    assert q.price > 0
    print(ticker, q.model_dump())


@pytest.mark.parametrize("tf", list(Timeframe))
def test_live_history_all_timeframes(tf):
    df = YahooProvider().get_history("NVDA", tf)
    assert len(df) > 10
    assert df["close"].gt(0).all()
    print(tf.value, len(df), df.index[0], df.index[-1])


def test_live_search():
    results = YahooProvider().search_symbol("nvidia")
    assert any(r.symbol == "NVDA" for r in results)
