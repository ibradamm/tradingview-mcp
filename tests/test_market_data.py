from datetime import datetime

import pandas as pd
import pytest

from trading_mcp.data.base import OHLCV_COLUMNS, Timeframe, normalize_ohlcv, resample_ohlcv, resolve_window
from trading_mcp.data.factory import CachedProvider
from trading_mcp.data.symbols import to_yahoo_symbol
from trading_mcp.data.synthetic import SyntheticProvider


@pytest.mark.parametrize(
    ("raw", "exchange", "expected"),
    [
        ("nvda", None, "NVDA"),
        ("NASDAQ:NVDA", None, "NVDA"),
        ("BTCUSD", None, "BTC-USD"),
        ("BINANCE:BTCUSDT", None, "BTC-USD"),
        ("BTC", None, "BTC-USD"),
        ("EUR/USD", None, "EURUSD=X"),
        ("EURUSD", None, "EURUSD=X"),
        ("ES1!", None, "ES=F"),
        ("SPX", None, "^GSPC"),
        ("MC", "PA", "MC.PA"),
        ("^VIX", None, "^VIX"),
    ],
)
def test_symbol_translation(raw, exchange, expected):
    assert to_yahoo_symbol(raw, exchange) == expected


def test_synthetic_history_shape_and_determinism():
    p = SyntheticProvider()
    a = p.get_history("NVDA", Timeframe.H1, datetime(2024, 1, 1), datetime(2024, 2, 1))
    b = p.get_history("NVDA", Timeframe.H1, datetime(2024, 1, 1), datetime(2024, 2, 1))
    assert list(a.columns) == OHLCV_COLUMNS
    assert str(a.index.tz) == "UTC"
    assert a.index.is_monotonic_increasing
    assert (a["high"] >= a[["open", "close"]].max(axis=1)).all()
    assert (a["low"] <= a[["open", "close"]].min(axis=1)).all()
    pd.testing.assert_frame_equal(a, b)


def test_quote_fields():
    q = SyntheticProvider().get_quote("AAPL")
    assert q.price > 0 and q.previous_close and q.change is not None
    assert q.source.startswith("synthetic")


def test_resample_to_4h():
    df = SyntheticProvider().get_history("X", Timeframe.H1, datetime(2024, 1, 1), datetime(2024, 1, 3))
    out = resample_ohlcv(df, Timeframe.H4)
    assert len(out) == 12
    assert out["volume"].sum() == pytest.approx(df["volume"].sum())
    assert out["high"].iloc[0] == df["high"].iloc[:4].max()


def test_normalize_handles_naive_index_and_duplicates():
    idx = pd.to_datetime(["2024-01-02", "2024-01-01", "2024-01-01"])
    raw = pd.DataFrame({"Open": 1, "High": 2, "Low": 0.5, "Close": [1, 2, 3], "Volume": 10}, index=idx)
    out = normalize_ohlcv(raw)
    assert list(out.index.strftime("%Y-%m-%d")) == ["2024-01-01", "2024-01-02"]
    assert str(out.index.tz) == "UTC"


def test_resolve_window_rejects_inverted_dates():
    with pytest.raises(ValueError):
        resolve_window(Timeframe.D1, datetime(2024, 2, 1), datetime(2024, 1, 1))


def test_cached_provider_returns_copies():
    calls = []

    class Counting(SyntheticProvider):
        def get_history(self, *a, **k):
            calls.append(1)
            return super().get_history(*a, **k)

    p = CachedProvider(Counting(), ttl_seconds=60)
    a = p.get_history("X", Timeframe.D1, datetime(2023, 1, 1), datetime(2023, 6, 1))
    a.iloc[0, 0] = -1
    b = p.get_history("X", Timeframe.D1, datetime(2023, 1, 1), datetime(2023, 6, 1))
    assert len(calls) == 1
    assert b.iloc[0, 0] != -1
