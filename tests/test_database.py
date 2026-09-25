import os

import pandas as pd
import pytest
from mcp import Client
from sqlalchemy import func, select

from trading_mcp.data.base import MarketDataError, Timeframe
from trading_mcp.data.factory import CachedProvider
from trading_mcp.data.synthetic import SyntheticProvider
from trading_mcp.db.models import PriceBar, ToolCall
from trading_mcp.db.repository import BarStore, list_runs
from trading_mcp.db.session import normalize_url, reset_engine, session_scope
from trading_mcp.server import create_mcp

WINDOW = (pd.Timestamp("2024-01-01", tz="UTC").to_pydatetime(), pd.Timestamp("2024-03-01", tz="UTC").to_pydatetime())


def test_normalize_url():
    assert normalize_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("postgresql://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("sqlite:///x.db") == "sqlite:///x.db"


def _roundtrip():
    df = SyntheticProvider().get_history("AAPL", Timeframe.D1, *WINDOW)
    store = BarStore()
    store.save_bars("AAPL", "1d", df)
    store.save_bars("AAPL", "1d", df)  # re-saving the same range must not duplicate rows
    back = store.load_bars("AAPL", "1d", None, None)
    pd.testing.assert_frame_equal(back, df, check_freq=False, check_names=False)
    with session_scope() as s:
        assert s.scalar(select(func.count()).select_from(PriceBar)) == len(df)


def test_bar_store_roundtrip_sqlite():
    _roundtrip()


@pytest.mark.skipif(not os.environ.get("TEST_POSTGRES_URL"), reason="TEST_POSTGRES_URL not set")
def test_bar_store_roundtrip_postgres(monkeypatch):
    from trading_mcp.config import get_settings

    monkeypatch.setenv("DATABASE_URL", os.environ["TEST_POSTGRES_URL"])
    get_settings.cache_clear()
    reset_engine()
    with session_scope() as s:
        s.execute(PriceBar.__table__.delete())
    _roundtrip()


def test_provider_falls_back_to_database_cache():
    class Flaky(SyntheticProvider):
        fail = False

        def get_history(self, *a, **k):
            if self.fail:
                raise MarketDataError("upstream down")
            return super().get_history(*a, **k)

    inner = Flaky()
    p = CachedProvider(inner, ttl_seconds=0, store=BarStore())
    fresh = p.get_history("MSFT", Timeframe.D1, *WINDOW)
    inner.fail = True
    cached = p.get_history("MSFT", Timeframe.D1, *WINDOW)
    assert len(cached) == len(fresh)
    assert "database cache" in cached.attrs["notes"][0]
    with pytest.raises(MarketDataError):
        p.get_history("NEVER_SEEN", Timeframe.D1, *WINDOW)


async def test_runs_and_audit_are_persisted():
    async with Client(create_mcp()) as client:
        r = await client.call_tool("backtest_strategy", {"ticker": "NVDA"})
        run_id = r.structured_content["run_id"]
        assert isinstance(run_id, int)
        r = await client.call_tool("get_backtest_run", {"run_id": run_id})
        assert r.structured_content["strategy"] == "sma_crossover"
        r = await client.call_tool("list_saved_runs", {"kind": "backtest"})
        assert r.structured_content["runs"][0]["id"] == run_id
        await client.call_tool("train_model", {"ticker": "NVDA", "save": False})
        assert list_runs("ml")[0]["model_type"] == "logistic_regression"
        bad = await client.call_tool("get_backtest_run", {"run_id": 999999})
        assert bad.is_error
    with session_scope() as s:
        tools = set(s.scalars(select(ToolCall.tool)).all())
        failures = s.scalar(select(func.count()).select_from(ToolCall).where(ToolCall.ok.is_(False)))
    assert {"backtest_strategy", "get_backtest_run", "train_model"} <= tools
    assert failures == 1
