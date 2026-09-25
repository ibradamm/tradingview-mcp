"""Shared fixtures. Tests run fully offline against the deterministic synthetic provider."""

from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ["MARKET_DATA_PROVIDER"] = "synthetic"
os.environ["MCP_AUTH_TOKEN"] = "test-token-0123456789abcdef"
os.environ["CACHE_TTL_SECONDS"] = "0"


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Point the DB and model store at a temp dir and reset cached singletons."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "models"))
    from trading_mcp.config import get_settings
    from trading_mcp.data.factory import get_provider

    get_settings.cache_clear()
    get_provider.cache_clear()
    try:
        from trading_mcp.db.session import reset_engine

        reset_engine()
    except ImportError:
        pass
    yield
    get_settings.cache_clear()
    get_provider.cache_clear()


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    """~3 years of synthetic daily bars."""
    from trading_mcp.data.base import Timeframe
    from trading_mcp.data.synthetic import SyntheticProvider

    return SyntheticProvider().get_history(
        "TEST", Timeframe.D1, pd.Timestamp("2021-01-01"), pd.Timestamp("2024-01-01")
    )
