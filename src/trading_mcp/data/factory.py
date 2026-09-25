"""Provider selection and an in-memory TTL cache wrapper."""

from __future__ import annotations

import threading
import time
from datetime import datetime
from functools import lru_cache

import pandas as pd

from trading_mcp.config import get_settings
from trading_mcp.data.base import MarketDataProvider, Quote, SymbolMatch, Timeframe


class CachedProvider(MarketDataProvider):
    """Decorates any provider with a short-lived in-memory cache for history requests.

    Quotes are never cached (they must stay fresh). Optionally persists history to the
    database through ``store`` (see :mod:`trading_mcp.db.repository`).
    """

    def __init__(self, inner: MarketDataProvider, ttl_seconds: int, store: object | None = None) -> None:
        self.inner = inner
        self.ttl = ttl_seconds
        self.store = store
        self.name = inner.name
        self._cache: dict[tuple[str, str, str, str], tuple[float, pd.DataFrame]] = {}
        self._lock = threading.Lock()

    def get_quote(self, ticker: str, exchange: str | None = None) -> Quote:
        return self.inner.get_quote(ticker, exchange)

    def search_symbol(self, query: str, limit: int = 10) -> list[SymbolMatch]:
        return self.inner.search_symbol(query, limit)

    def get_history(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        if self.ttl <= 0:
            return self.inner.get_history(ticker, timeframe, start, end)
        # Round the default "now" end to the TTL bucket so consecutive calls share an entry.
        key = (ticker.upper(), timeframe.value, str(start), str(end) if end else f"now/{int(time.time() // self.ttl)}")
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit[0] < self.ttl:
                return hit[1].copy()
        df = self.inner.get_history(ticker, timeframe, start, end)
        with self._lock:
            self._cache[key] = (now, df)
            if len(self._cache) > 512:  # crude bound on memory
                oldest = min(self._cache, key=lambda k: self._cache[k][0])
                del self._cache[oldest]
        if self.store is not None:
            self.store.save_bars(df.attrs.get("symbol", ticker.upper()), timeframe.value, df)  # type: ignore[attr-defined]
        out = df.copy()
        out.attrs = dict(df.attrs)
        return out


def build_provider(kind: str) -> MarketDataProvider:
    """Instantiate a raw provider by name."""
    if kind == "synthetic":
        from trading_mcp.data.synthetic import SyntheticProvider

        return SyntheticProvider()
    if kind == "yahoo":
        from trading_mcp.data.yahoo import YahooProvider

        return YahooProvider()
    raise ValueError(f"Unknown market data provider: {kind}")


@lru_cache
def get_provider() -> MarketDataProvider:
    """Process-wide provider configured by ``MARKET_DATA_PROVIDER``."""
    settings = get_settings()
    return CachedProvider(build_provider(settings.market_data_provider), settings.cache_ttl_seconds)
