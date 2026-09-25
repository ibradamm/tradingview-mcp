"""Provider selection, in-memory TTL cache and persistent fallback cache."""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime
from functools import lru_cache
from typing import Protocol

import pandas as pd

from trading_mcp.config import get_settings
from trading_mcp.data.base import MarketDataProvider, Quote, SymbolMatch, Timeframe

logger = logging.getLogger(__name__)


class BarStoreProtocol(Protocol):
    def save_bars(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None: ...

    def load_bars(self, symbol: str, timeframe: str, start: datetime | None, end: datetime | None) -> pd.DataFrame: ...


class CachedProvider(MarketDataProvider):
    """Decorates any provider with a short-lived in-memory cache for history requests.

    Quotes are never cached (they must stay fresh). When a ``store`` is given, fetched bars are
    persisted and served back (flagged as cached) if the upstream provider fails.
    """

    def __init__(self, inner: MarketDataProvider, ttl_seconds: int, store: BarStoreProtocol | None = None) -> None:
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

    def get_market_cap(self, ticker: str) -> float | None:
        return self.inner.get_market_cap(ticker)

    def get_history(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        key_ticker = ticker.strip().upper()
        bucket = f"now/{int(time.time() // self.ttl)}" if self.ttl > 0 else "nocache"
        key = (key_ticker, timeframe.value, str(start), str(end) if end else bucket)
        now = time.monotonic()
        if self.ttl > 0:
            with self._lock:
                hit = self._cache.get(key)
                if hit and now - hit[0] < self.ttl:
                    return _copy(hit[1])
        try:
            df = self.inner.get_history(ticker, timeframe, start, end)
        except Exception as exc:
            fallback = self._from_store(key_ticker, timeframe, start, end)
            if fallback is None:
                raise
            logger.warning("provider failed for %s %s (%s); serving database cache", ticker, timeframe.value, exc)
            fallback.attrs["notes"] = [f"Live provider failed ({exc}); served from the database cache - may be stale."]
            return fallback
        if self.ttl > 0:
            with self._lock:
                self._cache[key] = (now, df)
                if len(self._cache) > 512:  # crude bound on memory
                    oldest = min(self._cache, key=lambda k: self._cache[k][0])
                    del self._cache[oldest]
        if self.store is not None:
            self.store.save_bars(key_ticker, timeframe.value, df)
        return _copy(df)

    def _from_store(
        self, ticker: str, timeframe: Timeframe, start: datetime | None, end: datetime | None
    ) -> pd.DataFrame | None:
        if self.store is None:
            return None
        try:
            df = self.store.load_bars(ticker, timeframe.value, start, end)
        except Exception:  # noqa: BLE001
            logger.exception("database cache lookup failed")
            return None
        return df if not df.empty else None


def _copy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.attrs = {k: (list(v) if isinstance(v, list) else v) for k, v in df.attrs.items()}
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
    """Process-wide provider configured by ``MARKET_DATA_PROVIDER`` (+ caches)."""
    settings = get_settings()
    store = None
    if settings.persist_bars:
        from trading_mcp.db.repository import BarStore

        store = BarStore()
    return CachedProvider(build_provider(settings.market_data_provider), settings.cache_ttl_seconds, store)
