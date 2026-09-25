"""Provider-agnostic market data contract.

Every data source implements :class:`MarketDataProvider`. The rest of the code base only
depends on this interface, so swapping Yahoo for another vendor means writing one new class.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime, timedelta
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field

OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


class Timeframe(StrEnum):
    """Supported bar sizes."""

    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"

    @property
    def pandas_freq(self) -> str:
        return {
            "1m": "1min", "5m": "5min", "15m": "15min", "30m": "30min",
            "1h": "1h", "4h": "4h", "1d": "1D", "1w": "W-MON",
        }[self.value]

    @property
    def is_intraday(self) -> bool:
        return self not in (Timeframe.D1, Timeframe.W1)

    @property
    def periods_per_year(self) -> float:
        """Approximate bars per year (US equity session), used to annualise statistics."""
        return {
            "1m": 252 * 390, "5m": 252 * 78, "15m": 252 * 26, "30m": 252 * 13,
            "1h": 252 * 6.5, "4h": 252 * 1.625, "1d": 252, "1w": 52,
        }[self.value]

    @property
    def default_lookback(self) -> timedelta:
        """Default history window when the caller gives no start date."""
        return {
            "1m": timedelta(days=5), "5m": timedelta(days=30), "15m": timedelta(days=45),
            "30m": timedelta(days=55), "1h": timedelta(days=180), "4h": timedelta(days=360),
            "1d": timedelta(days=3 * 365), "1w": timedelta(days=10 * 365),
        }[self.value]


class Quote(BaseModel):
    """Latest available price snapshot."""

    ticker: str
    price: float
    previous_close: float | None = None
    change: float | None = None
    change_percent: float | None = None
    volume: float | None = None
    currency: str | None = None
    timestamp: datetime
    source: str
    delayed: bool = Field(default=True, description="Free sources are usually delayed.")


class SymbolMatch(BaseModel):
    """Result of a symbol search."""

    symbol: str
    name: str | None = None
    exchange: str | None = None
    asset_type: str | None = None


class MarketDataError(RuntimeError):
    """Raised when a provider cannot return the requested data."""


class MarketDataProvider(ABC):
    """Abstract data source. Implementations must return UTC-indexed OHLCV frames."""

    name: str = "abstract"

    @abstractmethod
    def get_quote(self, ticker: str, exchange: str | None = None) -> Quote:
        """Return the latest quote for ``ticker``."""

    @abstractmethod
    def get_history(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        """Return OHLCV bars indexed by UTC timestamp with columns :data:`OHLCV_COLUMNS`."""

    @abstractmethod
    def search_symbol(self, query: str, limit: int = 10) -> list[SymbolMatch]:
        """Search symbols by ticker or company name."""

    def get_market_cap(self, ticker: str) -> float | None:
        """Market capitalisation in the listing currency, or None when the source has no fundamentals."""
        return None


def resolve_window(
    timeframe: Timeframe, start: datetime | None, end: datetime | None
) -> tuple[datetime, datetime]:
    """Fill in missing start/end dates and force timezone-aware UTC datetimes."""
    end_dt = _as_utc(end) if end else datetime.now(UTC)
    start_dt = _as_utc(start) if start else end_dt - timeframe.default_lookback
    if start_dt >= end_dt:
        raise ValueError(f"start ({start_dt.isoformat()}) must be before end ({end_dt.isoformat()})")
    return start_dt, end_dt


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case columns, keep OHLCV only, UTC index, sorted, no duplicate/NaN-close rows."""
    if df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS, index=pd.DatetimeIndex([], tz="UTC", name="timestamp"))
    out = df.rename(columns=str.lower)
    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise MarketDataError(f"Provider data is missing columns: {missing}")
    out = out[OHLCV_COLUMNS].astype(float)
    index = pd.DatetimeIndex(out.index)
    out.index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    out.index.name = "timestamp"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["close"])


def resample_ohlcv(df: pd.DataFrame, timeframe: Timeframe) -> pd.DataFrame:
    """Aggregate bars into a coarser timeframe (e.g. 1h -> 4h)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(timeframe.pandas_freq, label="left", closed="left").agg(agg).dropna(subset=["close"])
