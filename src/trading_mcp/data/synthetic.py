"""Deterministic synthetic provider for offline development and automated tests.

Prices follow a geometric Brownian motion seeded from the ticker, so the same request always
returns the same bars. The data is FAKE and must never be used for real analysis.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

import numpy as np
import pandas as pd

from trading_mcp.data.base import (
    MarketDataProvider,
    Quote,
    SymbolMatch,
    Timeframe,
    resolve_window,
)

_UNIVERSE = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation", "NVDA": "NVIDIA Corporation",
    "AMZN": "Amazon.com, Inc.", "TSLA": "Tesla, Inc.", "META": "Meta Platforms, Inc.",
    "SPY": "SPDR S&P 500 ETF", "QQQ": "Invesco QQQ Trust", "BTC-USD": "Bitcoin USD",
    "ETH-USD": "Ethereum USD", "EURUSD=X": "EUR/USD",
}
_MAX_BARS = 50_000


def _seed(*parts: str) -> int:
    return int.from_bytes(hashlib.sha256("|".join(parts).encode()).digest()[:8], "little")


class SyntheticProvider(MarketDataProvider):
    """Generates reproducible fake OHLCV bars."""

    name = "synthetic (fake data, testing only)"

    def get_quote(self, ticker: str, exchange: str | None = None) -> Quote:
        df = self.get_history(ticker, Timeframe.D1)
        last, prev = df.iloc[-1], df.iloc[-2]
        change = float(last["close"] - prev["close"])
        return Quote(
            ticker=ticker.upper(),
            price=float(last["close"]),
            previous_close=float(prev["close"]),
            change=change,
            change_percent=change / float(prev["close"]) * 100,
            volume=float(last["volume"]),
            currency="USD",
            timestamp=df.index[-1].to_pydatetime(),
            source=self.name,
        )

    def get_history(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        start_dt, end_dt = resolve_window(timeframe, start, end)
        # Grid aligned on whole days so repeated calls with the same window return identical bars.
        grid_start = pd.Timestamp(start_dt).floor("D")
        index = pd.date_range(grid_start, pd.Timestamp(end_dt), freq=timeframe.pandas_freq, inclusive="left")
        index = index[index >= grid_start][-_MAX_BARS:]
        n = len(index)
        if n < 2:
            raise ValueError("Window too short for the requested timeframe.")

        rng = np.random.default_rng(_seed(ticker.upper(), timeframe.value, str(index[0])))
        drift, vol = 0.08 / timeframe.periods_per_year, 0.35 / np.sqrt(timeframe.periods_per_year)
        shocks = rng.standard_normal(n) * vol + drift - 0.5 * vol**2
        base = 50 + (_seed(ticker.upper()) % 400)
        close = np.exp(np.log(base) + np.cumsum(shocks))
        open_ = np.concatenate([[base], close[:-1]])
        wick = np.abs(rng.standard_normal(n)) * vol * close
        high = np.maximum(open_, close) + wick
        low = np.minimum(open_, close) - wick
        volume = np.round(np.exp(rng.normal(13, 0.4, n)))

        df = pd.DataFrame(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
            index=pd.DatetimeIndex(index, name="timestamp"),
        )
        df.attrs.update({"symbol": ticker.upper(), "source": self.name, "notes": ["FAKE DATA"], "adjusted": False})
        return df

    def search_symbol(self, query: str, limit: int = 10) -> list[SymbolMatch]:
        q = query.lower()
        return [
            SymbolMatch(symbol=s, name=n, exchange="SYNTH", asset_type="EQUITY")
            for s, n in _UNIVERSE.items()
            if q in s.lower() or q in n.lower()
        ][:limit]
