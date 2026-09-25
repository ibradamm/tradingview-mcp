"""Yahoo Finance provider (via the open-source ``yfinance`` package).

Data source disclosure: ``yfinance`` is an unofficial client of Yahoo's public web endpoints.
It is not affiliated with or endorsed by Yahoo, data may be delayed or wrong, and the endpoints
can change without notice. Yahoo's terms restrict usage to personal, non-commercial purposes.
Swap in a licensed provider (see ``MarketDataProvider``) for anything beyond personal research.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import pandas as pd
import yfinance as yf

from trading_mcp.data.base import (
    MarketDataError,
    MarketDataProvider,
    Quote,
    SymbolMatch,
    Timeframe,
    normalize_ohlcv,
    resample_ohlcv,
    resolve_window,
)
from trading_mcp.data.symbols import to_yahoo_symbol

logger = logging.getLogger(__name__)

# Yahoo interval and how far back that interval is available.
_INTERVALS: dict[Timeframe, tuple[str, timedelta]] = {
    Timeframe.M1: ("1m", timedelta(days=7)),
    Timeframe.M5: ("5m", timedelta(days=59)),
    Timeframe.M15: ("15m", timedelta(days=59)),
    Timeframe.M30: ("30m", timedelta(days=59)),
    Timeframe.H1: ("1h", timedelta(days=729)),
    Timeframe.H4: ("1h", timedelta(days=729)),  # resampled locally, Yahoo has no 4h bars
    Timeframe.D1: ("1d", timedelta(days=365 * 100)),
    Timeframe.W1: ("1wk", timedelta(days=365 * 100)),
}


class YahooProvider(MarketDataProvider):
    """Market data from Yahoo Finance: stocks, ETFs, indices, forex, crypto, futures."""

    name = "yahoo_finance (yfinance, unofficial)"

    def get_quote(self, ticker: str, exchange: str | None = None) -> Quote:
        symbol = to_yahoo_symbol(ticker, exchange)
        tk = yf.Ticker(symbol)
        daily = normalize_ohlcv(tk.history(period="10d", interval="1d", auto_adjust=False))
        if daily.empty:
            raise MarketDataError(f"No data returned by Yahoo for '{symbol}'. Check the symbol.")
        intraday = normalize_ohlcv(tk.history(period="1d", interval="1m", prepost=True, auto_adjust=False))

        last_daily_ts = daily.index[-1]
        if not intraday.empty:
            price, timestamp = float(intraday["close"].iloc[-1]), intraday.index[-1].to_pydatetime()
        else:
            price, timestamp = float(daily["close"].iloc[-1]), last_daily_ts.to_pydatetime()

        # Previous close = close of the last *completed* session before the latest daily bar.
        previous_close = float(daily["close"].iloc[-2]) if len(daily) >= 2 else None
        change = price - previous_close if previous_close else None
        currency = None
        try:
            currency = tk.fast_info.get("currency")
        except Exception:  # noqa: BLE001 - metadata is optional
            logger.debug("currency lookup failed for %s", symbol)

        return Quote(
            ticker=symbol,
            price=price,
            previous_close=previous_close,
            change=change,
            change_percent=(change / previous_close * 100) if change is not None and previous_close else None,
            volume=float(daily["volume"].iloc[-1]),
            currency=currency,
            timestamp=timestamp,
            source=self.name,
            delayed=True,
        )

    def get_history(
        self,
        ticker: str,
        timeframe: Timeframe,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame:
        symbol = to_yahoo_symbol(ticker)
        start_dt, end_dt = resolve_window(timeframe, start, end)
        interval, max_lookback = _INTERVALS[timeframe]
        earliest = datetime.now(UTC) - max_lookback
        notes: list[str] = []
        if start_dt < earliest:
            notes.append(
                f"Yahoo only serves {interval} bars for the last {max_lookback.days} days; "
                f"start moved from {start_dt.date()} to {earliest.date()}."
            )
            start_dt = earliest
        if start_dt >= end_dt:
            raise MarketDataError(f"Requested window is outside Yahoo's availability for {interval} bars.")

        raw = yf.Ticker(symbol).history(start=start_dt, end=end_dt, interval=interval, auto_adjust=True)
        df = normalize_ohlcv(raw)
        if df.empty:
            raise MarketDataError(f"No {timeframe.value} data returned by Yahoo for '{symbol}'.")
        if timeframe is Timeframe.H4:
            df = resample_ohlcv(df, timeframe)
        df.attrs.update({"symbol": symbol, "source": self.name, "notes": notes, "adjusted": True})
        return df

    def search_symbol(self, query: str, limit: int = 10) -> list[SymbolMatch]:
        search = yf.Search(query, max_results=limit, news_count=0, lists_count=0, raise_errors=True)
        return [
            SymbolMatch(
                symbol=item.get("symbol", ""),
                name=item.get("longname") or item.get("shortname"),
                exchange=item.get("exchDisp") or item.get("exchange"),
                asset_type=item.get("quoteType"),
            )
            for item in search.quotes[:limit]
            if item.get("symbol")
        ]
