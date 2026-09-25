"""Multi-criteria market screener built on :func:`compute_snapshot`."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

from pydantic import BaseModel, Field

from trading_mcp.analysis.snapshot import compute_snapshot
from trading_mcp.data.base import MarketDataProvider, Timeframe

logger = logging.getLogger(__name__)

UNIVERSES: dict[str, list[str]] = {
    "us_mega_caps": [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "BRK-B", "JPM", "LLY", "V", "UNH",
        "XOM", "MA", "JNJ", "PG", "HD", "COST", "ABBV", "WMT", "NFLX", "CRM", "BAC", "ORCL", "AMD", "KO",
        "PEP", "CVX", "MRK", "ADBE", "TMO", "CSCO", "ACN", "MCD", "INTC", "QCOM", "DIS", "IBM", "PLTR",
    ],
    "us_etfs": ["SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "SMH",
                "ARKK", "TLT", "GLD", "SLV", "USO", "HYG", "EEM"],
    "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "BNB-USD", "ADA-USD", "DOGE-USD", "AVAX-USD",
               "DOT-USD", "LINK-USD", "LTC-USD", "TRX-USD"],
    "forex_majors": ["EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X", "USDCAD=X", "NZDUSD=X",
                     "EURGBP=X", "EURJPY=X"],
    "cac40": ["AI.PA", "AIR.PA", "BNP.PA", "CAP.PA", "CS.PA", "DG.PA", "EL.PA", "KER.PA", "MC.PA", "OR.PA",
              "ORA.PA", "RI.PA", "SAF.PA", "SAN.PA", "SGO.PA", "SU.PA", "TTE.PA", "RMS.PA", "BN.PA", "ACA.PA"],
}
MAX_UNIVERSE = 100


class ScreenerFilters(BaseModel):
    """All filters are optional and combined with AND."""

    min_price: float | None = None
    max_price: float | None = None
    min_volume: float | None = Field(None, description="Minimum volume of the last bar")
    min_change_percent: float | None = Field(None, description="Last bar change vs previous close, in %")
    max_change_percent: float | None = None
    min_rsi: float | None = None
    max_rsi: float | None = None
    min_volatility: float | None = Field(None, description="Annualised 20-bar volatility, as a fraction (0.3 = 30%)")
    max_volatility: float | None = None
    min_market_cap: float | None = Field(None, description="In the listing currency; requires a fundamentals lookup")
    max_market_cap: float | None = None
    trend: Literal["up", "down", "sideways"] | None = None
    min_relative_volume: float | None = Field(None, description="Last bar volume / previous 20-bar average")
    breakout: Literal["up", "down", "any"] | None = Field(None, description="Close beyond the prior 20-bar high/low")
    min_distance_vwap_percent: float | None = None
    max_distance_vwap_percent: float | None = None
    min_distance_sma50_percent: float | None = None
    max_distance_sma50_percent: float | None = None
    min_distance_sma200_percent: float | None = None
    max_distance_sma200_percent: float | None = None
    min_distance_ema20_percent: float | None = None
    max_distance_ema20_percent: float | None = None

    def needs_market_cap(self) -> bool:
        return self.min_market_cap is not None or self.max_market_cap is not None


_RANGE_FIELDS = {
    "price": "price", "change_percent": "change_percent", "rsi": "rsi_14", "volatility": "volatility_20_annualized",
    "market_cap": "market_cap", "relative_volume": "relative_volume", "distance_vwap_percent": "distance_vwap_percent",
    "distance_sma50_percent": "distance_sma50_percent", "distance_sma200_percent": "distance_sma200_percent",
    "distance_ema20_percent": "distance_ema20_percent", "volume": "volume",
}


def passes(snap: dict[str, Any], f: ScreenerFilters) -> tuple[bool, list[str]]:
    """Return (matches, reasons_for_rejection)."""
    failed: list[str] = []
    for name, key in _RANGE_FIELDS.items():
        lo, hi = getattr(f, f"min_{name}", None), getattr(f, f"max_{name}", None)
        if lo is None and hi is None:
            continue
        value = snap.get(key)
        if value is None:
            failed.append(f"{name}: unavailable")
        elif lo is not None and value < lo:
            failed.append(f"{name} {value:.4g} < {lo}")
        elif hi is not None and value > hi:
            failed.append(f"{name} {value:.4g} > {hi}")
    if f.trend and snap.get("trend") != f.trend:
        failed.append(f"trend is {snap.get('trend')}")
    if f.breakout:
        b = snap.get("breakout_20")
        if (f.breakout == "any" and b == "none") or (f.breakout != "any" and b != f.breakout):
            failed.append(f"breakout is {b}")
    return not failed, failed


def run_screen(
    provider: MarketDataProvider,
    tickers: list[str],
    timeframe: Timeframe,
    filters: ScreenerFilters,
    max_workers: int = 8,
) -> dict[str, Any]:
    """Fetch data for every ticker in parallel, snapshot it and apply filters."""
    tickers = list(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not tickers:
        raise ValueError("Empty universe.")
    if len(tickers) > MAX_UNIVERSE:
        raise ValueError(f"Universe too large ({len(tickers)} > {MAX_UNIVERSE}).")

    def scan(ticker: str) -> dict[str, Any]:
        try:
            df = provider.get_history(ticker, timeframe)
            snap = compute_snapshot(df, timeframe)
            snap["ticker"] = ticker
            if filters.needs_market_cap():
                snap["market_cap"] = provider.get_market_cap(ticker)
            return snap
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not break the scan
            logger.info("screener skipped %s: %s", ticker, exc)
            return {"ticker": ticker, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        snaps = list(pool.map(scan, tickers))

    matches, rejected, errors = [], [], []
    for snap in snaps:
        if "error" in snap:
            errors.append({"ticker": snap["ticker"], "error": snap["error"]})
            continue
        ok, why = passes(snap, filters)
        if ok:
            matches.append(snap)
        else:
            rejected.append({"ticker": snap["ticker"], "reasons": why})
    return {"matches": matches, "rejected": rejected, "errors": errors}
