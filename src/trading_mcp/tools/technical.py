"""Technical analysis MCP tools. Each returns the latest value(s) plus a tail of the series."""

from __future__ import annotations

from typing import Any

import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from trading_mcp.data.base import Timeframe
from trading_mcp.indicators import technical as ta
from trading_mcp.tools.common import clean, frame_to_records, load_ohlcv, safe_tool, source_info

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
DEFAULT_TAIL = 30


def _respond(df: pd.DataFrame, result: pd.Series | pd.DataFrame, params: dict[str, Any], tail: int) -> dict[str, Any]:
    frame = result.to_frame() if isinstance(result, pd.Series) else result
    frame = frame.copy()
    frame["close"] = df["close"]
    valid = frame.drop(columns="close").dropna(how="all")
    latest = frame.loc[valid.index[-1]].to_dict() if len(valid) else {}
    return {
        "meta": source_info(df),
        "params": params,
        "latest": clean({"timestamp": valid.index[-1] if len(valid) else None, **latest}),
        "series": frame_to_records(frame, max(1, min(int(tail), 1000))),
    }


def _load(ticker: str, timeframe: str, start: str | None, end: str | None) -> pd.DataFrame:
    return load_ohlcv(ticker, timeframe, start, end)


@safe_tool
def calculate_rsi(
    ticker: str, timeframe: str = "1d", period: int = 14, start: str | None = None, end: str | None = None,
    tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Relative Strength Index (Wilder). Conventional reading: >70 overbought, <30 oversold (descriptive only)."""
    df = _load(ticker, timeframe, start, end)
    return _respond(df, ta.rsi(df["close"], period), {"period": period}, tail)


@safe_tool
def calculate_macd(
    ticker: str, timeframe: str = "1d", fast: int = 12, slow: int = 26, signal: int = 9,
    start: str | None = None, end: str | None = None, tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """MACD line, signal line and histogram."""
    df = _load(ticker, timeframe, start, end)
    return _respond(df, ta.macd(df["close"], fast, slow, signal), {"fast": fast, "slow": slow, "signal": signal}, tail)


@safe_tool
def calculate_sma(
    ticker: str, timeframe: str = "1d", period: int = 20, start: str | None = None, end: str | None = None,
    tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Simple moving average of the close."""
    df = _load(ticker, timeframe, start, end)
    return _respond(df, ta.sma(df["close"], period), {"period": period}, tail)


@safe_tool
def calculate_ema(
    ticker: str, timeframe: str = "1d", period: int = 20, start: str | None = None, end: str | None = None,
    tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Exponential moving average of the close."""
    df = _load(ticker, timeframe, start, end)
    return _respond(df, ta.ema(df["close"], period), {"period": period}, tail)


@safe_tool
def calculate_bollinger_bands(
    ticker: str, timeframe: str = "1d", period: int = 20, num_std: float = 2.0,
    start: str | None = None, end: str | None = None, tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Bollinger Bands: middle (SMA), upper/lower bands, bandwidth and %B."""
    df = _load(ticker, timeframe, start, end)
    return _respond(df, ta.bollinger_bands(df["close"], period, num_std), {"period": period, "num_std": num_std}, tail)


@safe_tool
def calculate_atr(
    ticker: str, timeframe: str = "1d", period: int = 14, start: str | None = None, end: str | None = None,
    tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Average True Range (Wilder) - absolute volatility in price units; also returned as % of close."""
    df = _load(ticker, timeframe, start, end)
    a = ta.atr(df, period)
    return _respond(df, pd.DataFrame({"atr": a, "atr_percent": a / df["close"] * 100}), {"period": period}, tail)


@safe_tool
def calculate_adx(
    ticker: str, timeframe: str = "1d", period: int = 14, start: str | None = None, end: str | None = None,
    tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """ADX with +DI/-DI. ADX > 25 is conventionally read as a trending market, < 20 as ranging."""
    df = _load(ticker, timeframe, start, end)
    return _respond(df, ta.adx(df, period), {"period": period}, tail)


@safe_tool
def calculate_stochastic(
    ticker: str, timeframe: str = "1d", k_period: int = 14, d_period: int = 3, smooth_k: int = 3,
    start: str | None = None, end: str | None = None, tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Slow stochastic oscillator %K / %D in [0, 100]."""
    df = _load(ticker, timeframe, start, end)
    params = {"k_period": k_period, "d_period": d_period, "smooth_k": smooth_k}
    return _respond(df, ta.stochastic(df, k_period, d_period, smooth_k), params, tail)


@safe_tool
def calculate_vwap(
    ticker: str, timeframe: str = "5m", anchor: str = "session", start: str | None = None, end: str | None = None,
    tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """VWAP. anchor='session' resets each UTC day (use with intraday bars); 'none' = cumulative over the window.

    Also returns the distance of the close from VWAP in %.
    """
    df = _load(ticker, timeframe, start, end)
    v = ta.vwap(df, anchor)
    return _respond(df, pd.DataFrame({"vwap": v, "distance_percent": (df["close"] / v - 1) * 100}),
                    {"anchor": anchor}, tail)


@safe_tool
def calculate_volatility(
    ticker: str, timeframe: str = "1d", window: int = 20, start: str | None = None, end: str | None = None,
    tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Rolling annualised close-to-close volatility (std of log returns x sqrt(bars per year))."""
    df = _load(ticker, timeframe, start, end)
    ppy = Timeframe(df.attrs["timeframe"]).periods_per_year
    return _respond(df, ta.volatility(df["close"], window, ppy), {"window": window, "periods_per_year": ppy}, tail)


@safe_tool
def calculate_returns(
    ticker: str, timeframe: str = "1d", kind: str = "simple", periods: int = 1,
    start: str | None = None, end: str | None = None, tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Period returns (kind='simple' or 'log') plus cumulative return over the window."""
    df = _load(ticker, timeframe, start, end)
    r = ta.returns(df["close"], kind, periods)
    total = float(df["close"].iloc[-1] / df["close"].iloc[0] - 1)
    out = _respond(df, r, {"kind": kind, "periods": periods}, tail)
    out["summary"] = clean({"total_return": total, "mean": r.mean(), "std": r.std(), "observations": int(r.count())})
    return out


@safe_tool
def calculate_drawdown(
    ticker: str, timeframe: str = "1d", start: str | None = None, end: str | None = None, tail: int = DEFAULT_TAIL,
) -> dict[str, Any]:
    """Drawdown from running peak, plus the maximum drawdown and its dates."""
    df = _load(ticker, timeframe, start, end)
    dd = ta.drawdown(df["close"])
    trough = dd["drawdown"].idxmin()
    peak_date = df["close"].loc[:trough].idxmax()
    out = _respond(df, dd, {}, tail)
    out["max_drawdown"] = clean({"value": dd["drawdown"].min(), "peak_date": peak_date, "trough_date": trough,
                                 "current_drawdown": dd["drawdown"].iloc[-1]})
    return out


@safe_tool
def calculate_correlation(
    tickers: list[str], timeframe: str = "1d", method: str = "pearson",
    start: str | None = None, end: str | None = None,
) -> dict[str, Any]:
    """Correlation matrix of returns between 2-20 symbols (pearson, spearman or kendall)."""
    if not 2 <= len(tickers) <= 20:
        raise ValueError("Provide between 2 and 20 tickers.")
    closes = pd.DataFrame({t: load_ohlcv(t, timeframe, start, end)["close"] for t in tickers}).dropna()
    corr = ta.correlation_matrix(closes, method)
    return clean({
        "method": method, "timeframe": timeframe, "observations": len(closes) - 1,
        "matrix": corr.to_dict(),
        "note": "Correlation of returns, not prices. Correlations are unstable and rise in market stress.",
    })


TOOLS = (
    calculate_rsi, calculate_macd, calculate_sma, calculate_ema, calculate_bollinger_bands, calculate_atr,
    calculate_adx, calculate_stochastic, calculate_vwap, calculate_volatility, calculate_returns,
    calculate_drawdown, calculate_correlation,
)


def register(mcp: MCPServer) -> None:
    for fn in TOOLS:
        mcp.add_tool(fn, annotations=READ_ONLY)
