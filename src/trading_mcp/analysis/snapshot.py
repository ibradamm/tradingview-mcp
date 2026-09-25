"""Point-in-time technical snapshot of one symbol on one timeframe.

Used by the multi-timeframe analysis and the screener. Every field is descriptive: it reports
observed conditions, never a buy/sell recommendation.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from trading_mcp.data.base import Timeframe
from trading_mcp.indicators import technical as ta

BREAKOUT_LOOKBACK = 20


def _last(series: pd.Series) -> float | None:
    if series.empty:
        return None
    value = series.iloc[-1]
    return None if value is None or (isinstance(value, float) and math.isnan(value)) else float(value)


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return (a / b - 1) * 100


def classify_trend(close: pd.Series) -> str:
    """'up', 'down' or 'sideways' from EMA20/EMA50 alignment and the EMA50 slope over 5 bars."""
    e20, e50 = ta.ema(close, 20), ta.ema(close, 50)
    if e50.dropna().shape[0] < 6:
        return "insufficient_data"
    c, a, b = close.iloc[-1], e20.iloc[-1], e50.iloc[-1]
    slope = e50.iloc[-1] - e50.iloc[-6]
    if c > b and a > b and slope > 0:
        return "up"
    if c < b and a < b and slope < 0:
        return "down"
    return "sideways"


def compute_snapshot(df: pd.DataFrame, timeframe: Timeframe) -> dict[str, Any]:
    """Compute the latest indicator values and descriptive signals for ``df``."""
    if len(df) < 3:
        raise ValueError("Need at least 3 bars for a snapshot.")
    close, volume = df["close"], df["volume"]
    price = float(close.iloc[-1])
    macd = ta.macd(close)
    ema20, ema50 = _last(ta.ema(close, 20)), _last(ta.ema(close, 50))
    sma50, sma200 = _last(ta.sma(close, 50)), _last(ta.sma(close, 200))
    vwap_series = ta.vwap(df, "session") if timeframe.is_intraday else _rolling_vwap(df, 20)
    vwap = _last(vwap_series)
    atr = _last(ta.atr(df, 14))
    rsi = _last(ta.rsi(close, 14))
    adx = ta.adx(df, 14)

    prior = df.iloc[-(BREAKOUT_LOOKBACK + 1) : -1]
    breakout = "none"
    if len(prior) == BREAKOUT_LOOKBACK:
        if price > prior["high"].max():
            breakout = "up"
        elif price < prior["low"].min():
            breakout = "down"

    snap: dict[str, Any] = {
        "timestamp": df.index[-1],
        "price": price,
        "change_percent": _pct(price, float(close.iloc[-2])),
        "volume": float(volume.iloc[-1]),
        "avg_volume_20": _last(volume.shift(1).rolling(20, min_periods=20).mean()),
        "relative_volume": _last(ta.relative_volume(volume, 20)),
        "rsi_14": rsi,
        "macd": _last(macd["macd"]),
        "macd_signal": _last(macd["signal"]),
        "macd_histogram": _last(macd["histogram"]),
        "ema_20": ema20,
        "ema_50": ema50,
        "sma_50": sma50,
        "sma_200": sma200,
        "distance_ema20_percent": _pct(price, ema20),
        "distance_sma50_percent": _pct(price, sma50),
        "distance_sma200_percent": _pct(price, sma200),
        "vwap": vwap,
        "vwap_type": "session (UTC day)" if timeframe.is_intraday else "rolling 20 bars",
        "distance_vwap_percent": _pct(price, vwap),
        "atr_14": atr,
        "atr_percent": (atr / price * 100) if atr else None,
        "volatility_20_annualized": _last(ta.volatility(close, 20, timeframe.periods_per_year)),
        "adx_14": _last(adx["adx"]),
        "trend": classify_trend(close),
        "breakout_20": breakout,
    }
    snap["signals"] = describe_signals(snap, macd)
    return snap


def _rolling_vwap(df: pd.DataFrame, window: int) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv = (typical * df["volume"]).rolling(window, min_periods=window).sum()
    return pv / df["volume"].rolling(window, min_periods=window).sum()


def describe_signals(s: dict[str, Any], macd: pd.DataFrame) -> list[str]:
    """Human-readable observations (not recommendations)."""
    out: list[str] = []
    rsi = s["rsi_14"]
    if rsi is not None:
        if rsi >= 70:
            out.append(f"RSI {rsi:.1f} is in the conventional overbought zone (>=70)")
        elif rsi <= 30:
            out.append(f"RSI {rsi:.1f} is in the conventional oversold zone (<=30)")
        else:
            out.append(f"RSI {rsi:.1f} is neutral (30-70)")
    hist = macd["histogram"].dropna()
    if len(hist) >= 2:
        if hist.iloc[-2] <= 0 < hist.iloc[-1]:
            out.append("MACD crossed above its signal line on the last bar")
        elif hist.iloc[-2] >= 0 > hist.iloc[-1]:
            out.append("MACD crossed below its signal line on the last bar")
        else:
            out.append(f"MACD is {'above' if hist.iloc[-1] > 0 else 'below'} its signal line")
    if s["ema_20"] and s["ema_50"]:
        out.append(f"EMA20 is {'above' if s['ema_20'] > s['ema_50'] else 'below'} EMA50")
    if s["sma_200"]:
        out.append(f"Price is {'above' if s['price'] > s['sma_200'] else 'below'} the 200-period SMA")
    if s["distance_vwap_percent"] is not None:
        out.append(f"Price is {s['distance_vwap_percent']:+.2f}% from VWAP ({s['vwap_type']})")
    if s["relative_volume"] is not None and s["relative_volume"] >= 1.5:
        out.append(f"Volume is {s['relative_volume']:.1f}x its 20-bar average")
    if s["adx_14"] is not None:
        adx = s["adx_14"]
        strength = "trending (ADX>25)" if adx > 25 else "weak/range-bound (ADX<20)" if adx < 20 else "moderate"
        out.append(f"ADX {s['adx_14']:.1f}: {strength}")
    if s["breakout_20"] != "none":
        out.append(f"Close broke the 20-bar {'high' if s['breakout_20'] == 'up' else 'low'}")
    return out
