"""Technical indicators implemented with pandas (no look-ahead: each value uses bars <= t only).

Conventions: input frames have lower-case ``open, high, low, close, volume`` columns and a
UTC ``DatetimeIndex``. RSI, ATR and ADX use Wilder's smoothing seeded with a simple average,
the same definition as TradingView's ``ta.rma``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check_period(period: int, name: str = "period") -> None:
    if int(period) < 1:
        raise ValueError(f"{name} must be >= 1 (got {period})")


def _wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing (TradingView ``ta.rma``): seeded with an SMA, then recursive."""
    out = pd.Series(np.nan, index=series.index, dtype=float)
    valid = series.dropna()
    if len(valid) < period:
        return out
    seeded = valid.iloc[period - 1 :].astype(float).copy()
    seeded.iloc[0] = valid.iloc[:period].mean()
    out.loc[seeded.index] = seeded.ewm(alpha=1 / period, adjust=False).mean()
    return out


def sma(close: pd.Series, period: int = 20) -> pd.Series:
    _check_period(period)
    return close.rolling(period, min_periods=period).mean().rename(f"sma_{period}")


def ema(close: pd.Series, period: int = 20) -> pd.Series:
    _check_period(period)
    return close.ewm(span=period, adjust=False, min_periods=period).mean().rename(f"ema_{period}")


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI in [0, 100]."""
    _check_period(period)
    delta = close.diff()
    gain = _wilder(delta.clip(lower=0), period)
    loss = _wilder(-delta.clip(upper=0), period)
    rs = gain / loss
    out = 100 - 100 / (1 + rs)
    out = out.where(loss != 0, 100.0).where(~(gain.eq(0) & loss.eq(0)), 50.0)
    return out.where(gain.notna()).rename(f"rsi_{period}")


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    for p, n in ((fast, "fast"), (slow, "slow"), (signal, "signal")):
        _check_period(p, n)
    if fast >= slow:
        raise ValueError("fast period must be smaller than slow period")
    line = ema(close, fast) - ema(close, slow)
    sig = line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": line, "signal": sig, "histogram": line - sig})


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    _check_period(period)
    mid = sma(close, period)
    std = close.rolling(period, min_periods=period).std(ddof=0)
    upper, lower = mid + num_std * std, mid - num_std * std
    return pd.DataFrame(
        {
            "middle": mid,
            "upper": upper,
            "lower": lower,
            "bandwidth": (upper - lower) / mid,
            "percent_b": (close - lower) / (upper - lower),
        }
    )


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    )
    return ranges.max(axis=1, skipna=False).fillna(df["high"] - df["low"]).rename("true_range")


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    _check_period(period)
    return _wilder(true_range(df), period).rename(f"atr_{period}")


def adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI."""
    _check_period(period)
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    atr_ = _wilder(true_range(df), period)
    plus_di = 100 * _wilder(plus_dm, period) / atr_
    minus_di = 100 * _wilder(minus_dm, period) / atr_
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return pd.DataFrame({"adx": _wilder(dx, period), "plus_di": plus_di, "minus_di": minus_di})


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, smooth_k: int = 3) -> pd.DataFrame:
    """Slow stochastic oscillator (%K smoothed by ``smooth_k``, %D = SMA of %K)."""
    for p, n in ((k_period, "k_period"), (d_period, "d_period"), (smooth_k, "smooth_k")):
        _check_period(p, n)
    low_min = df["low"].rolling(k_period, min_periods=k_period).min()
    high_max = df["high"].rolling(k_period, min_periods=k_period).max()
    raw_k = 100 * (df["close"] - low_min) / (high_max - low_min).replace(0, np.nan)
    k = raw_k.rolling(smooth_k, min_periods=smooth_k).mean()
    return pd.DataFrame({"k": k, "d": k.rolling(d_period, min_periods=d_period).mean()})


def vwap(df: pd.DataFrame, anchor: str = "session") -> pd.Series:
    """Volume-weighted average price.

    anchor="session" resets every UTC calendar day (meaningful for intraday bars);
    anchor="none" is cumulative over the whole window (e.g. for daily bars).
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    pv, vol = typical * df["volume"], df["volume"]
    if anchor == "session":
        groups = df.index.floor("D")
        cum_pv, cum_vol = pv.groupby(groups).cumsum(), vol.groupby(groups).cumsum()
    elif anchor == "none":
        cum_pv, cum_vol = pv.cumsum(), vol.cumsum()
    else:
        raise ValueError("anchor must be 'session' or 'none'")
    return (cum_pv / cum_vol.replace(0, np.nan)).rename("vwap")


def returns(close: pd.Series, kind: str = "simple", periods: int = 1) -> pd.Series:
    _check_period(periods, "periods")
    if kind == "simple":
        return close.pct_change(periods).rename("return")
    if kind == "log":
        return np.log(close / close.shift(periods)).rename("log_return")
    raise ValueError("kind must be 'simple' or 'log'")


def volatility(close: pd.Series, window: int = 20, periods_per_year: float = 252.0) -> pd.Series:
    """Rolling annualised volatility of log returns (close-to-close)."""
    _check_period(window, "window")
    return (returns(close, "log").rolling(window, min_periods=window).std() * np.sqrt(periods_per_year)).rename(
        f"volatility_{window}"
    )


def drawdown(close_or_equity: pd.Series) -> pd.DataFrame:
    """Running peak, drawdown (fraction, <= 0)."""
    peak = close_or_equity.cummax()
    return pd.DataFrame({"peak": peak, "drawdown": close_or_equity / peak - 1})


def max_drawdown(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    return float(drawdown(series)["drawdown"].min())


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Current volume divided by the average of the *previous* ``window`` bars (no look-ahead)."""
    _check_period(window, "window")
    return (volume / volume.shift(1).rolling(window, min_periods=window).mean()).rename("relative_volume")


def correlation_matrix(closes: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation of simple returns between columns (aligned on common timestamps)."""
    if method not in {"pearson", "spearman", "kendall"}:
        raise ValueError("method must be pearson, spearman or kendall")
    rets = closes.pct_change().dropna(how="any")
    if len(rets) < 3:
        raise ValueError("Not enough overlapping observations to compute correlations.")
    return rets.corr(method=method)
