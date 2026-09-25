"""Feature engineering and targets.

Leakage rules enforced here:
* every feature at time t is computed from bars <= t (causal indicators, no centred windows);
* the target at time t looks FORWARD (t -> t+horizon) and is only ever used as a label;
* rows whose target is unknown (the last ``horizon`` bars) are excluded from training.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from trading_mcp.indicators import technical as ta

FeatureFn = Callable[[pd.DataFrame, bool], pd.Series]


def _vwap_distance(df: pd.DataFrame, intraday: bool) -> pd.Series:
    if intraday:
        v = ta.vwap(df, "session")
    else:
        typical = (df["high"] + df["low"] + df["close"]) / 3
        v = (typical * df["volume"]).rolling(20).sum() / df["volume"].rolling(20).sum()
    return df["close"] / v - 1


FEATURES: dict[str, FeatureFn] = {
    "return_1": lambda df, _: ta.returns(df["close"], "simple", 1),
    "return_5": lambda df, _: ta.returns(df["close"], "simple", 5),
    "log_return_1": lambda df, _: ta.returns(df["close"], "log", 1),
    "momentum_10": lambda df, _: df["close"] / df["close"].shift(10) - 1,
    "momentum_20": lambda df, _: df["close"] / df["close"].shift(20) - 1,
    "rsi_14": lambda df, _: ta.rsi(df["close"], 14) / 100,
    "macd_hist_pct": lambda df, _: ta.macd(df["close"])["histogram"] / df["close"],
    "ema20_distance": lambda df, _: df["close"] / ta.ema(df["close"], 20) - 1,
    "sma50_distance": lambda df, _: df["close"] / ta.sma(df["close"], 50) - 1,
    "ema20_ema50_spread": lambda df, _: ta.ema(df["close"], 20) / ta.ema(df["close"], 50) - 1,
    "bb_percent_b": lambda df, _: ta.bollinger_bands(df["close"])["percent_b"],
    "bb_bandwidth": lambda df, _: ta.bollinger_bands(df["close"])["bandwidth"],
    "atr_pct": lambda df, _: ta.atr(df, 14) / df["close"],
    "adx_14": lambda df, _: ta.adx(df, 14)["adx"] / 100,
    "di_spread": lambda df, _: (lambda a: (a["plus_di"] - a["minus_di"]) / 100)(ta.adx(df, 14)),
    "volatility_20": lambda df, _: ta.returns(df["close"], "log").rolling(20).std(),
    "log_volume_change": lambda df, _: np.log(df["volume"].replace(0, np.nan)).diff(),
    "relative_volume": lambda df, _: ta.relative_volume(df["volume"], 20),
    "vwap_distance": _vwap_distance,
    "high_low_range": lambda df, _: (df["high"] - df["low"]) / df["close"],
}
DEFAULT_FEATURES = list(FEATURES)


def build_features(df: pd.DataFrame, features: list[str] | None = None, intraday: bool = False) -> pd.DataFrame:
    """Return a feature matrix aligned on ``df.index`` (warm-up rows contain NaN)."""
    names = features or DEFAULT_FEATURES
    unknown = [f for f in names if f not in FEATURES]
    if unknown:
        raise ValueError(f"Unknown features {unknown}. Available: {sorted(FEATURES)}")
    out = pd.DataFrame({name: FEATURES[name](df, intraday) for name in names}, index=df.index)
    return out.replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series, horizon: int, task: str, threshold: float = 0.0) -> pd.Series:
    """Forward return over ``horizon`` bars; classification label = 1 if it exceeds ``threshold``."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1 bar")
    fwd = close.shift(-horizon) / close - 1
    if task == "regression":
        return fwd.rename("target")
    if task == "classification":
        return (fwd > threshold).astype(float).where(fwd.notna()).rename("target")
    raise ValueError("task must be 'classification' or 'regression'")


def build_dataset(
    df: pd.DataFrame, features: list[str] | None, horizon: int, task: str, intraday: bool, threshold: float = 0.0
) -> tuple[pd.DataFrame, pd.Series]:
    """Features and target with incomplete rows removed (warm-up and unknown future)."""
    X = build_features(df, features, intraday)
    y = make_target(df["close"], horizon, task, threshold)
    mask = X.notna().all(axis=1) & y.notna()
    return X[mask], y[mask]
