"""Rule-based strategies.

A strategy maps OHLCV bars to a *target position* series in {-1, 0, 1} evaluated at each bar's
close, using only data available at that close. The engine executes the change on the NEXT bar's
open, so no strategy can trade on information it did not have.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from trading_mcp.indicators import technical as ta

SignalFn = Callable[..., pd.Series]


def _hold_state(entries: pd.Series, exits: pd.Series) -> pd.Series:
    """Convert entry/exit booleans into a 1/0 holding state (state machine, no look-ahead)."""
    state = pd.Series(np.nan, index=entries.index)
    state[entries.fillna(False).astype(bool)] = 1.0
    state[exits.fillna(False).astype(bool) & ~entries.fillna(False).astype(bool)] = 0.0
    return state.ffill().fillna(0.0)


def _directional(long_cond: pd.Series, allow_short: bool) -> pd.Series:
    long_cond = long_cond.fillna(False).astype(bool)
    pos = long_cond.astype(float)
    if allow_short:
        pos = pos.where(long_cond, -1.0)
    return pos


def sma_crossover(df: pd.DataFrame, fast: int = 20, slow: int = 50, allow_short: bool = False) -> pd.Series:
    if fast >= slow:
        raise ValueError("fast must be < slow")
    f, s = ta.sma(df["close"], fast), ta.sma(df["close"], slow)
    return _directional(f > s, allow_short).where(s.notna(), 0.0)


def ema_crossover(df: pd.DataFrame, fast: int = 12, slow: int = 26, allow_short: bool = False) -> pd.Series:
    if fast >= slow:
        raise ValueError("fast must be < slow")
    f, s = ta.ema(df["close"], fast), ta.ema(df["close"], slow)
    return _directional(f > s, allow_short).where(s.notna(), 0.0)


def macd_crossover(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9,
                   allow_short: bool = False) -> pd.Series:
    m = ta.macd(df["close"], fast, slow, signal)
    return _directional(m["macd"] > m["signal"], allow_short).where(m["signal"].notna(), 0.0)


def rsi_mean_reversion(df: pd.DataFrame, period: int = 14, lower: float = 30, upper: float = 70) -> pd.Series:
    """Long when RSI crosses below ``lower``; flat when RSI rises above ``upper``. Long only."""
    if lower >= upper:
        raise ValueError("lower must be < upper")
    r = ta.rsi(df["close"], period)
    return _hold_state(r < lower, r > upper)


def bollinger_reversion(df: pd.DataFrame, period: int = 20, num_std: float = 2.0) -> pd.Series:
    """Long when close falls below the lower band; flat when it reaches the middle band. Long only."""
    bb = ta.bollinger_bands(df["close"], period, num_std)
    return _hold_state(df["close"] < bb["lower"], df["close"] >= bb["middle"])


def donchian_breakout(df: pd.DataFrame, entry: int = 20, exit: int = 10) -> pd.Series:
    """Long on a close above the prior ``entry``-bar high; flat on a close below the prior ``exit``-bar low."""
    hi = df["high"].shift(1).rolling(entry, min_periods=entry).max()
    lo = df["low"].shift(1).rolling(exit, min_periods=exit).min()
    return _hold_state(df["close"] > hi, df["close"] < lo)


def buy_and_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


@dataclass(frozen=True)
class StrategySpec:
    fn: SignalFn
    defaults: dict[str, Any]
    description: str
    grid: dict[str, list[Any]] = field(default_factory=dict)  # used by walk-forward optimisation


STRATEGIES: dict[str, StrategySpec] = {
    "sma_crossover": StrategySpec(sma_crossover, {"fast": 20, "slow": 50, "allow_short": False},
                                  "Long while SMA(fast) > SMA(slow); optional short otherwise.",
                                  {"fast": [10, 20, 30], "slow": [50, 100, 200]}),
    "ema_crossover": StrategySpec(ema_crossover, {"fast": 12, "slow": 26, "allow_short": False},
                                  "Long while EMA(fast) > EMA(slow); optional short otherwise.",
                                  {"fast": [8, 12, 20], "slow": [26, 50, 100]}),
    "macd_crossover": StrategySpec(macd_crossover, {"fast": 12, "slow": 26, "signal": 9, "allow_short": False},
                                   "Long while MACD > signal line.", {"signal": [5, 9, 14]}),
    "rsi_mean_reversion": StrategySpec(rsi_mean_reversion, {"period": 14, "lower": 30, "upper": 70},
                                       "Buy oversold RSI, exit when overbought.",
                                       {"period": [7, 14], "lower": [25, 30, 35], "upper": [55, 65, 75]}),
    "bollinger_reversion": StrategySpec(bollinger_reversion, {"period": 20, "num_std": 2.0},
                                        "Buy below the lower band, exit at the middle band.",
                                        {"period": [10, 20, 30], "num_std": [1.5, 2.0, 2.5]}),
    "donchian_breakout": StrategySpec(donchian_breakout, {"entry": 20, "exit": 10},
                                      "Buy breakouts above the prior N-bar high, exit below the M-bar low.",
                                      {"entry": [20, 55], "exit": [10, 20]}),
    "buy_and_hold": StrategySpec(buy_and_hold, {}, "Always long (benchmark)."),
}


def build_signals(df: pd.DataFrame, strategy: str, params: dict[str, Any] | None = None) -> pd.Series:
    """Validate parameters and compute the target position series for ``strategy``."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {sorted(STRATEGIES)}")
    spec = STRATEGIES[strategy]
    params = {**spec.defaults, **(params or {})}
    unknown = set(params) - set(spec.defaults)
    if unknown:
        raise ValueError(f"Unknown parameters for {strategy}: {sorted(unknown)}. Allowed: {sorted(spec.defaults)}")
    return spec.fn(df, **params).reindex(df.index).fillna(0.0).rename("target_position")
