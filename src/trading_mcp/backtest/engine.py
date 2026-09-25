"""Bar-by-bar backtesting engine.

Execution model (documented because it drives every result):
* the target position is decided at bar t's close with data <= t;
* orders fill at bar t+1's OPEN, moved against the trader by ``slippage_bps``;
* ``fee_bps`` is charged on the notional of every fill (entry and exit);
* fractional quantities are allowed; no margin interest, borrow fees or dividends are modelled;
* a position still open on the last bar is closed at its close (with costs) and flagged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator

from trading_mcp.backtest.metrics import performance_stats, trade_stats
from trading_mcp.indicators.technical import returns


class BacktestConfig(BaseModel):
    initial_capital: float = Field(10_000, gt=0)
    fee_bps: float = Field(5.0, ge=0, le=500, description="Commission per fill, basis points of notional")
    slippage_bps: float = Field(5.0, ge=0, le=500, description="Adverse price move per fill, basis points")
    position_sizing: Literal["percent_equity", "fixed_cash", "volatility_target"] = "percent_equity"
    percent_equity: float = Field(1.0, gt=0, le=10, description="Fraction of equity per position (1 = 100%)")
    fixed_cash: float | None = Field(None, gt=0, description="Cash per position for fixed_cash sizing")
    target_volatility: float = Field(0.15, gt=0, le=2, description="Annualised target for volatility_target sizing")
    max_leverage: float = Field(1.0, gt=0, le=10)
    risk_free_rate: float = Field(0.0, ge=0, le=0.2)

    @model_validator(mode="after")
    def _check(self) -> BacktestConfig:
        if self.position_sizing == "fixed_cash" and self.fixed_cash is None:
            raise ValueError("fixed_cash sizing requires fixed_cash")
        return self


@dataclass
class BacktestResult:
    equity: pd.Series
    positions: pd.Series
    trades: list[dict[str, Any]]
    stats: dict[str, Any]
    benchmark: dict[str, Any]
    config: BacktestConfig
    notes: list[str] = field(default_factory=list)


def run_backtest(
    df: pd.DataFrame,
    target: pd.Series,
    config: BacktestConfig,
    periods_per_year: float,
) -> BacktestResult:
    """Simulate trading ``target`` positions (-1/0/1 decided at each close) on ``df``."""
    if len(df) < 3:
        raise ValueError("Need at least 3 bars to backtest.")
    target = target.reindex(df.index).fillna(0.0).clip(-1, 1)
    desired = np.sign(target.shift(1).fillna(0.0).to_numpy())  # executed on the next bar's open
    opens, closes = df["open"].to_numpy(float), df["close"].to_numpy(float)
    idx = df.index
    fee, slip = config.fee_bps / 1e4, config.slippage_bps / 1e4
    vol = (returns(df["close"], "log").rolling(20, min_periods=20).std() * math.sqrt(periods_per_year)).shift(1)
    vol_arr = vol.to_numpy()

    cash, qty = config.initial_capital, 0.0
    entry: dict[str, Any] | None = None
    equity = np.empty(len(df))
    pos = np.zeros(len(df))
    trades: list[dict[str, Any]] = []
    notes: list[str] = []

    def close_position(i: int, price: float, reason: str) -> None:
        nonlocal cash, qty, entry
        fill = price * (1 - slip * np.sign(qty))
        exit_fee = abs(qty) * fill * fee
        cash += qty * fill - exit_fee
        assert entry is not None
        pnl = (fill - entry["price"]) * qty - entry["fee"] - exit_fee
        trades.append({
            "direction": "long" if qty > 0 else "short",
            "entry_time": entry["time"], "entry_price": entry["price"],
            "exit_time": idx[i], "exit_price": fill, "quantity": abs(qty),
            "pnl": pnl, "return": pnl / (abs(qty) * entry["price"]),
            "fees": entry["fee"] + exit_fee, "bars_held": i - entry["bar"], "exit_reason": reason,
        })
        qty, entry = 0.0, None

    for i in range(len(df)):
        want = desired[i]
        if want != np.sign(qty):
            if qty != 0:
                close_position(i, opens[i], "signal")
            if want != 0 and cash > 0:
                fill = opens[i] * (1 + slip * want)
                alloc = _allocation(config, cash, vol_arr[i])
                units = alloc / (fill * (1 + fee))
                if units > 0:
                    qty = want * units
                    entry_fee = units * fill * fee
                    cash -= qty * fill + entry_fee
                    entry = {"time": idx[i], "price": fill, "fee": entry_fee, "bar": i}
        equity[i] = cash + qty * closes[i]
        pos[i] = np.sign(qty)
        if equity[i] <= 0:
            notes.append(f"Equity reached zero at {idx[i]}; simulation stopped.")
            equity[i:] = 0.0
            qty, entry = 0.0, None
            break

    if qty != 0:
        close_position(len(df) - 1, closes[-1], "end_of_data")
        equity[-1] = cash
        notes.append("A position open on the last bar was closed at its close price (costs included).")

    equity_s = pd.Series(equity, index=idx, name="equity")
    stats = {**performance_stats(equity_s, periods_per_year, config.risk_free_rate), **trade_stats(trades)}
    stats["exposure"] = float(np.mean(pos != 0))
    stats["total_fees"] = float(sum(t["fees"] for t in trades))
    stats["final_equity"] = float(equity_s.iloc[-1])
    bh = df["close"] / df["close"].iloc[0] * config.initial_capital
    benchmark = {"buy_and_hold_total_return": float(bh.iloc[-1] / bh.iloc[0] - 1),
                 "buy_and_hold_max_drawdown": float((bh / bh.cummax() - 1).min()),
                 "note": "Buy-and-hold benchmark excludes fees and slippage."}
    return BacktestResult(equity_s, pd.Series(pos, index=idx), trades, stats, benchmark, config, notes)


def _allocation(config: BacktestConfig, equity: float, realized_vol: float) -> float:
    cap = equity * config.max_leverage
    if config.position_sizing == "percent_equity":
        return min(equity * config.percent_equity, cap)
    if config.position_sizing == "fixed_cash":
        return min(float(config.fixed_cash or 0), cap)
    # volatility_target: scale exposure so expected annualised vol ~ target
    if realized_vol is None or not np.isfinite(realized_vol) or realized_vol <= 0:
        return min(equity, cap)
    return min(equity * config.target_volatility / realized_vol, cap)


def downsample(series: pd.Series, max_points: int = 250) -> pd.Series:
    """Evenly thin a series for transport, always keeping the last point."""
    if len(series) <= max_points:
        return series
    step = math.ceil(len(series) / max_points)
    thinned = series.iloc[::step]
    return thinned if thinned.index[-1] == series.index[-1] else pd.concat([thinned, series.iloc[[-1]]])
