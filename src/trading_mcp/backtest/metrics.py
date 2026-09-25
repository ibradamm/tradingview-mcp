"""Performance statistics for equity curves and trade logs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from trading_mcp.indicators.technical import max_drawdown


def performance_stats(
    equity: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0
) -> dict[str, float | None]:
    """Return-based statistics of an equity curve (one value per bar)."""
    if len(equity) < 2:
        raise ValueError("Equity curve needs at least 2 points.")
    rets = equity.pct_change().dropna()
    n = len(rets)
    total = float(equity.iloc[-1] / equity.iloc[0] - 1)
    years = n / periods_per_year
    annualized = (1 + total) ** (1 / years) - 1 if years > 0 and total > -1 else None
    vol = float(rets.std(ddof=1) * np.sqrt(periods_per_year)) if n > 1 else None
    rf_per_bar = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    excess = rets - rf_per_bar
    std = excess.std(ddof=1)
    sharpe = float(excess.mean() / std * np.sqrt(periods_per_year)) if n > 1 and std > 0 else None
    downside = np.sqrt((np.minimum(excess, 0) ** 2).mean())
    sortino = float(excess.mean() / downside * np.sqrt(periods_per_year)) if downside > 0 else None
    mdd = max_drawdown(equity)
    return {
        "total_return": total,
        "annualized_return": annualized,
        "annualized_volatility": vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "calmar_ratio": (annualized / abs(mdd)) if annualized is not None and mdd < 0 else None,
        "observations": n,
        "years": years,
    }


def trade_stats(trades: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Win rate, profit factor and average trade figures (net of fees and slippage)."""
    if not trades:
        return {"number_of_trades": 0, "win_rate": None, "profit_factor": None, "average_trade": None,
                "average_trade_return": None, "average_win": None, "average_loss": None,
                "largest_win": None, "largest_loss": None, "average_bars_held": None}
    pnl = np.array([t["pnl"] for t in trades], dtype=float)
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    gross_loss = -losses.sum()
    return {
        "number_of_trades": len(trades),
        "win_rate": float(len(wins) / len(pnl)),
        "profit_factor": float(wins.sum() / gross_loss) if gross_loss > 0 else None,
        "average_trade": float(pnl.mean()),
        "average_trade_return": float(np.mean([t["return"] for t in trades])),
        "average_win": float(wins.mean()) if len(wins) else None,
        "average_loss": float(losses.mean()) if len(losses) else None,
        "largest_win": float(pnl.max()),
        "largest_loss": float(pnl.min()),
        "average_bars_held": float(np.mean([t["bars_held"] for t in trades])),
    }
