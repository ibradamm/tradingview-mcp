"""Risk metrics: position sizing, VaR/CVaR, concentration, portfolio statistics."""

from __future__ import annotations

import math
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from scipy import stats

from trading_mcp.backtest.metrics import performance_stats


class PositionSizeRequest(BaseModel):
    capital: float = Field(gt=0)
    risk_percent: float = Field(gt=0, le=100, description="Max % of capital lost if the stop is hit")
    entry_price: float = Field(gt=0)
    stop_loss: float = Field(gt=0)
    max_position_percent: float = Field(100, gt=0, le=1000, description="Cap on exposure as % of capital")
    fee_bps: float = Field(0, ge=0, le=500, description="Round-trip-per-side costs included in the risk")
    allow_fractional: bool = False

    @model_validator(mode="after")
    def _check(self) -> PositionSizeRequest:
        if self.stop_loss == self.entry_price:
            raise ValueError("stop_loss must differ from entry_price")
        return self


def position_size(req: PositionSizeRequest) -> dict[str, Any]:
    """Fixed-fractional sizing: quantity such that hitting the stop loses ``risk_percent`` of capital."""
    direction = "long" if req.stop_loss < req.entry_price else "short"
    per_unit_move = abs(req.entry_price - req.stop_loss)
    per_unit_costs = (req.entry_price + req.stop_loss) * req.fee_bps / 1e4
    risk_per_unit = per_unit_move + per_unit_costs
    budget = req.capital * req.risk_percent / 100
    raw_qty = budget / risk_per_unit
    cap_qty = req.capital * req.max_position_percent / 100 / req.entry_price
    qty = min(raw_qty, cap_qty)
    if not req.allow_fractional:
        qty = math.floor(qty)
    exposure = qty * req.entry_price
    sign = 1 if direction == "long" else -1
    return {
        "direction": direction,
        "quantity": qty,
        "risk_per_unit": risk_per_unit,
        "amount_at_risk": qty * risk_per_unit,
        "risk_percent_of_capital": qty * risk_per_unit / req.capital * 100,
        "exposure": exposure,
        "exposure_percent_of_capital": exposure / req.capital * 100,
        "capped_by_max_position": raw_qty > cap_qty,
        "stop_distance_percent": per_unit_move / req.entry_price * 100,
        "reward_targets": {f"{r}R": req.entry_price + sign * r * per_unit_move for r in (1, 2, 3)},
        "notes": [
            "Assumes the stop fills at its price; gaps and slippage can make the real loss larger.",
            "Leverage (exposure > 100% of capital) requires margin and increases risk of ruin.",
        ],
    }


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> dict[str, Any]:
    """1-period VaR and CVaR (expected shortfall) as positive loss fractions, three methods."""
    r = returns.dropna()
    if len(r) < 30:
        raise ValueError("Need at least 30 return observations for VaR.")
    if not 0.8 <= confidence < 1:
        raise ValueError("confidence must be in [0.8, 1).")
    alpha = 1 - confidence
    hist_var = -np.quantile(r, alpha)
    tail = r[r <= -hist_var]
    mu, sigma = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(alpha)
    param_var = -(mu + z * sigma)
    param_cvar = -(mu - sigma * stats.norm.pdf(z) / alpha)
    s, k = stats.skew(r), stats.kurtosis(r)  # excess kurtosis
    z_cf = z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24 - (2 * z**3 - 5 * z) * s**2 / 36
    return {
        "confidence": confidence,
        "horizon": "1 bar",
        "historical_var": float(hist_var),
        "historical_cvar": float(-tail.mean()) if len(tail) else None,
        "parametric_normal_var": float(param_var),
        "parametric_normal_cvar": float(param_cvar),
        "cornish_fisher_var": float(-(mu + z_cf * sigma)),
        "skewness": float(s),
        "excess_kurtosis": float(k),
        "observations": int(len(r)),
        "methodology": (
            "VaR = loss not exceeded with the given confidence over one bar, as a fraction of value. "
            "Historical uses the empirical quantile (assumes the past distribution repeats); parametric "
            "assumes normal returns (underestimates fat tails, see excess_kurtosis); Cornish-Fisher adjusts "
            "for skew and kurtosis. Multi-period scaling by sqrt(time) assumes i.i.d. returns."
        ),
    }


def concentration(weights: pd.Series) -> dict[str, Any]:
    w = weights.abs() / weights.abs().sum()
    hhi = float((w**2).sum())
    return {"herfindahl_index": hhi, "effective_number_of_assets": 1 / hhi, "largest_weight": float(w.max()),
            "largest_position": str(w.idxmax())}


def portfolio_returns(
    rets: pd.DataFrame, weights: pd.Series, rebalance: Literal["every_bar", "none"] = "every_bar"
) -> pd.Series:
    """Portfolio return series. 'every_bar' keeps weights constant; 'none' lets them drift (buy and hold)."""
    w = weights.reindex(rets.columns).fillna(0.0)
    if rebalance == "every_bar":
        return rets.mul(w, axis=1).sum(axis=1)
    growth = (1 + rets).cumprod()
    value = growth.mul(w, axis=1).sum(axis=1)
    return value.pct_change().fillna(value.iloc[0] - 1)


def portfolio_statistics(
    rets: pd.DataFrame, weights: pd.Series, periods_per_year: float, risk_free_rate: float = 0.0,
    rebalance: Literal["every_bar", "none"] = "every_bar", confidence: float = 0.95,
) -> dict[str, Any]:
    port = portfolio_returns(rets, weights, rebalance)
    equity = pd.concat([pd.Series([1.0]), (1 + port).cumprod().reset_index(drop=True)])
    perf = performance_stats(equity, periods_per_year, risk_free_rate)
    cov = rets.cov() * periods_per_year
    w = weights.reindex(rets.columns).fillna(0.0)
    port_vol = float(np.sqrt(w @ cov @ w))
    asset_vol = np.sqrt(np.diag(cov))
    marginal = cov @ w / port_vol if port_vol > 0 else cov @ w * 0
    contrib = w * marginal
    per_asset = {}
    for i, col in enumerate(rets.columns):
        eq = (1 + rets[col]).cumprod()
        per_asset[col] = {
            "weight": float(w[col]),
            "annualized_return": performance_stats(pd.concat([pd.Series([1.0]), eq.reset_index(drop=True)]),
                                                   periods_per_year)["annualized_return"],
            "annualized_volatility": float(asset_vol[i]),
            "max_drawdown": float((eq / eq.cummax() - 1).min()),
            "risk_contribution_percent": float(contrib[col] / port_vol * 100) if port_vol > 0 else None,
        }
    return {
        "portfolio": {**perf, "var": value_at_risk(port, confidence)},
        "assets": per_asset,
        "correlation_matrix": rets.corr().to_dict(),
        "diversification": {
            "diversification_ratio": float((np.abs(w) @ asset_vol) / port_vol) if port_vol > 0 else None,
            "average_pairwise_correlation": _avg_offdiag(rets.corr()),
            **concentration(w),
        },
        "rebalance": rebalance,
    }


def _avg_offdiag(corr: pd.DataFrame) -> float | None:
    n = len(corr)
    if n < 2:
        return None
    values = corr.to_numpy()
    return float((values.sum() - n) / (n * n - n))
