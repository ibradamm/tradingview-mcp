"""Risk management and portfolio MCP tools."""

from __future__ import annotations

from typing import Any, Literal

import pandas as pd
from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations
from pydantic import BaseModel, Field

from trading_mcp.backtest.engine import BacktestConfig, run_backtest
from trading_mcp.backtest.strategies import build_signals
from trading_mcp.data.base import Timeframe
from trading_mcp.indicators import technical as ta
from trading_mcp.risk.risk import (
    PositionSizeRequest,
    concentration,
    portfolio_statistics,
    position_size,
    value_at_risk,
)
from trading_mcp.tools.common import RESEARCH_DISCLAIMER, clean, load_ohlcv, parse_timeframe, safe_tool

READ = ToolAnnotations(readOnlyHint=True, openWorldHint=True)


class Holding(BaseModel):
    ticker: str
    weight: float = Field(description="Portfolio weight; weights are normalised to sum to 1 (negative = short)")


def _returns_frame(
    tickers: list[str], timeframe: str, start: str | None, end: str | None
) -> tuple[pd.DataFrame, float]:
    closes = pd.DataFrame({t: load_ohlcv(t, timeframe, start, end)["close"] for t in tickers})
    # Different calendars (crypto trades on weekends): keep common timestamps only.
    closes = closes.dropna()
    if len(closes) < 31:
        raise ValueError("Fewer than 31 common observations across assets; widen the date range.")
    return closes.pct_change().dropna(), parse_timeframe(timeframe).periods_per_year


@safe_tool
def calculate_position_size(
    capital: float, risk_percent: float, entry_price: float, stop_loss: float,
    max_position_percent: float = 100, fee_bps: float = 0, allow_fractional: bool = False,
) -> dict[str, Any]:
    """Fixed-fractional position size: how many units so that hitting the stop loses ``risk_percent`` % of capital.

    Direction is inferred (stop below entry = long). Returns quantity, amount at risk, exposure, stop distance and
    1R/2R/3R price levels. Exposure is capped at ``max_position_percent`` of capital.
    """
    req = PositionSizeRequest(capital=capital, risk_percent=risk_percent, entry_price=entry_price,
                              stop_loss=stop_loss, max_position_percent=max_position_percent, fee_bps=fee_bps,
                              allow_fractional=allow_fractional)
    return clean(position_size(req))


@safe_tool
def risk_analysis(
    tickers: list[str],
    weights: list[float] | None = None,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    confidence: float = 0.95,
    capital: float | None = None,
    strategy: str | None = None,
    strategy_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Risk profile of one asset, a basket, or a strategy.

    Assets: volatility, max drawdown, VaR/CVaR (historical, normal, Cornish-Fisher) per asset and for the weighted
    basket, concentration (HHI, effective N), correlations. Equal weights if ``weights`` is omitted.
    Strategy: pass ``strategy`` (see list_strategies) with a single ticker to analyse the strategy's returns
    (fees/slippage included) instead of the raw asset.
    ``capital`` converts VaR fractions into amounts.
    """
    if not 1 <= len(tickers) <= 25:
        raise ValueError("Provide between 1 and 25 tickers.")
    if strategy:
        if len(tickers) != 1:
            raise ValueError("Strategy risk analysis takes exactly one ticker.")
        df = load_ohlcv(tickers[0], timeframe, start, end)
        ppy = Timeframe(df.attrs["timeframe"]).periods_per_year
        res = run_backtest(df, build_signals(df, strategy, strategy_params), BacktestConfig(), ppy)
        rets = res.equity.pct_change().dropna()
        var = value_at_risk(rets, confidence)
        return clean({
            "subject": f"strategy {strategy} on {tickers[0]}",
            "volatility_annualized": res.stats["annualized_volatility"], "max_drawdown": res.stats["max_drawdown"],
            "var": var, "var_amounts": _amounts(var, capital),
            "exposure": res.stats["exposure"], "number_of_trades": res.stats["number_of_trades"],
            "note": "Risk of the simulated strategy (backtest) - past risk is not a bound on future risk.",
            "disclaimer": RESEARCH_DISCLAIMER,
        })

    w = pd.Series(weights if weights else [1.0] * len(tickers), index=[t.upper() for t in tickers], dtype=float)
    if len(w) != len(tickers):
        raise ValueError("weights must have the same length as tickers.")
    if w.abs().sum() == 0:
        raise ValueError("weights cannot all be zero.")
    w = w / w.abs().sum()
    rets, ppy = _returns_frame([t.upper() for t in tickers], timeframe, start, end)
    per_asset = {}
    for col in rets.columns:
        eq = (1 + rets[col]).cumprod()
        per_asset[col] = {"volatility_annualized": float(rets[col].std() * ppy**0.5),
                          "max_drawdown": ta.max_drawdown(eq), "var": value_at_risk(rets[col], confidence)}
    basket = rets.mul(w, axis=1).sum(axis=1)
    var = value_at_risk(basket, confidence)
    return clean({
        "subject": "assets", "timeframe": timeframe, "observations": len(rets), "weights": w.to_dict(),
        "per_asset": per_asset,
        "portfolio": {"volatility_annualized": float(basket.std() * ppy**0.5),
                      "max_drawdown": ta.max_drawdown((1 + basket).cumprod()), "var": var,
                      "var_amounts": _amounts(var, capital)},
        "concentration": concentration(w),
        "correlation_matrix": rets.corr().to_dict() if len(rets.columns) > 1 else None,
        "disclaimer": RESEARCH_DISCLAIMER,
    })


def _amounts(var: dict[str, Any], capital: float | None) -> dict[str, float] | None:
    if not capital:
        return None
    keys = ("historical_var", "historical_cvar", "parametric_normal_var", "cornish_fisher_var")
    return {k: var[k] * capital for k in keys if var.get(k) is not None}


@safe_tool
def portfolio_analysis(
    holdings: list[Holding],
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    risk_free_rate: float = 0.0,
    rebalance: Literal["every_bar", "none"] = "every_bar",
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Portfolio statistics from a list of {ticker, weight}: return, annualised return and volatility, Sharpe,
    Sortino, max drawdown, VaR, per-asset risk contributions, correlation matrix, diversification ratio and
    concentration. rebalance='every_bar' keeps weights constant; 'none' lets them drift (buy and hold).
    Default window: 3 years of daily data.
    """
    if not 1 <= len(holdings) <= 25:
        raise ValueError("Provide between 1 and 25 holdings.")
    w = pd.Series({h.ticker.upper(): h.weight for h in holdings}, dtype=float)
    if w.abs().sum() == 0:
        raise ValueError("weights cannot all be zero.")
    normalised = w / w.abs().sum()
    rets, ppy = _returns_frame(list(w.index), timeframe, start, end)
    out = portfolio_statistics(rets, normalised, ppy, risk_free_rate, rebalance, confidence)
    return clean({"weights_normalised": normalised.to_dict(), "period": {"start": rets.index[0], "end": rets.index[-1],
                  "observations": len(rets)}, **out,
                  "notes": ["Historical statistics; they do not predict future risk or return.",
                            "Survivorship bias: holdings chosen today have, by construction, survived.",
                            "Transaction costs of rebalancing are not included."],
                  "disclaimer": RESEARCH_DISCLAIMER})


def register(mcp: MCPServer) -> None:
    mcp.add_tool(calculate_position_size, annotations=ToolAnnotations(readOnlyHint=True, openWorldHint=False))
    for fn in (risk_analysis, portfolio_analysis):
        mcp.add_tool(fn, annotations=READ)
