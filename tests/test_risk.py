import numpy as np
import pandas as pd
import pytest
from mcp import Client
from scipy import stats

from trading_mcp.risk.risk import (
    PositionSizeRequest,
    concentration,
    portfolio_returns,
    portfolio_statistics,
    position_size,
    value_at_risk,
)
from trading_mcp.server import create_mcp


def test_position_size_long():
    out = position_size(PositionSizeRequest(capital=10_000, risk_percent=1, entry_price=100, stop_loss=95))
    assert out["direction"] == "long" and out["quantity"] == 20
    assert out["amount_at_risk"] == pytest.approx(100) and out["exposure"] == 2000
    assert out["reward_targets"]["2R"] == 110


def test_position_size_short_cap_and_fees():
    out = position_size(PositionSizeRequest(capital=10_000, risk_percent=2, entry_price=50, stop_loss=50.5,
                                            max_position_percent=50))
    assert out["direction"] == "short" and out["capped_by_max_position"]
    assert out["exposure"] <= 5000
    fees = position_size(PositionSizeRequest(capital=10_000, risk_percent=1, entry_price=100, stop_loss=95,
                                             fee_bps=10, allow_fractional=True))
    assert fees["quantity"] < 20 and fees["amount_at_risk"] == pytest.approx(100)
    with pytest.raises(ValueError):
        PositionSizeRequest(capital=1, risk_percent=1, entry_price=10, stop_loss=10)


def test_var_normal_sample():
    r = pd.Series(np.random.default_rng(1).normal(0, 0.01, 100_000))
    v = value_at_risk(r, 0.95)
    expected = -stats.norm.ppf(0.05) * 0.01
    for key in ("historical_var", "parametric_normal_var", "cornish_fisher_var"):
        assert v[key] == pytest.approx(expected, rel=0.03), key
    assert v["historical_cvar"] > v["historical_var"]
    with pytest.raises(ValueError):
        value_at_risk(r.iloc[:10])


def test_concentration():
    c = concentration(pd.Series({"A": 0.5, "B": 0.5}))
    assert c["herfindahl_index"] == 0.5 and c["effective_number_of_assets"] == 2


def test_portfolio_rebalance_modes():
    rets = pd.DataFrame({"A": [0.10, 0.10], "B": [0.0, 0.0]})
    w = pd.Series({"A": 0.5, "B": 0.5})
    assert portfolio_returns(rets, w, "every_bar").tolist() == pytest.approx([0.05, 0.05])
    drift = portfolio_returns(rets, w, "none")
    assert drift.iloc[1] == pytest.approx((0.5 * 1.21 + 0.5) / (0.5 * 1.1 + 0.5) - 1)


def test_diversification_ratio_uncorrelated():
    rng = np.random.default_rng(0)
    rets = pd.DataFrame(rng.normal(0, 0.01, (5000, 4)), columns=list("ABCD"))
    out = portfolio_statistics(rets, pd.Series(0.25, index=list("ABCD")), 252)
    assert out["diversification"]["diversification_ratio"] == pytest.approx(2.0, rel=0.05)
    contribs = [a["risk_contribution_percent"] for a in out["assets"].values()]
    assert sum(contribs) == pytest.approx(100)


async def test_risk_tools_via_mcp():
    async with Client(create_mcp()) as client:
        r = await client.call_tool("calculate_position_size",
                                   {"capital": 25000, "risk_percent": 1, "entry_price": 120, "stop_loss": 114})
        assert not r.is_error and r.structured_content["quantity"] == 41
        r = await client.call_tool("risk_analysis", {"tickers": ["NVDA", "AAPL"], "capital": 10000})
        assert not r.is_error, r.content
        assert r.structured_content["portfolio"]["var_amounts"]["historical_var"] > 0
        r = await client.call_tool("risk_analysis", {"tickers": ["NVDA"], "strategy": "sma_crossover"})
        assert not r.is_error, r.content
        assert r.structured_content["subject"].startswith("strategy")
        r = await client.call_tool("portfolio_analysis", {"holdings": [
            {"ticker": "NVDA", "weight": 40}, {"ticker": "SPY", "weight": 40}, {"ticker": "BTC-USD", "weight": 20}]})
        assert not r.is_error, r.content
        d = r.structured_content
        assert d["weights_normalised"]["NVDA"] == pytest.approx(0.4)
        for key in ("total_return", "annualized_volatility", "sharpe_ratio", "max_drawdown"):
            assert key in d["portfolio"]
        assert d["diversification"]["diversification_ratio"] >= 1
