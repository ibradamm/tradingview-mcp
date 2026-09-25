import numpy as np
import pandas as pd
import pytest
from mcp import Client

from trading_mcp.backtest.engine import BacktestConfig, downsample, run_backtest
from trading_mcp.backtest.metrics import performance_stats, trade_stats
from trading_mcp.backtest.strategies import build_signals
from trading_mcp.backtest.walk_forward import make_folds, walk_forward_rules
from trading_mcp.server import create_mcp

NO_COSTS = BacktestConfig(initial_capital=1000, fee_bps=0, slippage_bps=0)


def _bars(opens, closes):
    idx = pd.date_range("2024-01-01", periods=len(opens), freq="D", tz="UTC")
    o, c = np.array(opens, float), np.array(closes, float)
    return pd.DataFrame({"open": o, "high": np.maximum(o, c), "low": np.minimum(o, c), "close": c, "volume": 1.0},
                        index=idx)


def test_buy_and_hold_executes_next_open_without_costs():
    df = _bars([10, 11, 12, 13], [10.5, 11.5, 12.5, 14])
    res = run_backtest(df, pd.Series(1.0, index=df.index), NO_COSTS, 252)
    # signal at bar0 close -> buy at bar1 open (11); final close 14
    assert res.equity.iloc[0] == 1000
    assert res.equity.iloc[1] == pytest.approx(1000 * 11.5 / 11)
    assert res.stats["final_equity"] == pytest.approx(1000 * 14 / 11)
    assert res.trades[0]["entry_price"] == 11 and res.trades[0]["exit_reason"] == "end_of_data"


def test_fees_and_slippage_are_applied_exactly():
    df = _bars([10, 10, 10, 10], [10, 10, 10, 10])
    cfg = BacktestConfig(initial_capital=1000, fee_bps=10, slippage_bps=20)
    res = run_backtest(df, pd.Series(1.0, index=df.index), cfg, 252)
    entry_fill, exit_fill = 10 * 1.002, 10 * 0.998
    units = 1000 / (entry_fill * 1.001)
    expected = 1000 - units * entry_fill * 0.001 + units * (exit_fill - entry_fill) - units * exit_fill * 0.001
    assert res.stats["final_equity"] == pytest.approx(expected)
    assert res.stats["final_equity"] < 1000
    assert res.trades[0]["fees"] == pytest.approx(units * (entry_fill + exit_fill) * 0.001)


def test_short_position_profits_when_price_falls():
    df = _bars([10, 10, 8, 6], [10, 9, 7, 5])
    res = run_backtest(df, pd.Series(-1.0, index=df.index), NO_COSTS, 252)
    assert res.trades[0]["direction"] == "short"
    assert res.stats["final_equity"] == pytest.approx(1000 + 100 * (10 - 5))


def test_no_lookahead_future_data_does_not_change_past(ohlcv):
    full = run_backtest(ohlcv, build_signals(ohlcv, "sma_crossover"), BacktestConfig(), 252)
    cut = 500
    part_df = ohlcv.iloc[:cut]
    part = run_backtest(part_df, build_signals(part_df, "sma_crossover"), BacktestConfig(), 252)
    # All bars except the last (forced close) must be identical.
    pd.testing.assert_series_equal(full.equity.iloc[: cut - 1], part.equity.iloc[: cut - 1])


def test_signal_on_last_bar_is_never_traded():
    df = _bars([10] * 5, [10] * 5)
    sig = pd.Series([0, 0, 0, 0, 1.0], index=df.index)
    res = run_backtest(df, sig, NO_COSTS, 252)
    assert res.trades == [] and res.stats["final_equity"] == 1000


def test_position_sizing_modes(ohlcv):
    sig = pd.Series(1.0, index=ohlcv.index)
    half = run_backtest(ohlcv, sig, BacktestConfig(percent_equity=0.5, fee_bps=0, slippage_bps=0), 252)
    full = run_backtest(ohlcv, sig, BacktestConfig(fee_bps=0, slippage_bps=0), 252)
    assert abs(half.stats["total_return"]) < abs(full.stats["total_return"])
    cash = run_backtest(ohlcv, sig, BacktestConfig(position_sizing="fixed_cash", fixed_cash=1000), 252)
    assert cash.trades[0]["quantity"] * cash.trades[0]["entry_price"] < 1001
    vt = run_backtest(ohlcv, sig, BacktestConfig(position_sizing="volatility_target", target_volatility=0.05), 252)
    assert vt.stats["number_of_trades"] == 1
    with pytest.raises(ValueError):
        BacktestConfig(position_sizing="fixed_cash")


def test_performance_stats_known_values():
    eq = pd.Series([100, 110, 99, 120], index=pd.date_range("2024", periods=4, freq="D"))
    s = performance_stats(eq, 252)
    assert s["total_return"] == pytest.approx(0.2)
    assert s["max_drawdown"] == pytest.approx(-0.1)
    rets = eq.pct_change().dropna()
    assert s["sharpe_ratio"] == pytest.approx(rets.mean() / rets.std() * np.sqrt(252))


def test_trade_stats():
    t = [{"pnl": 10, "return": 0.1, "bars_held": 2}, {"pnl": -5, "return": -0.05, "bars_held": 4}]
    s = trade_stats(t)
    assert s["win_rate"] == 0.5 and s["profit_factor"] == 2 and s["average_trade"] == 2.5
    assert trade_stats([])["number_of_trades"] == 0


def test_all_strategies_build(ohlcv):
    for name in ["sma_crossover", "ema_crossover", "macd_crossover", "rsi_mean_reversion",
                 "bollinger_reversion", "donchian_breakout", "buy_and_hold"]:
        s = build_signals(ohlcv, name)
        assert set(s.unique()) <= {-1.0, 0.0, 1.0}, name
    assert set(build_signals(ohlcv, "sma_crossover", {"allow_short": True}).unique()) <= {-1.0, 0.0, 1.0}
    with pytest.raises(ValueError):
        build_signals(ohlcv, "sma_crossover", {"bogus": 1})
    with pytest.raises(ValueError):
        build_signals(ohlcv, "nope")


def test_make_folds_are_ordered_and_disjoint():
    folds = make_folds(1000, 400, 100, 100)
    assert len(folds) == 5
    for f in folds:
        assert f.train.stop == f.validation.start and f.validation.stop == f.test.start
    assert folds[1].test.start == folds[0].test.stop
    with pytest.raises(ValueError):
        make_folds(100, 400, 100, 100)


def test_walk_forward_rules(ohlcv):
    out = walk_forward_rules(ohlcv, "sma_crossover", BacktestConfig(), 252, 300, 100, 100,
                             param_grid={"fast": [5, 10], "slow": [30, 60]})
    assert len(out["folds"]) >= 4
    assert out["candidates_tested"] == 4
    assert "degradation_ratio" in out["train_vs_test"]
    assert out["out_of_sample"]["observations"] > 0
    for f in out["folds"]:
        assert f["train_period"]["end"] < f["validation_period"]["start"] < f["test_period"]["start"]


def test_downsample_keeps_last():
    s = pd.Series(range(1000), index=pd.date_range("2024", periods=1000, freq="h"))
    d = downsample(s, 100)
    assert len(d) <= 101 and d.index[-1] == s.index[-1]


async def test_backtest_tools_via_mcp():
    async with Client(create_mcp()) as client:
        r = await client.call_tool("backtest_strategy", {
            "ticker": "NVDA", "strategy": "sma_crossover", "params": {"fast": 10, "slow": 30},
            "config": {"initial_capital": 5000, "fee_bps": 2, "slippage_bps": 3},
        })
        assert not r.is_error, r.content
        d = r.structured_content
        for key in ("total_return", "annualized_return", "annualized_volatility", "sharpe_ratio", "sortino_ratio",
                    "max_drawdown", "win_rate", "profit_factor", "number_of_trades", "average_trade",
                    "average_win", "average_loss"):
            assert key in d["statistics"], key
        assert d["equity_curve"] and d["warnings"]
        r = await client.call_tool("walk_forward_backtest", {"ticker": "NVDA", "train_bars": 250,
                                                             "validation_bars": 60, "test_bars": 60})
        assert not r.is_error, r.content
        assert r.structured_content["overfitting_assessment"]["verdict"]
        r = await client.call_tool("list_strategies", {})
        assert "rsi_mean_reversion" in r.structured_content
