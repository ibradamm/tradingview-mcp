"""Backtesting MCP tools."""

from __future__ import annotations

from typing import Any

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from trading_mcp.backtest.engine import BacktestConfig, run_backtest
from trading_mcp.backtest.strategies import STRATEGIES, build_signals
from trading_mcp.backtest.walk_forward import walk_forward_rules
from trading_mcp.data.base import Timeframe
from trading_mcp.tools.common import (
    BACKTEST_WARNINGS,
    RESEARCH_DISCLAIMER,
    clean,
    equity_records,
    load_ohlcv,
    safe_tool,
    source_info,
)

ANNOTATIONS = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)


def _persist(kind: str, payload: dict[str, Any]) -> int | None:
    """Store a run in the database when the persistence layer is available."""
    try:
        from trading_mcp.db.repository import save_backtest
    except ImportError:
        return None
    return save_backtest(kind, payload)


@safe_tool
def list_strategies() -> dict[str, Any]:
    """Available backtest strategies with their default parameters and walk-forward search grids."""
    return {name: {"description": s.description, "defaults": s.defaults, "walk_forward_grid": s.grid}
            for name, s in STRATEGIES.items()}


@safe_tool
def backtest_strategy(
    ticker: str,
    strategy: str = "sma_crossover",
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
    max_trades_returned: int = 50,
) -> dict[str, Any]:
    """Backtest a rule-based strategy with fees and slippage.

    strategy: see list_strategies (sma_crossover, ema_crossover, macd_crossover, rsi_mean_reversion,
    bollinger_reversion, donchian_breakout, buy_and_hold). params override the defaults, e.g. {"fast": 10, "slow": 30}.
    config: initial_capital, fee_bps, slippage_bps, position_sizing (percent_equity | fixed_cash | volatility_target),
    percent_equity, fixed_cash, target_volatility, max_leverage, risk_free_rate.
    Signals are computed at bar close and executed at the next bar's open (no look-ahead).
    Returns statistics, a (downsampled) equity curve and the trade log.
    """
    config = config or BacktestConfig()
    df = load_ohlcv(ticker, timeframe, start, end)
    tf = Timeframe(df.attrs["timeframe"])
    signals = build_signals(df, strategy, params)
    res = run_backtest(df, signals, config, tf.periods_per_year)
    payload = clean({
        "meta": source_info(df),
        "strategy": strategy,
        "params": {**STRATEGIES[strategy].defaults, **(params or {})},
        "config": config.model_dump(),
        "statistics": res.stats,
        "benchmark": res.benchmark,
        "equity_curve": equity_records(res.equity),
        "trades": res.trades[-max(0, min(int(max_trades_returned), 500)):],
        "trades_total": len(res.trades),
        "notes": res.notes,
        "warnings": BACKTEST_WARNINGS,
        "disclaimer": RESEARCH_DISCLAIMER,
    })
    payload["run_id"] = _persist("backtest", payload)
    return payload


@safe_tool
def walk_forward_backtest(
    ticker: str,
    strategy: str = "sma_crossover",
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    train_bars: int = 504,
    validation_bars: int = 126,
    test_bars: int = 126,
    param_grid: dict[str, list[Any]] | None = None,
    fixed_params: dict[str, Any] | None = None,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    """Walk-forward optimisation: TRAIN (grid search by Sharpe) -> VALIDATION (pick among top 3) -> TEST
    (out-of-sample), then roll forward by test_bars. Defaults on daily bars: 2y train, 6m validation, 6m test.

    Returns per-fold results, stitched out-of-sample performance, stability, train-vs-test comparison and an
    overfitting assessment. param_grid defaults to the strategy's grid (see list_strategies); max 60 combinations.
    """
    config = config or BacktestConfig()
    if start is None and timeframe in ("1d", "1w"):
        start = "2010-01-01"
    df = load_ohlcv(ticker, timeframe, start, end)
    tf = Timeframe(df.attrs["timeframe"])
    out = walk_forward_rules(df, strategy, config, tf.periods_per_year, train_bars, validation_bars, test_bars,
                             param_grid, fixed_params)
    out["equity_curve_oos"] = equity_records(out["equity_curve_oos"])
    payload = clean({"meta": source_info(df), "strategy": strategy, "config": config.model_dump(), **out,
                     "warnings": BACKTEST_WARNINGS, "disclaimer": RESEARCH_DISCLAIMER})
    payload["run_id"] = _persist("walk_forward", payload)
    return payload


@safe_tool
def list_saved_runs(kind: str = "all_backtests", limit: int = 20) -> dict[str, Any]:
    """History stored in the database. kind: all_backtests, backtest, walk_forward or ml (ML experiments)."""
    from trading_mcp.db.repository import list_runs

    if kind not in {"all_backtests", "backtest", "walk_forward", "ml"}:
        raise ValueError("kind must be all_backtests, backtest, walk_forward or ml")
    return clean({"kind": kind, "runs": list_runs(kind, max(1, min(int(limit), 100)))})


@safe_tool
def get_backtest_run(run_id: int) -> dict[str, Any]:
    """Full stored result of a previous backtest or walk-forward run (by run_id)."""
    from trading_mcp.db.repository import get_run

    return get_run(int(run_id))


def register(mcp: MCPServer) -> None:
    mcp.add_tool(list_strategies, annotations=ToolAnnotations(readOnlyHint=True))
    mcp.add_tool(list_saved_runs, annotations=ToolAnnotations(readOnlyHint=True))
    mcp.add_tool(get_backtest_run, annotations=ToolAnnotations(readOnlyHint=True))
    for fn in (backtest_strategy, walk_forward_backtest):
        mcp.add_tool(fn, annotations=ANNOTATIONS)
