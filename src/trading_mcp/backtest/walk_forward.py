"""Walk-forward evaluation: TRAIN -> VALIDATION -> TEST, then roll forward by the test length.

Only the TEST segments are out-of-sample. They are stitched together into one equity curve,
which is the honest estimate of the procedure (parameter search included).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from trading_mcp.backtest.engine import BacktestConfig, BacktestResult, run_backtest
from trading_mcp.backtest.metrics import performance_stats
from trading_mcp.backtest.strategies import STRATEGIES, build_signals

MAX_CANDIDATES = 60
TOP_K_FROM_TRAIN = 3


@dataclass(frozen=True)
class Fold:
    number: int
    train: slice
    validation: slice
    test: slice


def make_folds(n: int, train_bars: int, validation_bars: int, test_bars: int, max_folds: int = 20) -> list[Fold]:
    """Rolling (fixed-length) windows; each fold's test starts right after its validation."""
    if min(train_bars, test_bars) < 10 or validation_bars < 0:
        raise ValueError("train_bars and test_bars must be >= 10, validation_bars >= 0")
    folds, start = [], 0
    while start + train_bars + validation_bars + test_bars <= n:
        tr_end = start + train_bars
        va_end = tr_end + validation_bars
        test = slice(va_end, va_end + test_bars)
        folds.append(Fold(len(folds) + 1, slice(start, tr_end), slice(tr_end, va_end), test))
        start += test_bars
    if not folds:
        raise ValueError(
            f"Not enough data: {n} bars < train+validation+test = {train_bars + validation_bars + test_bars}."
        )
    return folds[-max_folds:]


def _score(stats: dict[str, Any]) -> float:
    s = stats.get("sharpe_ratio")
    return float(s) if s is not None and np.isfinite(s) else -np.inf


def summarize(folds: list[dict[str, Any]], oos_returns: pd.Series, periods_per_year: float,
              initial_capital: float) -> dict[str, Any]:
    """Aggregate fold results: stitched out-of-sample stats, stability and overfitting diagnostics."""
    oos_equity = initial_capital * (1 + oos_returns.fillna(0)).cumprod()
    oos_equity = pd.concat([pd.Series([initial_capital], index=[oos_equity.index[0] - pd.Timedelta(1, "s")]),
                            oos_equity])
    overall = performance_stats(oos_equity, periods_per_year)
    train_sh = [f["train"]["sharpe_ratio"] for f in folds if f["train"].get("sharpe_ratio") is not None]
    test_sh = [f["test"]["sharpe_ratio"] for f in folds if f["test"].get("sharpe_ratio") is not None]
    test_ret = [f["test"]["total_return"] for f in folds]
    mean_train, mean_test = (float(np.mean(x)) if x else None for x in (train_sh, test_sh))
    degradation = (mean_test / mean_train) if mean_train and mean_train > 0 and mean_test is not None else None

    flags = []
    if mean_train is not None and mean_test is not None and mean_train - mean_test > 0.5:
        flags.append(f"Average Sharpe drops from {mean_train:.2f} (train) to {mean_test:.2f} (test).")
    if degradation is not None and degradation < 0.5:
        flags.append("Test performance is less than half of train performance (degradation ratio < 0.5).")
    positive_share = float(np.mean([r > 0 for r in test_ret])) if test_ret else 0.0
    if positive_share < 0.5:
        flags.append("Fewer than half of the test periods were profitable.")
    if len(folds) < 4:
        flags.append("Few folds: the estimate is statistically weak.")

    return {
        "out_of_sample": overall,
        "stability": {
            "folds": len(folds),
            "share_of_profitable_test_periods": positive_share,
            "test_return_mean": float(np.mean(test_ret)) if test_ret else None,
            "test_return_std": float(np.std(test_ret, ddof=1)) if len(test_ret) > 1 else None,
        },
        "train_vs_test": {"mean_train_sharpe": mean_train, "mean_test_sharpe": mean_test,
                          "degradation_ratio": degradation},
        "overfitting_assessment": {
            "flags": flags,
            "verdict": "potential overfitting / unstable edge" if flags else "no strong overfitting signal detected "
                       "(this does not prove the edge is real)",
        },
        "equity_curve_oos": oos_equity,
    }


def walk_forward_rules(
    df: pd.DataFrame,
    strategy: str,
    config: BacktestConfig,
    periods_per_year: float,
    train_bars: int,
    validation_bars: int,
    test_bars: int,
    param_grid: dict[str, list[Any]] | None = None,
    fixed_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Optimise strategy parameters on TRAIN, pick among the top-K on VALIDATION, measure on TEST."""
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Available: {sorted(STRATEGIES)}")
    grid = param_grid if param_grid is not None else STRATEGIES[strategy].grid
    candidates = _expand(grid, fixed_params or {})
    # Signals are causal, so computing them once on the full history is identical to recomputing per window.
    signals: dict[int, pd.Series] = {}
    valid_candidates = []
    for i, params in enumerate(candidates):
        try:
            signals[i] = build_signals(df, strategy, params)
            valid_candidates.append((i, params))
        except ValueError:
            continue  # e.g. fast >= slow combinations
    if not valid_candidates:
        raise ValueError("No valid parameter combination in the grid.")

    def bt(i: int, sl: slice) -> BacktestResult:
        part = df.iloc[sl]
        return run_backtest(part, signals[i].iloc[sl], config, periods_per_year)

    fold_results, oos_parts = [], []
    for fold in make_folds(len(df), train_bars, validation_bars, test_bars):
        train_scores = [(i, p, bt(i, fold.train)) for i, p in valid_candidates]
        train_scores.sort(key=lambda x: _score(x[2].stats), reverse=True)
        shortlist = train_scores[:TOP_K_FROM_TRAIN]
        if validation_bars > 0:
            chosen = max(shortlist, key=lambda x: _score(bt(x[0], fold.validation).stats))
            val_stats = bt(chosen[0], fold.validation).stats
        else:
            chosen, val_stats = shortlist[0], None
        test_res = bt(chosen[0], fold.test)
        oos_parts.append(test_res.equity.pct_change().fillna(test_res.equity.iloc[0] / config.initial_capital - 1))
        fold_results.append({
            "fold": fold.number,
            "train_period": _period(df, fold.train), "validation_period": _period(df, fold.validation),
            "test_period": _period(df, fold.test),
            "selected_params": chosen[1],
            "train": _brief(chosen[2].stats), "validation": _brief(val_stats) if val_stats else None,
            "test": _brief(test_res.stats),
        })

    summary = summarize(fold_results, pd.concat(oos_parts), periods_per_year, config.initial_capital)
    param_changes = sum(
        1 for a, b in itertools.pairwise(fold_results) if a["selected_params"] != b["selected_params"]
    )
    summary["stability"]["parameter_changes_between_folds"] = param_changes
    return {"folds": fold_results, "candidates_tested": len(valid_candidates), **summary}


def _expand(grid: dict[str, list[Any]], fixed: dict[str, Any]) -> list[dict[str, Any]]:
    if not grid:
        return [dict(fixed)]
    keys = list(grid)
    combos = [dict(zip(keys, values, strict=True)) | fixed for values in itertools.product(*grid.values())]
    if len(combos) > MAX_CANDIDATES:
        raise ValueError(f"Parameter grid too large ({len(combos)} > {MAX_CANDIDATES} combinations).")
    return combos


def _period(df: pd.DataFrame, sl: slice) -> dict[str, Any] | None:
    part = df.index[sl]
    return {"start": part[0], "end": part[-1], "bars": len(part)} if len(part) else None


_BRIEF = ("total_return", "annualized_return", "sharpe_ratio", "max_drawdown", "number_of_trades", "win_rate")


def _brief(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: stats.get(k) for k in _BRIEF}

