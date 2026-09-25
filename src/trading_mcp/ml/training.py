"""Time-ordered training, evaluation and walk-forward validation for ML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline

from trading_mcp.backtest.engine import BacktestConfig, run_backtest
from trading_mcp.backtest.metrics import performance_stats
from trading_mcp.backtest.walk_forward import make_folds, summarize
from trading_mcp.ml.features import build_dataset
from trading_mcp.ml.models import create_model

MIN_TRAIN_ROWS = 100


@dataclass(frozen=True)
class SplitIndex:
    train: pd.Index
    validation: pd.Index
    test: pd.Index


def chronological_split(index: pd.Index, train_frac: float, val_frac: float, gap: int) -> SplitIndex:
    """Split ordered rows into train / validation / test with ``gap`` rows purged between them.

    The gap (= prediction horizon) removes rows whose forward-looking label would overlap the next set.
    """
    if not (0 < train_frac < 1 and 0 <= val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError("Fractions must satisfy 0 < train, 0 <= validation, train + validation < 1.")
    n = len(index)
    tr_end = int(n * train_frac)
    va_start, va_end = tr_end + gap, int(n * (train_frac + val_frac))
    te_start = va_end + gap if val_frac > 0 else tr_end + gap
    split = SplitIndex(index[:tr_end], index[va_start:va_end] if val_frac > 0 else index[:0], index[te_start:])
    if len(split.train) < MIN_TRAIN_ROWS or len(split.test) < 20:
        raise ValueError(f"Not enough rows ({n}) for a meaningful split. Use a longer history or a shorter timeframe.")
    return split


def classification_metrics(y: pd.Series, proba: np.ndarray) -> dict[str, Any]:
    pred = (proba >= 0.5).astype(int)
    both = len(np.unique(y)) > 1
    base_rate = float(y.mean())
    return {
        "observations": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if both else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)) if both else None,
        "log_loss": float(log_loss(y, np.clip(proba, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, proba)),
        "positive_rate": base_rate,
        "baseline_accuracy_majority_class": max(base_rate, 1 - base_rate),
        "calibration": calibration_table(y, proba),
    }


def calibration_table(y: pd.Series, proba: np.ndarray, bins: int = 5) -> list[dict[str, Any]]:
    """Mean predicted probability vs observed frequency per probability bucket."""
    edges = np.linspace(0, 1, bins + 1)
    table = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (proba >= lo) & ((proba < hi) if hi < 1 else (proba <= hi))
        if mask.sum():
            table.append({"bucket": f"{lo:.1f}-{hi:.1f}", "count": int(mask.sum()),
                          "mean_predicted": float(proba[mask].mean()),
                          "observed_frequency": float(np.asarray(y)[mask].mean())})
    return table


def regression_metrics(y: pd.Series, pred: np.ndarray) -> dict[str, Any]:
    ic = float(np.corrcoef(y, pred)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else None
    return {
        "observations": int(len(y)),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "r2": float(r2_score(y, pred)),
        "directional_accuracy": float(np.mean(np.sign(pred) == np.sign(y))),
        "information_coefficient": ic,
        "baseline_mae_zero_forecast": float(np.mean(np.abs(y))),
    }


def score(model: Pipeline, X: pd.DataFrame, y: pd.Series, task: str) -> dict[str, Any]:
    if task == "classification":
        return classification_metrics(y, model.predict_proba(X)[:, 1])
    return regression_metrics(y, model.predict(X))


def model_signal(model: Pipeline, X: pd.DataFrame, task: str, threshold: float = 0.5) -> pd.Series:
    """Long (1) when P(up) >= threshold (classification) or predicted return > 0 (regression), else flat."""
    if task == "classification":
        return pd.Series((model.predict_proba(X)[:, 1] >= threshold).astype(float), index=X.index)
    return pd.Series((model.predict(X) > 0).astype(float), index=X.index)


def train_and_evaluate(
    df: pd.DataFrame,
    model_type: str,
    task: str,
    horizon: int,
    features: list[str] | None,
    intraday: bool,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
    cv_splits: int = 5,
    periods_per_year: float = 252,
) -> dict[str, Any]:
    """Fit on TRAIN, report CV (TimeSeriesSplit on train), VALIDATION and held-out TEST metrics."""
    X, y = build_dataset(df, features, horizon, task, intraday)
    split = chronological_split(X.index, train_frac, val_frac, gap=horizon)
    Xtr, ytr = X.loc[split.train], y.loc[split.train]
    if task == "classification" and ytr.nunique() < 2:
        raise ValueError("Training labels contain a single class; use a longer period.")

    cv = TimeSeriesSplit(n_splits=cv_splits, gap=horizon)
    cv_scores = []
    for tr_i, va_i in cv.split(Xtr):
        m = create_model(model_type, task).fit(Xtr.iloc[tr_i], ytr.iloc[tr_i])
        s = score(m, Xtr.iloc[va_i], ytr.iloc[va_i], task)
        cv_scores.append(s.get("roc_auc") if task == "classification" else s.get("information_coefficient"))

    model = create_model(model_type, task).fit(Xtr, ytr)
    result: dict[str, Any] = {
        "model": model,
        "features": list(X.columns),
        "splits": {name: _span(idx) for name, idx in (("train", split.train), ("validation", split.validation),
                                                        ("test", split.test))},
        "cross_validation": {"method": f"TimeSeriesSplit(n_splits={cv_splits}, gap={horizon}) on train",
                             "metric": "roc_auc" if task == "classification" else "information_coefficient",
                             "scores": cv_scores,
                             "mean": _nanmean(cv_scores), "std": _nanstd(cv_scores)},
        "metrics": {"train": score(model, Xtr, ytr, task)},
        "feature_importance": feature_importance(model, list(X.columns)),
        "total_observations": len(X),
        "training_end": split.train[-1],
    }
    for name, idx in (("validation", split.validation), ("test", split.test)):
        if len(idx):
            result["metrics"][name] = score(model, X.loc[idx], y.loc[idx], task)
    # Trading view of the TEST set: go long on positive predictions, costs included.
    test_df = df.loc[split.test[0]: split.test[-1]]
    sig = model_signal(model, X.loc[split.test], task).reindex(test_df.index).fillna(0.0)
    bt = run_backtest(test_df, sig, BacktestConfig(), periods_per_year)
    result["test_trading_simulation"] = {k: bt.stats.get(k) for k in (
        "total_return", "sharpe_ratio", "max_drawdown", "number_of_trades", "win_rate", "exposure")}
    result["test_trading_simulation"]["buy_and_hold_total_return"] = bt.benchmark["buy_and_hold_total_return"]
    result["overfitting_check"] = overfitting_check(result["metrics"], task)
    return result


def overfitting_check(metrics: dict[str, Any], task: str) -> dict[str, Any]:
    key = "roc_auc" if task == "classification" else "information_coefficient"
    tr, te = metrics.get("train", {}).get(key), metrics.get("test", {}).get(key)
    gap = (tr - te) if tr is not None and te is not None else None
    flags = []
    if gap is not None and gap > 0.1:
        flags.append(f"{key} falls from {tr:.3f} (train) to {te:.3f} (test): the model memorised training noise.")
    if task == "classification" and te is not None and te < 0.55:
        flags.append(f"Test ROC-AUC {te:.3f} is close to 0.5 (coin flip): no reliable predictive signal.")
    if task == "regression" and te is not None and abs(te) < 0.05:
        flags.append(f"Test information coefficient {te:.3f} is ~0: no reliable predictive signal.")
    return {"metric": key, "train": tr, "test": te, "gap": gap, "flags": flags}


def feature_importance(model: Pipeline, names: list[str]) -> dict[str, float] | None:
    est = model.named_steps["model"]
    if hasattr(est, "feature_importances_"):
        values = est.feature_importances_
    elif hasattr(est, "coef_"):
        values = np.abs(np.ravel(est.coef_))
    else:
        return None
    ranked = sorted(zip(names, map(float, values), strict=True), key=lambda kv: kv[1], reverse=True)
    return dict(ranked)


def walk_forward_ml(
    df: pd.DataFrame,
    model_type: str,
    task: str,
    horizon: int,
    features: list[str] | None,
    intraday: bool,
    train_bars: int,
    test_bars: int,
    periods_per_year: float,
    config: BacktestConfig | None = None,
) -> dict[str, Any]:
    """Retrain on each rolling TRAIN window, predict the next TEST window, roll forward.

    A purge of ``horizon`` rows at the end of each training window prevents label overlap with the test window.
    """
    config = config or BacktestConfig()
    X, y = build_dataset(df, features, horizon, task, intraday)
    base = create_model(model_type, task)
    folds, oos_parts, metric_rows = [], [], []
    for fold in make_folds(len(X), train_bars, 0, test_bars):
        tr_idx = X.index[fold.train][:-horizon] if horizon < len(X.index[fold.train]) else X.index[fold.train]
        te_idx = X.index[fold.test]
        if task == "classification" and y.loc[tr_idx].nunique() < 2:
            continue
        model = clone(base).fit(X.loc[tr_idx], y.loc[tr_idx])
        train_s = score(model, X.loc[tr_idx], y.loc[tr_idx], task)
        test_s = score(model, X.loc[te_idx], y.loc[te_idx], task)
        seg = df.loc[te_idx[0]: te_idx[-1]]
        sig = model_signal(model, X.loc[te_idx], task).reindex(seg.index).fillna(0.0)
        bt = run_backtest(seg, sig, config, periods_per_year)
        tr_bt = run_backtest(df.loc[tr_idx[0]: tr_idx[-1]],
                             model_signal(model, X.loc[tr_idx], task).reindex(df.loc[tr_idx[0]: tr_idx[-1]].index)
                             .fillna(0.0), config, periods_per_year)
        oos_parts.append(bt.equity.pct_change().fillna(bt.equity.iloc[0] / config.initial_capital - 1))
        key = "roc_auc" if task == "classification" else "information_coefficient"
        metric_rows.append((train_s.get(key), test_s.get(key)))
        folds.append({
            "fold": fold.number,
            "train_period": {"start": tr_idx[0], "end": tr_idx[-1], "rows": len(tr_idx)},
            "test_period": {"start": te_idx[0], "end": te_idx[-1], "rows": len(te_idx)},
            "train_metric": train_s.get(key), "test_metric": test_s.get(key),
            "test_accuracy": test_s.get("accuracy", test_s.get("directional_accuracy")),
            "train": _brief(tr_bt.stats), "test": _brief(bt.stats),
        })
    if not folds:
        raise ValueError("No usable fold (not enough data or single-class labels).")
    summary = summarize(folds, pd.concat(oos_parts), periods_per_year, config.initial_capital)
    summary["model_metric"] = {
        "name": "roc_auc" if task == "classification" else "information_coefficient",
        "mean_train": _nanmean([a for a, _ in metric_rows]), "mean_test": _nanmean([b for _, b in metric_rows]),
    }
    summary["out_of_sample_buy_and_hold"] = performance_stats(
        df["close"].loc[folds[0]["test_period"]["start"]:folds[-1]["test_period"]["end"]], periods_per_year)
    return {"folds": folds, **summary}


def _brief(stats: dict[str, Any]) -> dict[str, Any]:
    return {k: stats.get(k) for k in ("total_return", "sharpe_ratio", "max_drawdown", "number_of_trades", "win_rate")}


def _span(idx: pd.Index) -> dict[str, Any] | None:
    return {"start": idx[0], "end": idx[-1], "rows": len(idx)} if len(idx) else None


def _nanmean(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def _nanstd(values: list[float | None]) -> float | None:
    v = [x for x in values if x is not None and np.isfinite(x)]
    return float(np.std(v, ddof=1)) if len(v) > 1 else None
