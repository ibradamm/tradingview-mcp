"""Machine-learning MCP tools. Outputs are probabilistic research estimates, never certainties."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer
from mcp_types import ToolAnnotations

from trading_mcp.backtest.engine import BacktestConfig
from trading_mcp.data.base import Timeframe
from trading_mcp.ml import store
from trading_mcp.ml.features import FEATURES, build_dataset, build_features
from trading_mcp.ml.models import available_models
from trading_mcp.ml.training import score, train_and_evaluate, walk_forward_ml
from trading_mcp.tools.common import (
    ML_WARNINGS,
    RESEARCH_DISCLAIMER,
    clean,
    equity_records,
    frame_to_records,
    load_ohlcv,
    safe_tool,
    source_info,
)

WRITE = ToolAnnotations(readOnlyHint=False, destructiveHint=False, openWorldHint=True)
READ = ToolAnnotations(readOnlyHint=True, openWorldHint=True)
Task = Literal["classification", "regression"]


def _default_start(timeframe: str, start: str | None) -> str | None:
    return start or ("2012-01-01" if timeframe in ("1d", "1w") else None)


def _record_experiment(payload: dict[str, Any]) -> None:
    try:
        from trading_mcp.db.repository import save_experiment
    except ImportError:
        return
    save_experiment(payload)


@safe_tool
def create_features(
    ticker: str, timeframe: str = "1d", features: list[str] | None = None, start: str | None = None,
    end: str | None = None, tail: int = 10,
) -> dict[str, Any]:
    """Build the ML feature matrix (all features use only past data). Lists available features and models,
    per-feature summary statistics and the last ``tail`` rows."""
    df = load_ohlcv(ticker, timeframe, start, end)
    X = build_features(df, features, Timeframe(df.attrs["timeframe"]).is_intraday)
    complete = X.dropna()
    return clean({
        "meta": source_info(df), "available_features": sorted(FEATURES), "available_models": available_models(),
        "features": list(X.columns), "rows_total": len(X), "rows_complete": len(complete),
        "summary": complete.describe().T[["mean", "std", "min", "max"]].to_dict(orient="index"),
        "last_rows": frame_to_records(complete, max(1, min(int(tail), 200))),
    })


@safe_tool
def train_model(
    ticker: str,
    model_type: str = "logistic_regression",
    task: Task = "classification",
    horizon: int = 1,
    timeframe: str = "1d",
    start: str | None = None,
    end: str | None = None,
    features: list[str] | None = None,
    train_fraction: float = 0.6,
    validation_fraction: float = 0.2,
    save: bool = True,
) -> dict[str, Any]:
    """Train a model to estimate the NEXT ``horizon`` bars' move (classification: P(return > 0);
    regression: future return).
    model_type: logistic_regression, random_forest, gradient_boosting (plus ridge for regression).

    Horizon is in bars of ``timeframe`` (e.g. timeframe=5m horizon=3 -> 15 minutes; timeframe=1h horizon=1 -> 1 hour).
    Chronological TRAIN/VALIDATION/TEST split with a purge gap of ``horizon`` bars, TimeSeriesSplit CV on train,
    no random shuffling. Returns metrics per split, a cost-inclusive trading simulation on TEST and overfitting flags.
    """
    start = _default_start(timeframe, start)
    df = load_ohlcv(ticker, timeframe, start, end)
    tf = Timeframe(df.attrs["timeframe"])
    res = train_and_evaluate(df, model_type, task, horizon, features, tf.is_intraday, train_fraction,
                             validation_fraction, periods_per_year=tf.periods_per_year)
    model_id = store.new_model_id()
    test_metric = res["overfitting_check"]["test"]
    metadata = clean({
        "ticker": ticker.upper(), "symbol": df.attrs.get("symbol"), "timeframe": tf.value, "model_type": model_type,
        "task": task, "horizon": horizon, "features": res["features"], "created_at": datetime.now(UTC),
        "training_end": res["training_end"], "splits": res["splits"], "test_metric": test_metric,
        "data_source": df.attrs.get("source"),
    })
    store.remember(model_id, res["model"], metadata)
    if save:
        store.save(model_id)
    payload = clean({
        "model_id": model_id, "saved": save, **{k: metadata[k] for k in ("ticker", "timeframe", "model_type", "task",
                                                                          "horizon")},
        "horizon_description": f"{horizon} bar(s) of {tf.value}",
        "splits": res["splits"], "total_observations": res["total_observations"],
        "cross_validation": res["cross_validation"], "metrics": res["metrics"],
        "test_trading_simulation": res["test_trading_simulation"],
        "overfitting_check": res["overfitting_check"], "feature_importance": res["feature_importance"],
        "warnings": ML_WARNINGS, "disclaimer": RESEARCH_DISCLAIMER,
    })
    _record_experiment(payload)
    return payload


@safe_tool
def evaluate_model(
    model_id: str,
    method: Literal["holdout", "walk_forward"] = "holdout",
    ticker: str | None = None,
    start: str | None = None,
    end: str | None = None,
    train_bars: int = 750,
    test_bars: int = 125,
) -> dict[str, Any]:
    """Evaluate a trained model.

    method="holdout": score the saved model on data AFTER its training period (default) or on [start, end]
    for ``ticker`` (defaults to the training ticker). Periods overlapping training are flagged as in-sample.
    method="walk_forward": retrain the same model type/features on rolling windows (train_bars) and test on the
    following test_bars, rolling forward; returns per-fold metrics, out-of-sample trading stats and stability.
    """
    stored = store.get(model_id)
    meta = stored.metadata
    tk, tf = ticker or meta["ticker"], Timeframe(meta["timeframe"])
    if method == "walk_forward":
        df = load_ohlcv(tk, tf.value, _default_start(tf.value, start), end)
        out = walk_forward_ml(df, meta["model_type"], meta["task"], int(meta["horizon"]), meta["features"],
                              tf.is_intraday, train_bars, test_bars, tf.periods_per_year, BacktestConfig())
        out["equity_curve_oos"] = equity_records(out["equity_curve_oos"])
        return clean({"model_id": model_id, "method": method, "ticker": tk, "meta": source_info(df), **out,
                      "warnings": ML_WARNINGS, "disclaimer": RESEARCH_DISCLAIMER})

    training_end = meta.get("training_end")
    df = load_ohlcv(tk, tf.value, start or training_end, end)
    X, y = build_dataset(df, meta["features"], int(meta["horizon"]), meta["task"], tf.is_intraday)
    if len(X) < 20:
        raise ValueError("Too few labelled rows in the evaluation window (need >= 20).")
    in_sample = bool(training_end and tk.upper() == meta["ticker"] and str(X.index[0]) <= str(training_end))
    return clean({
        "model_id": model_id, "method": method, "ticker": tk, "meta": source_info(df),
        "evaluation_window": {"start": X.index[0], "end": X.index[-1], "rows": len(X)},
        "overlaps_training_period": in_sample,
        "metrics": score(stored.model, X, y, meta["task"]),
        "note": "IN-SAMPLE: scores are optimistic." if in_sample else "Out-of-sample relative to training.",
        "warnings": ML_WARNINGS,
    })


@safe_tool
def predict(model_id: str, ticker: str | None = None) -> dict[str, Any]:
    """Apply a trained model to the most recent completed bar. Returns a probability (classification) or an
    expected return (regression) for the next ``horizon`` bars, with the model's historical test metrics for
    context. This is a statistical estimate, NOT a forecast to act on."""
    stored = store.get(model_id)
    meta = stored.metadata
    tk, tf = ticker or meta["ticker"], Timeframe(meta["timeframe"])
    df = load_ohlcv(tk, tf.value)
    X = build_features(df, meta["features"], tf.is_intraday).dropna()
    if X.empty:
        raise ValueError("Not enough recent data to compute features.")
    row = X.iloc[[-1]]
    out: dict[str, Any] = {
        "model_id": model_id, "ticker": tk, "timeframe": tf.value, "as_of_bar": row.index[0],
        "horizon": f"next {meta['horizon']} bar(s) of {tf.value}", "task": meta["task"],
        "model_test_metric": meta.get("test_metric"),
        "observations_used_for_training": (meta.get("splits") or {}).get("train", {}).get("rows"),
    }
    if meta["task"] == "classification":
        p = float(stored.model.predict_proba(row)[0, 1])
        out.update({"probability_up": p, "probability_down": 1 - p,
                    "confidence_note": _confidence_note(p, meta.get("test_metric"))})
    else:
        out["predicted_return"] = float(stored.model.predict(row)[0])
    out.update({"warnings": ML_WARNINGS, "disclaimer": RESEARCH_DISCLAIMER,
                "reminder": "Prediction != outcome. The last bar may still be forming for live intraday data."})
    return clean(out)


def _confidence_note(p: float, test_auc: float | None) -> str:
    edge = abs(p - 0.5)
    auc = f"test ROC-AUC {test_auc:.3f}" if test_auc is not None else "unknown test ROC-AUC"
    level = "weak" if edge < 0.05 else "moderate" if edge < 0.15 else "strong-looking"
    return (f"Probability is {level} ({p:.1%}); the model's {auc}. Probabilities are only as reliable as the "
            "calibration table from training shows; an AUC near 0.5 means no demonstrated skill.")


@safe_tool
def save_model(model_id: str, name: str | None = None) -> dict[str, Any]:
    """Persist a trained model (and optional human-readable name) to the model store."""
    return clean(store.save(model_id, name))


@safe_tool
def load_model(model_id: str) -> dict[str, Any]:
    """Load a saved model into memory and return its metadata (features, horizon, splits, test metric)."""
    return clean(store.get(model_id).metadata)


@safe_tool
def list_models() -> dict[str, Any]:
    """List trained models (id, ticker, timeframe, type, horizon, test metric)."""
    return clean({"models": store.list_models(), "available_model_types": available_models()})


def register(mcp: MCPServer) -> None:
    for fn in (create_features, predict, load_model, list_models):
        mcp.add_tool(fn, annotations=READ)
    for fn in (train_model, evaluate_model, save_model):
        mcp.add_tool(fn, annotations=WRITE)
