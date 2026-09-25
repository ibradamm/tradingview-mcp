"""Data access functions. Persistence failures are logged and never break a tool call."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy import delete, insert, select

from trading_mcp.db.models import BacktestRun, MLExperiment, PriceBar, ToolCall
from trading_mcp.db.session import session_scope

logger = logging.getLogger(__name__)


class BarStore:
    """Persistent OHLCV cache used by :class:`trading_mcp.data.factory.CachedProvider`."""

    def save_bars(self, symbol: str, timeframe: str, df: pd.DataFrame) -> None:
        if df.empty:
            return
        try:
            with session_scope() as s:
                s.execute(delete(PriceBar).where(
                    PriceBar.symbol == symbol, PriceBar.timeframe == timeframe,
                    PriceBar.ts >= df.index[0].to_pydatetime(), PriceBar.ts <= df.index[-1].to_pydatetime()))
                rows = df[["open", "high", "low", "close", "volume"]].reset_index(names="ts")
                rows["ts"] = rows["ts"].dt.to_pydatetime()
                rows = rows.assign(symbol=symbol, timeframe=timeframe, source=df.attrs.get("source"))
                s.execute(insert(PriceBar), rows.to_dict(orient="records"))  # executemany bulk insert
        except Exception:  # noqa: BLE001
            logger.exception("could not cache bars for %s %s", symbol, timeframe)

    def load_bars(self, symbol: str, timeframe: str, start: datetime | None, end: datetime | None) -> pd.DataFrame:
        with session_scope() as s:
            q = select(PriceBar).where(PriceBar.symbol == symbol, PriceBar.timeframe == timeframe)
            if start:
                q = q.where(PriceBar.ts >= start)
            if end:
                q = q.where(PriceBar.ts < end)
            rows = s.scalars(q.order_by(PriceBar.ts)).all()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame([{"timestamp": r.ts, "open": r.open, "high": r.high, "low": r.low, "close": r.close,
                            "volume": r.volume} for r in rows]).set_index("timestamp")
        df.index = pd.DatetimeIndex(df.index)
        df.index = df.index.tz_localize("UTC") if df.index.tz is None else df.index.tz_convert("UTC")
        df.attrs.update({"symbol": symbol, "source": f"{rows[-1].source} (database cache)"})
        return df


def _safe(fn):  # noqa: ANN001, ANN202 - tiny internal decorator
    def wrapper(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        try:
            return fn(*args, **kwargs)
        except Exception:  # noqa: BLE001
            logger.exception("persistence error in %s", fn.__name__)
            return None

    return wrapper


@_safe
def save_backtest(kind: str, payload: dict[str, Any]) -> int:
    meta = payload.get("meta") or {}
    with session_scope() as s:
        run = BacktestRun(kind=kind, ticker=meta.get("symbol"), timeframe=meta.get("timeframe"),
                          strategy=payload.get("strategy"), params=payload.get("params"),
                          statistics=payload.get("statistics") or payload.get("out_of_sample"), payload=payload)
        s.add(run)
        s.flush()
        return run.id


@_safe
def save_experiment(payload: dict[str, Any]) -> int:
    with session_scope() as s:
        exp = MLExperiment(model_id=payload["model_id"], ticker=payload.get("ticker"),
                           timeframe=payload.get("timeframe"), model_type=payload.get("model_type"),
                           task=payload.get("task"), horizon=payload.get("horizon"),
                           metrics=payload.get("metrics"), payload=payload)
        s.add(exp)
        s.flush()
        return exp.id


@_safe
def record_tool_call(tool: str, duration_ms: float, ok: bool, error: str | None = None) -> None:
    with session_scope() as s:
        s.add(ToolCall(tool=tool, duration_ms=duration_ms, ok=ok, error=(error or "")[:500] or None))


def list_runs(kind: str, limit: int = 20) -> list[dict[str, Any]]:
    with session_scope() as s:
        if kind == "ml":
            rows = s.scalars(select(MLExperiment).order_by(MLExperiment.id.desc()).limit(limit)).all()
            return [{"id": r.id, "created_at": r.created_at, "model_id": r.model_id, "ticker": r.ticker,
                     "timeframe": r.timeframe, "model_type": r.model_type, "task": r.task, "horizon": r.horizon,
                     "test_metrics": (r.metrics or {}).get("test")} for r in rows]
        q = select(BacktestRun).order_by(BacktestRun.id.desc()).limit(limit)
        if kind != "all_backtests":
            q = q.where(BacktestRun.kind == kind)
        rows = s.scalars(q).all()
        return [{"id": r.id, "created_at": r.created_at, "kind": r.kind, "ticker": r.ticker,
                 "timeframe": r.timeframe, "strategy": r.strategy, "params": r.params,
                 "statistics": r.statistics} for r in rows]


def get_run(run_id: int) -> dict[str, Any]:
    with session_scope() as s:
        run = s.get(BacktestRun, run_id)
        if run is None:
            raise KeyError(f"No backtest run with id {run_id}")
        return run.payload
