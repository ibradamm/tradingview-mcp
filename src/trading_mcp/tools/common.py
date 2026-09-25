"""Helpers shared by all MCP tool modules: error mapping, parsing and JSON serialisation."""

from __future__ import annotations

import functools
import logging
import math
from collections.abc import Callable
from datetime import datetime
from typing import Any, ParamSpec, TypeVar

import numpy as np
import pandas as pd
from mcp.server.mcpserver.exceptions import ToolError

from trading_mcp.data.base import MarketDataError, Timeframe
from trading_mcp.data.factory import get_provider

logger = logging.getLogger(__name__)

P = ParamSpec("P")
R = TypeVar("R")

RESEARCH_DISCLAIMER = (
    "Research/education only - not investment advice. Past and simulated performance does not "
    "predict future results."
)
BACKTEST_WARNINGS = [
    "Backtests are simulations on historical data: results are optimistic by construction.",
    "Overfitting: parameters tuned on the same data inflate performance; check out-of-sample results.",
    "Look-ahead bias: signals here use only past bars and execute on the NEXT bar's open.",
    "Survivorship bias: the data source only lists symbols that still trade.",
    "Fees and slippage are modelled with simple fixed rates; real costs vary (spread, liquidity, market impact).",
    "Market regimes change: a strategy that worked historically can stop working.",
]
ML_WARNINGS = [
    "ML outputs are probabilistic estimates, never certainties. Financial returns are mostly noise.",
    "Data leakage is avoided with time-ordered splits and features built from past bars only; "
    "results remain sensitive to the chosen period.",
    "Overfitting risk: compare train vs test metrics; a large gap indicates memorisation.",
    "Regime change: models trained on one market regime can fail in another.",
]


def safe_tool(fn: Callable[P, R]) -> Callable[P, R]:
    """Turn anticipated failures into ``ToolError`` so the model sees a useful message."""

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except (ValueError, KeyError, MarketDataError, FileNotFoundError) as exc:
            logger.info("tool %s rejected: %s", fn.__name__, exc)
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


def parse_timeframe(value: str) -> Timeframe:
    aliases = {"1wk": "1w", "60m": "1h", "240m": "4h", "d": "1d", "w": "1w", "1min": "1m", "5min": "5m"}
    key = aliases.get(value.strip().lower(), value.strip().lower())
    try:
        return Timeframe(key)
    except ValueError as exc:
        raise ValueError(f"Unsupported timeframe '{value}'. Use one of {[t.value for t in Timeframe]}.") from exc


def parse_date(value: str | None) -> datetime | None:
    """Parse ISO dates like '2024-01-31' or '2024-01-31T14:30:00Z'."""
    if value is None or str(value).strip() == "":
        return None
    try:
        ts = pd.Timestamp(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid date '{value}'. Use ISO format, e.g. 2024-01-31.") from exc
    return ts.to_pydatetime()


def load_ohlcv(ticker: str, timeframe: str, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """Fetch OHLCV bars through the configured provider."""
    tf = parse_timeframe(timeframe)
    df = get_provider().get_history(ticker, tf, parse_date(start), parse_date(end))
    if df.empty:
        raise MarketDataError(f"No data for {ticker} {tf.value}.")
    df.attrs.setdefault("timeframe", tf.value)
    df.attrs["timeframe"] = tf.value
    return df


def clean(value: Any) -> Any:
    """Recursively convert numpy/pandas objects into JSON-safe Python values (NaN -> None)."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        f = float(value)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def frame_to_records(df: pd.DataFrame | pd.Series, limit: int | None = None) -> list[dict[str, Any]]:
    """Serialise the last ``limit`` rows of a time-indexed frame/series."""
    frame = df.to_frame() if isinstance(df, pd.Series) else df
    if limit is not None:
        frame = frame.tail(limit)
    records = frame.reset_index().to_dict(orient="records")
    return clean(records)


def source_info(df: pd.DataFrame) -> dict[str, Any]:
    return clean(
        {
            "symbol": df.attrs.get("symbol"),
            "source": df.attrs.get("source"),
            "timeframe": df.attrs.get("timeframe"),
            "bars": len(df),
            "first_bar": df.index[0] if len(df) else None,
            "last_bar": df.index[-1] if len(df) else None,
            "notes": df.attrs.get("notes", []),
        }
    )
