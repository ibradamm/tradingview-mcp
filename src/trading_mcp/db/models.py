"""SQLAlchemy ORM models. Portable across SQLite and PostgreSQL (generic types only)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class PriceBar(Base):
    """Cached OHLCV bar."""

    __tablename__ = "price_bars"
    __table_args__ = (UniqueConstraint("symbol", "timeframe", "ts", name="uq_bar"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    symbol: Mapped[str] = mapped_column(String(40))
    timeframe: Mapped[str] = mapped_column(String(8))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    open: Mapped[float] = mapped_column(Float)
    high: Mapped[float] = mapped_column(Float)
    low: Mapped[float] = mapped_column(Float)
    close: Mapped[float] = mapped_column(Float)
    volume: Mapped[float] = mapped_column(Float)
    source: Mapped[str | None] = mapped_column(String(80))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    kind: Mapped[str] = mapped_column(String(20))  # backtest | walk_forward
    ticker: Mapped[str | None] = mapped_column(String(40))
    timeframe: Mapped[str | None] = mapped_column(String(8))
    strategy: Mapped[str | None] = mapped_column(String(40))
    params: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    statistics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class MLExperiment(Base):
    __tablename__ = "ml_experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
    model_id: Mapped[str] = mapped_column(String(12), index=True)
    ticker: Mapped[str | None] = mapped_column(String(40))
    timeframe: Mapped[str | None] = mapped_column(String(8))
    model_type: Mapped[str | None] = mapped_column(String(40))
    task: Mapped[str | None] = mapped_column(String(20))
    horizon: Mapped[int | None] = mapped_column(Integer)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSON)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ToolCall(Base):
    """Audit log of tool invocations (no arguments stored, to avoid logging sensitive inputs)."""

    __tablename__ = "tool_calls"
    __table_args__ = (Index("ix_tool_calls_ts", "ts"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    tool: Mapped[str] = mapped_column(String(60))
    duration_ms: Mapped[float] = mapped_column(Float)
    ok: Mapped[bool]
    error: Mapped[str | None] = mapped_column(Text)
