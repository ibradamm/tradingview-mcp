"""Model persistence (joblib + JSON metadata) with an in-process cache.

Security: joblib uses pickle, which can execute code when loading an untrusted file. Only files
written by this server into ``MODEL_DIR`` are ever loaded, and model ids are validated so a
caller cannot point the loader at another path.
"""

from __future__ import annotations

import json
import re
import threading
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from trading_mcp.config import get_settings

_ID = re.compile(r"^[a-f0-9]{12}$")
_cache: dict[str, StoredModel] = {}
_lock = threading.Lock()


@dataclass
class StoredModel:
    model_id: str
    model: Pipeline
    metadata: dict[str, Any]


def new_model_id() -> str:
    return uuid.uuid4().hex[:12]


def _paths(model_id: str) -> tuple[Path, Path]:
    if not _ID.match(model_id):
        raise ValueError(f"Invalid model_id '{model_id}'.")
    root = Path(get_settings().model_dir)
    return root / f"{model_id}.joblib", root / f"{model_id}.json"


def remember(model_id: str, model: Pipeline, metadata: dict[str, Any]) -> StoredModel:
    """Keep a trained model in memory (not yet on disk)."""
    stored = StoredModel(model_id, model, {**metadata, "model_id": model_id, "saved": False})
    with _lock:
        _cache[model_id] = stored
    return stored


def save(model_id: str, name: str | None = None) -> dict[str, Any]:
    """Persist a model held in memory to MODEL_DIR."""
    stored = get(model_id)
    model_path, meta_path = _paths(model_id)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    stored.metadata.update({"saved": True, "saved_at": datetime.now(UTC).isoformat()})
    if name:
        stored.metadata["name"] = name
    joblib.dump(stored.model, model_path)
    meta_path.write_text(json.dumps(stored.metadata, default=str, indent=2))
    return stored.metadata


def get(model_id: str) -> StoredModel:
    """Return a model from memory, loading it from disk if necessary."""
    with _lock:
        if model_id in _cache:
            return _cache[model_id]
    model_path, meta_path = _paths(model_id)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Model '{model_id}' not found. Train one with train_model or list_models.")
    stored = StoredModel(model_id, joblib.load(model_path), json.loads(meta_path.read_text()))
    with _lock:
        _cache[model_id] = stored
    return stored


def list_models() -> list[dict[str, Any]]:
    """Metadata of models on disk and in memory (most recent first)."""
    root = Path(get_settings().model_dir)
    found: dict[str, dict[str, Any]] = {}
    if root.exists():
        for meta in root.glob("*.json"):
            try:
                found[meta.stem] = json.loads(meta.read_text())
            except (OSError, json.JSONDecodeError):
                continue
    with _lock:
        for mid, stored in _cache.items():
            found.setdefault(mid, stored.metadata)
    keys = ("model_id", "name", "ticker", "timeframe", "model_type", "task", "horizon", "created_at", "saved",
            "test_metric")
    rows = [{k: m.get(k) for k in keys} for m in found.values()]
    return sorted(rows, key=lambda r: str(r.get("created_at")), reverse=True)


def clear_cache() -> None:
    with _lock:
        _cache.clear()
