import numpy as np
import pandas as pd
import pytest
from mcp import Client

from trading_mcp.ml import store
from trading_mcp.ml.features import DEFAULT_FEATURES, build_dataset, build_features, make_target
from trading_mcp.ml.models import available_models, create_model, register_model
from trading_mcp.ml.training import chronological_split, train_and_evaluate, walk_forward_ml
from trading_mcp.server import create_mcp


def _predictable(n=1500, seed=0):
    """Returns with strong positive autocorrelation: tomorrow's sign usually equals today's."""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    for i in range(1, n):
        r[i] = 0.8 * r[i - 1] + rng.normal(0, 0.01)
    close = 100 * np.exp(np.cumsum(r))
    idx = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": np.r_[close[0], close[:-1]], "high": close * 1.005, "low": close * 0.995,
                         "close": close, "volume": rng.integers(1e5, 2e5, n).astype(float)}, index=idx)


def test_features_are_causal(ohlcv):
    full = build_features(ohlcv)
    cut = build_features(ohlcv.iloc[:400])
    pd.testing.assert_frame_equal(full.iloc[:400], cut)
    assert set(DEFAULT_FEATURES) == set(full.columns)
    with pytest.raises(ValueError):
        build_features(ohlcv, ["nope"])


def test_target_looks_forward_only():
    close = pd.Series([100.0, 110, 99, 120])
    y = make_target(close, 1, "regression")
    assert y.iloc[0] == pytest.approx(0.1) and np.isnan(y.iloc[-1])
    assert make_target(close, 2, "classification").tolist()[:2] == [0.0, 1.0]


def test_dataset_drops_unknown_future(ohlcv):
    X, y = build_dataset(ohlcv, None, 5, "classification", False)
    assert X.index[-1] <= ohlcv.index[-6]
    assert not X.isna().any().any() and not y.isna().any()


def test_chronological_split_with_purge_gap():
    idx = pd.RangeIndex(1000)
    s = chronological_split(idx, 0.6, 0.2, gap=5)
    assert s.train[-1] < s.validation[0] - 4 and s.validation[-1] < s.test[0] - 4
    assert s.train[-1] == 599 and s.validation[0] == 605 and s.test[0] == 805


@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest", "gradient_boosting"])
def test_models_learn_a_real_signal(model_type):
    res = train_and_evaluate(_predictable(), model_type, "classification", 1, ["return_1", "return_5"], False)
    assert res["metrics"]["test"]["roc_auc"] > 0.75
    assert res["metrics"]["test"]["observations"] > 100


def test_random_walk_is_flagged_as_no_skill(ohlcv):
    res = train_and_evaluate(ohlcv, "logistic_regression", "classification", 1, None, False)
    assert 0.35 < res["metrics"]["test"]["roc_auc"] < 0.65
    assert res["overfitting_check"]["flags"]


def test_regression_task():
    res = train_and_evaluate(_predictable(), "ridge", "regression", 1, ["return_1"], False)
    assert res["metrics"]["test"]["information_coefficient"] > 0.5
    assert res["metrics"]["test"]["directional_accuracy"] > 0.6


def test_walk_forward_ml():
    out = walk_forward_ml(_predictable(), "logistic_regression", "classification", 1, ["return_1"], False,
                          train_bars=400, test_bars=200, periods_per_year=252)
    assert len(out["folds"]) >= 4
    assert out["model_metric"]["mean_test"] > 0.75
    for f in out["folds"]:
        assert f["train_period"]["end"] < f["test_period"]["start"]


def test_registry_is_extensible():
    from sklearn.dummy import DummyClassifier

    register_model("dummy", "classification", lambda: DummyClassifier())
    assert "dummy" in available_models()
    assert create_model("dummy", "classification") is not None
    with pytest.raises(ValueError):
        create_model("does_not_exist", "classification")


def test_store_roundtrip_and_id_validation():
    model = create_model("logistic_regression", "classification")
    mid = store.new_model_id()
    store.remember(mid, model, {"ticker": "X", "created_at": "2024"})
    store.save(mid, name="demo")
    store.clear_cache()
    loaded = store.get(mid)
    assert loaded.metadata["name"] == "demo" and loaded.metadata["saved"] is True
    assert any(m["model_id"] == mid for m in store.list_models())
    with pytest.raises(ValueError):
        store.get("../../etc/passwd")
    with pytest.raises(FileNotFoundError):
        store.get("0" * 12)


async def test_ml_tools_via_mcp():
    async with Client(create_mcp()) as client:
        r = await client.call_tool("create_features", {"ticker": "NVDA", "tail": 3})
        assert not r.is_error, r.content
        assert len(r.structured_content["last_rows"]) == 3

        r = await client.call_tool("train_model", {"ticker": "NVDA", "model_type": "random_forest", "horizon": 5})
        assert not r.is_error, r.content
        d = r.structured_content
        mid = d["model_id"]
        assert {"train", "validation", "test"} <= set(d["metrics"])
        assert d["warnings"] and d["overfitting_check"]["metric"] == "roc_auc"

        r = await client.call_tool("predict", {"model_id": mid})
        assert not r.is_error, r.content
        assert 0 <= r.structured_content["probability_up"] <= 1

        r = await client.call_tool("evaluate_model", {"model_id": mid, "method": "walk_forward",
                                                       "train_bars": 500, "test_bars": 250})
        assert not r.is_error, r.content
        assert r.structured_content["folds"]

        r = await client.call_tool("evaluate_model", {"model_id": mid, "ticker": "AAPL", "start": "2020-01-01"})
        assert not r.is_error, r.content

        store.clear_cache()
        r = await client.call_tool("load_model", {"model_id": mid})
        assert r.structured_content["horizon"] == 5
        r = await client.call_tool("save_model", {"model_id": mid, "name": "nvda-rf"})
        assert r.structured_content["name"] == "nvda-rf"
        r = await client.call_tool("list_models", {})
        assert r.structured_content["models"][0]["model_id"] == mid

        r = await client.call_tool("train_model", {"ticker": "NVDA", "task": "regression", "model_type": "ridge"})
        assert not r.is_error, r.content
        r = await client.call_tool("predict", {"model_id": r.structured_content["model_id"]})
        assert "predicted_return" in r.structured_content

        bad = await client.call_tool("load_model", {"model_id": "nope"})
        assert bad.is_error
