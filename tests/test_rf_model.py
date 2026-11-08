"""Tests for src.prediction.rf_model."""

from __future__ import annotations

import numpy as np
import pytest

from src.prediction.rf_model import HotspotRFModel


def _toy_data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 10)).astype(np.float32)
    y = (X[:, 0] > 0).astype(np.int64)
    return X, y


def test_fit_predict_save_load_round_trip(tmp_path):
    X, y = _toy_data()
    model = HotspotRFModel(n_estimators=10, random_state=0).fit(X, y)
    pred = model.predict(X)
    proba = model.predict_proba(X)

    path = tmp_path / "rf.joblib"
    model.save(str(path))
    loaded = HotspotRFModel.load(str(path))

    np.testing.assert_array_equal(loaded.predict(X), pred)
    np.testing.assert_allclose(loaded.predict_proba(X), proba)


def test_predict_proba_returns_positive_class_only():
    X, y = _toy_data()
    model = HotspotRFModel(n_estimators=10, random_state=0).fit(X, y)
    probs = model.predict_proba(X)
    assert probs.shape == (len(X),)
    assert (probs >= 0).all() and (probs <= 1).all()


def test_feature_importances_sum_to_one():
    X, y = _toy_data()
    model = HotspotRFModel(n_estimators=10, random_state=0).fit(X, y)
    names = [f"f{i}" for i in range(10)]
    imp = model.feature_importances(names)
    assert set(imp.keys()) == set(names)
    assert sum(imp.values()) == pytest.approx(1.0, rel=1e-5)
