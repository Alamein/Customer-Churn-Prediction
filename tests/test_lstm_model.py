"""Tests for src.prediction.lstm_model."""

from __future__ import annotations

import numpy as np

from src.prediction.lstm_model import HotspotLSTMWrapper, auto_device


def _toy_sequences(seq_len: int = 5, feat: int = 10, n: int = 40):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, seq_len, feat)).astype(np.float32)
    y = (X.mean(axis=(1, 2)) > 0).astype(np.int64)
    return X, y


def test_auto_device_returns_known_value():
    assert auto_device() in {"cuda", "mps", "cpu"}


def test_fit_predict_save_load_round_trip(tmp_path):
    X, y = _toy_sequences(seq_len=5)
    model = HotspotLSTMWrapper(
        input_dim=10, hidden_dim=16, num_layers=1, dropout=0.0,
        epochs=3, batch_size=8, sequence_length=5,
    ).fit(X, y)
    pred_before = model.predict(X)
    proba_before = model.predict_proba(X)

    path = tmp_path / "lstm.pt"
    model.save(str(path))
    loaded = HotspotLSTMWrapper.load(str(path))

    np.testing.assert_array_equal(loaded.predict(X), pred_before)
    np.testing.assert_allclose(loaded.predict_proba(X), proba_before, atol=1e-5)


def test_sequence_length_is_persisted(tmp_path):
    X, y = _toy_sequences(seq_len=7)
    model = HotspotLSTMWrapper(
        input_dim=10, hidden_dim=8, num_layers=1, dropout=0.0,
        epochs=1, batch_size=8, sequence_length=7,
    ).fit(X, y)
    path = tmp_path / "lstm.pt"
    model.save(str(path))
    loaded = HotspotLSTMWrapper.load(str(path))
    assert loaded.cfg["sequence_length"] == 7


def test_predict_proba_outputs_are_probabilities():
    X, y = _toy_sequences(seq_len=5)
    model = HotspotLSTMWrapper(
        input_dim=10, hidden_dim=8, num_layers=1, dropout=0.0,
        epochs=2, batch_size=8, sequence_length=5,
    ).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(X),)
    assert (proba >= 0).all() and (proba <= 1).all()
