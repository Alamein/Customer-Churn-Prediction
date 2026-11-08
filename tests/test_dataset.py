"""Tests for src.prediction.dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.prediction.dataset import (
    DEFAULT_FEATURES,
    build_sequences,
    build_tabular,
    derive_proxy_label,
    partition_sequences,
    split_train_test,
    time_bucket_cutoff,
)


def test_derive_proxy_label_within_expected_quantile_range(synthetic_features):
    label = derive_proxy_label(synthetic_features, threshold_quantile=0.7)
    assert label.dtype.kind == "i"
    assert set(label.unique()).issubset({0, 1})
    # 30 % expected; allow slack for ties at the quantile boundary.
    assert 0.20 <= label.mean() <= 0.40


def test_split_train_test_is_chronological(synthetic_features):
    train, test = split_train_test(synthetic_features, test_size=0.2)
    assert train["time_bucket"].max() <= test["time_bucket"].min()
    assert len(train) + len(test) == len(synthetic_features)


def test_build_tabular_shapes_and_dtypes(synthetic_features):
    synthetic_features["label"] = (synthetic_features.index % 2).astype(int)
    X, y = build_tabular(synthetic_features)
    assert X.shape == (len(synthetic_features), len(DEFAULT_FEATURES))
    assert y.shape == (len(synthetic_features),)
    assert X.dtype == np.float32
    assert y.dtype == np.int64


def test_build_sequences_returns_aligned_outputs(synthetic_features):
    synthetic_features["label"] = (synthetic_features.index % 2).astype(int)
    X, y, target_meta = build_sequences(synthetic_features, seq_len=5)
    assert X.ndim == 3
    assert X.shape[1] == 5
    assert X.shape[2] == len(DEFAULT_FEATURES)
    assert y.shape == (X.shape[0],)
    assert len(target_meta) == X.shape[0]
    # Single-scene → target_meta has the three expected columns.
    assert set(target_meta.columns) == {"grid_x", "grid_y", "target_bucket"}
    # 25 cells × (20 − 5 + 1) windows = 400 sequences
    assert X.shape[0] == 25 * (20 - 5 + 1)


def _two_scene_features(n_buckets: int = 20) -> pd.DataFrame:
    """Two scenes (A, B) × 3 cells × n_buckets time buckets. Cells in A and B
    deliberately share (grid_x, grid_y) values so scene-conflation bugs are
    catchable.
    """
    rows = []
    for scene in ("A", "B"):
        base = 1 if scene == "A" else 2
        for gx, gy in [(0, 0), (1, 0), (1, 1)]:
            for t in range(n_buckets):
                rows.append(dict(
                    sequence=scene,
                    grid_x=gx, grid_y=gy, time_bucket=t,
                    vehicle_count=base * (t + 1),
                    average_speed=base * 5.0,
                    speed_variance=1.0,
                    traffic_density=0.001,
                    sudden_stops=base,
                    lane_changes=base,
                    acceleration_events=base,
                    time_of_day=0, day_of_week=0, weather_code=0,
                    label=(t % 3 == 0),
                ))
    return pd.DataFrame(rows)


def test_split_train_test_per_scene_chronological():
    df = _two_scene_features(n_buckets=10)
    train, test = split_train_test(df, test_size=0.3)
    # Each scene contributes its own last 30% to test.
    for scene in ("A", "B"):
        scene_train_buckets = train[train["sequence"] == scene]["time_bucket"].unique()
        scene_test_buckets = test[test["sequence"] == scene]["time_bucket"].unique()
        assert max(scene_train_buckets) < min(scene_test_buckets)
    # Both scenes are present in train and in test.
    assert set(train["sequence"].unique()) == {"A", "B"}
    assert set(test["sequence"].unique()) == {"A", "B"}


def test_build_sequences_keeps_scenes_separate():
    df = _two_scene_features(n_buckets=10)
    X, y, target_meta = build_sequences(df, seq_len=4)
    # Multi-scene target_meta carries `sequence`.
    assert "sequence" in target_meta.columns
    # Each scene has 3 cells × (10 − 4 + 1) = 21 windows. Total 42.
    assert len(X) == 2 * 3 * (10 - 4 + 1)
    # No window's metadata mixes scenes (it's drawn from one group only).
    assert set(target_meta["sequence"].unique()) == {"A", "B"}


def test_partition_sequences_routes_targets_to_correct_split():
    df = _two_scene_features(n_buckets=10)
    train, test = split_train_test(df, test_size=0.3)
    _, _, target_meta = build_sequences(df, seq_len=4)
    partitions = partition_sequences(target_meta, train, test)
    assert set(partitions).issubset({"train", "test", "unknown"})
    # Every window must land somewhere known.
    assert (partitions != "unknown").all()
    # Partition must agree with target_bucket being on the train or test side
    # of the per-scene chronological cut.
    for i, row in enumerate(target_meta.itertuples(index=False)):
        scene_test = test[test["sequence"] == row.sequence]
        assert (partitions[i] == "test") == (row.target_bucket in scene_test["time_bucket"].values)


def test_time_bucket_cutoff_rejects_multi_scene_input():
    df = _two_scene_features(n_buckets=10)
    with pytest.raises(ValueError):
        time_bucket_cutoff(df, test_size=0.2)


def test_build_sequences_raises_when_no_cell_has_enough_history():
    rows = [
        dict(grid_x=0, grid_y=0, time_bucket=t, label=0,
             **{c: 1.0 for c in DEFAULT_FEATURES})
        for t in range(3)
    ]
    with pytest.raises(ValueError):
        build_sequences(pd.DataFrame(rows), seq_len=10)


def test_time_bucket_cutoff_matches_split(synthetic_features):
    # 500 rows / 25 cells = 20 buckets. 80 % cutoff at row index 400 → bucket 16.
    assert time_bucket_cutoff(synthetic_features, test_size=0.2) == 16
