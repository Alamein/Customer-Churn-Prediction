"""Tests for src.features.feature_extractor."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.features import FeatureExtractor


def test_output_columns(synthetic_trajectories):
    fe = FeatureExtractor(time_window=1.0, spatial_grid_size=50.0, fps=30.0)
    out = fe.extract(synthetic_trajectories)
    assert list(out.columns) == FeatureExtractor.OUTPUT_COLUMNS
    assert len(out) > 0


def test_grid_and_bucket_assignment():
    # cx=75, cy=125 with grid=50 → (1, 2). frame=30 at fps=30 = 1 s → bucket 0 at window=2 s.
    df = pd.DataFrame([dict(
        frame=30, track_id=0, class_id=2, class_name="car",
        x1=65, y1=115, x2=85, y2=135, cx=75, cy=125,
        confidence=0.9, speed=10.0,
    )])
    out = FeatureExtractor(time_window=2.0, spatial_grid_size=50.0, fps=30.0).extract(df)
    row = out.iloc[0]
    assert row["grid_x"] == 1
    assert row["grid_y"] == 2
    assert row["time_bucket"] == 0
    assert row["vehicle_count"] == 1


def test_sudden_stop_detected(synthetic_trajectories):
    # Single bucket covering all 60 frames, generous grid so the stop is captured.
    fe = FeatureExtractor(time_window=10.0, spatial_grid_size=500.0, fps=30.0)
    out = fe.extract(synthetic_trajectories)
    assert out["sudden_stops"].sum() >= 1


def test_temporal_features_use_video_start(synthetic_trajectories):
    start = datetime(2026, 5, 16, 14, 30)  # Saturday 14:30
    fe = FeatureExtractor(time_window=1.0, fps=30.0, video_start=start)
    out = fe.extract(synthetic_trajectories)
    assert (out["time_of_day"] == 14).all()
    assert (out["day_of_week"] == 5).all()


def test_empty_dataframe_returns_empty_frame_with_correct_columns():
    out = FeatureExtractor().extract(pd.DataFrame())
    assert out.empty
    assert list(out.columns) == FeatureExtractor.OUTPUT_COLUMNS
