"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_trajectories() -> pd.DataFrame:
    """3 tracks × 60 frames at 30 fps, with an injected sudden stop on track 1
    at frame 30 (speed 10 → 1, i.e. ~−270 m/s² deceleration)."""
    rng = np.random.default_rng(2026)
    rows = []
    for tid in range(3):
        speed = 10.0
        cx, cy = 100.0 + tid * 60.0, 200.0
        for f in range(60):
            if tid == 1 and f == 30:
                speed = 1.0
            else:
                speed = max(0.0, speed + rng.normal(0, 0.3))
            cx += speed / 30.0
            rows.append(dict(
                frame=f, track_id=tid, class_id=2, class_name="car",
                x1=cx - 10, y1=cy - 10, x2=cx + 10, y2=cy + 10,
                cx=cx, cy=cy, confidence=0.9, speed=float(speed),
            ))
    return pd.DataFrame(rows)


@pytest.fixture
def synthetic_features() -> pd.DataFrame:
    """500 feature rows on a 5×5 grid × 20 time buckets, with risky cells
    near the centre. Suitable for dataset / model unit tests."""
    rng = np.random.default_rng(42)
    rows = []
    for gx in range(5):
        for gy in range(5):
            risky = abs(gx - 2) + abs(gy - 2) <= 1
            base = 4 if risky else 1
            for t in range(20):
                rows.append(dict(
                    grid_x=gx, grid_y=gy, time_bucket=t,
                    vehicle_count=int(rng.poisson(12 if risky else 6)),
                    average_speed=float(rng.normal(10, 2)),
                    speed_variance=float(rng.exponential(2)),
                    traffic_density=0.001,
                    sudden_stops=int(rng.poisson(base)),
                    lane_changes=int(rng.poisson(base * 2)),
                    acceleration_events=int(rng.poisson(base)),
                    time_of_day=t % 24,
                    day_of_week=0,
                    weather_code=0,
                ))
    return pd.DataFrame(rows)
