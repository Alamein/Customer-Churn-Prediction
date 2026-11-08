"""Unit tests for VehicleTracker — covers logic that doesn't require YOLO.

The full ``track_frame`` / ``process_video`` paths need real weights and are
exercised by the dashboard's smoke test instead.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from src.tracking.vehicle_tracker import VehicleTracker


def _bare_tracker() -> VehicleTracker:
    """Construct a VehicleTracker without invoking __init__ (skips YOLO load)."""
    tracker = VehicleTracker.__new__(VehicleTracker)
    tracker.history = defaultdict(list)
    return tracker


def test_estimate_speed_pixel_distance_per_second():
    tracker = _bare_tracker()
    # 10 frames at fps=30: cx goes 0 → 100, cy fixed at 0. Distance 100 px in 9/30 s.
    for f in range(10):
        tracker.history[1].append((f, f * (100.0 / 9.0), 0.0))
    speed = tracker.estimate_speed(track_id=1, fps=30.0, window=10)
    assert speed is not None
    assert 320 < speed < 340  # ~333 px/s


def test_estimate_speed_returns_none_when_history_too_short():
    tracker = _bare_tracker()
    tracker.history[1] = [(0, 0.0, 0.0), (1, 1.0, 0.0)]
    assert tracker.estimate_speed(track_id=1, fps=30.0, window=5) is None


def test_estimate_speed_with_homography_uses_world_coords():
    tracker = _bare_tracker()
    # Identity homography → world coords == pixel coords; result should match
    # the pixel-only branch within float tolerance.
    H = np.eye(3, dtype=np.float32)
    for f in range(10):
        tracker.history[1].append((f, f * 10.0, 0.0))
    pix = tracker.estimate_speed(track_id=1, fps=30.0, window=10, homography=None)
    world = tracker.estimate_speed(track_id=1, fps=30.0, window=10, homography=H)
    assert pix is not None and world is not None
    assert abs(pix - world) < 1e-3


def test_draw_tracks_preserves_frame_shape():
    tracker = _bare_tracker()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tracks = [dict(
        track_id=42, class_name="car",
        bbox=[10.0, 20.0, 110.0, 120.0],
    )]
    out = tracker.draw_tracks(frame, tracks)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype
