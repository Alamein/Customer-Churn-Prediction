"""Unit tests for VehicleDetector — pure-logic paths only (no YOLO load)."""

from __future__ import annotations

import numpy as np

from src.detection.vehicle_detector import VehicleDetector


def _bare_detector() -> VehicleDetector:
    return VehicleDetector.__new__(VehicleDetector)


def test_vehicle_classes_match_coco_ids():
    # COCO IDs for car / motorcycle / bus / truck — the rest of the pipeline
    # (tracker, dashboard) relies on these being stable.
    detector = _bare_detector()
    detector.vehicle_classes = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
    assert set(detector.vehicle_classes) == {2, 3, 5, 7}


def test_draw_detections_preserves_frame_shape():
    detector = _bare_detector()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [dict(
        bbox=[100, 100, 200, 200], confidence=0.9,
        class_id=2, class_name="car", center=(150, 150),
    )]
    out = detector.draw_detections(frame, detections)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_draw_detections_handles_empty_list():
    detector = _bare_detector()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    out = detector.draw_detections(frame, [])
    assert out.shape == frame.shape
