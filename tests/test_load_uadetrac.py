"""Tests for src.data.load_uadetrac."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.data import load_uadetrac_xml

_SYNTHETIC_XML = """<?xml version="1.0" encoding="utf-8"?>
<sequence name="MVI_TEST" frame_count="2">
  <ignored_region>
    <box left="0" top="0" width="10" height="10"/>
  </ignored_region>
  <frame density="2" num="1">
    <target_list>
      <target id="1">
        <box left="100" top="50" width="40" height="30"/>
        <attribute orientation="0" speed="3.5" trajectory_length="0"
                   truncation_ratio="0" vehicle_type="car"/>
      </target>
      <target id="2">
        <box left="200" top="80" width="80" height="40"/>
        <attribute orientation="0" speed="5.0" trajectory_length="0"
                   truncation_ratio="0" vehicle_type="bus"/>
      </target>
    </target_list>
  </frame>
  <frame density="1" num="2">
    <target_list>
      <target id="1">
        <box left="110" top="55" width="40" height="30"/>
        <attribute orientation="0" speed="3.6" trajectory_length="1"
                   truncation_ratio="0" vehicle_type="car"/>
      </target>
    </target_list>
  </frame>
</sequence>
"""


@pytest.fixture
def synthetic_xml(tmp_path: Path) -> Path:
    p = tmp_path / "MVI_TEST.xml"
    p.write_text(_SYNTHETIC_XML)
    return p


def test_row_count_and_columns(synthetic_xml):
    df = load_uadetrac_xml(synthetic_xml)
    assert len(df) == 3  # 2 targets in frame 1 + 1 target in frame 2
    expected_cols = {
        "frame", "track_id", "class_id", "class_name",
        "x1", "y1", "x2", "y2", "cx", "cy",
        "confidence", "speed",
    }
    assert expected_cols.issubset(df.columns)


def test_class_mapping(synthetic_xml):
    df = load_uadetrac_xml(synthetic_xml)
    car_rows = df[df["class_name"] == "car"]
    bus_rows = df[df["class_name"] == "bus"]
    assert (car_rows["class_id"] == 2).all()
    assert (bus_rows["class_id"] == 5).all()


def test_bounding_box_arithmetic(synthetic_xml):
    df = load_uadetrac_xml(synthetic_xml)
    row = df.iloc[0]  # frame 1, track 1: left=100, top=50, w=40, h=30
    assert row["x1"] == 100 and row["y1"] == 50
    assert row["x2"] == 140 and row["y2"] == 80
    assert row["cx"] == 120 and row["cy"] == 65


def test_track_ids_are_stable_across_frames(synthetic_xml):
    df = load_uadetrac_xml(synthetic_xml)
    track_1 = df[df["track_id"] == 1].sort_values("frame")
    assert list(track_1["frame"]) == [1, 2]
    # The centroid should have moved between frames (vehicle is in motion).
    assert track_1.iloc[0]["cx"] != track_1.iloc[1]["cx"]


def test_confidence_is_ground_truth(synthetic_xml):
    df = load_uadetrac_xml(synthetic_xml)
    assert (df["confidence"] == 1.0).all()


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_uadetrac_xml("/tmp/this_file_does_not_exist.xml")
