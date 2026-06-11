"""Tests for the deterministic helpers in scripts/autoCorrespondancePoints:
label parsing, confidence filtering, grouping, field-coordinate lookup, and the
confidence-weighted detection average.
"""

import numpy as np
import pytest

import autoCorrespondancePoints as ac


def test_parse_regular_marker():
    parsed = ac.parse_yard_marker_label("fl1")
    assert parsed["near_far"] == "f"
    assert parsed["left_right"] == "l"
    assert parsed["yard_number"] == "1"   # NOTE: kept as a string for regular markers
    assert parsed["original_label"] == "fl1"


def test_parse_midfield_special_cases():
    n5 = ac.parse_yard_marker_label("n5")
    assert n5["near_far"] == "n"
    assert n5["left_right"] is None
    assert n5["yard_number"] == 5         # int for the no-side midfield markers


def test_parse_rejects_too_short():
    assert ac.parse_yard_marker_label("f") is None


def test_filter_detections_by_confidence():
    dets = [{"confidence": 0.9}, {"confidence": 0.5}, {"confidence": 0.7}, {}]
    kept = ac.filter_detections_by_confidence(dets, confidence_threshold=0.7)
    assert kept == [{"confidence": 0.9}, {"confidence": 0.7}]


def test_group_detections_by_marker():
    dets = [
        {"class": "fl1", "confidence": 0.9},
        {"class": "fl1", "confidence": 0.8},
        {"class": "nr2", "confidence": 0.7},
    ]
    grouped = ac.group_detections_by_marker(dets)
    assert set(grouped) == {"fl1", "nr2"}
    assert len(grouped["fl1"]) == 2
    assert len(grouped["nr2"]) == 1


def test_get_field_coordinates_uses_position_table():
    coords = ac.get_field_coordinates_for_marker("nl1")
    assert (coords["x"], coords["y"]) == ac.positionsDict["nl1"]
    assert coords["near_far"] == "n"
    assert coords["hash_side"] == "l"


def test_get_field_coordinates_returns_none_for_unparseable_label():
    # Labels shorter than 2 chars fail parsing -> None.
    assert ac.get_field_coordinates_for_marker("z") is None


def test_get_field_coordinates_unknown_marker_raises():
    # NOTE: documents current behavior -- a parseable but unrecognised marker
    # is NOT in positionsDict and raises KeyError rather than returning None.
    with pytest.raises(KeyError):
        ac.get_field_coordinates_for_marker("zz")


def test_average_detections_single_is_passthrough():
    det = {"class": "fl1", "bbox": {"center_x": 1, "center_y": 2}, "confidence": 0.9}
    assert ac.average_detections_simple([det]) is det


def test_average_detections_weighted_center():
    dets = [
        {"class": "fl1", "class_id": 0, "confidence": 1.0,
         "bbox": {"center_x": 0.0, "center_y": 0.0, "width": 10, "height": 10}},
        {"class": "fl1", "class_id": 0, "confidence": 3.0,
         "bbox": {"center_x": 4.0, "center_y": 0.0, "width": 10, "height": 10}},
    ]
    avg = ac.average_detections_simple(dets)
    # Confidence-weighted: (0*1 + 4*3) / 4 = 3.0
    assert avg["bbox"]["center_x"] == pytest.approx(3.0)
    assert avg["bbox"]["center_y"] == pytest.approx(0.0)
    assert avg["class"] == "fl1"
