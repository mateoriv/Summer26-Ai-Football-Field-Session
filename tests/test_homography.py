"""Tests for the deterministic helpers in scripts/perFrameHomographyTransform:
point transform, degeneracy guard, and windowed correspondence aggregation.
"""

import numpy as np
import pytest

import perFrameHomographyTransform as ph


def test_transform_point_identity():
    out = ph.transform_point((5.0, 7.0), np.eye(3, dtype=np.float32))
    assert out == pytest.approx((5.0, 7.0))


def test_transform_point_translation():
    H = np.array([[1, 0, 10], [0, 1, 20], [0, 0, 1]], dtype=np.float32)
    out = ph.transform_point((5.0, 7.0), H)
    assert out == pytest.approx((15.0, 27.0))


def test_transform_point_none_homography():
    assert ph.transform_point((1.0, 2.0), None) is None


def test_degenerate_too_few_points():
    assert ph.field_points_are_degenerate([[0, 0], [1, 1], [2, 2]]) is True


def test_degenerate_colinear_points():
    # Every point on the same hash line (y == 8) -> collapses the solve.
    colinear = [[10, 8], [20, 8], [30, 8], [40, 8]]
    assert ph.field_points_are_degenerate(colinear) is True


def test_non_degenerate_spanning_points():
    spanning = [[10, 8], [20, 8], [10, 45], [20, 45]]
    assert ph.field_points_are_degenerate(spanning) is False


def _cp(fx, fy, ix, iy, conf):
    return {
        "field_point": {"x": fx, "y": fy},
        "image_point": {"x": ix, "y": iy},
        "yard_marker_info": {"confidence": conf},
    }


def test_gather_window_excludes_out_of_window_frames():
    fc = {
        "10": [_cp(10, 8, 100, 200, 0.5)],
        "12": [_cp(10, 8, 105, 205, 0.9)],
        "30": [_cp(20, 8, 300, 200, 0.7)],  # 20 frames away -> dropped
    }
    img, fld = ph.gather_window_correspondences(fc, frame_number=10, window=15)
    # Only the (10,8) field point survives; keep the instance from the closest
    # frame (frame 10, distance 0) over frame 12.
    assert fld == [[10, 8]]
    assert img == [[100, 200]]


def test_gather_window_keeps_nearest_frame_per_field_point():
    fc = {
        "10": [_cp(10, 8, 100, 200, 0.5)],
        "12": [_cp(10, 8, 105, 205, 0.9)],
    }
    img, fld = ph.gather_window_correspondences(fc, frame_number=12, window=15)
    assert fld == [[10, 8]]
    assert img == [[105, 205]]  # frame 12 is closest to target 12
