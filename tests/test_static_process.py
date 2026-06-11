"""Tests for the deterministic helpers in scripts/staticProcess:
offense-side inference, the first-11-on-side extractor, and the geometric
feature builder (shape + finiteness).

Importing staticProcess pulls in torch; these tests still avoid any GPU / model
files and run on tiny synthetic inputs.
"""

import numpy as np
import pytest

import staticProcess as sp


def _det(cls, cx):
    return {"class": cls, "bbox": {"center_x": cx, "center_y": 100.0}}


def test_offense_side_left_of_defense():
    dets = [_det("offense", 200), _det("offense", 300), _det("offense", 400),
            _det("defense", 1000), _det("defense", 1100)]
    assert sp.get_offense_side_from_positions(dets, image_width=1920) == "left"


def test_offense_side_right_of_defense():
    dets = [_det("offense", 1500), _det("offense", 1600),
            _det("defense", 800), _det("defense", 900)]
    assert sp.get_offense_side_from_positions(dets, image_width=1920) == "right"


def test_offense_side_ignores_refs():
    dets = [_det("ref", 50), _det("offense", 200), _det("defense", 1000)]
    # The ref must not be counted as offense.
    assert sp.get_offense_side_from_positions(dets, image_width=1920) == "left"


def test_offense_side_none_when_no_offense():
    dets = [_det("defense", 1000)]
    assert sp.get_offense_side_from_positions(dets, image_width=1920) is None


def _hdet(nx, ny):
    return {"normalized_position": {"x": nx, "y": ny},
            "original_bbox": {"center_x": nx * 10, "center_y": ny * 10}}


def test_take_first_11_on_left_picks_smallest_x_sorted_by_y():
    dets = [_hdet(float(i), float(11 - i)) for i in range(12)]  # nx 0..11
    pts = sp.take_first_11_on_side(dets, "left")
    assert len(pts) == 11
    nx_values = sorted(p[0] for p in pts)
    assert nx_values == [float(i) for i in range(11)]   # the 11 smallest nx
    ny_values = [p[1] for p in pts]
    assert ny_values == sorted(ny_values)               # final sort is by y


def test_take_first_11_on_right_returns_eleven():
    dets = [_hdet(float(i), float(i)) for i in range(12)]
    pts = sp.take_first_11_on_side(dets, "right")
    assert len(pts) == 11


def test_take_first_11_empty():
    assert sp.take_first_11_on_side([], "left") == []


def test_extract_geometric_features_shape_and_finite():
    # 11 players * (x, y) = 22-dim raw input -> 34-dim feature vector:
    # 22 base + 2 span + 1 pairwise + 2 centroid + 2 eigvals + 5 qb.
    raw = np.arange(22, dtype=np.float32)
    feats = sp.extract_geometric_features(raw)
    assert tuple(feats.shape) == (1, 34)
    import torch
    assert torch.isfinite(feats).all()


def test_normalize_class_alias_matches_shared_helper():
    import ioutils
    assert sp._normalize_class is ioutils.normalize_class
    assert sp._normalize_class("  Wide Receiver ") == "wide_receiver"
