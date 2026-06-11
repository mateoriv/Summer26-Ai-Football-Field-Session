"""Tests for the QB/TE geometry in formations/qb_anchored_matcher.

These are the deterministic, numpy-only primitives that orient a snap into
(lateral, depth) yards and recover the QB / TE candidates -- the features fed to
the offense-positions MLP.
"""

import numpy as np
import pytest

import qb_anchored_matcher as qam


def test_orient_finds_five_linemen(standard_formation_points):
    pts = standard_formation_points["points"]
    oriented, ol_idx, los_depth, ol_span = qam._orient_in_yards(pts)

    assert oriented.shape == (11, 2)
    assert len(set(ol_idx)) == 5
    assert los_depth == 0.0
    # The five OL span x in [-2, 2] -> ~4 yd laterally (small PCA tilt allowed).
    assert 3.0 < ol_span < 5.5


def test_orient_picks_the_actual_linemen(standard_formation_points):
    pts = standard_formation_points["points"]
    idx = standard_formation_points["index"]
    _, ol_idx, _, _ = qam._orient_in_yards(pts)
    expected = {idx[n] for n in ("ol0", "ol1", "ol2", "ol3", "ol4")}
    assert set(ol_idx) == expected


def test_identify_qb_recovers_the_centered_back(standard_formation_points):
    pts = standard_formation_points["points"]
    idx = standard_formation_points["index"]
    oriented, ol_idx, _, _ = qam._orient_in_yards(pts)
    qb_idx = qam._identify_qb(oriented, ol_idx)
    assert qb_idx == idx["qb"]


def test_identify_qb_returns_none_when_no_back(standard_formation_points):
    # An all-on-line look (no one in the 1-7 yd backfield band) -> no QB.
    pts = np.array([[float(x), 0.0] for x in range(-5, 6)], dtype=float)
    oriented, ol_idx, _, _ = qam._orient_in_yards(pts)
    assert qam._identify_qb(oriented, ol_idx) is None


def test_identify_te_finds_the_wing(standard_formation_points):
    pts = standard_formation_points["points"]
    oriented, ol_idx, _, _ = qam._orient_in_yards(pts)
    qb_idx = qam._identify_qb(oriented, ol_idx)
    tes = qam._identify_te(oriented, ol_idx, qb_idx)
    # Exactly the one tight end just outside the tackle (split WRs are too wide).
    assert len(tes) == 1
    _, side = tes[0]
    assert side in ("left", "right")


def test_qb_features_shape_and_signal(standard_formation_points):
    pts = standard_formation_points["points"]
    feats = qam.qb_features_for_points(pts)
    assert feats.shape == (len(qam.QB_FEATURE_NAMES),)
    qb_lat, qb_depth, te_left, te_right, ol_span = feats
    assert abs(qb_lat) < 1.0          # QB is centered
    assert qb_depth < 0.0             # QB sits behind the line (negative depth)
    assert te_left + te_right == 1.0  # exactly one TE flagged
    assert 3.0 < ol_span < 5.5


def test_qb_features_zeroed_for_too_few_points():
    feats = qam.qb_features_for_points(np.zeros((3, 2)))
    assert feats.shape == (len(qam.QB_FEATURE_NAMES),)
    assert np.all(feats == 0.0)


def test_qb_features_zeroed_for_wrong_dimension():
    feats = qam.qb_features_for_points(np.zeros((11, 3)))
    assert np.all(feats == 0.0)
