"""Tests for the torch-free geometry in formations/template_matcher.

Covered: canonicalize() invariances and the mirror-invariant match()/cost.
"""

import math

import numpy as np
import pytest

import template_matcher as tm


def _square_plus_tail():
    # A non-degenerate, asymmetric point set (so PCA axes are well-defined).
    return np.array(
        [[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [0.0, 2.0],
         [2.0, -3.0], [1.0, 1.0]],
        dtype=float,
    )


def test_canonicalize_returns_none_for_too_few_points():
    canon, span = tm.canonicalize(np.array([[0.0, 0.0], [1.0, 1.0]]))
    assert canon is None and span is None


def test_canonicalize_returns_none_for_coincident_points():
    pts = np.zeros((6, 2))
    canon, span = tm.canonicalize(pts)
    assert canon is None and span is None


def test_canonicalize_output_is_rms_normalized_and_centered():
    canon, span = tm.canonicalize(_square_plus_tail())
    assert canon is not None
    # Centered: mean ~ 0.
    assert np.allclose(canon.mean(axis=0), 0.0, atol=1e-9)
    # RMS-normalized: mean of squared row norms == 1.
    rms = math.sqrt((canon ** 2).sum(axis=1).mean())
    assert rms == pytest.approx(1.0, abs=1e-9)
    assert span > 0.0


def test_canonicalize_is_translation_invariant():
    pts = _square_plus_tail()
    a, _ = tm.canonicalize(pts)
    b, _ = tm.canonicalize(pts + np.array([100.0, -50.0]))
    assert np.allclose(a, b, atol=1e-9)


def test_canonicalize_is_scale_invariant():
    pts = _square_plus_tail()
    a, _ = tm.canonicalize(pts)
    b, _ = tm.canonicalize(pts * 5.0)
    assert np.allclose(a, b, atol=1e-9)


def test_assignment_cost_zero_for_identical_sets():
    canon, _ = tm.canonicalize(_square_plus_tail())
    assert tm._assignment_cost(canon, canon) == pytest.approx(0.0, abs=1e-9)


def test_match_ranks_identical_template_first():
    canon, _ = tm.canonicalize(_square_plus_tail())
    other = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, -2], [2, 2]], float)
    other_canon, _ = tm.canonicalize(other)
    templates = [
        {"name": "self", "canon": canon},
        {"name": "other", "canon": other_canon},
    ]
    ranking = tm.match(canon, templates)
    # Best match (highest score) is the identical template, distance ~ 0.
    name, score, dist = ranking[0]
    assert name == "self"
    assert dist == pytest.approx(0.0, abs=1e-9)
    assert score == pytest.approx(1.0, abs=1e-9)


def test_match_is_mirror_invariant():
    canon, _ = tm.canonicalize(_square_plus_tail())
    mirrored = canon.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    templates = [{"name": "self", "canon": canon}]
    # A left-right mirror of the detected set still matches its template at dist 0.
    _, _, dist = tm.match(mirrored, templates)[0]
    assert dist == pytest.approx(0.0, abs=1e-9)
