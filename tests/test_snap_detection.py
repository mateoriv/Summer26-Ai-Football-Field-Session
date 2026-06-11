"""Tests for the deterministic helpers in scripts/snapDetection:
nearest-neighbour player matching, candidate clustering, hard-count masking.
"""

import numpy as np
import pytest

import snapDetection as sd


def test_match_players_uniform_shift():
    prev = np.array([[0.0, 0.0], [10.0, 10.0]])
    curr = np.array([[1.0, 0.0], [11.0, 10.0]])
    deltas = sd.match_players(prev, curr)
    assert deltas.shape == (2, 2)
    assert np.allclose(deltas, [[1.0, 0.0], [1.0, 0.0]])


def test_match_players_empty_inputs():
    assert sd.match_players(np.zeros((0, 2)), np.array([[1.0, 1.0]])).shape == (0, 2)
    assert sd.match_players(np.array([[1.0, 1.0]]), np.zeros((0, 2))).shape == (0, 2)


def test_cluster_candidates_collapses_adjacent_frames():
    candidates = [
        {"frame": 10, "confidence": 0.5},
        {"frame": 11, "confidence": 0.9},   # same cluster as 10, higher conf
        {"frame": 50, "confidence": 0.3},   # separate cluster
    ]
    out = sd.cluster_candidates(candidates, gap_frames=5)
    assert [c["frame"] for c in out] == [11, 50]


def test_cluster_candidates_empty():
    assert sd.cluster_candidates([], gap_frames=5) == []


def test_build_hardcount_mask_constant_is_all_false():
    # A constant signal is one long "spike" (length >= return_frames), so it is
    # never flagged as a short hard-count.
    smoothed = np.full(40, 0.2)
    mask = sd.build_hardcount_mask(smoothed, fps=30)
    assert mask.dtype == bool
    assert mask.shape == (40,)
    assert not mask.any()


def test_build_hardcount_mask_flags_short_spike_not_sustained_burst():
    calm = 0.1
    hi = 1.0
    smoothed = np.concatenate([
        np.full(8, calm),    # 0-7
        np.full(3, hi),      # 8-10  short spike (hard count)
        np.full(10, calm),   # 11-20 returns to calm
        np.full(15, hi),     # 21-35 sustained burst (the real snap motion)
        np.full(9, calm),    # 36-44
    ])
    mask = sd.build_hardcount_mask(smoothed, fps=30)
    assert mask[9]      # inside the short spike -> flagged
    assert not mask[30]  # inside the sustained burst -> not flagged
    assert not mask[44]  # quiet tail -> not flagged
