"""Tests for formations/line_count_classifier -- the geometry-first offense
front / strength reader.

Convention used by ``classify``: x is the along-field (LOS->backfield) axis and
y is lateral. The offense here lines up at x=50 and attacks toward +x (defense
sits downfield at x=55).
"""

import numpy as np
import pytest

import line_count_classifier as lc


CENTER_Y = 26.0
LOS_X = 50.0


def _off(role, x, y):
    return {"role": role, "team": "offense", "x": float(x), "y": float(y)}


def _def(x, y):
    return {"role": "defense", "team": "defense", "x": float(x), "y": float(y)}


def _line():
    # 5 interior linemen on the LOS, centered on y=26.
    return [_off("oline", LOS_X, CENTER_Y + dy) for dy in (-2, -1, 0, 1, 2)]


def _backs():
    # QB 3 yd behind the line; RB 6 yd behind -- both centered.
    return [_off("qb", LOS_X - 3, CENTER_Y), _off("running_back", LOS_X - 6, CENTER_Y)]


def _defense():
    return [_def(LOS_X + 5, CENTER_Y + dy) for dy in (-3, 0, 3)]


def test_classify_left_strong_3x1():
    receivers = [
        _off("wide_receiver", LOS_X, CENTER_Y + 7),
        _off("wide_receiver", LOS_X, CENTER_Y + 9),
        _off("wide_receiver", LOS_X, CENTER_Y + 11),   # 3 to the left
        _off("wide_receiver", LOS_X, CENTER_Y - 9),     # 1 to the right
    ]
    players = _line() + _backs() + receivers + _defense()
    result = lc.classify(players)

    assert result["qb_recovered"] is True
    assert result["recv_left"] == 3
    assert result["recv_right"] == 1
    assert result["strength"] == "LEFT"
    assert result["bucket"] == "3x1"
    assert result["on_line_count"] == 9   # 5 OL + 4 split receivers at LOS depth
    assert result["n_offense"] == 11
    assert result["n_defense"] == 3
    assert result["reliable"] is True


def test_classify_balanced_2x2():
    receivers = [
        _off("wide_receiver", LOS_X, CENTER_Y + 8),
        _off("wide_receiver", LOS_X, CENTER_Y + 10),    # 2 left
        _off("wide_receiver", LOS_X, CENTER_Y - 8),
        _off("wide_receiver", LOS_X, CENTER_Y - 10),    # 2 right
    ]
    players = _line() + _backs() + receivers + _defense()
    result = lc.classify(players)

    assert result["recv_left"] == 2
    assert result["recv_right"] == 2
    assert result["strength"] == "BALANCED"
    assert result["bucket"] == "2x2"


def test_classify_rejects_too_few_offense():
    players = [_off("oline", LOS_X, CENTER_Y + dy) for dy in (-1, 0, 1)]
    result = lc.classify(players)
    assert result["on_line_count"] is None
    assert "offense" in result["reason"]


def test_dedup_merges_close_same_team_and_prefers_anchor_label():
    players = [
        _off("wide_receiver", 10.0, 10.0),
        _off("oline", 10.3, 10.0),   # within DEDUP_YD of the first, same team
    ]
    kept = lc._dedup_same_team(players)
    assert len(kept) == 1
    # The more informative OL/QB anchor label wins on merge.
    assert kept[0]["role"] == "oline"


def test_dedup_keeps_different_teams_apart():
    players = [
        _off("wide_receiver", 10.0, 10.0),
        {"role": "defense", "team": "defense", "x": 10.1, "y": 10.0},
    ]
    kept = lc._dedup_same_team(players)
    assert len(kept) == 2


def test_on_field_bounds():
    assert lc._on_field(LOS_X, CENTER_Y) is True
    assert lc._on_field(-5.0, CENTER_Y) is False          # behind the back endzone margin
    assert lc._on_field(LOS_X, lc.FIELD_WIDTH_YD + 5) is False  # outside the sideline
