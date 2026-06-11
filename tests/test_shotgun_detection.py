"""Tests for the QB-depth classifier in scripts/shotgunDetection.

``_classify`` maps the QB's measured depth behind the offensive line (yards) to
an alignment label, with thresholds UNDER_CENTER_MAX=1.5 and PISTOL_MAX=4.0.
"""

import pytest

import shotgunDetection as sg


def test_classify_under_center():
    assert sg._classify(0.0) == "under_center"
    assert sg._classify(1.49) == "under_center"


def test_classify_pistol():
    assert sg._classify(1.5) == "pistol"
    assert sg._classify(3.9) == "pistol"


def test_classify_shotgun():
    assert sg._classify(4.0) == "shotgun"
    assert sg._classify(7.0) == "shotgun"


def test_classify_thresholds_match_constants():
    # Guard the boundary semantics against accidental threshold edits.
    assert sg._classify(sg.UNDER_CENTER_MAX) == "pistol"
    assert sg._classify(sg.PISTOL_MAX) == "shotgun"
