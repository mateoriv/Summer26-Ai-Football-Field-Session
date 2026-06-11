"""Tests for the shared scripts/ioutils helpers."""

import json
import os

import pytest

import ioutils


def test_normalize_class_trims_lowercases_and_underscores():
    assert ioutils.normalize_class("  Wide Receiver ") == "wide_receiver"
    assert ioutils.normalize_class("QB") == "qb"
    assert ioutils.normalize_class("Tight End") == "tight_end"


def test_normalize_class_handles_none_and_empty():
    assert ioutils.normalize_class(None) == ""
    assert ioutils.normalize_class("") == ""


def test_save_then_load_json_roundtrip(tmp_path):
    data = {"a": 1, "nested": {"b": [1, 2, 3]}, "s": "x"}
    target = tmp_path / "sub" / "deep" / "out.json"  # parent dirs don't exist yet
    ioutils.save_json(data, str(target))

    assert target.exists()
    assert ioutils.load_json(str(target)) == data


def test_save_json_creates_parent_directories(tmp_path):
    target = tmp_path / "created" / "by" / "save.json"
    ioutils.save_json([1, 2], str(target))
    assert os.path.isdir(os.path.dirname(str(target)))


def test_load_json_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        ioutils.load_json(str(tmp_path / "does_not_exist.json"))
