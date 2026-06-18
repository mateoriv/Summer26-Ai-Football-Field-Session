"""Tests for formations/verified_store -- the human-verification persistence
layer (QB clicks). Pure stdlib + tmp_path; no Qt, no cv2."""

import csv
import json
import os

import verified_store as vs


QB = {"frame": 143, "x": 612.4, "y": 388.1, "snapped": True,
      "track_id": 7, "detected_class": "oline",
      "bbox": {"x1": 590, "y1": 360, "x2": 634, "y2": 420}}


def test_save_load_roundtrip(tmp_path):
    path = vs.save_qb_verification("Wide - Clip 012", "F", str(tmp_path), QB)
    assert os.path.exists(path)
    data = vs.load_verified("Wide - Clip 012", "F", str(tmp_path))
    assert data["clip"] == "Wide - Clip 012"
    assert data["version"] == 1
    assert data["qb"]["track_id"] == 7
    assert data["qb"]["verified_at"]          # stamped automatically
    assert data["qb"]["verified_by"] == "ui_click"


def test_load_missing_and_corrupt(tmp_path):
    assert vs.load_verified("nope", "F", str(tmp_path)) is None
    p = vs.verified_path("bad", "F", str(tmp_path))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        f.write("{not json")
    assert vs.load_verified("bad", "F", str(tmp_path)) is None


def test_merge_preserves_future_keys(tmp_path):
    # A future "center" verification must survive a QB re-verify.
    p = vs.verified_path("c1", "F", str(tmp_path))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"version": 1, "clip": "c1", "center": {"frame": 10}}, f)
    vs.save_qb_verification("c1", "F", str(tmp_path), QB)
    data = vs.load_verified("c1", "F", str(tmp_path))
    assert data["center"] == {"frame": 10}
    assert data["qb"]["frame"] == 143


def test_clear_qb(tmp_path):
    vs.save_qb_verification("c1", "F", str(tmp_path), QB)
    assert vs.clear_qb_verification("c1", "F", str(tmp_path)) is True
    assert vs.load_verified("c1", "F", str(tmp_path)) is None  # file removed
    assert vs.clear_qb_verification("c1", "F", str(tmp_path)) is False


def test_clear_qb_keeps_other_facts(tmp_path):
    p = vs.verified_path("c1", "F", str(tmp_path))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump({"version": 1, "clip": "c1", "center": {"frame": 10}}, f)
    vs.save_qb_verification("c1", "F", str(tmp_path), QB)
    vs.clear_qb_verification("c1", "F", str(tmp_path))
    data = vs.load_verified("c1", "F", str(tmp_path))
    assert data is not None and "qb" not in data and data["center"]["frame"] == 10


def test_formation_history_accumulates(tmp_path):
    # Each confirmation/correction must be kept; the latest stays in "formation".
    vs.save_formation_verification("c1", "F", str(tmp_path),
        {"formation": "DETROIT", "system_pick": "DETROIT", "agreed": True})
    vs.save_formation_verification("c1", "F", str(tmp_path),
        {"formation": "TREY Y OFF", "system_pick": "DETROIT", "agreed": False})
    vs.save_formation_verification("c1", "F", str(tmp_path),
        {"formation": "DOUBLES", "system_pick": "DETROIT", "agreed": False})
    data = vs.load_verified("c1", "F", str(tmp_path))
    assert data["formation"]["formation"] == "DOUBLES"   # latest = current truth
    hist = vs.load_formation_history("c1", "F", str(tmp_path))
    assert [h["formation"] for h in hist] == ["DETROIT", "TREY Y OFF", "DOUBLES"]
    assert [h["seq"] for h in hist] == [1, 2, 3]
    assert hist[1]["previous_formation"] == "DETROIT"
    assert hist[2]["previous_formation"] == "TREY Y OFF"


def test_formation_history_csv_is_append_only(tmp_path):
    base = str(tmp_path)
    vs.append_formation_history("F", base, {
        "folder": "F", "clip": "c1", "seq": 1, "chosen_formation": "DETROIT",
        "previous_formation": "", "system_pick": "DETROIT", "agreed": True})
    path = vs.append_formation_history("F", base, {
        "folder": "F", "clip": "c1", "seq": 2, "chosen_formation": "DOUBLES",
        "previous_formation": "DETROIT", "system_pick": "DETROIT", "agreed": False})
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2                       # nothing replaced
    assert rows[0]["chosen_formation"] == "DETROIT"
    assert rows[1]["previous_formation"] == "DETROIT"


def test_formation_history_survives_qb_reverify(tmp_path):
    vs.save_formation_verification("c1", "F", str(tmp_path),
        {"formation": "DETROIT", "agreed": True})
    vs.save_qb_verification("c1", "F", str(tmp_path), QB)
    hist = vs.load_formation_history("c1", "F", str(tmp_path))
    assert len(hist) == 1 and hist[0]["formation"] == "DETROIT"


def test_training_label_append_and_replace(tmp_path):
    row = {"folder": "F", "clip": "c1", "frame": 143, "x": 612.4, "y": 388.1,
           "matched_track_id": 7, "matched_class": "oline", "verified_at": "t1"}
    path = vs.append_training_label("F", str(tmp_path), row)
    vs.append_training_label("F", str(tmp_path), {**row, "clip": "c2"})
    # re-verify c1 -> replaces, not duplicates
    vs.append_training_label("F", str(tmp_path), {**row, "frame": 150, "verified_at": "t2"})
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    c1 = next(r for r in rows if r["clip"] == "c1")
    assert c1["frame"] == "150" and c1["verified_at"] == "t2"
