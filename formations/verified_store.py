#!/usr/bin/env python3
"""
Persistent store for HUMAN-VERIFIED facts about a clip (v1: the QB).

One coach click on the video verifies the QB at the snap frame. That
verification is the formation reader's anchor (attack direction, RB cut,
reliability gate, jersey-colour team naming), so it is persisted OUTSIDE the
pipeline's stage directories: re-running any stage re-derives everything
*anchored on* the saved verification, and never destroys it.

Layout:
  cache/<folder>/verified/<clip>_verified.json     one file per clip
  cache/<folder>/verified/qb_training_labels.csv   data flywheel (one row per
                                                   clip, replaced on re-verify)

Schema (room to grow: "center", "team_colors" later):
  {
    "version": 1,
    "clip": "Wide - Clip 012",
    "qb": {
      "frame": 143,                 # frame the coach clicked on
      "x": 612.4, "y": 388.1,      # VIDEO pixels (matched detection centre
                                    # when snapped, raw click otherwise)
      "snapped": true,
      "track_id": 7,                # nullable
      "detected_class": "oline",   # what the detector THOUGHT it was
      "bbox": {...},                # nullable
      "verified_at": "2026-06-11T10:30:00Z",
      "verified_by": "ui_click"
    }
  }

Stdlib-only so line_count_classifier stays import-safe in the sandbox.
"""

import csv
import json
import os
from datetime import datetime, timezone

VERIFIED_DIR = "verified"
TRAINING_CSV = "qb_training_labels.csv"
TRAINING_COLUMNS = ["folder", "clip", "frame", "x", "y",
                    "matched_track_id", "matched_class", "verified_at"]

# Formation flywheel: one row per clip the coach has confirmed, recording what
# the system guessed vs what the coach chose (so "agreed" is the live real-world
# accuracy meter, and the chosen label is clean training truth).
FORMATION_CSV = "formation_training_labels.csv"
FORMATION_COLUMNS = ["folder", "clip", "chosen_formation", "system_pick",
                     "system_confidence", "agreed", "verified_at"]


def verified_path(video_name, folder_name, base_cache_dir):
    return os.path.join(base_cache_dir, folder_name, VERIFIED_DIR,
                        f"{video_name}_verified.json")


def load_verified(video_name, folder_name, base_cache_dir):
    """The clip's verified record, or None (missing/corrupt files are not
    errors -- an absent verification just means the geometric path runs)."""
    path = verified_path(video_name, folder_name, base_cache_dir)
    try:
        with open(path) as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def save_qb_verification(video_name, folder_name, base_cache_dir, qb_record):
    """Write/update the clip's QB verification. Merges into the existing file
    so future keys (center, team colours) survive a QB re-verify."""
    data = load_verified(video_name, folder_name, base_cache_dir) or {}
    data.setdefault("version", 1)
    data["clip"] = video_name
    qb = dict(qb_record)
    qb.setdefault("verified_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    qb.setdefault("verified_by", "ui_click")
    data["qb"] = qb
    path = verified_path(video_name, folder_name, base_cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def clear_qb_verification(video_name, folder_name, base_cache_dir):
    """Remove the QB verification (keeps any other verified facts)."""
    data = load_verified(video_name, folder_name, base_cache_dir)
    if not data or "qb" not in data:
        return False
    data.pop("qb")
    path = verified_path(video_name, folder_name, base_cache_dir)
    remaining = [k for k in data if k not in ("version", "clip")]
    if remaining:
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    else:
        os.remove(path)
    return True


def save_formation_verification(video_name, folder_name, base_cache_dir, formation_record):
    """Write/update the clip's coach-confirmed formation. Merged into the same
    verified JSON as the QB (so neither destroys the other), and persisted
    OUTSIDE the stage dirs so a formation-only re-run never erases it."""
    data = load_verified(video_name, folder_name, base_cache_dir) or {}
    data.setdefault("version", 1)
    data["clip"] = video_name
    rec = dict(formation_record)
    rec.setdefault("verified_at", datetime.now(timezone.utc).isoformat(timespec="seconds"))
    rec.setdefault("verified_by", "ui_click")
    data["formation"] = rec
    path = verified_path(video_name, folder_name, base_cache_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def append_formation_training_label(folder_name, base_cache_dir, row):
    """Record the formation confirmation in the flywheel CSV (one row per clip,
    re-confirm REPLACES the row, so the file is always current truth)."""
    path = os.path.join(base_cache_dir, folder_name, VERIFIED_DIR, FORMATION_CSV)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    if os.path.exists(path):
        try:
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f)
                        if r.get("clip") != row.get("clip")]
        except (OSError, ValueError):
            rows = []
    clean = {c: ("" if row.get(c) is None else row.get(c)) for c in FORMATION_COLUMNS}
    rows.append(clean)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FORMATION_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in FORMATION_COLUMNS})
    return path


def append_training_label(folder_name, base_cache_dir, row):
    """Record the verification in the data-flywheel CSV (one row per clip --
    a re-verify REPLACES the clip's previous row, so the file is always the
    current truth, ready for a detector fine-tune)."""
    path = os.path.join(base_cache_dir, folder_name, VERIFIED_DIR, TRAINING_CSV)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rows = []
    if os.path.exists(path):
        try:
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f)
                        if r.get("clip") != row.get("clip")]
        except (OSError, ValueError):
            rows = []
    clean = {c: ("" if row.get(c) is None else row.get(c)) for c in TRAINING_COLUMNS}
    rows.append(clean)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRAINING_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in TRAINING_COLUMNS})
    return path
