#!/usr/bin/env python3
"""
The user's target design: identify a formation by COUNTS (front-line count +
per-side receiver counts), and when several CSV formations share the same
counts, return ALL of them as a shortlist for the coach to pick from.

The metric that matters here is NOT top-1 exact. It is:
  * COVERAGE  -- how often the true formation is inside the predicted count-class
                (i.e. the coach's shortlist contains the right answer), and
  * SHORTLIST -- how many options the coach is shown (smaller = better).

Both signatures (template truth and observed geometry) use the SAME definition:
  front  = # offense on the line of scrimmage
  side   = # receivers (non-OL, non-QB, non-central-back) each side, ordered
           strong/weak so a left/right flip doesn't matter.

Ground truth: models/offense_positions/play_predictions.csv `actual_play`.
Always keyed to the 17 templates in offense_formation_coordinates_17.csv.

Usage:
  PYTHONPATH=scripts:formations .venv/bin/python3 formations/eval_count_shortlist.py
"""

import collections
import csv as _csv
import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import line_count_classifier as L              # noqa: E402
from validate_line_count import _norm_formname  # noqa: E402

TEMPLATE_CSV = L.os.path.join(HERE, "offense_formation_coordinates_17.csv")
LABELS_CSV = os.path.join(ROOT, "models", "offense_positions", "play_predictions.csv")
RECV = ["W", "S", "X", "Y", "Z", "U"]


def _template_sig(row):
    """(front_line_count, strong_side#, weak_side#) for one template row."""
    on_line_skill = left = right = 0
    for r in RECV:
        x, y = row.get(f"{r}_x"), row.get(f"{r}_y")
        if x in ("", None) or y in ("", None):
            continue
        x, y = float(x), float(y)
        if abs(y) < 0.5:
            on_line_skill += 1
        if x < -0.5:
            left += 1
        elif x > 0.5:
            right += 1
    return (5 + on_line_skill, max(left, right), min(left, right))


def _template_groups():
    groups = collections.defaultdict(list)
    sig_of = {}
    with open(TEMPLATE_CSV, newline="") as f:
        for row in _csv.DictReader(f):
            name = (row.get("formation") or "").strip()
            if not name:
                continue
            s = _template_sig(row)
            groups[s].append(name.lower())
            sig_of[name.lower()] = s
    return groups, sig_of


def _observed_sig(result):
    """Same signature, read from the line-count geometry of one clip."""
    if result.get("on_line_count") is None:
        return None
    front = result["on_line_count"]
    l, r = result.get("recv_left", 0), result.get("recv_right", 0)
    return (front, max(l, r), min(l, r))


def main():
    groups, sig_of = _template_groups()
    labels = pd.read_csv(LABELS_CSV, index_col=0)["actual_play"].to_dict()
    clip_folder = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "cache", "*", ""))):
        folder = os.path.basename(d.rstrip("/"))
        for snap in glob.glob(os.path.join(d, "snap_detection", "*_snap_detection.json")):
            clip = os.path.basename(snap)[: -len("_snap_detection.json")]
            if (os.path.exists(os.path.join(d, "positions", clip + "_position.json")) and
                    os.path.exists(os.path.join(d, "correspondence", clip + "_correspondence.json"))):
                clip_folder.setdefault(clip, folder)

    base = os.path.join(ROOT, "cache")
    n = read = cover_exactsig = cover_frontonly = 0
    shortlist_sizes = []
    confusions = []
    for clip, actual in labels.items():
        folder = clip_folder.get(clip)
        if folder is None:
            continue
        true_key = _norm_formname(actual)
        true_sig = sig_of.get(true_key)
        if true_sig is None:          # label has no template (e.g. EMPTY)
            continue
        n += 1
        res = L.recognize_from_cache(clip, folder, base)
        obs = _observed_sig(res)
        if obs is None:
            confusions.append((clip, actual, "no-read", None))
            continue
        read += 1
        shortlist = groups.get(obs, [])
        in_list = true_key in shortlist
        cover_exactsig += in_list
        # looser: count-class with matching FRONT only (bigger shortlist)
        front_list = [f for s, fs in groups.items() if s[0] == obs[0] for f in fs]
        cover_frontonly += (true_key in front_list)
        shortlist_sizes.append(len(shortlist))
        if not in_list:
            confusions.append((clip, actual, f"obs{obs}", f"true{true_sig}"))

    print(f"labeled+templated clips: {n}   produced a count-read: {read}\n")
    print("=== SHORTLIST COVERAGE (true formation inside predicted count-class) ===")
    if read:
        print(f"  exact-signature match:  {cover_exactsig}/{read} = {cover_exactsig/read:5.1%}"
              f"   (avg shortlist {sum(shortlist_sizes)/len(shortlist_sizes):.1f} formations)")
        print(f"  front-count-only match: {cover_frontonly}/{read} = {cover_frontonly/read:5.1%}"
              f"   (bigger shortlist, just the front number)")
    print(f"\n  of {n} clips: {n-read} gave no count-read (geometry rejected/failed)")
    print(f"  end-to-end exact-sig coverage over ALL labeled: {cover_exactsig}/{n} = {cover_exactsig/n:5.1%}")

    print("\n=== misses (observed count-class != true) — where the read drifts ===")
    for clip, actual, obs, true in confusions[:20]:
        print(f"  {clip:18s} actual={actual:16s} {obs}  {true or ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
