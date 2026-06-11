#!/usr/bin/env python3
"""
Test the INTENDED pipeline: template match -> eliminate candidates whose SHAPE
(line-count bucket/strength) contradicts what we observe -> pick best survivor.

This is not the matcher alone and not the shape reader alone -- it is the
re-ranking ensemble: the matcher proposes ranked formation candidates, and the
line-count shape read prunes the structurally-impossible ones before we commit.

Baseline   = matcher top-1.
Shape-filt = matcher top-1 among candidates whose template bucket matches the
             observed bucket (only when the shape read is reliable; else fall
             back to the unfiltered top-1, so the filter can only help or tie).

Usage:
  PYTHONPATH=scripts:formations .venv/bin/python3 formations/eval_shape_filter.py
"""

import glob
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import template_matcher as tm           # noqa: E402
import line_count_classifier as L       # noqa: E402
from validate_line_count import _template_structures  # noqa: E402
from validate_formations import normalize_pred, base_family  # noqa: E402

LABELS_CSV = os.path.join(ROOT, "models", "offense_positions", "play_predictions.csv")


def main():
    struct = _template_structures()                       # template name -> {bucket, shape}
    labels = pd.read_csv(LABELS_CSV, index_col=0)["actual_play"].to_dict()

    # clip -> cache folder (processed only)
    clip_folder = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "cache", "*", ""))):
        folder = os.path.basename(d.rstrip("/"))
        for snap in glob.glob(os.path.join(d, "snap_detection", "*_snap_detection.json")):
            clip = os.path.basename(snap)[: -len("_snap_detection.json")]
            pos = os.path.join(d, "positions", clip + "_position.json")
            corr = os.path.join(d, "correspondence", clip + "_correspondence.json")
            if os.path.exists(pos) and os.path.exists(corr):
                clip_folder.setdefault(clip, folder)

    base_dir = os.path.join(ROOT, "cache")
    rows, n_filter_applied, n_changed = [], 0, 0
    for clip, actual in labels.items():
        folder = clip_folder.get(clip)
        if folder is None:
            continue
        mt = tm.recognize_from_cache(clip, folder, base_dir)
        if not mt.get("formation"):
            continue
        ranking = mt.get("ranking") or [(mt["formation"], mt.get("score", 0))]
        baseline = mt["formation"]

        lc = L.recognize_from_cache(clip, folder, base_dir)
        obs_bucket = lc.get("bucket") if lc.get("reliable") else None

        if obs_bucket is not None:
            survivors = [n for n, _ in ranking if struct.get(n, {}).get("bucket") == obs_bucket]
            filtered = survivors[0] if survivors else baseline
            n_filter_applied += 1
            n_changed += (filtered != baseline)
        else:
            filtered = baseline

        a = str(actual)
        rows.append({
            "clip": clip, "actual": a,
            "base_pred": normalize_pred(baseline),
            "filt_pred": normalize_pred(filtered),
            "obs_bucket": obs_bucket,
            "changed": filtered != baseline,
        })

    df = pd.DataFrame(rows)
    n = len(df)
    print(f"labeled + processed + matcher-read clips: {n}")
    print(f"shape filter applicable (reliable shape): {n_filter_applied}")
    print(f"  of those, filter CHANGED the pick:      {n_changed}\n")

    def acc(col):
        exact = (df[col] == df.actual).mean()
        base = df.apply(lambda r: base_family(r[col]) == base_family(r.actual), axis=1).mean()
        return exact, base

    be, bb = acc("base_pred")
    fe, fb = acc("filt_pred")
    print("=== matcher ALONE (baseline) ===")
    print(f"  exact: {be:5.1%}   base-family: {bb:5.1%}")
    print("=== matcher + SHAPE ELIMINATION (your method) ===")
    print(f"  exact: {fe:5.1%}   base-family: {fb:5.1%}")
    print(f"\n  delta: exact {fe-be:+.1%}   base-family {fb-bb:+.1%}")

    # Show the clips the filter changed, so the effect is auditable.
    ch = df[df.changed]
    if len(ch):
        print("\n=== clips the shape filter changed ===")
        for _, r in ch.iterrows():
            base_ok = "ok" if base_family(r["base_pred"]) == base_family(r["actual"]) else "x"
            filt_ok = "ok" if base_family(r["filt_pred"]) == base_family(r["actual"]) else "x"
            verdict = "FIXED" if (filt_ok == "ok" and base_ok == "x") else \
                      ("BROKE" if (filt_ok == "x" and base_ok == "ok") else "same")
            print(f"  {r['clip']:18s} actual={r['actual']:16s} {r['base_pred']:14s}[{base_ok}] -> "
                  f"{r['filt_pred']:14s}[{filt_ok}]  {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
