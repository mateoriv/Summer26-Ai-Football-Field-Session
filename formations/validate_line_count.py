#!/usr/bin/env python3
"""
Validate the LINE-COUNT formation reader (line_count_classifier) against the
coach's recorded formations.

This complements validate_formations.py (which scores the *template matcher*).
The line-count reader doesn't name the 17 variants -- it reports robust
STRUCTURE (front count, strength, 3x1/2x2 bucket). So we score it on the
structure the coach's label implies, not on the exact name:

  ground truth label  ->  structure via the 17 templates
  (models/offense_positions/play_predictions.csv `actual_play`)
  (formations/offense_formation_coordinates_17.csv)

Clips are keyed directly by name in play_predictions.csv -- no positional join,
so the comparison is exact. The same label vocabulary is what breakdown.xlsx
`OFF FORM` uses; this file is the clip-keyed subset of it.

Targets (both sign-independent, surviving the attack-relative left/right axis):
  * bucket -- 3x1 (trips-strong) vs 2x2 (balanced)
  * shape  -- balanced vs one-sided strength

Usage:
  .venv/bin/python3 formations/validate_line_count.py [--cache-dir cache] [--csv out.csv]
"""

import argparse
import glob
import os
import re
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import line_count_classifier as L  # noqa: E402

TEMPLATE_CSV = os.path.join(HERE, "offense_formation_coordinates_17.csv")
LABELS_CSV = os.path.join(ROOT, "models", "offense_positions", "play_predictions.csv")
BREAKDOWN_XLSX = os.path.join(ROOT, "data", "CSAI_FORMATIONS", "breakdown.xlsx")

# Template columns: Q=QB, T=back; the rest are receivers/TEs whose lateral sign
# (x, negative = one side) gives the strength side.
RECEIVER_COLS = ["W", "S", "X", "Y", "Z", "U"]


def _template_structures():
    """name -> {bucket, shape} derived from the 17 coordinate templates."""
    df = pd.read_csv(TEMPLATE_CSV)
    out = {}
    for _, row in df.iterrows():
        left = right = 0
        for c in RECEIVER_COLS:
            xc = f"{c}_x"
            if xc in row and not pd.isna(row[xc]):
                x = float(row[xc])
                if x < -0.5:
                    left += 1
                elif x > 0.5:
                    right += 1
        hi, lo = max(left, right), min(left, right)
        out[row["formation"].strip().lower()] = {
            "bucket": "3x1" if (hi >= 3 and hi - lo >= 2) else "2x2",
            "shape": "BALANCED" if left == right else "ONE-SIDED",
        }
    return out


def _norm_formname(name):
    """'TREY Y OFF' -> 'trey_y_off'; map coach spellings onto template keys."""
    if not isinstance(name, str):
        return None
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    key = key.replace("tite", "tight")          # coach 'TITE' == template 'tight'
    key = re.sub(r"^empty_", "", key)            # EMPTY TRIPS OPEN -> trips_open
    return key


def _breakdown_labels():
    """Coach labels from breakdown.xlsx: row N is 'Wide - Clip N' (confirmed
    with the coach -- the breakdown numbers every play 1..1025 in clip order;
    OFF FORM is the formation call). Returns {clip_name: label}."""
    bd = pd.read_excel(BREAKDOWN_XLSX)
    out = {}
    for i, v in enumerate(bd["OFF FORM"].tolist(), start=1):
        if isinstance(v, str) and v.strip():
            out[f"Wide - Clip {i:03d}"] = v.strip()
    return out


def _pred_structure(result):
    if result.get("on_line_count") is None:
        return None
    return {
        "bucket": result.get("bucket"),
        "shape": "BALANCED" if result.get("strength") == "BALANCED" else "ONE-SIDED",
        "reliable": bool(result.get("reliable")),
    }


def main():
    ap = argparse.ArgumentParser(description="Validate line-count reader vs coach labels")
    ap.add_argument("--cache-dir", default=os.path.join(ROOT, "cache"))
    ap.add_argument("--csv", default=os.path.join(HERE, "line_count_validation.csv"))
    args = ap.parse_args()

    templates = _template_structures()
    # Official ground truth: the coach's breakdown (1025 plays, clip order).
    # play_predictions.csv is kept only as a fallback -- its labels were found
    # partly misaligned (30/89 agree with the breakdown).
    if os.path.exists(BREAKDOWN_XLSX):
        labels = _breakdown_labels()
    else:
        labels = pd.read_csv(LABELS_CSV, index_col=0)["actual_play"].to_dict()

    # Map each labeled clip to its cache folder (if processed).
    clip_folder = {}
    for d in sorted(glob.glob(os.path.join(args.cache_dir, "*", ""))):
        folder = os.path.basename(d.rstrip("/"))
        for snap in glob.glob(os.path.join(d, "snap_detection", "*_snap_detection.json")):
            clip = os.path.basename(snap)[: -len("_snap_detection.json")]
            pos = os.path.join(d, "positions", clip + "_position.json")
            corr = os.path.join(d, "correspondence", clip + "_correspondence.json")
            if os.path.exists(pos) and os.path.exists(corr):
                clip_folder.setdefault(clip, folder)

    rep = []
    for clip, actual in labels.items():
        folder = clip_folder.get(clip)
        if folder is None:
            continue  # labeled but not processed through the pipeline yet
        res = L.recognize_from_cache(clip, folder, args.cache_dir)
        pred = _pred_structure(res)
        gt = templates.get(_norm_formname(actual))
        rep.append({
            "clip": clip,
            "actual": actual,
            "gt_bucket": gt["bucket"] if gt else None,
            "gt_shape": gt["shape"] if gt else None,
            "pred_bucket": pred["bucket"] if pred else None,
            "pred_shape": pred["shape"] if pred else None,
            "reliable": pred["reliable"] if pred else False,
            "read": "yes" if pred else "rejected",
            "reason": "" if pred else res.get("reason", ""),
        })

    rdf = pd.DataFrame(rep)
    rdf.to_csv(args.csv, index=False)

    labeled_processed = len(rdf)
    print(f"labeled clips processed through pipeline: {labeled_processed} "
          f"(of {len(labels)} labeled total)")

    scored = rdf[(rdf.reliable) & rdf.gt_bucket.notna()]
    print("\n=== STRUCTURE ACCURACY (reliable reads, label maps to a template) ===")
    print(f"clips scored: {len(scored)}")
    if len(scored):
        ba = (scored.pred_bucket == scored.gt_bucket).mean()
        sa = (scored.pred_shape == scored.gt_shape).mean()
        print(f"  bucket (3x1 vs 2x2):      {ba:5.1%}  ({int((scored.pred_bucket==scored.gt_bucket).sum())}/{len(scored)})")
        print(f"  shape  (balanced vs not): {sa:5.1%}  ({int((scored.pred_shape==scored.gt_shape).sum())}/{len(scored)})")
        print("\n  bucket confusion (rows=coach, cols=ours):")
        print(pd.crosstab(scored.gt_bucket, scored.pred_bucket).to_string())

    print("\n=== COVERAGE FUNNEL ===")
    print(f"  labeled + processed:   {labeled_processed}")
    print(f"  produced a read:       {(rdf.read=='yes').sum()}")
    print(f"  reliable read:         {int(rdf.reliable.sum())}")
    print(f"  reliable + scorable:   {len(scored)}")
    print(f"\nper-clip report -> {os.path.relpath(args.csv, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
