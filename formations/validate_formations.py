#!/usr/bin/env python3
"""
Validate the template formation matcher against the coaches' recorded formations.

Ground truth is `models/offense_positions/play_predictions.csv` (`actual_play` =
what the coaches recorded down). For each processed clip we run the geometry
matcher, compare to that label two ways -- exact label and base family -- and
sweep the match score to recommend a confidence cutoff below which the
prediction should be flagged "unsure, don't rely on it."

Usage:
    # after processing the target clips through the YOLO pipeline:
    python formations/validate_formations.py --folder CSAI_FORMATIONS
    python formations/validate_formations.py --folder CSAI_FORMATIONS --all-processed

NOTE: a robust threshold needs many clips. With ~10 the numbers are directional;
re-run on more clips before committing the cutoff in production.
"""

import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import template_matcher as tm  # noqa: E402

# The 10 diverse wide-angle labeled clips chosen as the first validation target.
DEFAULT_CLIPS = [
    "Wide - Clip 275", "Wide - Clip 188", "Wide - Clip 854", "Wide - Clip 114",
    "Wide - Clip 933", "Wide - Clip 278", "Wide - Clip 305", "Wide - Clip 564",
    "Wide - Clip 1021", "Wide - Clip 021",
]

LABELS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "models", "offense_positions", "play_predictions.csv")

# Base families we actually have templates for. Recorded labels outside this set
# (e.g. EMPTY) cannot be recognized by design -- flagged, not silently scored.
TEMPLATE_BASES = {"DETROIT", "TRIPS", "DALLAS", "TREY", "SLOT", "DENVER"}


def normalize_pred(template_name):
    """Map a template name (lowercase, '_') to the coaches' label space."""
    if not template_name:
        return ""
    name = template_name.upper().replace("_", " ").replace("TIGHT", "TITE").strip()
    if name == "PRO":  # pro == denver_u_off geometrically (identical template rows)
        name = "DENVER U OFF"
    return name


def base_family(label):
    return (label or "").split()[0] if label else ""


def load_labels():
    df = pd.read_csv(LABELS_CSV, index_col=0)
    return {str(k): str(v) for k, v in df["actual_play"].items()}


def evaluate(clips, folder, base_cache_dir, labels, templates):
    """Run the matcher on each clip; return per-clip result rows."""
    rows = []
    for clip in clips:
        actual = labels.get(clip)
        res = tm.recognize_from_cache(clip, folder, base_cache_dir, templates=templates)
        if not res.get("formation"):
            rows.append({"clip": clip, "actual": actual, "pred": None,
                         "score": None, "reliable": None, "status": res.get("reason")})
            continue
        pred = normalize_pred(res["formation"])
        rows.append({
            "clip": clip,
            "actual": actual,
            "pred": pred,
            "score": res["score"],
            "reliable": res["reliable"],
            "exact": (actual is not None and pred == actual),
            "base_ok": (actual is not None and base_family(pred) == base_family(actual)),
            "has_template": base_family(actual) in TEMPLATE_BASES if actual else None,
            "status": "ok",
        })
    return rows


def threshold_sweep(scored, thresholds=None):
    """For each cutoff, report accuracy and coverage over reliable clips kept."""
    if thresholds is None:
        thresholds = [round(0.30 + 0.05 * i, 2) for i in range(11)]  # 0.30..0.80
    total = len(scored)
    out = []
    for t in thresholds:
        kept = [r for r in scored if r["reliable"] and r["score"] >= t]
        if not kept:
            out.append((t, 0, 0.0, None, None))
            continue
        exact = sum(r["exact"] for r in kept) / len(kept)
        base = sum(r["base_ok"] for r in kept) / len(kept)
        out.append((t, len(kept), len(kept) / total if total else 0.0, exact, base))
    return out


def recommend_threshold(scored, target_base_acc=0.8, min_coverage=0.4):
    """Smallest cutoff giving >= target base accuracy with enough coverage."""
    for t, n, cov, exact, base in threshold_sweep(scored):
        if base is not None and base >= target_base_acc and cov >= min_coverage:
            return t, base, cov
    return None, None, None


def main():
    ap = argparse.ArgumentParser(description="Validate template matcher vs recorded formations")
    ap.add_argument("--folder", default="CSAI_FORMATIONS")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--all-processed", action="store_true",
                    help="evaluate every processed+labeled clip in the folder, not just the 10")
    ap.add_argument("--clips", nargs="*", default=None, help="explicit clip names")
    args = ap.parse_args()

    base_cache = args.cache_dir if os.path.isabs(args.cache_dir) else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.cache_dir)

    labels = load_labels()
    templates = tm.load_templates()

    if args.all_processed:
        processed = {os.path.basename(p).replace("_position.json", "")
                     for p in glob.glob(os.path.join(base_cache, args.folder, "positions", "*_position.json"))}
        clips = sorted(processed & set(labels))
    else:
        clips = args.clips or DEFAULT_CLIPS

    rows = evaluate(clips, args.folder, base_cache, labels, templates)

    # ---- Per-clip table ----
    print(f"\n=== Per-clip comparison (folder={args.folder}) ===")
    print(f"{'CLIP':20s} {'RECORDED':18s} {'OURS':18s} {'SCORE':>6s} {'REL':>4s} {'EXACT':>6s} {'BASE':>5s}")
    print("-" * 86)
    scored = []
    not_processed = 0
    for r in rows:
        if r["status"] != "ok":
            print(f"{r['clip']:20s} {str(r['actual']):18s} {'--- not processed ---':18s}  ({r['status']})")
            not_processed += 1
            continue
        flag = "" if r["has_template"] else "  [no template for this family]"
        print(f"{r['clip']:20s} {str(r['actual']):18s} {str(r['pred']):18s} "
              f"{r['score']:6.2f} {str(r['reliable'])[0]:>4s} "
              f"{('Y' if r['exact'] else 'n'):>6s} {('Y' if r['base_ok'] else 'n'):>5s}{flag}")
        if r["actual"] is not None:
            scored.append(r)

    if not scored:
        print(f"\n{not_processed}/{len(rows)} target clips are not processed yet.")
        print("Run the YOLO pipeline on them first (see process_validation_clips.sh), then re-run this.")
        return 1

    # ---- Accuracy ----
    n = len(scored)
    exact_acc = sum(r["exact"] for r in scored) / n
    base_acc = sum(r["base_ok"] for r in scored) / n
    with_tpl = [r for r in scored if r["has_template"]]
    print(f"\n=== Accuracy over {n} labeled+processed clips ===")
    print(f"  Exact-label accuracy : {exact_acc:5.1%}  ({sum(r['exact'] for r in scored)}/{n})")
    print(f"  Base-family accuracy : {base_acc:5.1%}  ({sum(r['base_ok'] for r in scored)}/{n})")
    if len(with_tpl) != n:
        wt = len(with_tpl)
        print(f"  (excluding {n-wt} clips whose recorded family has no template):")
        if wt:
            print(f"    Exact : {sum(r['exact'] for r in with_tpl)/wt:5.1%} | "
                  f"Base : {sum(r['base_ok'] for r in with_tpl)/wt:5.1%}  over {wt} clips")

    # ---- Confidence threshold sweep ----
    print(f"\n=== Confidence threshold sweep (reliable clips only) ===")
    print(f"{'CUTOFF':>7s} {'KEPT':>5s} {'COVERAGE':>9s} {'EXACT':>7s} {'BASE':>7s}")
    for t, k, cov, ex, ba in threshold_sweep(scored):
        ex_s = f"{ex:6.0%}" if ex is not None else "   -- "
        ba_s = f"{ba:6.0%}" if ba is not None else "   -- "
        print(f"{t:7.2f} {k:5d} {cov:8.0%} {ex_s:>7s} {ba_s:>7s}")

    t, ba, cov = recommend_threshold(scored)
    print("\n=== Recommendation ===")
    if t is not None:
        print(f"  Suggested confidence cutoff: {t:.2f}")
        print(f"  At/above it: base-family accuracy {ba:.0%}, keeping {cov:.0%} of clips.")
        print(f"  Below {t:.2f}: flag as 'AI unsure' -- coaches should verify manually.")
    else:
        print("  No cutoff reaches the accuracy/coverage target on this set.")
        print("  Treat all predictions as advisory until validated on more clips.")
    print(f"\n  NOTE: n={n}. This is directional; re-run with --all-processed once more")
    print("  clips are processed before fixing the cutoff in production.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
