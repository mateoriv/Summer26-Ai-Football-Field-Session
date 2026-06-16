#!/usr/bin/env python3
"""
Ensemble offense-formation predictor: template matcher x playbook prior x
line-count structure.

Why an ensemble: each signal alone is weak on wide-angle footage --
  * the TEMPLATE MATCHER's top-1 is ~14% exact (the discriminative depth cues
    are below the homography's resolution), but its top-5 holds the right
    answer ~45% of the time -- it is a good CANDIDATE GENERATOR;
  * the PLAYBOOK PRIOR (how often the coach actually calls each formation,
    from the breakdown's 1025 OFF FORM labels) is a strong tiebreaker -- this
    team's calls are heavily concentrated (DETROIT + TRIPS OPEN ~= 37%);
  * the LINE-COUNT reader sees STRUCTURE (3x1/2x2 bucket, balanced/one-sided)
    at ~60-70% -- soft evidence that re-ranks candidates.

Measured on the 56 cached clips with breakdown labels:
  matcher top-1 14.3% exact / 26.8% family
  ensemble      26.8% exact / 33.9% family   (prior-only floor: 19.4%/25.7%)

The prior ships as formation_priors.json (regenerate with
`python formations/formation_predictor.py --rebuild-priors` when the coach
adds plays to breakdown.xlsx).
"""

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))

PRIORS_PATH = os.path.join(HERE, "formation_priors.json")
BREAKDOWN_XLSX = os.path.join(HERE, "..", "data", "CSAI_FORMATIONS", "breakdown.xlsx")

# Soft-evidence weights (log-space bonuses when a candidate's template
# structure agrees with the observed line-count read). Coarse, monotone
# values -- tuned only to the nearest power of two on the 56 eval clips.
BUCKET_BONUS = 2.0
SHAPE_BONUS = 1.0
UNRELIABLE_W = 0.4   # discount structure evidence on unreliable reads
PRIOR_FLOOR = 1e-4   # for formations never seen in the breakdown


def _norm_formname(name):
    import re
    if not isinstance(name, str):
        return None
    key = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    key = key.replace("tite", "tight")
    key = re.sub(r"^empty_", "", key)
    return key


def rebuild_priors(xlsx_path=BREAKDOWN_XLSX, out_path=PRIORS_PATH):
    """Recompute the playbook prior from the coach's breakdown OFF FORM column
    (rows are in clip order, one row per play)."""
    import pandas as pd
    from validate_line_count import _template_structures
    struct = _template_structures()
    bd = pd.read_excel(xlsx_path)
    counts = {}
    for v in bd["OFF FORM"].dropna().astype(str):
        k = _norm_formname(v)
        if k in struct:
            counts[k] = counts.get(k, 0) + 1
    total = sum(counts.values())
    priors = {k: c / total for k, c in counts.items()}
    with open(out_path, "w") as f:
        json.dump({"source": os.path.basename(xlsx_path), "n_labels": total,
                   "priors": priors}, f, indent=2)
    return priors


def load_priors(path=PRIORS_PATH):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f).get("priors", {})


def predict(video_name, folder_name, base_cache_dir):
    """Best-guess formation for one cached clip.

    Returns {"formation", "score", "reliable", "ranking", "sources"} or
    {"formation": None, "reason": ...}. `score` is the matcher score of the
    chosen candidate (so the app's confidence display stays comparable).
    """
    import template_matcher as TM
    import line_count_classifier as LC
    from validate_line_count import _template_structures

    tm = TM.recognize_from_cache(video_name, folder_name, base_cache_dir)
    if not tm or not tm.get("ranking"):
        return {"formation": None,
                "reason": (tm or {}).get("reason", "no matcher read")}

    lc = LC.recognize_from_cache(video_name, folder_name, base_cache_dir)
    if not lc or lc.get("on_line_count") is None:
        lc = None

    struct = _template_structures()
    priors = load_priors()

    best, best_s, scores = None, -math.inf, {}
    for name, score in tm["ranking"]:
        s = math.log(max(float(score), 1e-6))
        s += math.log(priors.get(name, PRIOR_FLOOR))
        if lc:
            w = 1.0 if lc.get("reliable") else UNRELIABLE_W
            st = struct.get(name, {})
            if st.get("bucket") == lc.get("bucket"):
                s += BUCKET_BONUS * w
            obs_shape = "BALANCED" if lc.get("strength") == "BALANCED" else "ONE-SIDED"
            if st.get("shape") == obs_shape:
                s += SHAPE_BONUS * w
        scores[name] = s
        if s > best_s:
            best_s, best = s, name

    matcher_score = dict(tm["ranking"]).get(best)
    return {
        "formation": best,
        "score": matcher_score,
        "reliable": bool(tm.get("reliable", True)),
        "ranking": sorted(scores, key=scores.get, reverse=True),
        "sources": {
            "matcher_top1": tm["ranking"][0][0],
            "line_count": ({"bucket": lc.get("bucket"), "strength": lc.get("strength"),
                            "reliable": lc.get("reliable")} if lc else None),
            "prior_used": bool(priors),
        },
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-priors", action="store_true")
    ap.add_argument("--video-name")
    ap.add_argument("--folder-name")
    ap.add_argument("--cache-dir", default=os.path.join(HERE, "..", "cache"))
    args = ap.parse_args()
    if args.rebuild_priors:
        priors = rebuild_priors()
        print(f"wrote {PRIORS_PATH} ({len(priors)} formations)")
        return 0
    if args.video_name and args.folder_name:
        print(json.dumps(predict(args.video_name, args.folder_name, args.cache_dir), indent=2))
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
