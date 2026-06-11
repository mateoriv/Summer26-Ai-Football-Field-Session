#!/usr/bin/env python3
"""
The user's method: ALWAYS match against the 17-template CSV, but use the
line-count GEOMETRY to identify the non-discriminative roles (QB + the back next
to him) and pin them out, so the RECEIVERS -- the part that actually separates
the formations, TE included -- drive the match.

Why this should help: in offense_formation_coordinates_17.csv every template has
Q=(0,-4) and T=(1,-4) -- identical across all 17. They carry no discriminative
signal, yet the baseline matcher's free Hungarian assignment lets detected
receivers get assigned into the Q/T slots (and vice versa), scrambling the score.
Geometry knows which detected player is the QB and which is the central back, so
we drop them from BOTH sides and match receiver-cloud vs receiver-cloud.

Compares three things against the same CSV, on the same labeled+processed clips:
  baseline   = current template_matcher (free assignment over all skill points)
  role-anchored = receivers-only match (QB + back removed via geometry)

Usage:
  PYTHONPATH=scripts:formations .venv/bin/python3 formations/eval_role_anchored.py
"""

import csv as _csv
import glob
import math
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import template_matcher as tm           # noqa: E402
import line_count_classifier as L       # noqa: E402
from validate_formations import normalize_pred, base_family  # noqa: E402

LABELS_CSV = os.path.join(ROOT, "models", "offense_positions", "play_predictions.csv")
TEMPLATE_CSV = tm.DEFAULT_TEMPLATE_CSV
# Roles that actually discriminate: everything except the QB(Q) and his back(T).
RECEIVER_ROLES = ["W", "S", "X", "Y", "Z", "U"]
COUNT_PENALTY = 0.15   # per missing/extra receiver, added to mean assign distance


def _receiver_templates():
    """name -> RMS-normalized receiver-only point cloud (lateral x, depth y)."""
    out = []
    with open(TEMPLATE_CSV, newline="") as f:
        for row in _csv.DictReader(f):
            name = (row.get("formation") or "").strip()
            if not name:
                continue
            pts = []
            for r in RECEIVER_ROLES:
                xs = (row.get(f"{r}_x") or "").strip()
                ys = (row.get(f"{r}_y") or "").strip()
                if xs and ys:
                    pts.append((float(xs), float(ys)))
            if len(pts) < 2:
                continue
            arr = np.asarray(pts, float)
            arr = arr - arr.mean(axis=0)
            rms = math.sqrt(float((arr ** 2).sum(axis=1).mean())) or 1.0
            out.append({"name": name, "canon": arr / rms, "n": len(pts)})
    return out


def _norm(arr):
    arr = np.asarray(arr, float)
    arr = arr - arr.mean(axis=0)
    rms = math.sqrt(float((arr ** 2).sum(axis=1).mean())) or 1.0
    return arr / rms


def _cost(det, tpl):
    c = np.linalg.norm(det[:, None, :] - tpl[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(c)
    base = float(c[rows, cols].mean())
    return base + COUNT_PENALTY * abs(len(det) - len(tpl))


def _role_anchored_pred(result, rec_templates):
    """Receivers-only match against the CSV; QB + central back removed by geometry."""
    if result.get("on_line_count") is None:
        return None
    los = result["los"]
    los_x, cen, ad = los["x_yd"], los["center_y_yd"], los["attack_dir_x"]
    recv = [p for p in result["players"]
            if p["team"] == "offense" and p.get("grp") == "recv"]
    if len(recv) < 2:
        return None
    # canonical frame: lateral = y - center, depth = (x - LOS)*attack_dir
    det = np.array([[(p["y"] - cen), (p["x"] - los_x) * ad] for p in recv], float)
    det = _norm(det)
    mir = det.copy(); mir[:, 0] = -mir[:, 0]
    ranking = []
    for t in rec_templates:
        d = min(_cost(det, t["canon"]), _cost(mir, t["canon"]))
        ranking.append((t["name"], math.exp(-d / 0.5)))
    ranking.sort(key=lambda r: r[1], reverse=True)
    return ranking[0][0]


def _skill_role_templates():
    """name -> (pts, roles) for QB+back+receivers, RMS-normalized together.

    roles: 'Q' (QB), 'T' (back), 'R' (receiver/TE). Keeps the QB/back depth
    anchor that exact-variant discrimination needs.
    """
    out = []
    with open(TEMPLATE_CSV, newline="") as f:
        for row in _csv.DictReader(f):
            name = (row.get("formation") or "").strip()
            if not name:
                continue
            pts, roles = [], []
            for r in tm.ROLE_ORDER:  # Q, T, W, S, X, Y, Z, U
                xs = (row.get(f"{r}_x") or "").strip()
                ys = (row.get(f"{r}_y") or "").strip()
                if xs and ys:
                    pts.append((float(xs), float(ys)))
                    roles.append("Q" if r == "Q" else "T" if r == "T" else "R")
            if len(pts) < 3:
                continue
            out.append({"name": name, "canon": _norm(pts), "roles": roles})
    return out


def _cost_constrained(det, det_roles, tpl, tpl_roles, pen=5.0):
    c = np.linalg.norm(det[:, None, :] - tpl[None, :, :], axis=2)
    for i, dr in enumerate(det_roles):
        for j, tr in enumerate(tpl_roles):
            if dr != tr:                       # QB only to Q, back only to T, R to R
                c[i, j] += pen
    rows, cols = linear_sum_assignment(c)
    return float(c[rows, cols].mean()) + COUNT_PENALTY * abs(len(det) - len(tpl))


def _role_constrained_pred(result, skill_templates):
    """Keep QB+back+receivers; force role-correct assignment against the CSV."""
    if result.get("on_line_count") is None:
        return None
    los = result["los"]
    los_x, cen, ad = los["x_yd"], los["center_y_yd"], los["attack_dir_x"]
    grp2role = {"qb": "Q", "rb": "T", "recv": "R"}
    skill = [(p, grp2role[p["grp"]]) for p in result["players"]
             if p["team"] == "offense" and p.get("grp") in grp2role]
    if len(skill) < 3:
        return None
    det = _norm([[(p["y"] - cen), (p["x"] - los_x) * ad] for p, _ in skill])
    det_roles = [r for _, r in skill]
    mir = det.copy(); mir[:, 0] = -mir[:, 0]
    ranking = []
    for t in skill_templates:
        d = min(_cost_constrained(det, det_roles, t["canon"], t["roles"]),
                _cost_constrained(mir, det_roles, t["canon"], t["roles"]))
        ranking.append((t["name"], math.exp(-d / 0.5)))
    ranking.sort(key=lambda r: r[1], reverse=True)
    return ranking[0][0]


def main():
    rec_templates = _receiver_templates()
    skill_templates = _skill_role_templates()
    labels = pd.read_csv(LABELS_CSV, index_col=0)["actual_play"].to_dict()
    clip_folder = {}
    for d in sorted(glob.glob(os.path.join(ROOT, "cache", "*", ""))):
        folder = os.path.basename(d.rstrip("/"))
        for snap in glob.glob(os.path.join(d, "snap_detection", "*_snap_detection.json")):
            clip = os.path.basename(snap)[: -len("_snap_detection.json")]
            if (os.path.exists(os.path.join(d, "positions", clip + "_position.json")) and
                    os.path.exists(os.path.join(d, "correspondence", clip + "_correspondence.json"))):
                clip_folder.setdefault(clip, folder)

    base_dir = os.path.join(ROOT, "cache")
    rows = []
    for clip, actual in labels.items():
        folder = clip_folder.get(clip)
        if folder is None:
            continue
        mt = tm.recognize_from_cache(clip, folder, base_dir)
        lc = L.recognize_from_cache(clip, folder, base_dir)
        base_pred = normalize_pred(mt["formation"]) if mt.get("formation") else None
        ra = _role_anchored_pred(lc, rec_templates)
        rc = _role_constrained_pred(lc, skill_templates)
        rows.append({"clip": clip, "actual": str(actual),
                     "base_pred": base_pred,
                     "ra_pred": normalize_pred(ra) if ra else None,
                     "rc_pred": normalize_pred(rc) if rc else None})

    df = pd.DataFrame(rows)

    def score(col, sub=None):
        d = df if sub is None else sub
        d = d[d[col].notna()]
        n = len(d)
        if not n:
            return 0, 0, 0
        ex = (d[col] == d.actual).mean()
        ba = d.apply(lambda r: base_family(r[col]) == base_family(r.actual), axis=1).mean()
        return n, ex, ba

    print(f"labeled + processed clips: {len(df)}\n")
    nb, be, bb = score("base_pred")
    print(f"=== BASELINE matcher (free assignment, all skill pts) — CSV ===")
    print(f"  n={nb}  exact {be:5.1%}  base-family {bb:5.1%}")

    nr, re_, rb = score("ra_pred")
    print(f"\n=== ROLE-ANCHORED (geometry removes QB+back, receivers vs CSV) ===")
    print(f"  n={nr}  exact {re_:5.1%}  base-family {rb:5.1%}")

    nc, ce, cb = score("rc_pred")
    print(f"\n=== ROLE-CONSTRAINED (keep QB+back+recv, forced role assignment vs CSV) ===")
    print(f"  n={nc}  exact {ce:5.1%}  base-family {cb:5.1%}")

    # Head-to-head on clips where ALL methods produced a prediction.
    both = df[df.base_pred.notna() & df.ra_pred.notna() & df.rc_pred.notna()]
    _, be2, bb2 = score("base_pred", both)
    _, re2, rb2 = score("ra_pred", both)
    _, ce2, cb2 = score("rc_pred", both)
    print(f"\n=== HEAD-TO-HEAD (same {len(both)} clips all read) ===")
    print(f"  baseline          exact {be2:5.1%}  base-family {bb2:5.1%}")
    print(f"  role-anchored     exact {re2:5.1%}  base-family {rb2:5.1%}  (Δfam {rb2-bb2:+.1%})")
    print(f"  role-constrained  exact {ce2:5.1%}  base-family {cb2:5.1%}  (Δfam {cb2-bb2:+.1%}, Δexact {ce2-be2:+.1%})")
    df.to_csv(os.path.join(HERE, "role_anchored_eval.csv"), index=False)
    print(f"\nper-clip -> {os.path.relpath(os.path.join(HERE, 'role_anchored_eval.csv'), ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
