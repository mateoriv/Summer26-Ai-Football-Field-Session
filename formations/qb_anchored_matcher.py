#!/usr/bin/env python3
"""
QB-anchored offense formation recognizer.

The legacy template_matcher treats the 11 detected offense points as
role-blind — Hungarian assignment can swap QB with a deep RB or slot WR if
the formation is ambiguous (e.g. some templates have T at the same depth as
QB, so the algorithm has no way to tell them apart).

This matcher uses simple geometry to identify the QB *before* matching, then
pins QB→Q during assignment so the role of the most distinctive backfield
player is fixed. A tight-end candidate is also identified (depth ≈ LOS,
outside the tackles but not split wide) and pinned to Y or U based on the
detected side; this also fixes the mirror.

Strategy (no model, no labels):
  1. Take the same 11 offense field-yard points the legacy matcher takes
     (extract_offense_points_from_cache).
  2. Project into a (lateral, depth) yard frame where backfield is negative
     depth, the 5 forwardmost points are the OL line, and the OL midpoint
     defines lateral = 0.
  3. QB = the non-OL point with the smallest |lateral| (most directly
     behind C) whose depth is in [-7, -1] yd. (T sits next to QB at the
     same depth, so use lateral centeredness, not depth.)
  4. TE candidate = a non-OL point at depth ≈ 0 (within 1.5 yd of the OL
     line) that is outside the tackles (|lateral| > 2.5 yd) but not split
     wide (|lateral| < 6 yd). 0, 1, or 2 candidates.
  5. For each template, build a role-tagged 11-point set (6 skill in CSV
     order + 5 OL). Try each consistent assignment of QB→Q and (if
     present) TE→Y, TE→U; remaining points → remaining template indices
     via Hungarian on a reduced cost matrix. Pick the best total score.
  6. Both detected and template are canonicalized into a shared
     RMS-normalized frame using the same canonicalize() the legacy
     matcher uses, so scores live on the same scale.

If QB identification fails (e.g. wildcat / direct-snap formations with no
clear deep-centered player), the matcher returns None and the caller is
expected to fall back to the legacy result.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

# Reuse the legacy matcher's primitives so scoring is comparable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import template_matcher as _tm  # canonicalize, OL_TEMPLATE, ROLE_ORDER, SCORE_SCALE, etc.

QB_DEPTH_MIN_YD = 1.0    # QB sits at least 1 yd behind the LOS
QB_DEPTH_MAX_YD = 7.0    # deep shotgun caps around 7 yd; anything deeper is a WR
QB_MAX_LATERAL_YD = 2.5  # QB has to be within ~tackle-width of center
TE_DEPTH_BAND_YD = 1.5   # TE is "on the line" within +/- 1.5 yd of LOS
TE_INNER_LATERAL_YD = 2.5
TE_OUTER_LATERAL_YD = 6.0


# --------------------------------------------------------------------------- #
# Geometry: identify QB and TE candidates from 11 field-yard points
# --------------------------------------------------------------------------- #

LOS_BAND_YD = 1.5  # points within +/- this depth of the LOS median are "on the line"


def _orient_in_yards(points_field: np.ndarray):
    """Return (lateral, depth) in yards relative to the OL midpoint.

    PCA picks the lateral (sideline-to-sideline) axis as the major-variance
    axis and depth (LOS-to-backfield) as the perpendicular. The LOS depth
    is set so the median of the LOS-band points is at depth 0, with the
    backfield negative. The OL is then the 5 most-laterally-clustered
    LOS-band points (smallest 5-point lateral span sliding window), and
    the OL median lateral becomes lateral = 0.

    This is the right cluster heuristic for football: WRs and TEs often
    sit at LOS depth too, so "5 tightest in depth" picks the wrong 5 when
    >5 players are on the line. The 5 OL are always the most laterally
    contiguous group of LOS-band players.

    Returns (oriented (11,2) yd, ol_idx list, los_depth_yd=0, ol_lat_span_yd).
    """
    pts = np.asarray(points_field, dtype=float)
    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    lateral_unit = eigvecs[:, int(np.argmax(eigvals))]
    depth_unit = np.array([-lateral_unit[1], lateral_unit[0]])

    lateral_all = centered @ lateral_unit
    depth_all = centered @ depth_unit

    # Pick the LOS depth as the side of the depth distribution where most
    # players cluster. Median of all depths is a decent first cut, but the
    # "true" LOS is where the majority of the LOS-band players sit. Search
    # both sides of the median and pick the band with more points.
    def _band(center):
        return [i for i in range(pts.shape[0]) if abs(depth_all[i] - center) <= LOS_BAND_YD]

    order_by_depth = np.argsort(depth_all)
    lo_center = float(np.median(depth_all[order_by_depth[:6]]))
    hi_center = float(np.median(depth_all[order_by_depth[-6:]]))
    lo_band = _band(lo_center)
    hi_band = _band(hi_center)
    los_band = hi_band if len(hi_band) >= len(lo_band) else lo_band
    los_center = hi_center if len(hi_band) >= len(lo_band) else lo_center
    if len(los_band) < 5:
        # Fallback: top-5 closest to LOS center even if outside the band.
        los_band = sorted(range(pts.shape[0]),
                          key=lambda i: abs(depth_all[i] - los_center))[:5]

    # 5 OL = the most-laterally-clustered window of 5 LOS-band players.
    los_sorted = sorted(los_band, key=lambda i: lateral_all[i])
    if len(los_sorted) <= 5:
        ol_idx = los_sorted[:5]
    else:
        best_span = float("inf")
        ol_idx = los_sorted[:5]
        for k in range(len(los_sorted) - 4):
            window = los_sorted[k:k + 5]
            span = float(lateral_all[window[-1]] - lateral_all[window[0]])
            if span < best_span:
                best_span = span
                ol_idx = window

    # Recenter: lateral 0 = OL median, depth 0 = OL median, backfield negative.
    ol_lat_med = float(np.median(lateral_all[ol_idx]))
    ol_dep_med = float(np.median(depth_all[ol_idx]))
    lateral_all = lateral_all - ol_lat_med
    depth_all = depth_all - ol_dep_med
    skill = [i for i in range(pts.shape[0]) if i not in ol_idx]
    if depth_all[skill].mean() > 0:
        depth_all = -depth_all

    ol_lat_span = float(np.ptp(lateral_all[ol_idx]))
    oriented = np.column_stack([lateral_all, depth_all])
    return oriented, ol_idx, 0.0, ol_lat_span


def _identify_qb(oriented: np.ndarray, ol_idx) -> Optional[int]:
    """Pick the QB index from oriented (lateral, depth) yards.

    QB = backfield non-OL point most directly behind center.
    Returns the index into the original 11 points, or None.
    """
    skill_idx = [i for i in range(oriented.shape[0]) if i not in ol_idx]
    candidates = []
    for i in skill_idx:
        lat, dep = oriented[i]
        if dep > -QB_DEPTH_MIN_YD or dep < -QB_DEPTH_MAX_YD:
            continue
        if abs(lat) > QB_MAX_LATERAL_YD:
            continue
        candidates.append((abs(lat), i))
    if not candidates:
        return None
    candidates.sort()
    # Reject if two candidates are equally centered (ambiguous, e.g. I-form QB+FB
    # would both be near 0 lateral but at different depths — pick the deeper).
    if len(candidates) >= 2 and abs(candidates[0][0] - candidates[1][0]) < 0.4:
        # Among the centered candidates, prefer the deeper (more negative depth) one.
        tied = [c for c in candidates if c[0] - candidates[0][0] < 0.4]
        tied.sort(key=lambda c: oriented[c[1], 1])  # most negative depth first
        return tied[0][1]
    return candidates[0][1]


def _identify_te(oriented: np.ndarray, ol_idx, qb_idx: int):
    """Pick TE candidates (0, 1, or 2) from oriented yards.

    TE = non-OL point at depth ≈ 0 that is outside the tackles
    (|lateral| > 2.5) but not split wide (|lateral| < 6).
    Returns list of (idx, side) where side ∈ {"left", "right"} by lateral
    sign (positive = right).
    """
    out = []
    for i in range(oriented.shape[0]):
        if i in ol_idx or i == qb_idx:
            continue
        lat, dep = oriented[i]
        if abs(dep) > TE_DEPTH_BAND_YD:
            continue
        if not (TE_INNER_LATERAL_YD < abs(lat) < TE_OUTER_LATERAL_YD):
            continue
        out.append((i, "right" if lat > 0 else "left"))
    return out


# --------------------------------------------------------------------------- #
# Role-tagged templates
# --------------------------------------------------------------------------- #

def _load_templates_with_roles(csv_path: str = _tm.DEFAULT_TEMPLATE_CSV):
    """Like template_matcher.load_templates, but also returns role tags.

    Returns list of dicts:
      {
        "name": str,
        "canon": (11, 2) array — canonicalized, same as legacy matcher,
        "roles": [str] — length 11, role at each index (e.g. "Q","T","W",..."OL","OL",...),
        "qb_idx": int,
        "te_indices": {"Y": int|None, "U": int|None},
      }
    """
    out = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("formation") or "").strip()
            if not name:
                continue
            skill_pts, skill_roles = [], []
            for role in _tm.ROLE_ORDER:
                xs = (row.get(f"{role}_x") or "").strip()
                ys = (row.get(f"{role}_y") or "").strip()
                if xs == "" or ys == "":
                    continue
                skill_pts.append((float(xs), float(ys)))
                skill_roles.append(role)
            if not skill_pts:
                continue
            pts = np.vstack([np.asarray(skill_pts, dtype=float), _tm.OL_TEMPLATE])
            canon, _ = _tm.canonicalize(pts)
            if canon is None:
                continue
            roles = skill_roles + ["OL"] * 5
            qb_idx = roles.index("Q") if "Q" in roles else None
            te_indices = {
                "Y": roles.index("Y") if "Y" in roles else None,
                "U": roles.index("U") if "U" in roles else None,
            }
            if qb_idx is None:
                continue  # every real formation has a QB; skip otherwise
            out.append({
                "name": name,
                "canon": canon,
                "roles": roles,
                "qb_idx": qb_idx,
                "te_indices": te_indices,
            })
    if not out:
        raise ValueError(f"No templates loaded from {csv_path}")
    return out


# --------------------------------------------------------------------------- #
# Constrained matching
# --------------------------------------------------------------------------- #

def _constrained_cost(detected_canon: np.ndarray, template_canon: np.ndarray, forced):
    """Mean per-point assignment cost with `forced` (det_idx, tpl_idx) pairs.

    Forced pairs are removed from the cost matrix and their distances added
    back to the total. Hungarian runs on the remainder.
    """
    n = detected_canon.shape[0]
    forced_d = forced or []
    det_keep = [i for i in range(n) if i not in {d for d, _ in forced_d}]
    tpl_keep = [j for j in range(n) if j not in {t for _, t in forced_d}]
    sub = np.linalg.norm(
        detected_canon[det_keep][:, None, :] - template_canon[tpl_keep][None, :, :],
        axis=2,
    )
    if sub.size == 0:
        forced_dist = sum(
            float(np.linalg.norm(detected_canon[d] - template_canon[t]))
            for d, t in forced_d
        )
        return forced_dist / max(n, 1)
    rows, cols = linear_sum_assignment(sub)
    total = float(sub[rows, cols].sum())
    for d, t in forced_d:
        total += float(np.linalg.norm(detected_canon[d] - template_canon[t]))
    return total / n


def _best_mirror_cost(detected_canon, template, forced_no_mirror, forced_with_mirror):
    """Pick the lower of the two mirror orientations.

    forced_no_mirror and forced_with_mirror are the constraint lists for the
    un-mirrored and mirrored detections respectively (they can differ when
    TE side is fixed by sign).
    """
    mirrored = detected_canon.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    a = _constrained_cost(detected_canon, template["canon"], forced_no_mirror)
    b = _constrained_cost(mirrored, template["canon"], forced_with_mirror)
    return min(a, b)


def _match_constrained(detected_canon, qb_det_idx, te_det, templates):
    """Rank templates with QB→Q (always) and TE→{Y,U} (when possible)."""
    results = []
    for tpl in templates:
        q_t = tpl["qb_idx"]
        # Forced lists in both mirrors. For TE side: in the un-mirrored frame the
        # detected lateral sign maps directly to the template lateral sign; in
        # the mirrored frame it flips. So we can only "fix" the mirror when a
        # TE is detected AND the template has a TE on the matching side.
        forced_a = [(qb_det_idx, q_t)]
        forced_b = [(qb_det_idx, q_t)]
        for det_idx, side in te_det:
            # Determine which template TE slot (Y or U) sits on which side, in
            # *canonical* coords (post-canonicalize sign is ambiguous, so we
            # treat both sides per template). We let the constrained_cost try
            # each TE pairing and keep only consistent ones.
            for role in ("Y", "U"):
                t_idx = tpl["te_indices"].get(role)
                if t_idx is None:
                    continue
                forced_a.append((det_idx, t_idx))
                forced_b.append((det_idx, t_idx))
                break  # one TE -> one role; if 2 TEs, second loop iter picks U

        # If the same det_idx ended up forced to multiple template indices
        # (shouldn't, given the break, but guard), keep only the first.
        seen_det = set()
        forced_a = [p for p in forced_a if not (p[0] in seen_det or seen_det.add(p[0]))]
        seen_det = set()
        forced_b = [p for p in forced_b if not (p[0] in seen_det or seen_det.add(p[0]))]

        dist = _best_mirror_cost(detected_canon, tpl, forced_a, forced_b)
        score = math.exp(-dist / _tm.SCORE_SCALE)
        results.append((tpl["name"], score, dist))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


# --------------------------------------------------------------------------- #
# Public API: same shape as template_matcher.recognize_points
# --------------------------------------------------------------------------- #

_TEMPLATES_CACHE = None


def _templates():
    global _TEMPLATES_CACHE
    if _TEMPLATES_CACHE is None:
        _TEMPLATES_CACHE = _load_templates_with_roles()
    return _TEMPLATES_CACHE


def recognize_points(points_field, templates=None):
    """Recognize the formation by anchoring on a geometrically-identified QB.

    Returns a dict with the same keys the legacy matcher returns
    ({"formation","score","dist","reliable","depth_span_yd","n_points",
      "ranking","method","qb_idx","te_count"}) or
    {"formation": None, "reason": ...} when QB identification fails.
    """
    if templates is None:
        templates = _templates()
    pts = np.asarray(points_field, dtype=float)
    if pts.shape[0] < 6:
        return {"formation": None, "reason": f"only {pts.shape[0]} offense points"}
    if pts.shape[0] != 11:
        return {"formation": None, "reason": f"need 11 points, got {pts.shape[0]}"}

    oriented, ol_idx, _, ol_span = _orient_in_yards(pts)
    qb_idx = _identify_qb(oriented, ol_idx)
    if qb_idx is None:
        return {"formation": None, "reason": "no geometric QB candidate"}
    te_det = _identify_te(oriented, ol_idx, qb_idx)

    detected_canon, depth_span = _tm.canonicalize(pts)
    if detected_canon is None:
        return {"formation": None, "reason": "degenerate geometry"}

    ranking = _match_constrained(detected_canon, qb_idx, te_det, templates)
    name, score, dist = ranking[0]
    reliable = depth_span <= _tm.MAX_RELIABLE_DEPTH_SPAN_YD
    return {
        "formation": name,
        "score": round(float(score), 3),
        "dist": round(float(dist), 3),
        "reliable": bool(reliable),
        "depth_span_yd": round(float(depth_span), 1),
        "n_points": int(pts.shape[0]),
        "ranking": [(n, round(float(s), 3)) for n, s, _ in ranking[:5]],
        "method": "qb_anchor",
        "qb_idx": int(qb_idx),
        "te_count": int(len(te_det)),
        "ol_lat_span_yd": round(float(ol_span), 1),
    }


def recognize_from_cache(video_name, folder_name, base_cache_dir):
    """Same input contract as template_matcher.recognize_from_cache, but
    routes through the QB-anchored matcher. Returns the same dict shape."""
    pts, err = _tm.extract_offense_points_from_cache(video_name, folder_name, base_cache_dir)
    if pts is None:
        return {"formation": None, "reason": err}
    return recognize_points(pts)


# --------------------------------------------------------------------------- #
# Geometric feature vector for the MLP (training + inference)
# --------------------------------------------------------------------------- #

# Order MUST match _QB_FEATURE_NAMES; do not reorder without bumping the
# model metadata's feature list.
QB_FEATURE_NAMES = (
    "qb_lateral_yd",
    "qb_depth_yd",
    "te_left",
    "te_right",
    "ol_lat_span_yd",
)


def qb_features_for_points(points_field):
    """Return a 5-vector of QB-derived geometric features for one snap.

    Always returns a length-5 vector. When the geometry is degenerate or the
    QB cannot be identified, the QB-specific entries are zeroed (so the MLP
    sees a clean "no signal" placeholder rather than a NaN). The order is
    fixed by QB_FEATURE_NAMES so train/inference stay in lockstep.

    Inputs:
      points_field : array-like (N, 2) of (x, y) field yards. N should be
                     11 for normal snaps; <6 returns zeros.
    """
    pts = np.asarray(points_field, dtype=float)
    out = np.zeros(len(QB_FEATURE_NAMES), dtype=np.float32)
    if pts.shape[0] < 6 or pts.shape[1] != 2:
        return out
    oriented, ol_idx, _, ol_span = _orient_in_yards(pts)
    out[4] = float(ol_span)
    qb_idx = _identify_qb(oriented, ol_idx)
    if qb_idx is None:
        return out
    out[0] = float(oriented[qb_idx, 0])
    out[1] = float(oriented[qb_idx, 1])
    for _, side in _identify_te(oriented, ol_idx, qb_idx):
        if side == "left":
            out[2] = 1.0
        else:
            out[3] = 1.0
    return out


def main():
    ap = argparse.ArgumentParser(description="QB-anchored offense formation recognition")
    ap.add_argument("--video-name", required=True)
    ap.add_argument("--folder-name", required=True)
    ap.add_argument("--cache-dir", default="cache",
                    help="absolute, or relative to the project root")
    args = ap.parse_args()

    if os.path.isabs(args.cache_dir):
        base = args.cache_dir
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.cache_dir)

    result = recognize_from_cache(args.video_name, args.folder_name, base)
    print(json.dumps(result, indent=2))
    return 0 if result.get("formation") else 1


if __name__ == "__main__":
    sys.exit(main())
