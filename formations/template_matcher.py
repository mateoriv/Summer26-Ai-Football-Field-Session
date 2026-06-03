#!/usr/bin/env python3
"""
Offense formation recognition by template matching (no training, no labels).

The repo ships 17 canonical offensive formations in
`formations/offense_formation_coordinates_17.csv`. Each row gives the ideal
positions of the **6 skill players** (the 11 offense minus the 5 OL) in a
ball-relative frame:

    x = lateral offset from the ball, in yards (+/- = the two sides)
    y = depth, in yards (0 = line of scrimmage, negative = backfield; QB at -4)

To recognize a play we:

  1. take the 11 offense players at the snap in field yards (the generic player
     detector reliably finds all 11; the role-tagged detector is too sparse at
     the snap, so we don't depend on roles -- see git history / docstring note),
  2. build an 11-point template per formation by adding 5 OL anchored on the LOS,
  3. canonicalize both detected and template point sets into a shared frame
     (PCA -> lateral/depth axes, center, isotropic RMS scale-normalize) so the
     match is invariant to where on the field the play is, its orientation, and
     the schematic-vs-measured scale difference,
  4. assign detected points to template points with an optimal (Hungarian)
     assignment, mirror-invariant, and return the closest formation + a score.

Template matching is deliberate: with only ~89 noisy labeled clips a trained
classifier overfits. These 17 clean templates encode the formations directly,
need no labels, and the per-formation distances stay interpretable.

The reliable 11-point extraction lives in `scripts/staticProcess.py`
(`get_offense_points_for_video`); this module consumes those points, so the
pipeline can recognize without re-projecting.
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

# The 8 template role columns, in CSV order.
ROLE_ORDER = ["Q", "T", "W", "S", "X", "Y", "Z", "U"]

# 5 offensive linemen anchored on the LOS (depth 0), ~1.25 yd splits, in the
# templates' schematic scale. They anchor the LOS/frame; the 6 skill positions
# carry the discriminative signal.
OL_TEMPLATE = np.array(
    [[-2.5, 0.0], [-1.25, 0.0], [0.0, 0.0], [1.25, 0.0], [2.5, 0.0]], dtype=float
)

# Convert mean per-player assignment distance (in RMS-normalized units, ~0-1) to
# a 0-1 score. A clean match lands ~0.2; ~0.6 is a poor fit.
SCORE_SCALE = 0.5

# If the pre-normalization depth spread exceeds this (yards), the field
# projection is distorted (e.g. a bad homography frame); the result is flagged
# unreliable rather than trusted.
MAX_RELIABLE_DEPTH_SPAN_YD = 12.0

DEFAULT_TEMPLATE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "offense_formation_coordinates_17.csv"
)


# --------------------------------------------------------------------------- #
# Templates
# --------------------------------------------------------------------------- #

def load_templates(csv_path=DEFAULT_TEMPLATE_CSV):
    """Load the canonical formations.

    Returns a list of dicts: {"name": str, "canon": (11, 2) array} where each
    template is the 6 skill points plus 5 LOS-anchored OL, already canonicalized
    into the shared matching frame.
    """
    templates = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("formation") or "").strip()
            if not name:
                continue
            skill = []
            for role in ROLE_ORDER:
                xs = (row.get(f"{role}_x") or "").strip()
                ys = (row.get(f"{role}_y") or "").strip()
                if xs == "" or ys == "":
                    continue
                skill.append((float(xs), float(ys)))
            if not skill:
                continue
            pts = np.vstack([np.asarray(skill, dtype=float), OL_TEMPLATE])
            canon, _ = canonicalize(pts)
            if canon is None:
                continue
            templates.append({"name": name, "canon": canon})
    if not templates:
        raise ValueError(f"No templates loaded from {csv_path}")
    return templates


# --------------------------------------------------------------------------- #
# Canonical frame
# --------------------------------------------------------------------------- #

def canonicalize(points):
    """Map a point set into the shared matching frame.

    Steps: center; rotate so the major-spread (lateral) axis is x and the
    perpendicular (depth) axis is y; orient depth so the sparse backfield tail is
    negative (template convention); recenter; isotropic RMS scale-normalize.

    Returns (canon (n,2) array, depth_span_yd) or (None, None) if degenerate.
    The lateral *sign* is left ambiguous and resolved by mirror-invariant
    matching. `depth_span_yd` is the pre-normalization depth spread, used to flag
    distorted projections.
    """
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] < 3:
        return None, None

    centered = pts - pts.mean(axis=0, keepdims=True)
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    lateral_unit = eigvecs[:, int(np.argmax(eigvals))]
    depth_unit = np.array([-lateral_unit[1], lateral_unit[0]])

    lateral = centered @ lateral_unit
    depth = centered @ depth_unit
    # Orient depth so the extreme (deepest back) is on the negative side.
    if abs(depth.max()) > abs(depth.min()):
        depth = -depth

    depth_span_yd = float(np.ptp(depth))

    frame = np.column_stack([lateral, depth])
    frame = frame - frame.mean(axis=0, keepdims=True)
    rms = math.sqrt(float((frame ** 2).sum(axis=1).mean()))
    if rms < 1e-6:
        return None, None
    return frame / rms, depth_span_yd


# --------------------------------------------------------------------------- #
# Matching
# --------------------------------------------------------------------------- #

def _assignment_cost(detected, template):
    """Mean optimal-assignment distance between two canonicalized point sets."""
    cost = np.linalg.norm(detected[:, None, :] - template[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    return float(cost[rows, cols].mean())


def match(detected_canon, templates):
    """Rank templates against a canonicalized detected set, mirror-invariant.

    Returns a list of (name, score, dist), best first.
    """
    detected = np.asarray(detected_canon, dtype=float)
    mirrored = detected.copy()
    mirrored[:, 0] = -mirrored[:, 0]

    results = []
    for tpl in templates:
        dist = min(_assignment_cost(detected, tpl["canon"]),
                   _assignment_cost(mirrored, tpl["canon"]))
        score = math.exp(-dist / SCORE_SCALE)
        results.append((tpl["name"], score, dist))
    results.sort(key=lambda r: r[1], reverse=True)
    return results


def recognize_points(points_field, templates=None):
    """Recognize the formation from 11 offense points in field yards.

    `points_field` is an (n, 2) array-like of (x, y) field coordinates (the
    `nx`/`ny` columns from offense_positions.csv / get_offense_points_for_video).

    Returns a result dict (see recognize_from_cache) or
    {"formation": None, "reason": ...}.
    """
    if templates is None:
        templates = load_templates()
    pts = np.asarray(points_field, dtype=float)
    if pts.shape[0] < 6:
        return {"formation": None, "reason": f"only {pts.shape[0]} offense points"}

    detected, depth_span = canonicalize(pts)
    if detected is None:
        return {"formation": None, "reason": "degenerate geometry"}

    ranking = match(detected, templates)
    name, score, dist = ranking[0]
    reliable = depth_span <= MAX_RELIABLE_DEPTH_SPAN_YD
    return {
        "formation": name,
        "score": round(float(score), 3),
        "dist": round(float(dist), 3),
        "reliable": bool(reliable),
        "depth_span_yd": round(float(depth_span), 1),
        "n_points": int(pts.shape[0]),
        "ranking": [(n, round(float(s), 3)) for n, s, _ in ranking[:5]],
    }


# --------------------------------------------------------------------------- #
# Cache-driven recognition (CLI / standalone use)
# --------------------------------------------------------------------------- #

# --- Torch-free offense-11 extraction from the cache JSON --------------------- #
# Mirrors the (torch-free) logic in scripts/staticProcess.py so this module can
# extract points without importing staticProcess (which imports torch at module
# load). Keep these in sync with staticProcess.get_offense_points_for_video.

_FIELD_WIDTH_YD = 160 / 3


def _norm_cls(name):
    return (name or "").strip().lower().replace(" ", "_")


def _offense_side(position_detections, image_width):
    import pandas as pd
    offense_x, defense_x = [], []
    for det in position_detections:
        cls = _norm_cls(det.get("class") or "")
        cx = (det.get("bbox") or {}).get("center_x")
        if cx is None or cls == "ref":
            continue
        (defense_x if cls == "defense" else offense_x).append(float(cx))
    if not offense_x:
        return None
    if defense_x:
        dmed = float(pd.Series(defense_x).median())
        left = sum(1 for x in offense_x if x < dmed)
        right = sum(1 for x in offense_x if x > dmed)
        if left != right:
            return "left" if left > right else "right"
    return "right" if float(pd.Series(offense_x).median()) > image_width / 2.0 else "left"


def _first_11_on_side(detections, side):
    pts = []
    for det in detections:
        npos = det.get("normalized_position") or {}
        bbox = det.get("original_bbox") or {}
        nx, ny = npos.get("x"), npos.get("y")
        ox, oy = bbox.get("center_x"), bbox.get("center_y")
        if None in (nx, ny, ox, oy):
            continue
        pts.append([float(nx), float(ny), float(ox), float(oy)])
    if not pts:
        return []
    pts.sort(key=lambda p: p[0])
    pts = pts[:11] if side == "left" else pts[-11:]
    pts.sort(key=lambda p: p[1])
    if side == "right":  # mirror right-side attack onto the left, as staticProcess does
        x_min = min(p[0] for p in pts)
        for p in pts:
            p[0] = float(p[0] - 2 * (p[0] - x_min))
    return pts


def extract_offense_points_from_cache(video_name, folder_name, base_cache_dir):
    """Return the 11 offense (nx, ny) field points at the snap, or (None, reason)."""
    snap_p = os.path.join(base_cache_dir, folder_name, "snap_detection", f"{video_name}_snap_detection.json")
    pos_p = os.path.join(base_cache_dir, folder_name, "positions", f"{video_name}_position.json")
    homo_p = os.path.join(base_cache_dir, folder_name, "homography", f"{video_name}_normalized_positions.json")
    for p in (snap_p, pos_p, homo_p):
        if not os.path.exists(p):
            return None, f"missing {os.path.basename(p)}"

    snaps = (_load(snap_p).get("snaps") or [])
    if not snaps:
        return None, "no snap frame"
    snap_frame = snaps[0].get("frame")

    pdata = _load(pos_p)
    width = float((pdata.get("video_info") or {}).get("width") or 1920)
    pos_dets = []
    for fr in (pdata.get("frames") or []):
        if fr.get("frame_number") == snap_frame:
            pos_dets = fr.get("detections") or []
            break
    side = _offense_side(pos_dets, width)
    if side is None:
        return None, "could not determine offense side"

    normalized = (_load(homo_p).get("normalized_positions") or {}).get(str(int(snap_frame))) or []
    pts = _first_11_on_side(normalized, side)
    if len(pts) < 11:
        return None, f"only {len(pts)} players on offense side"
    return [(p[0], p[1]) for p in pts], None


def _load(path):
    with open(path, "r") as f:
        return json.load(f)


# --- Track-aggregated, role-aware recognition (the smarter matcher) ---------- #
# When positionDetection ran with ByteTrack (over a window around the snap),
# each detection carries a stable track_id. We aggregate per-track classes
# (confidence-weighted majority vote across the window), which denoises the
# detector's single-frame label noise -- the same QB visible in many frames
# but mis-labeled differently per frame collapses to a single stable role.
# We then project the cleaned tracks to field yards with the per-frame
# homography, use the REAL OL line as the canonical (lateral, depth) frame,
# and match skill players against the 6-skill (no-OL) templates. This
# sidesteps both the count-noise of single-frame detections AND the OL-
# dilution of full-11-point matching.

SKILL_CLASSES = frozenset({"qb", "running_back", "wide_receiver", "tight_end"})
OL_CLASS_NAME = "oline"


def _aggregate_tracks(positions_data, snap_frame, window=None):
    """Group detections by track_id over [snap-window, snap+window] and vote.

    Returns: list of {class, bbox, track_id, confidence, frames_seen}.
    The bbox is the one from the frame closest to the snap among detections of
    the voted-winner class for that track.
    """
    frames = positions_data.get("frames", [])
    if window is None:
        window = int((positions_data.get("tracking") or {}).get("window") or 30)
    lo = snap_frame - window
    hi = snap_frame + window

    by_id = {}
    snap_only = []  # detections with no track_id -- only kept at the snap itself
    for fr in frames:
        n = fr.get("frame_number")
        if n is None or n < lo or n > hi:
            continue
        for d in fr.get("detections", []):
            tid = d.get("track_id")
            if tid is None:
                if n == snap_frame:
                    snap_only.append(d)
                continue
            by_id.setdefault(tid, []).append((n, d))

    cleaned = []
    for tid, dets in by_id.items():
        votes = {}
        for _, d in dets:
            c = (d.get("class") or "").lower()
            if not c:
                continue
            votes[c] = votes.get(c, 0.0) + float(d.get("confidence") or 1.0)
        if not votes:
            continue
        cls = max(votes, key=votes.get)
        cands = [(n, d) for n, d in dets if (d.get("class") or "").lower() == cls]
        cands.sort(key=lambda x: abs(x[0] - snap_frame))
        chosen = cands[0][1]
        cleaned.append({
            "class": cls,
            "bbox": chosen.get("bbox") or {},
            "track_id": tid,
            "confidence": votes[cls],
            "frames_seen": len(dets),
        })
    # Untracked snap-frame detections kept as fallback (e.g. legacy JSON, or new
    # tracks that didn't get an ID at the snap exactly).
    for d in snap_only:
        cleaned.append({
            "class": (d.get("class") or "").lower(),
            "bbox": d.get("bbox") or {},
            "track_id": None,
            "confidence": float(d.get("confidence") or 0.0),
            "frames_seen": 1,
        })
    return cleaned


def _project_cleaned(cleaned, H):
    """Project cleaned tracks to field yards via H. Returns {ol, skill, qb}."""
    from perFrameHomographyTransform import transform_point  # torch-free
    out = {"ol": [], "skill": [], "qb": []}
    for t in cleaned:
        b = t.get("bbox") or {}
        cx, cy = b.get("center_x"), b.get("center_y")
        if cx is None or cy is None:
            continue
        p = transform_point((cx, cy), H)
        if p is None:
            continue
        cls = t["class"]
        if cls == OL_CLASS_NAME:
            out["ol"].append(p)
        elif cls in SKILL_CLASSES:
            out["skill"].append(p)
            if cls == "qb":
                out["qb"].append(p)
    return out


def _ol_canonical_frame(skill_field, ol_field):
    """Express skill points in (lateral, depth) yards using the OL line.

    OL line direction -> lateral axis; perpendicular -> depth; depth oriented
    so the backfield (skill centroid relative to OL) is negative -- the same
    convention the templates use.
    """
    skill = np.asarray(skill_field, dtype=float)
    if skill.shape[0] == 0:
        return None
    ol = np.asarray(ol_field, dtype=float)
    if ol.shape[0] >= 2:
        origin = np.median(ol, axis=0)
        centered = ol - ol.mean(axis=0)
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        lateral = eigvecs[:, int(np.argmax(eigvals))]
    else:
        # No OL line -- fall back to skill PCA so we still return something.
        origin = np.median(skill, axis=0)
        c = skill - skill.mean(axis=0)
        cov = c.T @ c
        eigvals, eigvecs = np.linalg.eigh(cov)
        lateral = eigvecs[:, int(np.argmax(eigvals))]
    depth = np.array([-lateral[1], lateral[0]])

    rel = skill - origin
    L = rel @ lateral
    D = rel @ depth
    if D.mean() > 0:
        D = -D
    return np.column_stack([L, D])


_SKILL_TEMPLATES_CACHE = None


def _load_skill_templates(csv_path=DEFAULT_TEMPLATE_CSV):
    """6-skill (no-OL) canonicalized templates for role-aware matching."""
    global _SKILL_TEMPLATES_CACHE
    if _SKILL_TEMPLATES_CACHE is not None:
        return _SKILL_TEMPLATES_CACHE
    out = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            name = (row.get("formation") or "").strip()
            if not name:
                continue
            pts = []
            for r in ROLE_ORDER:
                xs = (row.get(f"{r}_x") or "").strip()
                ys = (row.get(f"{r}_y") or "").strip()
                if xs and ys:
                    pts.append((float(xs), float(ys)))
            if len(pts) < 3:
                continue
            arr = np.asarray(pts, dtype=float)
            arr = arr - arr.mean(axis=0)
            rms = math.sqrt(float((arr ** 2).sum(axis=1).mean()))
            if rms < 1e-6:
                continue
            out.append({"name": name, "canon": arr / rms})
    _SKILL_TEMPLATES_CACHE = out
    return out


def _canon_skill(skill_yd):
    """Center + isotropic RMS scale a skill point set already in OL-yard frame."""
    P = np.asarray(skill_yd, dtype=float) - np.asarray(skill_yd, dtype=float).mean(axis=0)
    rms = math.sqrt(float((P ** 2).sum(axis=1).mean()))
    if rms < 1e-6:
        return None
    return P / rms


def recognize_from_tracks(positions_data, frame_correspondences, snap_frame,
                          window=None, skill_templates=None):
    """Tracked + role-aware formation recognition (skill-only matching)."""
    from perFrameHomographyTransform import (
        gather_window_correspondences, get_homography_matrix, field_points_are_degenerate,
    )

    if skill_templates is None:
        skill_templates = _load_skill_templates()

    # Best homography frame within +/-20 of snap.
    H = None
    h_frame = snap_frame
    for fr in range(max(0, snap_frame - 20), snap_frame + 21):
        ip, fp = gather_window_correspondences(frame_correspondences, fr, 15)
        if len(ip) < 4 or field_points_are_degenerate(fp):
            continue
        H = get_homography_matrix(ip, fp)
        h_frame = fr
        break
    if H is None:
        return {"formation": None, "reason": "no valid homography near snap"}

    cleaned = _aggregate_tracks(positions_data, snap_frame, window=window)
    if not cleaned:
        return {"formation": None, "reason": "no tracks in window"}

    proj = _project_cleaned(cleaned, H)
    n_sk, n_ol, n_qb = len(proj["skill"]), len(proj["ol"]), len(proj["qb"])
    if n_sk < 3:
        return {"formation": None, "reason": f"too few skill players ({n_sk}) after track aggregation"}

    skill_yd = _ol_canonical_frame(proj["skill"], proj["ol"])
    if skill_yd is None:
        return {"formation": None, "reason": "could not build OL frame"}
    detected = _canon_skill(skill_yd)
    if detected is None:
        return {"formation": None, "reason": "degenerate skill geometry"}

    # Mirror-invariant Hungarian over skill-only templates.
    ranking = []
    mirrored = detected.copy()
    mirrored[:, 0] = -mirrored[:, 0]
    for tpl in skill_templates:
        dist = min(_assignment_cost(detected, tpl["canon"]),
                   _assignment_cost(mirrored, tpl["canon"]))
        ranking.append((tpl["name"], math.exp(-dist / SCORE_SCALE), dist))
    ranking.sort(key=lambda r: r[1], reverse=True)
    name, score, dist = ranking[0]

    # Reliability: enough cleaned offense to trust the geometry.
    reliable = (n_sk >= 5 and n_ol >= 3)
    return {
        "formation": name,
        "score": round(float(score), 3),
        "dist": round(float(dist), 3),
        "reliable": bool(reliable),
        "method": "tracked",
        "n_skill": n_sk,
        "n_ol": n_ol,
        "qb_detected": bool(n_qb > 0),
        "homography_frame": h_frame,
        "ranking": [(n, round(float(s), 3)) for n, s, _ in ranking[:5]],
    }


def recognize_from_cache(video_name, folder_name, base_cache_dir, templates=None):
    """Recognize the formation for one clip from the cache JSON.

    Dispatches: when the positions JSON contains track_ids (new tracked
    detector), uses the role-aware tracked path; otherwise falls back to the
    legacy 11-generic-point path. Same output shape either way -- consumers
    just read `formation` / `score` / `reliable`.
    """
    snap_p = os.path.join(base_cache_dir, folder_name, "snap_detection", f"{video_name}_snap_detection.json")
    pos_p = os.path.join(base_cache_dir, folder_name, "positions", f"{video_name}_position.json")
    corr_p = os.path.join(base_cache_dir, folder_name, "correspondence", f"{video_name}_correspondence.json")
    for p in (snap_p, pos_p, corr_p):
        if not os.path.exists(p):
            return {"formation": None, "reason": f"missing {os.path.basename(p)}"}

    pdata = _load(pos_p)
    snaps = (_load(snap_p).get("snaps") or [])
    if not snaps:
        return {"formation": None, "reason": "no snap frame"}
    snap_frame = snaps[0].get("frame")

    has_tracks = any(
        d.get("track_id") is not None
        for fr in pdata.get("frames", [])
        for d in fr.get("detections", [])
    )
    if has_tracks:
        corr = _load(corr_p).get("frame_correspondences", {})
        return recognize_from_tracks(pdata, corr, snap_frame)

    # Legacy fallback: 11 generic offense points at the snap.
    field_pts, err = extract_offense_points_from_cache(video_name, folder_name, base_cache_dir)
    if field_pts is None:
        return {"formation": None, "reason": err}
    return recognize_points(field_pts, templates=templates)


def main():
    ap = argparse.ArgumentParser(description="Offense formation recognition by template matching")
    ap.add_argument("--video-name", required=True)
    ap.add_argument("--folder-name", required=True)
    ap.add_argument("--cache-dir", default="cache",
                    help="absolute, or relative to the project root")
    ap.add_argument("--templates", default=DEFAULT_TEMPLATE_CSV)
    args = ap.parse_args()

    if os.path.isabs(args.cache_dir):
        base = args.cache_dir
    else:
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.cache_dir)

    result = recognize_from_cache(
        args.video_name, args.folder_name, base, templates=load_templates(args.templates)
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("formation") else 1


if __name__ == "__main__":
    sys.exit(main())
