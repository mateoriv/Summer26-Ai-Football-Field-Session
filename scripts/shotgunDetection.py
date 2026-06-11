#!/usr/bin/env python3
"""
QB alignment / shotgun detection (geometric -- no model, no labels).

Shotgun is defined by how far the QB lines up behind the offensive line, so it
is a measurement, not a learned pattern. This reads the position detections
(which carry distinct `qb` and `oline` classes) at/near the snap frame,
projects the QB and OL to field yards via the per-frame homography, and
classifies the QB's depth behind the line.

Robustness:
  * QB detection is sparse, so search a window around the snap for the best
    frame that has a QB + enough OL + a valid (non-degenerate) homography.
  * If no QB is found in the window, fall back to the deepest offensive-skill
    player behind the OL (the QB by geometry, regardless of label).
  * Report which path was used so low-confidence cases are visible.
"""

import argparse
import json
import os
import sys

import numpy as np

# Reuse the homography helpers (incl. the temporal aggregation + degeneracy guard).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from perFrameHomographyTransform import (
    gather_window_correspondences,
    get_homography_matrix,
    transform_point,
    field_points_are_degenerate,
)
from ioutils import load_json

OFFENSE_SKILL = frozenset({"qb", "running_back", "wide_receiver", "tight_end"})

# Depth thresholds in yards (tunable on real clips).
UNDER_CENTER_MAX = 1.5
PISTOL_MAX = 4.0


def _frame_dets(positions, frame_number):
    for f in positions.get("frames", []):
        if f.get("frame_number") == frame_number:
            return f.get("detections", [])
    return []


def _center(det):
    b = det.get("bbox", {})
    return b.get("center_x"), b.get("center_y")


def _homography_for(frame_corr, frame_number, window):
    ip, fp = gather_window_correspondences(frame_corr, frame_number, window)
    if len(ip) < 4 or field_points_are_degenerate(fp):
        return None
    return get_homography_matrix(ip, fp)


def _classify(depth_yd):
    if depth_yd < UNDER_CENTER_MAX:
        return "under_center"
    if depth_yd < PISTOL_MAX:
        return "pistol"
    return "shotgun"


def detect_alignment(snap_frame, positions, frame_corr, search_radius=60, window=15):
    """Return a dict describing QB alignment at/near the snap frame."""
    # 1) Prefer a frame with a real QB + OL + valid homography, closest to snap.
    best = None  # (score_tuple, frame, qb_det, ol_dets, H)
    for fr in range(max(0, snap_frame - search_radius), snap_frame + search_radius + 1):
        dets = _frame_dets(positions, fr)
        if not dets:
            continue
        qbs = [d for d in dets if d.get("class") == "qb"]
        ols = [d for d in dets if d.get("class") == "oline"]
        if not qbs or len(ols) < 3:
            continue
        H = _homography_for(frame_corr, fr, window)
        if H is None:
            continue
        qb = max(qbs, key=lambda d: d.get("confidence", 0.0))  # highest-confidence QB
        score = (len(ols), -abs(fr - snap_frame))
        cand = (score, fr, qb, ols, H)
        if best is None or score > best[0]:
            best = cand

    if best is not None:
        _, fr, qb, ols, H = best
        qb_f = transform_point(_center(qb), H)
        ol_f = [transform_point(_center(o), H) for o in ols]
        ol_f = [p for p in ol_f if p is not None]
        if qb_f is not None and ol_f:
            ol_x = float(np.median([p[0] for p in ol_f]))
            depth = abs(qb_f[0] - ol_x)
            return {
                "alignment": _classify(depth),
                "qb_depth_yd": round(depth, 2),
                "frame_used": fr,
                "method": "qb",
                "qb_confidence": round(float(qb.get("confidence", 0.0)), 3),
                "ol_count": len(ol_f),
            }

    # 2) Fallback: deepest offensive-skill player behind the OL.
    for fr in range(max(0, snap_frame - search_radius), snap_frame + search_radius + 1):
        dets = _frame_dets(positions, fr)
        ols = [d for d in dets if d.get("class") == "oline"]
        skill = [d for d in dets if d.get("class") in OFFENSE_SKILL]
        defense = [d for d in dets if d.get("class") == "defense"]
        if len(ols) < 3 or not skill or not defense:
            continue
        H = _homography_for(frame_corr, fr, window)
        if H is None:
            continue
        ol_f = [transform_point(_center(o), H) for o in ols]
        ol_f = [p for p in ol_f if p is not None]
        def_f = [transform_point(_center(d), H) for d in defense]
        def_f = [p for p in def_f if p is not None]
        sk_f = [(d, transform_point(_center(d), H)) for d in skill]
        sk_f = [(d, p) for d, p in sk_f if p is not None]
        if not ol_f or not def_f or not sk_f:
            continue
        ol_x = float(np.median([p[0] for p in ol_f]))
        def_x = float(np.median([p[0] for p in def_f]))
        # Offense is on the opposite side of the OL from the defense.
        behind_sign = 1.0 if ol_x > def_x else -1.0
        # deepest skill player behind the line (largest depth on the offense side)
        candidates = [
            (behind_sign * (p[0] - ol_x), d, p) for d, p in sk_f
            if behind_sign * (p[0] - ol_x) > 0
        ]
        if not candidates:
            continue
        depth, det, _ = max(candidates, key=lambda t: t[0])
        # Sanity cap: a real QB is at most ~7 yd behind the line. A deeper
        # "back" is a downfield receiver, not the QB -- don't fake a reading.
        if depth > 7.0:
            return {
                "alignment": "unknown",
                "qb_depth_yd": round(float(depth), 2),
                "frame_used": fr,
                "method": "fallback_rejected",
                "reason": f"deepest back ({det.get('class')}) is {depth:.1f} yd behind OL -- too deep to be a QB; no QB detected near snap",
                "ol_count": len(ol_f),
            }
        return {
            "alignment": _classify(depth),
            "qb_depth_yd": round(float(depth), 2),
            "frame_used": fr,
            "method": "fallback_deepest_back",
            "fallback_class": det.get("class"),
            "ol_count": len(ol_f),
        }

    return {
        "alignment": "unknown",
        "qb_depth_yd": None,
        "frame_used": None,
        "method": "none",
        "reason": "no frame with QB or deepest-back + OL + valid homography in window",
    }


def main():
    ap = argparse.ArgumentParser(description="Geometric QB alignment / shotgun detection")
    ap.add_argument("--positions", required=True, help="positionDetection JSON (has qb/oline classes)")
    ap.add_argument("--correspondence", required=True, help="correspondence points JSON")
    ap.add_argument("--snap-detection", required=True, help="snap detection JSON")
    ap.add_argument("--search-radius", type=int, default=60, help="frames each side of snap to search")
    ap.add_argument("--window", type=int, default=15, help="homography aggregation window")
    args = ap.parse_args()

    snaps = (load_json(args.snap_detection).get("snaps") or [])
    if not snaps:
        print(json.dumps({"alignment": "unknown", "reason": "no snap frame"}))
        return 1
    snap_frame = snaps[0].get("frame")

    result = detect_alignment(
        snap_frame,
        load_json(args.positions),
        load_json(args.correspondence).get("frame_correspondences", {}),
        search_radius=args.search_radius,
        window=args.window,
    )
    result["snap_frame"] = snap_frame
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
