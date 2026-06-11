#!/usr/bin/env python3
"""
Formation classification by COUNTING, not by name-matching.

The expensive part of recognising an offensive formation -- discriminating
~17 named variants -- is fragile because it depends on every skill player
being detected and projected accurately (see the project notes: variant
accuracy is capped by detection recall, not by the matcher). This module
takes the robust route the coach actually uses to read a formation:

  1. capture the pre-snap snapshot of the ATTACKING team (offense),
  2. count how many offense players are ON the line of scrimmage
     (across the full width -- the "front" count, typically 5/6/7/8),
  3. resolve formation STRENGTH (left / right / balanced) from the
     off-the-line skill players relative to the ball,

and reports those primitives. Counting is detection-tolerant: a missed or
mislabeled player shifts a count by one rather than flipping the whole
named guess, and the count alone already eliminates most formation families.

Everything is geometry in FIELD YARDS (via the per-frame homography), so the
same numbers drive both the CSV and the virtual-field overlay. No torch, no
labels, no training -- this module is import-safe in the sandbox and has a
CLI for validating against the cache.

Coordinate frame (matches perFrameHomographyTransform / virtualField):
  x = field length, 0..120 yd (goal line to goal line, endzones included)
  y = field width,  0..53.33 yd
The line of scrimmage runs along y at a fixed x; the offense attacks along x
toward the defense. "Strength left/right" is from the offense's point of view.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

# Make the torch-free homography + shared I/O helpers importable.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from ioutils import load_json, normalize_class

# --------------------------------------------------------------------------- #
# Classes
# --------------------------------------------------------------------------- #

# Offense roles emitted by the position detector. We classify by GEOMETRY, not
# by trusting these labels (a TE often reads as OL/WR); the labels only split
# offense from defense and pick the LOS anchor.
OL_CLASS = "oline"
QB_CLASS = "qb"
OFFENSE_CLASSES = frozenset({OL_CLASS, QB_CLASS, "running_back", "wide_receiver", "tight_end"})
DEFENSE_CLASS = "defense"
IGNORE_CLASSES = frozenset({"ref"})

# --------------------------------------------------------------------------- #
# Tunables (yards)
# --------------------------------------------------------------------------- #

# A player counts as "on the line of scrimmage" when their depth (distance
# behind the LOS, along the attack axis) is within this band. On-ball players
# (linemen, split ends, attached TEs) sit ~0; off-ball players (slots, backs,
# QB) sit >= ~1.3 yd back in the templates. 1.2 yd separates them with margin
# for homography noise.
ON_LINE_TOL_YD = 1.2

# A player further back than the LOS band but still ahead of this is "off the
# line" eligible (slot/wing); deeper than this is clearly backfield (QB/RB).
# Both off-line and backfield skill players contribute to strength; only the
# QB is excluded (it sits over the ball and carries no side).
BACKFIELD_TOL_YD = ON_LINE_TOL_YD

# Lateral dead-zone around the ball: a player within this of the center line is
# treated as "over the ball" (center/QB) and does not vote for a side.
CENTER_DEADZONE_YD = 0.75

# Interior box half-width: on-line players within this of center are "in the
# box" (the 5 O-line + any attached TE/wing). Split ends (~8 yd wide) fall
# outside and count as receivers. Sets the 5/6/7 "middle" count.
BOX_HALFWIDTH_YD = 4.5

# Strength is BALANCED unless one side has at least this many more off-line
# skill players than the other.
STRENGTH_MARGIN = 1

# Pre-snap, no offensive player is downfield of the line of scrimmage. When the
# homography is collapsed or rotated, receivers project several yards forward of
# the line -- physically impossible -- so a read past this margin is refused.
FORWARD_TOL_YD = 2.5

# Only players at RECEIVER depth vote a side. The ground-truth templates count
# W/S/X/Y/Z/U (split ends, slots, wings: on the line or <= ~2.5 yd off it) and
# exclude the back (T) entirely -- so a player at back depth (>= ~3.5 yd, where
# shotgun backs and QB/RB track fragments live) must not vote.
MAX_RECEIVER_DEPTH_YD = 3.5

# Two side-voting receivers closer than this (yards, straight-line) are one
# fragmented player, not a stack -- tighter than any real split in the playbook.
RECV_DEDUP_YD = 1.1

# Field bounds (+ margin) an offense point must fall in to be trusted.
FIELD_LENGTH_YD = 120.0
FIELD_WIDTH_YD = 160.0 / 3.0
FIELD_MARGIN_YD = 2.0

# ByteTrack fragments one player into several near-identical detections. Merge
# same-team points closer than this so duplicates don't inflate the count. Kept
# well below a real offensive-line split (~1 yd center-to-center) so adjacent
# linemen are never collapsed into one.
DEDUP_YD = 0.6


# Module-local aliases so existing call sites stay unchanged.
_norm_cls = normalize_class
_load = load_json


# --------------------------------------------------------------------------- #
# Snapshot extraction: classed detections -> field yards
# --------------------------------------------------------------------------- #

def _aggregate_tracks(positions_data, snap_frame, window):
    """Vote each track's class over [snap-window, snap+window].

    Returns list of {class, center_x, center_y} in PIXELS (one per track, the
    bbox nearest the snap among the winning-class detections). Falls back to
    raw snap-frame detections when no track_ids exist.
    """
    frames = positions_data.get("frames", [])
    lo, hi = snap_frame - window, snap_frame + window

    by_id = {}
    untracked = []
    for fr in frames:
        n = fr.get("frame_number")
        if n is None or n < lo or n > hi:
            continue
        for d in fr.get("detections", []):
            tid = d.get("track_id")
            if tid is None:
                if n == snap_frame:
                    untracked.append(d)
                continue
            by_id.setdefault(tid, []).append((n, d))

    out = []
    for tid, dets in by_id.items():
        votes = {}
        for _, d in dets:
            c = _norm_cls(d.get("class"))
            if c:
                votes[c] = votes.get(c, 0.0) + float(d.get("confidence") or 1.0)
        if not votes:
            continue
        cls = max(votes, key=votes.get)
        cands = [(n, d) for n, d in dets if _norm_cls(d.get("class")) == cls]
        cands.sort(key=lambda x: abs(x[0] - snap_frame))
        b = cands[0][1].get("bbox") or {}
        if b.get("center_x") is None:
            continue
        out.append({"class": cls, "center_x": b["center_x"], "center_y": b["center_y"]})

    if not out:  # no tracks -> single snap frame
        for d in untracked:
            b = d.get("bbox") or {}
            if b.get("center_x") is None:
                continue
            out.append({"class": _norm_cls(d.get("class")),
                        "center_x": b["center_x"], "center_y": b["center_y"]})
    return out


def _best_homography(frame_correspondences, snap_frame):
    """Pick a valid homography matrix within +/-20 frames of the snap."""
    from perFrameHomographyTransform import (
        gather_window_correspondences, get_homography_matrix, field_points_are_degenerate,
    )
    for fr in range(max(0, snap_frame - 20), snap_frame + 21):
        ip, fp = gather_window_correspondences(frame_correspondences, fr, 15)
        if len(ip) < 4 or field_points_are_degenerate(fp):
            continue
        H = get_homography_matrix(ip, fp)
        if H is not None:
            return H, fr
    return None, snap_frame


def _on_field(x, y):
    return (-FIELD_MARGIN_YD <= x <= FIELD_LENGTH_YD + FIELD_MARGIN_YD and
            -FIELD_MARGIN_YD <= y <= FIELD_WIDTH_YD + FIELD_MARGIN_YD)


def jersey_team_points(positions_data, snap_frame, video_name, folder_name):
    """Two-team split by JERSEY COLOUR at the snap frame (jersey_color module).

    Returns [(center_x_px, center_y_px, 'offense'|'defense'), ...] for the
    snap-frame detections, or None when the split is unavailable/unreliable
    (no cv2, no source video, kits too alike) -- the caller then keeps the
    detector-class split. Colour fixes the leak where a mislabeled defender
    joins the offense and corrupts the front/strength counts.
    """
    try:
        import jersey_color as jc
        if jc.cv2 is None:
            return None
        video = jc._resolve_video(positions_data, video_name, folder_name)
        if video is None:
            return None
        frame = jc.read_frame(video, snap_frame)
        if frame is None:
            return None
        res = jc.assign_teams(frame, jc._detections_at(positions_data, snap_frame))
        if not res.get("reliable"):
            return None
        pts = [(float(p["bbox"]["center_x"]), float(p["bbox"]["center_y"]), p["team"])
               for p in res["players"]
               if p.get("team") and (p.get("bbox") or {}).get("center_x") is not None]
        return pts or None
    except Exception:
        return None  # never let the colour read break the geometric one


def _nearest_jersey_team(points, cx, cy, tol_px=40.0):
    best, best_d = None, tol_px * tol_px
    for px, py, team in points:
        dd = (px - cx) ** 2 + (py - cy) ** 2
        if dd < best_d:
            best_d, best = dd, team
    return best


def project_snapshot(positions_data, frame_correspondences, snap_frame, window=None,
                     jersey_teams=None):
    """Project the pre-snap classed detections to field yards.

    Returns (players, H, h_frame) where `players` is a list of
    {role, team, x, y} in field yards, on-field only, refs dropped. team is
    'offense' or 'defense'. Returns ([], None, frame) when no homography.

    `jersey_teams` (optional, from jersey_team_points): pixel-space team split
    by jersey colour; when given it OVERRIDES the detector-class team per
    player. The labeled QB is exempt -- a differently-coloured QB kit must not
    flip the one anchor the geometry depends on.
    """
    from perFrameHomographyTransform import transform_point

    if window is None:
        window = int((positions_data.get("tracking") or {}).get("window") or 30)

    H, h_frame = _best_homography(frame_correspondences, snap_frame)
    if H is None:
        return [], None, h_frame

    players = []
    for d in _aggregate_tracks(positions_data, snap_frame, window):
        cls = d["class"]
        if cls in IGNORE_CLASSES:
            continue
        if cls == DEFENSE_CLASS:
            team = "defense"
        elif cls in OFFENSE_CLASSES:
            team = "offense"
        else:
            continue
        if jersey_teams and cls != QB_CLASS:
            jt = _nearest_jersey_team(jersey_teams, d["center_x"], d["center_y"])
            if jt:
                team = jt
        p = transform_point((d["center_x"], d["center_y"]), H)
        if p is None or not _on_field(p[0], p[1]):
            continue
        players.append({"role": cls, "team": team, "x": float(p[0]), "y": float(p[1])})
    return _dedup_same_team(players), H, h_frame


def _dedup_same_team(players, tol=DEDUP_YD):
    """Greedily merge same-team detections within `tol` yards (fragmented
    tracks). Keeps the first point; prefers a specific role over a generic one
    when a duplicate carries the OL/QB label."""
    kept = []
    for p in players:
        dup = None
        for q in kept:
            if q["team"] != p["team"]:
                continue
            if math.hypot(q["x"] - p["x"], q["y"] - p["y"]) <= tol:
                dup = q
                break
        if dup is None:
            kept.append(dict(p))
        elif dup["role"] not in (OL_CLASS, QB_CLASS) and p["role"] in (OL_CLASS, QB_CLASS):
            dup["role"] = p["role"]  # keep the more informative anchor label
    return kept


# --------------------------------------------------------------------------- #
# The count
# --------------------------------------------------------------------------- #

def _densest_x_cluster(xs, width=2 * ON_LINE_TOL_YD):
    """Return (center_x, member_mask) for the densest `width`-yd window in xs.

    The 5 offensive linemen sit packed at one attack-axis depth, so they are the
    tightest cluster along x -- a far more robust line-of-scrimmage anchor than
    the median, which split receivers and deep backs drag off the line whenever
    few linemen happen to carry the `oline` label.
    """
    xs = np.asarray(xs, dtype=float)
    if xs.size == 0:
        return None, None
    best_n, best_c, best_mask = -1, None, None
    for x0 in xs:
        mask = np.abs(xs - x0) <= width / 2.0
        n = int(mask.sum())
        if n > best_n:
            best_n, best_c, best_mask = n, float(np.median(xs[mask])), mask
    return best_c, best_mask


def classify(players):
    """Read the offense front from field-yard players, geometry-first.

    The detector's role labels are unreliable at the snap (QB rarely found,
    TEs read as OL/WR), so we anchor on GEOMETRY:
      1. find the line of scrimmage + attack direction,
      2. RECOVER the QB as the central player just behind the line (it also
         fixes the center line and the attack direction -- the QB is always on
         the offense's side of the ball),
      3. take the INTERIOR 5 (the O-line) as the 5 on-line players closest to
         center -- not whatever was labeled `oline` -- so a mislabeled TE is
         not swallowed into the line,
      4. cut interior-5 + QB + RB, and let everything else (split ends,
         attached TEs, slots) vote its SIDE -> the 3x1 / 2x2 receiver split.

    `players`: list of {role, team, x, y} (output of project_snapshot).
    Returns a result dict or {"on_line_count": None, "reason": ...}.
    """
    offense = [p for p in players if p["team"] == "offense"]
    defense = [p for p in players if p["team"] == "defense"]
    if len(offense) < 5:
        return {"on_line_count": None, "reason": f"only {len(offense)} offense players"}

    n = len(offense)
    off_x = np.array([p["x"] for p in offense], dtype=float)
    off_y = np.array([p["y"] for p in offense], dtype=float)

    # --- Line of scrimmage = the densest x-cluster of offense (the O-line) -- #
    # Not the median: split receivers and deep backs drag the median off the
    # line when few linemen carry the `oline` label. The 5 packed linemen are
    # the tightest cluster along the attack axis, which the homography preserves.
    los_x, line_mask = _densest_x_cluster(off_x)
    if los_x is None:
        return {"on_line_count": None, "reason": "no offense x positions"}
    center0 = float(np.median(off_y[line_mask]))

    # Provisional attack direction from the defense (refined by the QB below).
    if defense:
        attack_dir = 1.0 if np.median([p["x"] for p in defense]) > los_x else -1.0
    else:
        fwd = off_x - los_x
        attack_dir = 1.0 if abs(fwd.max()) <= abs(fwd.min()) else -1.0

    def depths(ad):
        return (off_x - los_x) * ad
    depth = depths(attack_dir)

    # --- Recover the QB: central offense player just behind the line ------ #
    # Candidates sit 1-7 yd behind the LOS (under center to shotgun). Prefer a
    # labeled QB if one happens to be there; else the most CENTRAL back.
    cand = [i for i in range(n) if -7.0 <= depth[i] <= -1.0]
    qb_idx = None
    if cand:
        labeled = [i for i in cand if offense[i]["role"] == QB_CLASS]
        pool = labeled if labeled else cand
        qb_idx = min(pool, key=lambda i: abs(off_y[i] - center0))

    if qb_idx is not None and abs(off_x[qb_idx] - los_x) > 0.3:
        # The QB is on the offense's side of the ball, so the LOS is on the
        # attack side of the QB -- a defense-independent direction fix.
        attack_dir = 1.0 if los_x > off_x[qb_idx] else -1.0
        depth = depths(attack_dir)

    # --- Reject degenerate / rotated projections -------------------------- #
    # With the LOS anchored on the line, no offensive player should sit downfield
    # of it pre-snap. One or two forward points are tolerated as projection noise
    # (a single misprojected detection), but a player absurdly downfield or THREE
    # forward at once means the homography collapsed (361) or rotated the field
    # (1002/015) -- refuse the read rather than emit a wrong one.
    n_forward = int((depth > FORWARD_TOL_YD).sum())
    if float(depth.max()) > 10.0 or n_forward >= 3:
        return {"on_line_count": None,
                "reason": f"bad projection: {n_forward} player(s) up to "
                          f"{float(depth.max()):.1f}yd downfield of LOS"}

    # --- Interior 5 = the O-line (geometry, not the label) ---------------- #
    on_line_idx = [i for i in range(n) if abs(depth[i]) <= ON_LINE_TOL_YD]
    interior = set(sorted(on_line_idx, key=lambda i: abs(off_y[i] - center0))[:5])

    # Center the box on the interior LINE, not the QB: the QB can sit a couple
    # yards off the snapper (e.g. a pistol back or an offset shotgun), which
    # narrows the box and drops real linemen out of the interior count.
    if len(interior) >= 3:
        center_y = float(np.median([off_y[i] for i in interior]))
    elif qb_idx is not None:
        center_y = float(off_y[qb_idx])
    else:
        center_y = center0

    on_line_count = len(on_line_idx)  # full width (every offense player at LOS depth)
    box_count = sum(1 for i in on_line_idx
                    if abs(off_y[i] - center_y) <= BOX_HALFWIDTH_YD)  # interior 5/6/7

    # The CENTER (the snapper, the player the QB works with) = the interior
    # lineman on the middle of the line. Marked so the overlays can label it.
    center_idx = (min(interior, key=lambda i: abs(off_y[i] - center_y))
                  if interior else None)
    if center_idx is not None:
        offense[center_idx]["_ctr"] = True

    # --- RB = the most central back (cut it as part of the middle group) -- #
    backfield_idx = [i for i in range(n) if depth[i] < -ON_LINE_TOL_YD and i != qb_idx]
    rb_idx = (min(backfield_idx, key=lambda i: abs(off_y[i] - center_y))
              if backfield_idx else None)

    # --- Receivers = everyone left after cutting the middle group --------- #
    cut = set(interior)
    if qb_idx is not None:
        cut.add(qb_idx)
    if rb_idx is not None:
        cut.add(rb_idx)

    voters = []
    for i, p in enumerate(offense):
        # geometric position tag (for the field overlay)
        if abs(depth[i]) <= ON_LINE_TOL_YD:
            p["_pos"] = "line"
        elif i == qb_idx:
            p["_pos"] = "qb"
        elif depth[i] < -BACKFIELD_TOL_YD:
            p["_pos"] = "backfield"
        else:
            p["_pos"] = "offline"
        if i in cut:
            p["_grp"] = "interior" if i in interior else ("qb" if i == qb_idx else "rb")
            continue
        p["_grp"] = "recv"
        # Only a player at RECEIVER depth may vote a side: not downfield of the
        # line (a leaked defender / projection noise -- up to two are tolerated
        # before the read is refused) and not at back depth (the templates'
        # receiver columns are split ends / slots / wings; backs carry no side,
        # and QB/RB track fragments live at that depth too).
        if depth[i] > FORWARD_TOL_YD or depth[i] < -MAX_RECEIVER_DEPTH_YD:
            p["_grp"] = "noise"
            continue
        # Offense-perspective lateral: facing the attack, left == (y-center)*dir>0.
        if abs(off_y[i] - center_y) < CENTER_DEADZONE_YD:
            p["_side"] = "middle"
            continue
        voters.append(i)

    # A fragmented track can survive the global dedup as two different-class
    # detections ~1 yd apart; no receiver split in these templates packs that
    # tight, so two voters closer than this are one player, one vote.
    left = right = 0
    counted = []
    for i in voters:
        p = offense[i]
        p["_side"] = "left" if (off_y[i] - center_y) * attack_dir > 0 else "right"
        if any(math.hypot(off_x[i] - off_x[j], off_y[i] - off_y[j]) <= RECV_DEDUP_YD
               for j in counted):
            p["_grp"] = "dup"
            continue
        counted.append(i)
        if p["_side"] == "left":
            left += 1
        else:
            right += 1

    hi, lo = max(left, right), min(left, right)
    # Same rule the 17 templates use (validate_line_count._template_structures):
    # trips needs a REAL 3-man surplus side, not just a 2-receiver difference --
    # a noisy (2,0) read is a 2x2 with one missed receiver, not a 3x1.
    bucket = "3x1" if (hi >= 3 and hi - lo >= 2) else "2x2"
    if left - right >= STRENGTH_MARGIN:
        strength = "LEFT"
    elif right - left >= STRENGTH_MARGIN:
        strength = "RIGHT"
    else:
        strength = "BALANCED"

    # Trust it when the geometry is well-formed: a real interior, a recovered
    # QB anchor, and a plausible box count.
    reliable = (n >= 9 and len(interior) >= 5 and qb_idx is not None
                and 5 <= box_count <= 7)

    return {
        "on_line_count": on_line_count,
        "box_count": box_count,
        "strength": strength,
        "bucket": bucket,
        "recv_left": left,
        "recv_right": right,
        "off_line_left": left,   # back-compat aliases
        "off_line_right": right,
        "qb_recovered": qb_idx is not None,
        "n_offense": n,
        "n_defense": len(defense),
        "attack_dir_x": int(attack_dir),
        "los_x_yd": round(los_x, 1),
        "center_y_yd": round(center_y, 1),
        "reliable": bool(reliable),
        "qb_found": qb_idx is not None,
        "center_found": center_idx is not None,
        "players": [
            {"role": p["role"], "team": p["team"],
             "x": round(p["x"], 2), "y": round(p["y"], 2),
             "pos": p.get("_pos"), "side": p.get("_side"), "grp": p.get("_grp"),
             "ctr": bool(p.get("_ctr"))}
            for p in players
        ],
        "los": {"x_yd": round(los_x, 1), "center_y_yd": round(center_y, 1),
                "attack_dir_x": int(attack_dir)},
    }


# --------------------------------------------------------------------------- #
# Cache-driven entry point
# --------------------------------------------------------------------------- #

def recognize_from_cache(video_name, folder_name, base_cache_dir, window=None):
    """Classify one clip's offense front + strength from the cache JSON."""
    snap_p = os.path.join(base_cache_dir, folder_name, "snap_detection", f"{video_name}_snap_detection.json")
    pos_p = os.path.join(base_cache_dir, folder_name, "positions", f"{video_name}_position.json")
    corr_p = os.path.join(base_cache_dir, folder_name, "correspondence", f"{video_name}_correspondence.json")
    for p in (snap_p, pos_p, corr_p):
        if not os.path.exists(p):
            return {"on_line_count": None, "reason": f"missing {os.path.basename(p)}"}

    snaps = (_load(snap_p).get("snaps") or [])
    if not snaps:
        return {"on_line_count": None, "reason": "no snap frame"}
    snap_frame = snaps[0].get("frame")

    pdata = _load(pos_p)
    corr = _load(corr_p).get("frame_correspondences", {})

    # Two team splits, best-of: jersey colour beats the detector class when its
    # geometry holds up (it removes leaked defenders from the counts), but when
    # the colour split breaks the read (real linemen flipped off the offense)
    # we fall back to the class split rather than lose the clip.
    jersey = jersey_team_points(pdata, snap_frame, video_name, folder_name)
    attempts = ([("jersey_color", jersey)] if jersey else []) + [("detector_class", None)]

    best = None
    for source, jt in attempts:
        players, H, h_frame = project_snapshot(pdata, corr, snap_frame, window=window,
                                               jersey_teams=jt)
        if H is None:
            return {"on_line_count": None, "reason": "no valid homography near snap"}
        if not players:
            result = {"on_line_count": None, "reason": "no players projected onto field"}
        else:
            result = classify(players)
        result["snap_frame"] = snap_frame
        result["homography_frame"] = h_frame
        result["team_source"] = source
        if result.get("on_line_count") is not None and result.get("reliable"):
            return result
        if best is None or (best.get("on_line_count") is None
                            and result.get("on_line_count") is not None):
            best = result
    return best


def save_snapshot(result, video_name, folder_name, base_cache_dir):
    """Write the classified pre-snap snapshot (field-yard players + LOS) to
    cache/<folder>/formation/<clip>_formation.json so the virtual field can
    render which team is attacking, which is defending, and the LOS clearly."""
    if result.get("on_line_count") is None:
        return None
    out_dir = os.path.join(base_cache_dir, folder_name, "formation")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{video_name}_formation.json")
    snapshot = {
        "snap_frame": result.get("snap_frame"),
        "homography_frame": result.get("homography_frame"),
        "on_line_count": result["on_line_count"],
        "box_count": result.get("box_count"),
        "strength": result["strength"],
        "bucket": result.get("bucket"),
        "recv_left": result.get("recv_left"),
        "recv_right": result.get("recv_right"),
        "qb_recovered": result.get("qb_recovered"),
        "center_found": result.get("center_found"),
        "n_offense": result.get("n_offense"),
        "n_defense": result.get("n_defense"),
        "team_source": result.get("team_source"),
        "attack_dir_x": result["attack_dir_x"],
        "reliable": result["reliable"],
        "los": result["los"],
        "players": result["players"],
    }
    with open(out_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Formation classification by counting the offense front")
    ap.add_argument("--video-name", required=True)
    ap.add_argument("--folder-name", required=True)
    ap.add_argument("--cache-dir", default="cache", help="absolute, or relative to the project root")
    args = ap.parse_args()

    base = (args.cache_dir if os.path.isabs(args.cache_dir)
            else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.cache_dir))
    result = recognize_from_cache(args.video_name, args.folder_name, base)
    # Drop the bulky per-player list from the printed summary.
    summary = {k: v for k, v in result.items() if k != "players"}
    print(json.dumps(summary, indent=2))
    return 0 if result.get("on_line_count") is not None else 1


if __name__ == "__main__":
    sys.exit(main())
