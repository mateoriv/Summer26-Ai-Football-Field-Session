#!/usr/bin/env python3
"""
Split the two teams by JERSEY COLOR instead of trusting the detector's
offense/defense class.

Why: today `line_count_classifier.project_snapshot` sets team = (YOLO class ==
"defense"). When the detector mislabels a defender as "oline"/"wide_receiver"
that player leaks onto the offense and corrupts the front/strength counts -- the
leak the LOS filter was added to paper over (see project notes
`formation-accuracy-bottleneck`). Jersey colour is a stronger, perception-level
signal: the two teams wear different colours, so clustering the per-player
jersey colour into two groups separates them regardless of class noise.

Approach (the ordering matters -- colour is a PIXEL-space read):
  1. For each detection, sample the TORSO patch from the raw frame (upper-centre
     of the bbox, below the helmet, above the legs) and reduce it to one
     representative colour, masking out grass-green and field-line white.
  2. Cluster those colours into k=2 (the two teams). Colour does the SPLIT.
  3. NAME the two clusters offense/defense by majority of the detector's own
     class labels inside each cluster (the cluster holding more offense-classed
     boxes is the offense). Class noise only has to be right on average per
     cluster, not per player -- far weaker a requirement than today.

This module is geometry-free and homography-free; it needs only the frame image
and the pixel bboxes. Depends on numpy + cv2 (already used by vision_reranker).
Has a CLI to run over the cache and compare colour-split vs class-split.
"""

import argparse
import glob
import os
import sys

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - cv2 always present in the app env
    cv2 = None

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from ioutils import load_json, normalize_class

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Offense vs defense detector classes -- used ONLY to name the colour clusters,
# never to assign an individual player. Mirrors line_count_classifier.
OFFENSE_CLASSES = frozenset(
    {"oline", "qb", "running_back", "wide_receiver", "tight_end"})
DEFENSE_CLASS = "defense"
IGNORE_CLASSES = frozenset({"ref"})

# --------------------------------------------------------------------------- #
# Torso patch + representative colour
# --------------------------------------------------------------------------- #

# Vertical slice of the bbox that is jersey: skip the top (helmet) and the
# bottom (pants/legs/grass). Fractions of the box height from the top.
TORSO_TOP = 0.18
TORSO_BOT = 0.55
# Horizontal inset to drop the arms/background at the bbox edges.
TORSO_SIDE = 0.22

# A patch needs at least this many non-grass pixels to be trusted.
MIN_JERSEY_PIXELS = 12

# Reliability gates for the two-team split. `separation` is the Lab distance
# between the cluster centres; below this the kits look too alike to trust.
MIN_SEPARATION = 20.0
# Both clusters should hold a real share of the players. A heavily lopsided
# split (e.g. 18 vs 1) is the signature of similar kits collapsing into one
# colour group, not a genuine offense/defense split. Counts here are RAW
# (pre-dedup) detections, so ByteTrack fragmentation can push a real team's
# count past 11 -- the ratio, not the absolute count, is the trustworthy signal.
MIN_CLUSTER_SHARE = 0.30


def torso_patch(frame, bbox):
    """Crop the jersey region (upper-centre) of a detection's bbox.

    `bbox` is the position-JSON dict with x1/y1/x2/y2 in pixels. Returns an
    HxWx3 BGR array, or None if the box is degenerate / off-frame.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
    bw, bh = (x2 - x1), (y2 - y1)
    if bw <= 1 or bh <= 1:
        return None
    cx1 = int(round(x1 + TORSO_SIDE * bw))
    cx2 = int(round(x2 - TORSO_SIDE * bw))
    cy1 = int(round(y1 + TORSO_TOP * bh))
    cy2 = int(round(y1 + TORSO_BOT * bh))
    cx1, cx2 = max(0, cx1), min(w, cx2)
    cy1, cy2 = max(0, cy1), min(h, cy2)
    if cx2 - cx1 < 1 or cy2 - cy1 < 1:
        return None
    return frame[cy1:cy2, cx1:cx2]


def _grass_mask(patch_bgr):
    """Boolean mask of pixels that are NOT grass-green and NOT line-white.

    Grass: green channel dominant with decent saturation. Lines/paint: very
    bright and desaturated. Everything else is treated as jersey/skin.
    """
    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    # OpenCV hue is 0..179; grass green ~35..85.
    grass = (h >= 35) & (h <= 85) & (s >= 60) & (v >= 40)
    line = (s < 35) & (v > 200)  # bright desaturated -> field paint / glare
    return ~(grass | line)


def representative_color(patch_bgr):
    """One representative BGR colour for a torso patch (median over jersey
    pixels). Returns None when too few jersey pixels survive the grass mask."""
    if patch_bgr is None or patch_bgr.size == 0:
        return None
    mask = _grass_mask(patch_bgr)
    pix = patch_bgr[mask]
    if pix.shape[0] < MIN_JERSEY_PIXELS:
        pix = patch_bgr.reshape(-1, 3)  # fallback: use the whole patch
        if pix.shape[0] == 0:
            return None
    return np.median(pix.astype(np.float32), axis=0)  # BGR


# --------------------------------------------------------------------------- #
# Two-team clustering
# --------------------------------------------------------------------------- #

def _to_lab(bgr_colors):
    """Convert an (N,3) array of BGR colours to Lab (perceptually uniform, so
    Euclidean distance ~ colour difference)."""
    arr = np.clip(bgr_colors, 0, 255).astype(np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    return lab


def _kmeans2(features):
    """k=2 clustering. Returns (labels, centers, separation). `separation` is
    the distance between the two cluster centres -- small => the teams look
    alike and the split is unreliable."""
    n = features.shape[0]
    if n < 2:
        return np.zeros(n, dtype=int), features.copy(), 0.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # Several inits; cv2 picks the best compactness internally with KMEANS_PP.
    _, labels, centers = cv2.kmeans(
        features, 2, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
    labels = labels.flatten()
    sep = float(np.linalg.norm(centers[0] - centers[1]))
    return labels, centers, sep


def assign_teams(frame, detections, offense_anchor_index=None):
    """Cluster detections into two teams by jersey colour, then name the
    clusters offense/defense by detector-class majority.

    `detections` : list of dicts with at least `class` and `bbox` (pixel
    x1/y1/x2/y2), e.g. straight from a positions-JSON frame.

    `offense_anchor_index` (optional): index into `detections` of a player
    KNOWN to be on offense (e.g. the human-verified QB). Its cluster is named
    offense outright -- a certain anchor replacing the class-majority vote.

    Returns dict:
      players   : list of {class, bbox, color(BGR list|None), cluster(0/1|None),
                           team('offense'|'defense'|None)}
      separation: Lab distance between cluster centres (>~25 is a clean split)
      offense_cluster, defense_cluster : the cluster ids, or None
      reliable  : bool -- both clusters non-empty and separation adequate
    """
    if cv2 is None:
        raise RuntimeError("cv2 is required for jersey-colour team assignment")

    players = []
    feats, idx = [], []
    for i, d in enumerate(detections):
        cls = normalize_class(d.get("class"))
        b = d.get("bbox") or {}
        rec = {"class": cls, "bbox": b, "color": None, "cluster": None, "team": None}
        players.append(rec)
        if cls in IGNORE_CLASSES or b.get("x1") is None:
            continue
        col = representative_color(torso_patch(frame, b))
        if col is None:
            continue
        rec["color"] = [float(c) for c in col]
        feats.append(col)
        idx.append(i)

    result = {"players": players, "separation": 0.0,
              "offense_cluster": None, "defense_cluster": None, "reliable": False}
    if len(feats) < 2:
        return result

    labels, _, sep = _kmeans2(_to_lab(np.array(feats, dtype=np.float32)))
    for j, i in enumerate(idx):
        players[i]["cluster"] = int(labels[j])
    result["separation"] = sep

    # Name the clusters. A known-offense anchor (human-verified QB) decides
    # outright; otherwise fall back to detector-class majority.
    off_c = None
    if (offense_anchor_index is not None
            and 0 <= offense_anchor_index < len(players)
            and players[offense_anchor_index]["cluster"] is not None):
        off_c = players[offense_anchor_index]["cluster"]
    if off_c is None:
        score = {0: 0, 1: 0}
        for i in idx:
            c = players[i]["cluster"]
            cls = players[i]["class"]
            if cls == DEFENSE_CLASS:
                score[c] -= 1
            elif cls in OFFENSE_CLASSES:
                score[c] += 1
        off_c = 0 if score[0] >= score[1] else 1
    def_c = 1 - off_c
    result["offense_cluster"], result["defense_cluster"] = off_c, def_c
    for i in idx:
        players[i]["team"] = "offense" if players[i]["cluster"] == off_c else "defense"

    # Reliability: the split is trustworthy only when the kits are distinct
    # (separation) AND the two clusters are plausible team sizes (balance). A
    # lopsided or oversized cluster means colour latched onto lighting, not team
    # -- the failure mode seen on same-coloured kits (e.g. both teams in white).
    counts = [sum(1 for i in idx if players[i]["cluster"] == c) for c in (0, 1)]
    total = sum(counts)
    share = min(counts) / total if total else 0.0
    result["counts"] = counts
    result["reliable"] = sep >= MIN_SEPARATION and share >= MIN_CLUSTER_SHARE
    return result


# --------------------------------------------------------------------------- #
# Frame access + CLI (run over the cache, compare colour-split vs class-split)
# --------------------------------------------------------------------------- #

def _resolve_video(positions_data, clip_name, folder):
    """Find the source mp4 for a clip, trying the JSON path then data/<folder>."""
    p = (positions_data.get("video_info") or {}).get("path")
    cands = [p]
    if p and not os.path.isabs(p):
        cands.append(os.path.join(REPO_ROOT, p))
    cands.append(os.path.join(REPO_ROOT, "data", folder, clip_name + ".mp4"))
    for cand in cands:
        if cand and os.path.exists(cand):
            return cand
    return None


def read_frame(video_path, frame_number):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _detections_at(positions_data, frame_number):
    for fr in positions_data.get("frames", []):
        if fr.get("frame_number") == frame_number:
            return fr.get("detections", [])
    return []


def _snap_frame(folder, clip_name):
    f = os.path.join(REPO_ROOT, "cache", folder, "snap_detection",
                     clip_name + "_snap_detection.json")
    if not os.path.exists(f):
        return 0
    snaps = (load_json(f).get("snaps") or [])
    return int(snaps[0]["frame"]) if snaps else 0


def run_folder(folder, limit=None):
    pos_dir = os.path.join(REPO_ROOT, "cache", folder, "positions")
    files = sorted(glob.glob(os.path.join(pos_dir, "*_position.json")))
    if limit:
        files = files[:limit]
    print(f"{'clip':<22} {'sep':>5} {'off':>4} {'def':>4}  {'class-off':>9}  {'flip%':>6}  reliable")
    print("-" * 72)
    for pf in files:
        clip = os.path.basename(pf)[: -len("_position.json")]
        data = load_json(pf)
        snap = _snap_frame(folder, clip)
        vid = _resolve_video(data, clip, folder)
        if not vid:
            print(f"{clip:<22} (no video)")
            continue
        frame = read_frame(vid, snap)
        if frame is None:
            print(f"{clip:<22} (no frame {snap})")
            continue
        dets = _detections_at(data, snap)
        res = assign_teams(frame, dets)
        players = [p for p in res["players"] if p["team"]]
        off = sum(1 for p in players if p["team"] == "offense")
        dfn = sum(1 for p in players if p["team"] == "defense")
        # How often the colour-split disagrees with the raw class label:
        class_off = sum(1 for p in players if p["class"] in OFFENSE_CLASSES)
        flips = sum(1 for p in players
                    if (p["class"] in OFFENSE_CLASSES) != (p["team"] == "offense"))
        flip_pct = 100.0 * flips / max(1, len(players))
        print(f"{clip:<22} {res['separation']:5.0f} {off:4d} {dfn:4d}  "
              f"{class_off:9d}  {flip_pct:5.0f}%  {res['reliable']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folder", default="CSAI_FORMATIONS")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    run_folder(args.folder, args.limit)


if __name__ == "__main__":
    main()
