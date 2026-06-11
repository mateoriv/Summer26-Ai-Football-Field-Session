#!/usr/bin/env python3
"""
Image-space line counter -- count the offensive line by 2D bounding-box overlap,
NO homography (so it dodges the depth foreshortening that makes the projected
front count over-shoot on wide footage).

Idea (user's): the 5 offensive linemen stand shoulder-to-shoulder, so their
boxes form one connected clump in the image; receivers split wide are isolated
boxes; the backfield is its own small clump. Team-split first (offense vs
defense lock up at the LOS), dedup ByteTrack fragments, then the largest
connected clump of OFFENSE boxes is the line.

Outputs the line count + the receiver boxes (for a side split). Pure 2D; the
homography side of the read is layered on separately.
"""

import collections
import json
import os

# Offense detector classes (everything else that isn't 'defense'/'ref').
OFFENSE_CLASSES = frozenset({"oline", "qb", "running_back", "wide_receiver", "tight_end"})
DEFENSE_CLASS = "defense"
IGNORE = frozenset({"ref"})


def _norm(cls):
    return (cls or "").strip().lower().replace(" ", "_")


def snap_boxes(clip, folder, cache_dir, window=15):
    """One box per track within `window` frames of the snap; full bbox kept."""
    sp = os.path.join(cache_dir, folder, "snap_detection", f"{clip}_snap_detection.json")
    pp = os.path.join(cache_dir, folder, "positions", f"{clip}_position.json")
    if not (os.path.exists(sp) and os.path.exists(pp)):
        return None, []
    snaps = (json.load(open(sp)).get("snaps") or [])
    if not snaps:
        return None, []
    sf = snaps[0].get("frame")
    pos = json.load(open(pp))

    by_id, untracked = {}, []
    for fr in pos.get("frames", []):
        n = fr.get("frame_number")
        if n is None or abs(n - sf) > window:
            continue
        for d in fr.get("detections", []):
            tid = d.get("track_id")
            if tid is None:
                if n == sf:
                    untracked.append(d)
            else:
                by_id.setdefault(tid, []).append((abs(n - sf), d))

    boxes = []
    for tid, ds in by_id.items():
        ds.sort(key=lambda x: x[0])
        d = ds[0][1]
        b = d.get("bbox") or {}
        if b.get("width") is None:
            continue
        boxes.append({"cls": _norm(d.get("class")), **{k: b[k] for k in
                      ("x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y")}})
    if not boxes:  # untracked fallback (single-frame cache)
        for d in untracked:
            b = d.get("bbox") or {}
            if b.get("width") is None:
                continue
            boxes.append({"cls": _norm(d.get("class")), **{k: b[k] for k in
                          ("x1", "y1", "x2", "y2", "width", "height", "center_x", "center_y")}})
    return sf, boxes


def _iou(a, b):
    ix1, iy1 = max(a["x1"], b["x1"]), max(a["y1"], b["y1"])
    ix2, iy2 = min(a["x2"], b["x2"]), min(a["y2"], b["y2"])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / (a["width"] * a["height"] + b["width"] * b["height"] - inter)


def dedup_fragments(boxes, iou_thr=0.45):
    """Drop ByteTrack fragments: same-class boxes with high IoU are one player."""
    kept = []
    for b in sorted(boxes, key=lambda b: -b["width"] * b["height"]):
        if any(b["cls"] == k["cls"] and _iou(b, k) > iou_thr for k in kept):
            continue
        kept.append(b)
    return kept


def _gap(a, b):
    """Pixel gap between two rectangles (0 if they overlap)."""
    dx = max(0.0, max(a["x1"] - b["x2"], b["x1"] - a["x2"]))
    dy = max(0.0, max(a["y1"] - b["y2"], b["y1"] - a["y2"]))
    return (dx * dx + dy * dy) ** 0.5


def _components(boxes, scale):
    """Connected components: link i,j if their gap < scale * min(box height)."""
    n = len(boxes)
    adj = collections.defaultdict(list)
    for i in range(n):
        for j in range(i + 1, n):
            thr = scale * min(boxes[i]["height"], boxes[j]["height"])
            if _gap(boxes[i], boxes[j]) <= thr:
                adj[i].append(j)
                adj[j].append(i)
    seen, comps = set(), []
    for i in range(n):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            u = stack.pop()
            if u in seen:
                continue
            seen.add(u)
            comp.append(u)
            stack += adj[u]
        comps.append(comp)
    return sorted(comps, key=len, reverse=True)


def count_line(clip, folder, cache_dir, scale=0.6, iou_thr=0.45):
    """Box-overlap line count. Returns dict or {'line_count': None, 'reason':...}."""
    sf, boxes = snap_boxes(clip, folder, cache_dir)
    if not boxes:
        return {"line_count": None, "reason": "no snap boxes"}
    offense = dedup_fragments([b for b in boxes if b["cls"] in OFFENSE_CLASSES], iou_thr)
    if len(offense) < 5:
        return {"line_count": None, "reason": f"only {len(offense)} offense boxes"}
    comps = _components(offense, scale)
    line_idx = comps[0]                       # largest connected clump = the line
    line = [offense[i] for i in line_idx]
    rest = [offense[i] for i in range(len(offense)) if i not in set(line_idx)]
    # line column (median image-x of the clump) lets the rest vote a side
    lx = sorted(b["center_x"] for b in line)[len(line) // 2]
    left = sum(1 for b in rest if b["center_x"] < lx)
    right = len(rest) - left
    return {
        "line_count": len(line),
        "n_offense": len(offense),
        "n_rest": len(rest),
        "recv_left": left,
        "recv_right": right,
        "clump_sizes": [len(c) for c in comps],
        "snap_frame": sf,
        "line_roles": collections.Counter(b["cls"] for b in line),
    }
