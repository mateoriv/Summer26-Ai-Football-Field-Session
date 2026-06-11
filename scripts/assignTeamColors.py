#!/usr/bin/env python3
"""
Stable whole-clip OFFENSE/DEFENSE marking by jersey colour -- a PROCESSING step.

The role detector only runs in the snap window, so app-side colouring goes grey
outside it. Jersey colour works on every frame. This script:
  1. LOCKS the offense vs defense team colours once at the snap (where roles
     exist, via jersey_color.assign_teams), so the naming never flickers;
  2. colours every player in every frame of the homography normalized_positions
     by NEAREST locked team colour;
  3. writes `team` ('offense'|'defense'|None) into each entry, so the app reads
     it directly and shows red/blue from the first frame to the last.

Run per clip or per folder; it augments the existing
cache/<folder>/homography/<clip>_normalized_positions.json in place.

Usage:
  .venv/bin/python3 scripts/assignTeamColors.py --folder CSAI_FORMATIONS --clip "Wide - Clip 275"
  .venv/bin/python3 scripts/assignTeamColors.py --folder CSAI_FORMATIONS            # whole folder
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, os.path.join(ROOT, "formations"))
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import jersey_color as jc  # noqa: E402


def _lab_one(bgr):
    return jc._to_lab(np.array([bgr], dtype=np.float32))[0]


def _lock_team_colors(video, snap_frame, pos_detections):
    """Lab colour of the offense and defense teams, fixed at the snap."""
    frame = jc.read_frame(video, snap_frame)
    if frame is None:
        return None
    res = jc.assign_teams(frame, pos_detections)
    off = [p["color"] for p in res["players"] if p.get("team") == "offense" and p.get("color")]
    deff = [p["color"] for p in res["players"] if p.get("team") == "defense" and p.get("color")]
    if not off or not deff:
        return None
    off_lab = np.median(jc._to_lab(np.array(off, dtype=np.float32)), axis=0)
    def_lab = np.median(jc._to_lab(np.array(deff, dtype=np.float32)), axis=0)
    return off_lab, def_lab, float(res.get("separation", 0.0)), bool(res.get("reliable", False))


def assign_clip(folder, clip):
    base = os.path.join(ROOT, "cache", folder)
    norm_p = os.path.join(base, "homography", clip + "_normalized_positions.json")
    pos_p = os.path.join(base, "positions", clip + "_position.json")
    if not (os.path.exists(norm_p) and os.path.exists(pos_p)):
        return {"clip": clip, "ok": False, "reason": "missing cache"}

    with open(norm_p) as f:
        norm = json.load(f)
    with open(pos_p) as f:
        pos = json.load(f)

    snap = jc._snap_frame(folder, clip)
    video = jc._resolve_video(pos, clip, folder)
    if video is None:
        return {"clip": clip, "ok": False, "reason": "no source video"}

    locked = _lock_team_colors(video, snap, jc._detections_at(pos, snap))
    if locked is None:
        return {"clip": clip, "ok": False, "reason": "no team split at snap"}
    off_lab, def_lab, sep, reliable = locked

    npos = norm.get("normalized_positions", {})
    frame_keys = {int(k) for k in npos if str(k).isdigit()}
    if not frame_keys:
        return {"clip": clip, "ok": False, "reason": "no frames"}
    max_f = max(frame_keys)

    counts = {"offense": 0, "defense": 0, "none": 0}
    cap = cv2.VideoCapture(video)
    f = 0
    while f <= max_f:
        ok, img = cap.read()
        if not ok:
            break
        k = str(f)
        if k in npos:
            for p in npos[k]:
                bb = p.get("original_bbox")
                team = None
                if bb:
                    col = jc.representative_color(jc.torso_patch(img, bb))
                    if col is not None:
                        cl = _lab_one(col)
                        if np.linalg.norm(cl - off_lab) <= np.linalg.norm(cl - def_lab):
                            team = "offense"
                        else:
                            team = "defense"
                p["team"] = team
                counts[team or "none"] += 1
        f += 1
    cap.release()

    norm["team_color_meta"] = {"separation": round(sep, 1), "reliable": reliable,
                               "snap_frame": snap}
    with open(norm_p, "w") as f:
        json.dump(norm, f)
    return {"clip": clip, "ok": True, "counts": counts, "separation": round(sep, 1),
            "reliable": reliable, "frames": len(frame_keys)}


def main():
    ap = argparse.ArgumentParser(description="Stable offense/defense colouring by jersey colour")
    ap.add_argument("--folder", required=True)
    ap.add_argument("--clip", default=None, help="one clip; omit for the whole folder")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    if args.clip:
        clips = [args.clip]
    else:
        pos_dir = os.path.join(ROOT, "cache", args.folder, "positions")
        clips = [os.path.basename(p)[: -len("_position.json")]
                 for p in sorted(glob.glob(os.path.join(pos_dir, "*_position.json")))]
        if args.limit:
            clips = clips[: args.limit]

    ok = 0
    for clip in clips:
        r = assign_clip(args.folder, clip)
        if r.get("ok"):
            ok += 1
            print(f"[OK] {clip}: {r['counts']} sep={r['separation']} "
                  f"reliable={r['reliable']} frames={r['frames']}")
        else:
            print(f"[--] {clip}: {r.get('reason')}")
    print(f"\ndone: {ok}/{len(clips)} clips coloured")
    return 0


if __name__ == "__main__":
    sys.exit(main())
