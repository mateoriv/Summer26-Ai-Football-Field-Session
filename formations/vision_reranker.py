#!/usr/bin/env python3
"""
Vision re-ranker: template proposes candidates, Claude vision picks among them.

Pipeline insight that motivates this (measured on 46 labeled clips):
  - template matcher top-1 family accuracy ~22% (at baseline),
  - BUT the correct family is in the template TOP-5 ~57% of the time.
So the bottleneck is *selection*, not *generation*. The count/side-split
re-ranker made it worse (too noisy). A vision model is a far stronger selector:
it reads the snap image directly (the perception the YOLO detector fumbles) and
only has to choose among ~5 candidates, not name from scratch.

This combines all three signals:
  - TEMPLATE matcher  -> candidate formations (recall@5 ~57%)
  - count/geometry    -> a structured hint + an impossible-candidate filter
  - VISION            -> the precise pick among survivors

Run (needs `pip install anthropic` and ANTHROPIC_API_KEY):
    python formations/vision_reranker.py --folder CSAI_FORMATIONS --limit 46
    python formations/vision_reranker.py ... --dry-run   # build prompts, no API
"""

import argparse
import base64
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

import cv2
import template_matcher as tm
import line_count_classifier as lc

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
LABELS_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "models", "offense_positions", "play_predictions.csv")

# One-line shape hints so the model knows what each candidate looks like.
FORMATION_HINTS = {
    "detroit": "2x2, balanced, no tight end attached",
    "trips_open": "3x1, three receivers to one side, empty backfield-ish",
    "dallas": "2x2 with a tight end attached on the line",
    "dallas_y_off": "2x2, tight end off the line (off-ball)",
    "trey": "3x1 with a tight end to the 3-receiver side",
    "trips": "3x1, trips bunch to one side, a back in the backfield",
    "trey_y_off": "3x1, tight end off the line on the strong side",
    "slot_open": "2x2 with a slot receiver, spread",
    "denver": "2x2, two tight ends / heavier middle (6-7 in the box)",
    "denver_y_off": "2x2 heavy, one tight end off the line",
    "denver_u_off": "2x2 heavy, off-ball wing",
    "dallas_wg": "2x2 with a wing back next to the tackle",
    "trey_y_off_tight": "3x1 tight, compressed splits",
    "trey_wg": "3x1 with a wing",
    "trey_tight": "3x1 compressed, tight end on the line",
    "pro": "2x2 pro set, two backs / tight end",
    "slot": "2x2 slot, balanced with a slot receiver",
}


def base_family(name):
    return (name or "").split("_")[0].upper()


def snap_frame_image(video_name, snap_frame, pad=0):
    """Grab the snap frame, return BGR image or None."""
    vid_path = None
    for folder in os.listdir(DATA_DIR):
        cand = os.path.join(DATA_DIR, folder, f"{video_name}.mp4")
        if os.path.exists(cand):
            vid_path = cand
            break
    if vid_path is None:
        return None
    cap = cv2.VideoCapture(vid_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def build_item(video_name, folder, base_cache_dir, top_k=5):
    """Assemble {candidates, hint, image_path} for one clip, or None."""
    snap_p = os.path.join(base_cache_dir, folder, "snap_detection", f"{video_name}_snap_detection.json")
    if not os.path.exists(snap_p):
        return None
    snap_frame = (json.load(open(snap_p)).get("snaps") or [{}])[0].get("frame")
    if snap_frame is None:
        return None

    tr = tm.recognize_from_cache(video_name, folder, base_cache_dir)
    ranking = tr.get("ranking") or []
    if not ranking:
        return None
    candidates = [n for n, _ in ranking[:top_k]]

    lcr = lc.recognize_from_cache(video_name, folder, base_cache_dir)
    hint = None
    if lcr.get("on_line_count") is not None:
        hint = (f"our geometry estimate (noisy, advisory): ~{lcr['box_count']} in the box, "
                f"receivers {lcr['recv_left']} left / {lcr['recv_right']} right, "
                f"offense attacking { 'right' if lcr['attack_dir_x']>0 else 'left' }")

    frame = snap_frame_image(video_name, snap_frame)
    if frame is None:
        return None
    img_path = f"/tmp/vr_{video_name.replace(' ','_')}.jpg"
    cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return {"video": video_name, "snap_frame": snap_frame,
            "candidates": candidates, "hint": hint, "image": img_path}


def build_prompt(item):
    lines = [
        "This is a single pre-snap frame from American football (wide TV angle).",
        "The OFFENSE is the team about to snap the ball; identify THEIR formation.",
        "",
        "Pick the single best match from these candidate formations:",
    ]
    for i, c in enumerate(item["candidates"], 1):
        lines.append(f"  {i}. {c} -- {FORMATION_HINTS.get(c, '')}")
    if item["hint"]:
        lines.append("")
        lines.append(item["hint"])
    lines += [
        "",
        "Reply as strict JSON: "
        '{"formation": "<one candidate exactly>", "offense_side": "left|right", '
        '"confidence": 0.0-1.0, "why": "<short>"}',
    ]
    return "\n".join(lines)


def call_claude(item, model):
    import anthropic
    client = anthropic.Anthropic()
    with open(item["image"], "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    msg = client.messages.create(
        model=model,
        max_tokens=400,
        messages=[{"role": "user", "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}},
            {"type": "text", "text": build_prompt(item)},
        ]}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    try:
        s = text[text.index("{"):text.rindex("}") + 1]
        return json.loads(s)
    except Exception:
        return {"formation": None, "raw": text}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="CSAI_FORMATIONS")
    ap.add_argument("--cache-dir", default="cache")
    ap.add_argument("--limit", type=int, default=46)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--model", default="claude-opus-4-8")
    ap.add_argument("--dry-run", action="store_true", help="build prompts/images, skip the API")
    args = ap.parse_args()

    base = args.cache_dir if os.path.isabs(args.cache_dir) else \
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", args.cache_dir)
    labels = {r[""].strip(): r["actual_play"].strip()
              for r in csv.DictReader(open(LABELS_CSV))}

    snaps = sorted(glob.glob(os.path.join(base, args.folder, "snap_detection", "*_snap_detection.json")))
    vids = [os.path.basename(p).replace("_snap_detection.json", "") for p in snaps]
    VALID = {"TREY", "TRIPS", "DETROIT", "DALLAS", "SLOT", "DENVER", "PRO"}

    tpl_top1 = tpl_top5 = vis_ok = n = 0
    for vid in vids:
        actual = labels.get(vid)
        if not actual or actual.split()[0] not in VALID:
            continue
        item = build_item(vid, args.folder, base, top_k=args.top_k)
        if item is None:
            continue
        n += 1
        cf = actual.split()[0]
        tpl_top1 += (base_family(item["candidates"][0]) == cf)
        tpl_top5 += (cf in {base_family(c) for c in item["candidates"]})
        if args.dry_run:
            print(f"[{n}] {vid}: candidates={item['candidates']}  img={item['image']}")
            continue
        pred = call_claude(item, args.model)
        ok = base_family(pred.get("formation")) == cf
        vis_ok += ok
        print(f"[{n}] {vid:<16} coach={cf:<8} vision={pred.get('formation'):<16} "
              f"{'OK' if ok else '.'}  conf={pred.get('confidence')}")
        if n >= args.limit:
            break

    print(f"\n=== {n} clips ===")
    print(f"template top-1 family:  {tpl_top1}/{n} = {tpl_top1/n*100:.0f}%")
    print(f"template top-5 ceiling: {tpl_top5}/{n} = {tpl_top5/n*100:.0f}%")
    if not args.dry_run:
        print(f"VISION re-rank family:  {vis_ok}/{n} = {vis_ok/n*100:.0f}%")


if __name__ == "__main__":
    main()
