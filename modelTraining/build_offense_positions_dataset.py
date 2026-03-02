#!/usr/bin/env python3
"""
Build a training CSV from cache: snap frame + positions (to find offense side)
+ players (first 11 on that side) → 22 inputs; label from folder CSV "OFF FORM".

Usage:
    python build_offense_positions_dataset.py \
        --cache-dir ../cache \
        --folder-name "Testing Footage" \
        --csv-path "../cache/Testing Footage/Testing Footage_data.csv" \
        --output offense_positions.csv

If --csv-path is omitted, uses {cache_dir}/{folder_name}/{folder_name}_data.csv.
Clip matching: if CSV has "CLIP NAME" (or "Clip Name"), rows match by that column;
otherwise row i is matched to the i-th video when sorted by clip name (label column "OFF FORM").
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
# Position classes we treat as offense (not defense, not ref)
OFFENSE_CLASSES = frozenset({
    "quarterback", "qb", "running_back", "rb", "wide_receiver", "wr",
    "tight_end", "te", "offensive_line", "ol", "center", "guard", "tackle",
})


def _normalize_class(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


def get_snap_frame(snap_detection_path: str) -> int | None:
    """Return snap frame number from snap_detection JSON, or None if missing/empty."""
    if not os.path.exists(snap_detection_path):
        return None
    with open(snap_detection_path, "r") as f:
        data = json.load(f)
    snaps = data.get("snaps") or []
    if not snaps:
        return None
    return snaps[0].get("frame")


def get_offense_side_from_positions(position_detections: list, image_width: float) -> str | None:
    """
    Determine which side of the field the offense is on by comparing
    offensive player x-positions relative to the defense.

    If more offensive players are to the left of the defense's median x,
    return "left"; if more are to the right, return "right".
    If we can't make that comparison, fall back to comparing the
    offense median to the field midpoint.
    """
    offense_x: list[float] = []
    defense_x: list[float] = []

    for det in position_detections:
        cls_name = _normalize_class(det.get("class") or "")
        bbox = det.get("bbox") or {}
        cx = bbox.get("center_x")
        if cx is None:
            continue

        if cls_name == "ref":
            continue
        if cls_name == "defense":
            defense_x.append(float(cx))
        else:
            # Treat anything else (QB, WR, RB, etc.) as offense
            offense_x.append(float(cx))

    if not offense_x:
        return None

    # If we have defensive positions, use them as the reference.
    if defense_x:
        defense_median = float(pd.Series(defense_x).median())
        offense_left = sum(1 for x in offense_x if x < defense_median)
        offense_right = sum(1 for x in offense_x if x > defense_median)

        if offense_left > offense_right:
            return "left"
        if offense_right > offense_left:
            return "right"

    # Fallback: compare offense median to field midpoint
    offense_median = float(pd.Series(offense_x).median())
    return "right" if offense_median > (image_width / 2.0) else "left"


def get_positions_at_snap(positions_path: str, snap_frame: int) -> tuple[list, float]:
    """
    Load position JSON and return (detections at snap frame, image width).
    Position file may have a single frame (snap only); match by frame_number.
    """
    with open(positions_path, "r") as f:
        data = json.load(f)
    frames = data.get("frames") or []
    width = float((data.get("video_info") or {}).get("width") or 1920)
    for fr in frames:
        if fr.get("frame_number") == snap_frame:
            return (fr.get("detections") or [], width)
    return ([], width)


def get_normalized_positions(players_path: str, snap_frame: int) -> list:
    """Return list of player detections (with bbox) for the given frame."""
    try:    
        with open(players_path, "r") as f:
            data = json.load(f)
            frames = data.get("normalized_positions") or []
            return frames[str(snap_frame)] or []
    except Exception as e:
        print(f"[ERROR] Error loading normalized positions: {e}")
        return []   


def take_first_11_on_side(detections: list, side: str, image_width: float) -> list[tuple[float, float]]:
    """
    From player detections (each with bbox.center_x, center_y), take the 11
    players on the given side ("left" or "right"), ordered along the x-axis.
    else:
        options = ("x2","y2")
    points = []
    Returns list of (x, y) pairs; length may be < 11.
    """
    points = []
    for det in detections:
        nx = det["normalized_position"]["x"]
        ny = det["normalized_position"]["y"]
        ox = det["original_bbox"]["center_x"]
        oy = det["original_bbox"]["center_y"]
        if ox is None or oy is None or nx is None or ny is None:
            continue
        points.append([float(nx), float(ny), float(ox), float(oy)])
    if not points:
        return []
    # Order purely along the x-axis on that side
    points.sort(key=lambda p: p[0])
    points = points[:11] if side == "left" else points[-11:]
    points.sort(key=lambda p: p[1])

    #Normalize attack from the left side to the right side
    if side == "right":
        x_min = []
        #Find Min x value
        for point in points:
            x_min.append(point[0])
        x_min_val = np.array(x_min).min()
        #Subtract 2 distance from LOS from all x values
        for point in points:
            dist = point[0] - x_min_val
            point[0] = float(point[0] - 2*dist)
    return points


def load_folder_csv(
    csv_path: str,
    clip_column: str,
    label_column: str,
) -> tuple[pd.DataFrame | None, bool]:
    """
    Load CSV. Returns (df, has_clip_col).
    If has_clip_col, we match by clip name; else caller can match by row order.
    """
    if not os.path.exists(csv_path):
        return None, False
    df = pd.read_csv(csv_path)
    if label_column not in df.columns:
        return None, False
    for key in [clip_column, "CLIP NAME", "Clip Name", "clip name"]:
        if key in df.columns:
            df = df.rename(columns={key: "CLIP NAME"})
            return df, True
    # No clip column: caller can match by row order (sorted video stems)
    return df, False


def get_label_for_clip(
    df: pd.DataFrame | None,
    video_stem: str,
    label_column: str,
    match_by_order: bool,
    video_stem_to_index: dict[str, int] | None,
) -> str | None:
    """
    Get label for this clip. If match_by_order and video_stem_to_index is set,
    use row index; else match by _clip column.
    """
    
    if df is None or label_column not in df.columns:
        return None
    for _, row in df.iterrows():
        clip_raw = str(row.get("CLIP NAME", "")).strip()
        # Normalize CSV clip name: drop any path and file extension
        clip_base = os.path.basename(clip_raw)
        clip_stem = os.path.splitext(clip_base)[0]
        if clip_stem == video_stem or clip_raw == video_stem:
            val = row.get(label_column)
            return str(val).strip() if pd.notna(val) else None
    return None


def build_dataset(
    cache_dir: str,
    folder_name: str,
    csv_path: str | None,
    clip_column: str,
    label_column: str,
    output_path: str,
) -> int:
    """
    Scan cache folder for videos with snap_detection + positions + players;
    for each, compute 22 features and label from CSV. Write combined CSV.
    Returns number of rows written.
    """
    base_dir = os.path.join(cache_dir, folder_name)
    if not os.path.isdir(base_dir):
        print(f"[ERROR] Folder not found: {base_dir}", file=sys.stderr)
        return 0

    # Resolve CSV
    if not csv_path:
        csv_path = os.path.join(base_dir, f"{folder_name}_data.csv")
    if not os.path.exists(csv_path):
        print(f"[WARNING] CSV not found: {csv_path}. Labels will be missing.", file=sys.stderr)
    df_csv, has_clip_col = (
        load_folder_csv(csv_path, clip_column, label_column) if os.path.exists(csv_path) else (None, False)
    )
    match_by_order = df_csv is not None and not has_clip_col

    snap_dir = os.path.join(base_dir, "snap_detection")
    positions_dir = os.path.join(base_dir, "positions")
    normalized_positions_dir = os.path.join(base_dir, "homography")

    # Build sorted list of video stems that have snap files (for row-order matching)
    snap_files = [n for n in os.listdir(snap_dir or []) if n.endswith("_snap_detection.json")]
   
    video_stems_sorted = sorted(
        n.replace("_snap_detection.json", "") for n in snap_files
    )
    video_stem_to_index = {s: i for i, s in enumerate(video_stems_sorted)} if match_by_order else None
   
    rows = []
    for snap_name in sorted(snap_files):
        video_stem = snap_name.replace("_snap_detection.json", "")
        snap_path = os.path.join(snap_dir, snap_name)
        positions_path = os.path.join(positions_dir, f"{video_stem}_position.json")
        normalized_positions_path = os.path.join(normalized_positions_dir, f"{video_stem}_normalized_positions.json")

        snap_frame = get_snap_frame(snap_path)
        if snap_frame is None:
            print(f"[SKIP] No snap frame for: {video_stem}")
            continue

        position_detections, image_width = get_positions_at_snap(positions_path, snap_frame)
        offense_side = get_offense_side_from_positions(position_detections, image_width)
       
        if offense_side is None:
            print(f"[SKIP] Could not determine offense side for: {video_stem}")
            continue
        
        normalized_positions = get_normalized_positions(normalized_positions_path, snap_frame)
        points = take_first_11_on_side(normalized_positions, offense_side, image_width)
        if len(points) < 11:
            print(f"[SKIP] Only {len(points)} players on offense side for: {video_stem}")
            continue
        label = (
            get_label_for_clip(
                df_csv, video_stem, label_column,
                match_by_order=match_by_order,
                video_stem_to_index=video_stem_to_index,
            )
            if df_csv is not None else None
        )
        if label is None:
            label = ""

        # Include clip identifier in the output for easier debugging/tracing
        row = {"clip_name": video_stem}
        for i in range(11):
            row[f"nx{i + 1}"] = points[i][0]
            row[f"ny{i + 1}"] = points[i][1]
            row[f"ox{i + 1}"] = points[i][2]
            row[f"oy{i + 1}"] = points[i][3]
        row["label"] = label
        rows.append(row)

    if not rows:
        print("[WARNING] No rows produced.", file=sys.stderr)
        return 0

    out_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(output_path, index=False)
    print(f"[INFO] Wrote {len(rows)} rows to {output_path}")
    return len(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Build offense positions CSV from cache (snap + positions + players + folder CSV)."
    )
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Cache root (e.g. cache or ../cache)")
    parser.add_argument("--folder-name", type=str, required=True,
                        help='Footage folder name (e.g. "Testing Footage")')
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to folder CSV with CLIP NAME and OFF FORM. Default: {cache_dir}/{folder_name}/{folder_name}_data.csv")
    parser.add_argument("--clip-column", type=str, default="CLIP NAME",
                        help="CSV column that matches video name (default: CLIP NAME)")
    parser.add_argument("--label-column", type=str, default="OFF FORM",
                        help="CSV column for label (default: OFF FORM)")
    parser.add_argument("--output", type=str, default="offense_positions.csv",
                        help="Output CSV path. By default this will be placed "
                             "in {cache-dir}/{folder-name}/offense_positions.csv.")
    args = parser.parse_args()

    # If user did not override the default output name, place it in the cache
    # folder for this footage (e.g. cache/Testing Footage/offense_positions.csv).
    if args.output == "offense_positions.csv":
        args.output = os.path.join(args.cache_dir, args.folder_name, args.output)

    n = build_dataset(
        cache_dir=args.cache_dir,
        folder_name=args.folder_name,
        csv_path=args.csv_path,
        clip_column=args.clip_column,
        label_column=args.label_column,
        output_path=args.output,
    )
    return 0 if n else 1


if __name__ == "__main__":
    sys.exit(main())
