#!/usr/bin/env python3
"""
Utility to add a clip-name column to a CSV based on the videos in a folder.

Assumes:
  - The CSV rows are in the same order as the clips.
  - Clip i in sorted filename order corresponds to row i in the CSV.

Example (for your testing footage):

    python add_clip_names_to_csv.py ^
        --video-dir "FootballFootage/Testing Footage" ^
        --csv-path "FootballFootage/TestingFootage.csv" ^
        --column-name "CLIP NAME"

By default this overwrites the CSV in place. Use --output to write to a new file.
"""

import argparse
import os
from typing import Sequence

import pandas as pd


VIDEO_EXTS: Sequence[str] = (".mp4", ".avi", ".mov", ".mkv", ".wmv")


def list_clips(video_dir: str) -> list[str]:
    """Return sorted list of video basenames in the given directory."""
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"Video directory not found: {video_dir}")

    names: list[str] = []
    for name in os.listdir(video_dir):
        lower = name.lower()
        if any(lower.endswith(ext) for ext in VIDEO_EXTS):
            names.append(name)

    names.sort()
    return names


def add_clip_column(
    video_dir: str,
    csv_path: str,
    column_name: str = "CLIP NAME",
    output_path: str | None = None,
) -> None:
    clips = list_clips(video_dir)
    if not clips:
        raise RuntimeError(f"No video files found in: {video_dir}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    n_rows = len(df)
    n_clips = len(clips)

    if n_rows != n_clips:
        # Use the min count but warn; user can adjust if needed
        print(
            f"[WARNING] Row/clip count mismatch: {n_rows} rows vs {n_clips} clips. "
            f"Using first {min(n_rows, n_clips)}."
        )

    limit = min(n_rows, n_clips)
    df = df.iloc[:limit].copy()

    # If the column already exists, overwrite it; else insert at the front.
    if column_name in df.columns:
        df[column_name] = clips[:limit]
    else:
        df.insert(0, column_name, clips[:limit])

    out_path = output_path or csv_path
    df.to_csv(out_path, index=False)
    print(f"[INFO] Wrote CSV with '{column_name}' to: {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a clip-name column to a CSV based on videos in a folder."
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        required=True,
        help="Directory containing video files (e.g. FootballFootage/Testing Footage).",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the CSV to update (e.g. FootballFootage/TestingFootage.csv).",
    )
    parser.add_argument(
        "--column-name",
        type=str,
        default="CLIP NAME",
        help="Name of the column to add/overwrite (default: 'CLIP NAME').",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path. If omitted, CSV is overwritten in place.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    add_clip_column(
        video_dir=args.video_dir,
        csv_path=args.csv_path,
        column_name=args.column_name,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

