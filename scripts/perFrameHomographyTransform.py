#!/usr/bin/env python3
"""
Per-Frame Homography Transformation Script
Transforms player position detections using frame-specific correspondence points
"""

import argparse
import json
import os
import sys
import numpy as np
import cv2

# NCAA Field dimensions (in yards) - consistent with virtualField.py
FIELD_LENGTH_YD = 120.0  # 120 yards (0-120)
FIELD_WIDTH_YD = 160.0 / 3.0  # ~53.33 yards (0-53.33)

def load_json_data(file_path):
    """Load JSON data from a file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json_data(data, file_path):
    """Save JSON data to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)

def get_homography_matrix(image_points, field_points):
    """
    Calculate the homography matrix from image points to field points.
    Args:
        image_points (np.array): Nx2 array of (x, y) pixel coordinates.
        field_points (np.array): Nx2 array of (x, y) field coordinates (yards).
    Returns:
        np.array: 3x3 homography matrix, or None if calculation fails.
    """
    if len(image_points) < 4 or len(field_points) < 4:
        return None

    # Ensure points are float32
    image_points = np.array(image_points, dtype=np.float32)
    field_points = np.array(field_points, dtype=np.float32)

    H, _ = cv2.findHomography(image_points, field_points, cv2.LMEDS, 5.0)
    return H


def field_points_are_degenerate(field_points, min_distinct=2):
    """
    A homography needs source/destination points spanning 2D. If the field
    points are (near-)colinear -- e.g. every detected yard number sits on the
    same hash line (all y==8) -- the solve collapses every player onto that
    line. Reject that case up front.

    Returns True when the field points do not span both axes.
    """
    if len(field_points) < 4:
        return True
    fp = np.array(field_points, dtype=np.float32)
    distinct_x = len(np.unique(np.round(fp[:, 0], 1)))
    distinct_y = len(np.unique(np.round(fp[:, 1], 1)))
    return distinct_x < min_distinct or distinct_y < min_distinct


def gather_window_correspondences(frame_correspondences, frame_number, window):
    """
    Collect correspondence points from frames within +/- `window` of
    `frame_number`, deduplicated by field position. For each distinct field
    point, keep the instance from the frame *closest* to the target (smallest
    camera-motion error), breaking ties by detection confidence.

    The camera is nearly static over a sub-second window around the snap, so
    this recovers both hash lines even when any single frame only sees one.

    Returns (image_points, field_points) as plain lists.
    """
    best = {}  # field-key -> (sort_key, image_point, field_point)
    for fn_str, cps in frame_correspondences.items():
        try:
            fn = int(fn_str)
        except (TypeError, ValueError):
            continue
        dist = abs(fn - frame_number)
        if dist > window:
            continue
        for cp in cps:
            fp = cp.get('field_point') or {}
            ip = cp.get('image_point') or {}
            if 'x' not in fp or 'x' not in ip:
                continue
            key = (round(fp['x'], 1), round(fp['y'], 1))
            conf = (cp.get('yard_marker_info') or {}).get('confidence', 0.0)
            sort_key = (dist, -conf)  # nearest frame first, then highest conf
            if key not in best or sort_key < best[key][0]:
                best[key] = (sort_key, [ip['x'], ip['y']], [fp['x'], fp['y']])

    image_points = [v[1] for v in best.values()]
    field_points = [v[2] for v in best.values()]
    return image_points, field_points

def transform_point(point_px, H):
    """
    Transform a single pixel point to field coordinates using homography matrix.
    Args:
        point_px (tuple): (x, y) pixel coordinates.
        H (np.array): 3x3 homography matrix.
    Returns:
        tuple: (x, y) field coordinates (yards), or None if transformation fails.
    """
    if H is None:
        return None
    
    point_px_homogeneous = np.array([[point_px[0]], [point_px[1]], [1]], dtype=np.float32)
    point_field_homogeneous = H @ point_px_homogeneous
    
    # Normalize by the third coordinate
    if point_field_homogeneous[2][0] != 0:
        x_field = point_field_homogeneous[0][0] / point_field_homogeneous[2][0]
        y_field = point_field_homogeneous[1][0] / point_field_homogeneous[2][0]
        return (x_field, y_field)
    return None

def process_per_frame_homography(position_detections_path, correspondence_points_path, output_path, window=15):
    """
    Process player positions detections and correspondence points per frame to generate normalized positions.
    Args:
        position_detections_path (str): Path to position detections JSON file.
        correspondence_points_path (str): Path to correspondence points JSON file.
        output_path (str): Path to save the output normalized positions JSON file.
    """
    print(f"Loading position detections from: {position_detections_path}")
    position_data = load_json_data(position_detections_path)
    
    print(f"Loading correspondence points from: {correspondence_points_path}")
    correspondence_data = load_json_data(correspondence_points_path)

    total_frames = position_data.get('total_frames', 0)
    if not total_frames:
        total_frames = len(position_data.get('frames', []))
    
    frame_correspondences = correspondence_data.get('frame_correspondences', {})
    
    normalized_positions_output = {
        "total_frames": total_frames,
        "frames_processed": 0,
        "frames_with_sufficient_markers": 0,
        "normalized_positions": {}
    }
    
    frames_with_sufficient_markers = 0
    
    print(f"Processing {total_frames} frames for per-frame homography transformation...")

    for i, frame_data in enumerate(position_data.get('frames', [])):
        frame_number = frame_data.get('frame_number', i)
        position_detections = frame_data.get('detections', [])
        
        # Aggregate correspondence points across a window of nearby frames so a
        # frame that only saw one hash line can borrow the other from a neighbour.
        image_points, field_points = gather_window_correspondences(
            frame_correspondences, frame_number, window
        )

        if len(image_points) < 4:
            # Skip frame if not enough correspondence points
            normalized_positions_output['normalized_positions'][str(frame_number)] = []
            if i % 50 == 0:
                progress = (i + 1) / total_frames * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_number}: Skipping (not enough markers: {len(image_points)})")
            continue

        # Reject colinear point sets (all markers on one hash line) before they
        # collapse every player onto that line.
        if field_points_are_degenerate(field_points):
            normalized_positions_output['normalized_positions'][str(frame_number)] = []
            if i % 50 == 0:
                progress = (i + 1) / total_frames * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_number}: Skipping (colinear markers, would collapse)")
            continue

        # Calculate homography matrix for this frame
        H = get_homography_matrix(image_points, field_points)

        if H is None:
            normalized_positions_output['normalized_positions'][str(frame_number)] = []
            if i % 50 == 0:
                progress = (i + 1) / total_frames * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_number}: Skipping (homography failed)")
            continue

        # Transform position detections
        transformed_positions = []
        for position_det in position_detections:
            bbox = position_det.get('bbox', {})
            if 'center_x' in bbox and 'center_y' in bbox:
                player_pixel_point = (bbox['center_x'], bbox['center_y'])
                normalized_pos = transform_point(player_pixel_point, H)

                if normalized_pos:
                    transformed_positions.append({
                        "frame_number": frame_number,
                        "object_label": position_det.get('class'),
                        "normalized_position": {"x": normalized_pos[0], "y": normalized_pos[1]},
                        "original_bbox": bbox,
                        "confidence": position_det.get('confidence', 0.0)
                    })

        # Output guards: drop the frame rather than emit garbage when the solve
        # went bad despite the input checks (happens when the camera panned
        # within the aggregation window, so mixed pixel coords give a wrong H).
        if transformed_positions:
            ys = [p["normalized_position"]["y"] for p in transformed_positions]
            xs = [p["normalized_position"]["x"] for p in transformed_positions]
            # (a) collapsed onto ~one line
            collapsed = float(np.std(ys)) < 1.0
            # (b) most players landed off the field (x in ~[0,120], y in ~[0,53.3])
            in_bounds = sum(
                1 for x, y in zip(xs, ys)
                if -5.0 <= x <= 125.0 and -3.0 <= y <= 56.0
            )
            mostly_off_field = in_bounds < 0.6 * len(transformed_positions)
            if collapsed or mostly_off_field:
                normalized_positions_output['normalized_positions'][str(frame_number)] = []
                if i % 50 == 0:
                    reason = "collapsed to one line" if collapsed else "resolved off-field"
                    progress = (i + 1) / total_frames * 100
                    print(f"Progress: {progress:.1f}% - Frame {frame_number}: Skipping ({reason})")
                continue

        frames_with_sufficient_markers += 1
        normalized_positions_output['normalized_positions'][str(frame_number)] = transformed_positions
        
        if i % 50 == 0:
            progress = (i + 1) / total_frames * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_number}: Processed {len(transformed_positions)} players")

    normalized_positions_output['frames_processed'] = total_frames
    normalized_positions_output['frames_with_sufficient_markers'] = frames_with_sufficient_markers
    
    save_json_data(normalized_positions_output, output_path)
    print(f"Normalized positions saved to: {output_path}")
    print(f"Total frames: {total_frames}, Frames with sufficient markers: {frames_with_sufficient_markers}")

def main():
    parser = argparse.ArgumentParser(description='Perform per-frame homography transformation')
    parser.add_argument('--position-detections', required=True,
                        help='Path to position detections JSON file')
    parser.add_argument('--correspondence-points', required=True,
                        help='Path to per-frame correspondence points JSON file')
    parser.add_argument('--output', required=True,
                        help='Path to save the output normalized positions JSON file')
    parser.add_argument('--window', type=int, default=15,
                        help='Frames each side to aggregate correspondence points over (default: 15)')

    args = parser.parse_args()

    process_per_frame_homography(args.position_detections, args.correspondence_points, args.output, window=args.window)

if __name__ == "__main__":
    sys.exit(main())