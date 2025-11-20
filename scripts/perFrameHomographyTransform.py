#!/usr/bin/env python3
"""
Per-Frame Homography Transformation Script
Transforms player detections using frame-specific correspondence points
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

def process_per_frame_homography(player_detections_path, correspondence_points_path, output_path):
    """
    Process player detections and correspondence points per frame to generate normalized positions.
    Args:
        player_detections_path (str): Path to player detections JSON file.
        correspondence_points_path (str): Path to correspondence points JSON file.
        output_path (str): Path to save the output normalized positions JSON file.
    """
    print(f"Loading player detections from: {player_detections_path}")
    player_data = load_json_data(player_detections_path)
    
    print(f"Loading correspondence points from: {correspondence_points_path}")
    correspondence_data = load_json_data(correspondence_points_path)

    total_frames = player_data.get('total_frames', 0)
    if not total_frames:
        total_frames = len(player_data.get('frames', []))
    
    frame_correspondences = correspondence_data.get('frame_correspondences', {})
    
    normalized_positions_output = {
        "total_frames": total_frames,
        "frames_processed": 0,
        "frames_with_sufficient_markers": 0,
        "normalized_positions": {}
    }
    
    frames_with_sufficient_markers = 0
    
    print(f"Processing {total_frames} frames for per-frame homography transformation...")

    for i, frame_data in enumerate(player_data.get('frames', [])):
        frame_number = frame_data.get('frame_number', i)
        player_detections = frame_data.get('detections', [])
        
        current_frame_correspondences = frame_correspondences.get(str(frame_number), [])
        
        image_points = []
        field_points = []
        
        # Collect image and field points from correspondence data
        for cp in current_frame_correspondences:
            image_points.append([cp['image_point']['x'], cp['image_point']['y']])
            field_points.append([cp['field_point']['x'], cp['field_point']['y']])
        
        if len(image_points) < 4:
            # Skip frame if not enough correspondence points
            normalized_positions_output['normalized_positions'][str(frame_number)] = []
            if i % 50 == 0:
                progress = (i + 1) / total_frames * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_number}: Skipping (not enough markers: {len(image_points)})")
            continue
        
        # Calculate homography matrix for this frame
        H = get_homography_matrix(image_points, field_points)
        
        if H is None:
            normalized_positions_output['normalized_positions'][str(frame_number)] = []
            if i % 50 == 0:
                progress = (i + 1) / total_frames * 100
                print(f"Progress: {progress:.1f}% - Frame {frame_number}: Skipping (homography failed)")
            continue
        
        frames_with_sufficient_markers += 1
        
        # Transform player detections
        transformed_players = []
        for player_det in player_detections:
            bbox = player_det.get('bbox', {})
            if 'center_x' in bbox and 'center_y' in bbox:
                player_pixel_point = (bbox['center_x'], bbox['center_y'])
                normalized_pos = transform_point(player_pixel_point, H)
                
                if normalized_pos:
                    transformed_players.append({
                        "frame_number": frame_number,
                        "object_label": player_det.get('class', 'player'),
                        "normalized_position": {"x": normalized_pos[0], "y": normalized_pos[1]},
                        "original_bbox": bbox,
                        "confidence": player_det.get('confidence', 0.0)
                    })
        
        normalized_positions_output['normalized_positions'][str(frame_number)] = transformed_players
        
        if i % 50 == 0:
            progress = (i + 1) / total_frames * 100
            print(f"Progress: {progress:.1f}% - Frame {frame_number}: Processed {len(transformed_players)} players")

    normalized_positions_output['frames_processed'] = total_frames
    normalized_positions_output['frames_with_sufficient_markers'] = frames_with_sufficient_markers
    
    save_json_data(normalized_positions_output, output_path)
    print(f"Normalized positions saved to: {output_path}")
    print(f"Total frames: {total_frames}, Frames with sufficient markers: {frames_with_sufficient_markers}")

def main():
    parser = argparse.ArgumentParser(description='Perform per-frame homography transformation')
    parser.add_argument('--player-detections', required=True,
                        help='Path to player detections JSON file')
    parser.add_argument('--correspondence-points', required=True,
                        help='Path to per-frame correspondence points JSON file')
    parser.add_argument('--output', required=True,
                        help='Path to save the output normalized positions JSON file')
    
    args = parser.parse_args()
    
    process_per_frame_homography(args.player_detections, args.correspondence_points, args.output)

if __name__ == "__main__":
    sys.exit(main())