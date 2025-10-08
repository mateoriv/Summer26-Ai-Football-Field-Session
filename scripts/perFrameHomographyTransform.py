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

def load_player_detections(detection_json_path):
    """Load player detection data from JSON file"""
    try:
        with open(detection_json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading player detections: {e}")
        return None

def load_correspondence_points(correspondence_json_path):
    """Load correspondence points data from JSON file"""
    try:
        with open(correspondence_json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading correspondence points: {e}")
        return None

def create_homography_matrix(correspondence_points):
    """
    Create homography matrix from correspondence points
    
    Args:
        correspondence_points: List of correspondence points with image_point and field_point
    
    Returns:
        Homography matrix or None if insufficient points
    """
    if len(correspondence_points) < 4:
        return None
    
    # Extract image points (pixel coordinates)
    image_points = []
    field_points = []
    
    for point in correspondence_points:
        img_pt = point.get('image_point', {})
        field_pt = point.get('field_point', {})
        
        if 'x' in img_pt and 'y' in img_pt and 'x' in field_pt and 'y' in field_pt:
            image_points.append([img_pt['x'], img_pt['y']])
            field_points.append([field_pt['x'], field_pt['y']])
    
    if len(image_points) < 4:
        return None
    
    # Convert to numpy arrays
    image_points = np.array(image_points, dtype=np.float32)
    field_points = np.array(field_points, dtype=np.float32)
    
    # Calculate homography matrix
    try:
        homography_matrix, mask = cv2.findHomography(image_points, field_points, cv2.RANSAC)
        return homography_matrix
    except Exception as e:
        print(f"Error calculating homography matrix: {e}")
        return None

def transform_player_positions(players, homography_matrix):
    """
    Transform player positions using homography matrix
    
    Args:
        players: List of player detection dictionaries
        homography_matrix: 3x3 homography matrix
    
    Returns:
        List of transformed player positions
    """
    if homography_matrix is None:
        return []
    
    transformed_players = []
    
    for player in players:
        bbox = player.get('bbox', {})
        if 'center_x' in bbox and 'center_y' in bbox:
            # Get player center coordinates
            x = bbox['center_x']
            y = bbox['center_y']
            
            # Transform using homography matrix
            point = np.array([[x, y]], dtype=np.float32).reshape(-1, 1, 2)
            transformed_point = cv2.perspectiveTransform(point, homography_matrix)
            
            # Extract transformed coordinates
            tx = float(transformed_point[0][0][0])
            ty = float(transformed_point[0][0][1])
            
            # Create transformed player data
            transformed_player = {
                "frame_number": player.get("frame_number", 0),
                "object_label": "player",
                "normalized_position": {
                    "x": tx,
                    "y": ty
                },
                "original_bbox": bbox,
                "confidence": player.get("confidence", 0.0)
            }
            
            transformed_players.append(transformed_player)
    
    return transformed_players

def process_per_frame_homography(player_detections, correspondence_data):
    """
    Process homography transformation for each frame
    
    Args:
        player_detections: Player detection data
        correspondence_data: Correspondence points data
    
    Returns:
        Dictionary with normalized player positions per frame
    """
    print("Processing per-frame homography transformation...")
    
    # Get frame correspondences
    frame_correspondences = correspondence_data.get('frame_correspondences', {})
    total_frames = correspondence_data.get('total_frames', 0)
    
    # Get player frames
    player_frames = player_detections.get('frames', [])
    
    # Process each frame
    normalized_positions = {}
    frames_processed = 0
    frames_with_sufficient_markers = 0
    
    for frame_data in player_frames:
        frame_number = frame_data.get('frame_number', 0)
        
        # Get correspondence points for this frame
        correspondence_points = frame_correspondences.get(str(frame_number), [])
        
        if len(correspondence_points) < 4:
            # Skip frame if insufficient markers
            normalized_positions[frame_number] = []
            continue
        
        # Create homography matrix for this frame
        homography_matrix = create_homography_matrix(correspondence_points)
        
        if homography_matrix is None:
            normalized_positions[frame_number] = []
            continue
        
        frames_with_sufficient_markers += 1
        
        # Get players for this frame
        players = frame_data.get('detections', [])
        
        # Transform player positions
        transformed_players = transform_player_positions(players, homography_matrix)
        normalized_positions[frame_number] = transformed_players
        
        frames_processed += 1
        
        # Progress update
        if frame_number % 50 == 0:
            progress = (frame_number + 1) / total_frames * 100
            print(f"Progress: {progress:.1f}% - Processed {frame_number + 1}/{total_frames} frames")
    
    print(f"Homography transformation completed!")
    print(f"Frames processed: {frames_processed}")
    print(f"Frames with sufficient markers: {frames_with_sufficient_markers}")
    print(f"Success rate: {frames_with_sufficient_markers/frames_processed*100:.1f}%")
    
    return {
        "total_frames": total_frames,
        "frames_processed": frames_processed,
        "frames_with_sufficient_markers": frames_with_sufficient_markers,
        "normalized_positions": normalized_positions
    }

def main():
    parser = argparse.ArgumentParser(description='Per-frame homography transformation')
    parser.add_argument('--player-detections', required=True,
                       help='Path to player detection JSON file')
    parser.add_argument('--correspondence-points', required=True,
                       help='Path to correspondence points JSON file')
    parser.add_argument('--output', required=True,
                       help='Path for output JSON file')
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.player_detections):
        print(f"Error: Player detection file not found: {args.player_detections}")
        return 1
    
    if not os.path.exists(args.correspondence_points):
        print(f"Error: Correspondence points file not found: {args.correspondence_points}")
        return 1
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load data
    print(f"Loading player detections from: {args.player_detections}")
    player_detections = load_player_detections(args.player_detections)
    if not player_detections:
        return 1
    
    print(f"Loading correspondence points from: {args.correspondence_points}")
    correspondence_data = load_correspondence_points(args.correspondence_points)
    if not correspondence_data:
        return 1
    
    # Process homography transformation
    result = process_per_frame_homography(player_detections, correspondence_data)
    
    # Save results
    print(f"Saving normalized positions to: {args.output}")
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)
    
    print("Per-frame homography transformation completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
