#!/usr/bin/env python3
"""
Static Process Script

Takes a video name and loads snap detection and homography data,
then extracts player data for snap frames.
"""

import json
import os
import sys
import argparse


def load_data(file_path):
    """
    Load JSON data from a file.
    
    Args:
        file_path: Absolute path to the JSON file
    
    Returns:
        Dictionary with loaded JSON data, or None if not found
    """
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"[INFO] Loaded data from: {file_path}")
        return data
    except Exception as e:
        print(f"[ERROR] Failed to load data from {file_path}: {e}")
        return None


def get_player_data_for_frame(video_name, folder_name=None, cache_dir="cache", project_root=None):
    """
    Get player detection data for snap frames.
    
    Args:
        video_name: Name of the video (without extension)
        folder_name: Name of the folder containing the video (optional, will try to find)
        cache_dir: Cache directory name (default: "cache")
        project_root: Project root directory (defaults to parent of script directory)
    
    Returns:
        Dictionary with processed frame data for all snap frames, or None if not found
    """
    # Get project root if not provided
    if project_root is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
    
   
    
    if folder_name is None:
        print(f"[ERROR] Could not find folder containing snap detection for video: {video_name}")
        return None
    
    # Construct absolute paths
    snap_file_path = os.path.join(project_root, cache_dir, folder_name, "snap_detection", f"{video_name}_snap_detection.json")
    homography_file_path = os.path.join(project_root, cache_dir, folder_name, "homography", f"{video_name}_normalized_positions.json")
    
    # Load snap detection data
    snap_data = load_data(snap_file_path)
    if snap_data is None:
        return None
    
    # Load homography data
    homography_data = load_data(homography_file_path)
    if homography_data is None:
        return None
    
    # Get snap frames
    snaps = snap_data.get('snaps', [])
    if not snaps:
        print(f"[WARNING] No snap frames found in snap detection")
        return None
    
    # Get normalized positions
    normalized_positions = homography_data.get('normalized_positions', {})
    
    # Process each snap frame
    results = []

    snap_frame_number = snaps[0].get('frame')
    snap_time = snaps[0].get('time', 0.0)
    
    # Find frame data in normalized positions (keys are strings)
    frame_key = str(snap_frame_number)
    frame_detections = normalized_positions.get(frame_key, [])
    
    if frame_detections:
        print(f"[SUCCESS] Found player data for snap frame {snap_frame_number}")
        results = {
            "snap_frame": snap_frame_number,
            "snap_time": snap_time,
            "detections": frame_detections
        }
    else:
        # Try to find closest frame
        closest_frame = None
        min_diff = float('inf')
        for frame_key_str, detections in normalized_positions.items():
            try:
                frame_num = int(frame_key_str)
                diff = abs(frame_num - snap_frame_number)
                if diff < min_diff:
                    min_diff = diff
                    closest_frame = (frame_num, detections)
            except ValueError:
                continue
        
        if closest_frame and min_diff <= 5:  # Within 5 frames
            frame_num, detections = closest_frame
            print(f"[INFO] Using closest frame {frame_num} for snap frame {snap_frame_number} (difference: {min_diff})")
            results.append({
                "snap_frame": snap_frame_number,
                "snap_time": snap_time,
                "actual_frame": frame_num,
                "frame_difference": min_diff,
                "detections": detections
            })
        else:
            print(f"[WARNING] Could not find player data for snap frame {snap_frame_number}")
    
    if not results:
        print(f"[ERROR] No player data found for any snap frames")
        return None
    
    return results


def process_frame_data(frame_data):
    """
    Process frame data to get player data for snap frames.
    
    Args:
        frame_data: Dictionary with frame data containing player detections
    
    Returns:
        Dictionary with processed frame data
    """
    if frame_data is None:
        return None
    print(f"[SUCCESS] Processed frame data: {frame_data}")
    return frame_data


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Load player detection data for snap frames")
    parser.add_argument("--video-name", type=str, required=True,
                        help="Name of the video (without extension)")
    parser.add_argument("--folder-name", type=str, default=None,
                        help="Name of the folder containing the video (optional)")
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Cache directory name (default: cache)")
    
    args = parser.parse_args()
    
    # Get player data for snap frames
    frame_data = get_player_data_for_frame(
        video_name=args.video_name,
        folder_name=args.folder_name,
        cache_dir=args.cache_dir
    )
    
    if frame_data is None:
        print("[ERROR] Failed to retrieve player data for snap frames")
        sys.exit(1)
    
    processed_frame_data = process_frame_data(frame_data)
    
   
    return None


if __name__ == "__main__":
    main()
