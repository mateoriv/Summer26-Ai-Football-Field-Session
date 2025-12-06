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
import pandas as pd
import numpy as np


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
    # Determine cache directory path
    # If cache_dir is absolute, use it directly; otherwise use project_root
    if os.path.isabs(cache_dir):
        base_cache_dir = cache_dir
    else:
        # Get project root if not provided
        if project_root is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
        base_cache_dir = os.path.join(project_root, cache_dir)
    
    if folder_name is None:
        print(f"[ERROR] Could not find folder containing snap detection for video: {video_name}")
        return None
    
    # Construct absolute paths
    snap_file_path = os.path.join(base_cache_dir, folder_name, "snap_detection", f"{video_name}_snap_detection.json")
    homography_file_path = os.path.join(base_cache_dir, folder_name, "homography", f"{video_name}_normalized_positions.json")
    
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


def process_frame_data(frame_data, video_name, folder_name=None, cache_dir="cache", project_root=None):
    """
    Process frame data and update the associated CSV file.
    
    Args:
        frame_data: Dictionary with frame data containing player detections
        video_name: Name of the video (without extension) - used to find the CSV row
        folder_name: Name of the folder containing the video
        cache_dir: Cache directory name (default: "cache")
        project_root: Project root directory (defaults to parent of script directory)
    
    Returns:
        Dictionary with processed frame data and update status
    """
    if frame_data is None:
        return None
    
    # Determine cache directory path
    # If cache_dir is absolute, use it directly; otherwise use project_root
    if os.path.isabs(cache_dir):
        base_cache_dir = cache_dir
    else:
        # Get project root if not provided
        if project_root is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
        base_cache_dir = os.path.join(project_root, cache_dir)
    
    if folder_name is None:
        print(f"[ERROR] Folder name required to access CSV file")
        return None
    
    # Construct CSV file path
    csv_file_path = os.path.join(base_cache_dir, folder_name, f"{folder_name}_data.csv")
    print(f"[INFO] CSV file path: {csv_file_path}")
    if not os.path.exists(csv_file_path):
        print(f"[ERROR] CSV file not found: {csv_file_path}")
        return None
    
    try:
        # Load CSV file
        df = pd.read_csv(csv_file_path)
        print(f"[INFO] Loaded CSV file: {csv_file_path}")
        
        # Find the row matching the video name
        # The CSV has a 'CLIP NAME' column that should match the video name
        video_row_index = None
        for idx, row in df.iterrows():
            clip_name = str(row.get('CLIP NAME', '')).strip()
            if clip_name == video_name:
                video_row_index = idx
                break
        
        if video_row_index is None:
            print(f"[WARNING] Video '{video_name}' not found in CSV file")
            print(f"[INFO] Available clip names: {df['CLIP NAME'].tolist()[:10]}...")  # Show first 10
            return frame_data
        
        print(f"[INFO] Found video '{video_name}' at row {video_row_index}")
        
        # Extract x positions from player detections
        detections = frame_data.get('detections', [])
        x_positions = []
        
        for detection in detections:
            normalized_pos = detection.get('normalized_position', {})
            x = normalized_pos.get('x')
            if x is not None:
                x_positions.append(x)
        
        # Calculate median x position and round to nearest integer
        if x_positions:
            median_x = np.median(x_positions)
            yard_line = int(round(median_x))
            print(f"[INFO] Calculated median x position: {median_x:.2f}, rounded to yard line: {yard_line}")
            
            # Update CSV row with yard line
            df.at[video_row_index, 'YARD LINE'] = yard_line
        else:
            print(f"[WARNING] No x positions found in detections, skipping yard line update")
        
        # Save the updated CSV
        df.to_csv(csv_file_path, index=False)
        print(f"[SUCCESS] Updated and saved CSV file: {csv_file_path}")
        
        return {
            "frame_data": frame_data,
            "csv_updated": True,
            "row_index": video_row_index
        }
        
    except Exception as e:
        print(f"[ERROR] Failed to process CSV file: {e}")
        import traceback
        traceback.print_exc()
        return frame_data


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description="Load player detection data for snap frames")
    parser.add_argument("--video-name", type=str, required=True,
                        help="Name of the video (without extension)")
    parser.add_argument("--folder-name", type=str, default=None,
                        help="Name of the folder containing the video (optional)")
    parser.add_argument("--cache-dir", type=str, default="cache",
                        help="Cache directory path (can be absolute or relative to project root)")
    args = parser.parse_args()
    
    print(f"Cache directory: {args.cache_dir}")
    print(f"Video name: {args.video_name}")
    print(f"Folder name: {args.folder_name}")
    
    # Get player data for snap frames
    frame_data = get_player_data_for_frame(
        video_name=args.video_name,
        folder_name=args.folder_name,
        cache_dir=args.cache_dir
    )
    
    if frame_data is None:
        print("[ERROR] Failed to retrieve player data for snap frames")
        sys.exit(1)
    
    # Process frame data and update CSV
    processed_frame_data = process_frame_data(
        frame_data=frame_data,
        video_name=args.video_name,
        folder_name=args.folder_name,
        cache_dir=args.cache_dir
    )
    
    if processed_frame_data:
        print("[SUCCESS] Processing completed successfully")
    else:
        print("[WARNING] Processing completed but CSV update may have failed")
    
    return processed_frame_data


if __name__ == "__main__":
    main()
