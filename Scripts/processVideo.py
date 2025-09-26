#!/usr/bin/env python3
"""
Process Video Script
Processes a selected video file through the detection and tracking pipeline
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

def process_video(video_path, output_dir="cache/processed_videos"):
    """
    Process a video file through the detection and tracking pipeline
    
    Args:
        video_path (str): Path to the video file to process
        output_dir (str): Directory to save processed outputs
    
    Returns:
        dict: Processing results and output paths
    """
    print(f"Processing video: {video_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video filename without extension
    video_name = Path(video_path).stem
    
    # Step 1: Player Detection
    print("Step 1: Running player detection...")
    detection_output = f"{output_dir}/{video_name}_detection.json"
    detection_cmd = ["python3", "Scripts/playerDetection.py", "--video", video_path, "--output", detection_output]
    print(f"Running: {' '.join(detection_cmd)}")
    
    try:
        result = subprocess.run(detection_cmd, capture_output=True, text=True, check=True)
        print("✅ Player detection completed successfully")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Player detection failed: {e.stderr}")
        raise Exception(f"Player detection failed: {e.stderr}")
    
    # Step 2: Homography Transformation (if correspondence points exist)
    print("Step 2: Checking for homography transformation...")
    correspondence_file = "cache/correspondence/correspondencePoints.json"
    if os.path.exists(correspondence_file):
        print("Correspondence points found, running homography transformation...")
        homography_output = f"{output_dir}/{video_name}_homography.json"
        homography_cmd = ["python3", "Scripts/homographyTransform.py", "--input", detection_output, "--correspondence", correspondence_file, "--output", homography_output]
        print(f"Running: {' '.join(homography_cmd)}")
        
        try:
            result = subprocess.run(homography_cmd, capture_output=True, text=True, check=True)
            print("✅ Homography transformation completed successfully")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Homography transformation failed: {e.stderr}")
            raise Exception(f"Homography transformation failed: {e.stderr}")
    else:
        print("No correspondence points found, skipping homography transformation")
        homography_output = None
    
    # Step 3: Render Field Video (if homography was successful)
    if homography_output and os.path.exists(homography_output):
        print("Step 3: Rendering field video...")
        field_video_output = f"{output_dir}/{video_name}_field.mp4"
        render_cmd = ["python3", "Scripts/renderFieldVideo.py", "--input", homography_output, "--output", field_video_output]
        print(f"Running: {' '.join(render_cmd)}")
        
        try:
            result = subprocess.run(render_cmd, capture_output=True, text=True, check=True)
            print("✅ Field video rendering completed successfully")
            print(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"❌ Field video rendering failed: {e.stderr}")
            raise Exception(f"Field video rendering failed: {e.stderr}")
    else:
        print("Skipping field video rendering (no homography data)")
        field_video_output = None
    
    # Return processing results
    results = {
        "video_path": video_path,
        "video_name": video_name,
        "detection_output": detection_output,
        "homography_output": homography_output,
        "field_video_output": field_video_output,
        "status": "completed"
    }
    
    print(f"Processing completed for: {video_name}")
    return results

def main():
    """Main function for command line usage"""
    parser = argparse.ArgumentParser(description="Process a video file through the detection and tracking pipeline")
    parser.add_argument("--video", required=True, help="Path to the video file to process")
    parser.add_argument("--output-dir", default="cache/processed_videos", help="Output directory for processed files")
    
    args = parser.parse_args()
    
    # Check if video file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    # Process the video
    try:
        results = process_video(args.video, args.output_dir)
        
        # Save results to JSON
        results_file = f"{args.output_dir}/{Path(args.video).stem}_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {results_file}")
        
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
