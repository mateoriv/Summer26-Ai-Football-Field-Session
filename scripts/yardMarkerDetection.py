#!/usr/bin/env python3
"""
Yard Marker Detection Script

Detects yard markers in football video using YOLO model.
Classes: nr1,nr2,nr3,nr4,n5,nl1,nl2,nl3,nl4,fr1,fr2,fr3,fr4,f5,fl1,fl2,fl3,fl4

Input: video file
Output: JSON file with frame-by-frame yard marker detection data
"""

import cv2
import json
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO
import numpy as np

# Class mapping for yard marker classes (matches model training)
YARD_MARKER_CLASSES = {
    0: 'f5', 1: 'fl1', 2: 'fl2', 3: 'fl3', 4: 'fl4',
    5: 'fr1', 6: 'fr2', 7: 'fr3', 8: 'fr4', 9: 'n5',
    10: 'nl1', 11: 'nl2', 12: 'nl3', 13: 'nl4',
    14: 'nr1', 15: 'nr2', 16: 'nr3', 17: 'nr4'
}

def yardMarkerDetection(video_path, model_path="yolo_models/TRYTHISONE.pt", confidence_threshold=0.5):
    """
    Detect yard markers in video frames using a trained YOLO model
    
    Args:
        video_path: Path to input video file
        model_path: Path to YOLO model file
        confidence_threshold: Minimum confidence for detections
    
    Returns:
        Dictionary with detection results
    """
    print(f"[INFO] Starting yard marker detection for: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load YOLO model
    print(f"[INFO] Loading YOLO model: {model_path}")
    try:
        model = YOLO(model_path)
        print("[SUCCESS] Model loaded successfully")
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model: {e}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Video info: {total_frames} frames, {fps} FPS, {width}x{height}")
    
    # Initialize results
    results = {
        "video_info": {
            "path": video_path,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height
        },
        "detection_info": {
            "model_path": model_path,
            "confidence_threshold": confidence_threshold,
            "classes": list(YARD_MARKER_CLASSES.values())
        },
        "frames": []
    }
    
    frame_number = 0
    detections_count = 0
    
    print("[INFO] Processing frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run YOLO detection
        try:
            yolo_results = model(frame, conf=confidence_threshold, verbose=False)
            
            # Process detections
            frame_detections = []
            if yolo_results and len(yolo_results) > 0:
                result = yolo_results[0]  # Get first (and only) result
                
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()  # Get bounding boxes
                    confidences = result.boxes.conf.cpu().numpy()  # Get confidences
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)  # Get class IDs
                    
                    for i, (box, conf, class_id) in enumerate(zip(boxes, confidences, class_ids)):
                        # Get class name
                        class_name = YARD_MARKER_CLASSES.get(class_id, f"unknown_{class_id}")
                        
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box
                        width = x2 - x1
                        height = y2 - y1
                        center_x = (x1 + x2) / 2
                        center_y = (y1 + y2) / 2
                        
                        detection = {
                            "class": class_name,
                            "class_id": int(class_id),
                            "confidence": float(conf),
                            "bbox": {
                                "x1": float(x1),
                                "y1": float(y1),
                                "x2": float(x2),
                                "y2": float(y2),
                                "width": float(width),
                                "height": float(height),
                                "center_x": float(center_x),
                                "center_y": float(center_y)
                            }
                        }
                        
                        frame_detections.append(detection)
                        detections_count += 1
            
            # Store frame results
            frame_result = {
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "detections": frame_detections
            }
            
            results["frames"].append(frame_result)
            
            # Progress update
            if frame_number % 5 == 0:  # Update every 30 frames
                print(f"Processed frame {frame_number}/{total_frames}", flush=True)
            
            frame_number += 1
            
        except Exception as e:
            print(f"[WARNING] Error processing frame {frame_number}: {e}")
            # Add empty frame result
            frame_result = {
                "frame_number": frame_number,
                "timestamp": frame_number / fps,
                "detections": []
            }
            results["frames"].append(frame_result)
            frame_number += 1
            continue
    
    # Clean up
    cap.release()
    
    # Add summary statistics
    results["summary"] = {
        "total_frames_processed": frame_number,
        "total_detections": detections_count,
        "frames_with_detections": len([f for f in results["frames"] if f["detections"]]),
        "average_detections_per_frame": detections_count / frame_number if frame_number > 0 else 0
    }
    
    print(f"[SUCCESS] Yard marker detection completed!")
    print(f"[INFO] Summary: {detections_count} total detections across {frame_number} frames")
    print(f"[INFO] Frames with detections: {results['summary']['frames_with_detections']}")
    
    return results

def save_results(results, output_path):
    """Save detection results to JSON file"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"[SUCCESS] Results saved to: {output_path}")

def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Yard Marker Detection Module')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='cache/yardMarkerDetection.json', 
                       help='Path to output JSON file')
    parser.add_argument('--model', type=str, default='yolo_models/TRYTHISONE.pt',
                       help='Path to YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Confidence threshold for detections')
    
    args = parser.parse_args()
    
    try:
        # Run detection
        results = yardMarkerDetection(
            video_path=args.video,
            model_path=args.model,
            confidence_threshold=args.confidence
        )
        
        # Save results
        save_results(results, args.output)
        
        print("[SUCCESS] Yard marker detection completed successfully!")
        
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()