#!/usr/bin/env python3
"""
Automated Correspondence Points Detection Script
Uses yard marker detection JSON to automatically generate correspondence points
based on NCAA football field standards.

Input: Yard marker detection JSON from yardMarkerDetection.py
Output: Correspondence points JSON for homography transformation

Features:
- Filters detections by confidence threshold (70% default)
- Averages multiple detections of same marker using line of best fit
- Detects significant camera movement to trigger recalculation
- Maps yard marker classes to real-world field coordinates

Yard marker format: (near/far)(left/right)(yardNumber)
Possible values: fl1,fl2,fl3,fl4,f5,nl1,nl2,nl3,nl4,n5,nr1,nr2,nr3,nr4,nr5

NCAA Standards:
- Yard-line numbers: max 6 feet height, 4 feet width
- Tops of numbers: 9 yards from sidelines
- Field width: 160 feet (53.33 yards)
- Field length: 120 yards (100 + 2 endzones of 10 yards each)
"""

import json
import os
import numpy as np
from collections import defaultdict

# NCAA Field dimensions (in feet)
FIELD_LENGTH_YD = 100 # 100 yards
FIELD_WIDTH_YD = 160/3  # 160 feet to yards
YARD_MARKER_DIST_YD = 8  # Center is 8 yards from sideline
positionsDict = {
    "nl1": (10, YARD_MARKER_DIST_YD),
    "nl2": (20, YARD_MARKER_DIST_YD),
    "nl3": (30, YARD_MARKER_DIST_YD),
    "nl4": (40, YARD_MARKER_DIST_YD),
    "n5": (50, YARD_MARKER_DIST_YD),
    "nr4": (FIELD_LENGTH_YD - 40, YARD_MARKER_DIST_YD),
    "nr3": (FIELD_LENGTH_YD - 30, YARD_MARKER_DIST_YD),
    "nr2": (FIELD_LENGTH_YD - 20, YARD_MARKER_DIST_YD),
    "nr1": (FIELD_LENGTH_YD - 10, YARD_MARKER_DIST_YD),
    "fl1": (10, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),
    "fl2": (20, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),
    "fl3": (30, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),
    "fl4": (40, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),
    "f5": (50, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),
    "fr4": (FIELD_LENGTH_YD - 40, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),
    "fr3": (FIELD_LENGTH_YD - 30, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),
    "fr2": (FIELD_LENGTH_YD - 20, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),
    "fr1": (FIELD_LENGTH_YD - 10, FIELD_WIDTH_YD - YARD_MARKER_DIST_YD),

}


def load_yard_marker_detections(detection_json_path):
    """
    Load yard marker detection data from JSON file
    
    Args:
        detection_json_path: Path to yard marker detection JSON file
    
    Returns:
        Dictionary containing detection data
    """
    try:
        with open(detection_json_path, 'r') as f:
            detection_data = json.load(f)
        print(f"Loaded yard marker detections: {len(detection_data.get('frames', []))} frames")
        return detection_data
    except Exception as e:
        print(f"Error loading yard marker detections: {e}")
        return None

def filter_detections_by_confidence(detections, confidence_threshold=0.7):
    """
    Filter detections by confidence threshold
    
    Args:
        detections: List of detection dictionaries
        confidence_threshold: Minimum confidence score (default: 0.7)
    
    Returns:
        Filtered list of detections
    """
    filtered = [det for det in detections if det.get('confidence', 0) >= confidence_threshold]
    return filtered

def group_detections_by_marker(detections):
    """
    Group detections by yard marker class (e.g., all 'fl1' detections together)
    
    Args:
        detections: List of detection dictionaries
    
    Returns:
        Dictionary with marker classes as keys and detection lists as values
    """
    grouped = defaultdict(list)
    for detection in detections:
        marker_class = detection.get('class', 'unknown')
        grouped[marker_class].append(detection)
    
    return dict(grouped)

def get_field_coordinates_for_marker(marker_class):
    """
    Get field coordinates for a yard marker class
    
    Args:
        marker_class: Yard marker class (e.g., 'fl1', 'nr2')
    
    Returns:
        Dictionary with field coordinates in feet
    """
    # Parse the marker label
    print(f"Parsing marker class: {marker_class}")
    parsed = parse_yard_marker_label(marker_class)
    if not parsed:
        return None
    
    # Calculate field coordinates based on NCAA standards
    near_far = parsed["near_far"]
    left_right = parsed["left_right"] 
    yard_number = parsed["yard_number"]

    # Calculate yard line position
    # fl1 = far left 10-yard marker, fl2 = far left 20-yard marker, etc.
    yard_line = yard_number * 10  # 10, 20, 30, 40, 50
    
    field_x = positionsDict[marker_class][0]
    field_y = positionsDict[marker_class][1]

    return {
        "x": field_x,
        "y": field_y,
        "yard_line": yard_line,
        "hash_side": left_right,
        "near_far": near_far,
        "yard_number": yard_number
    }

def average_detections_simple(detections):
    """
    Average multiple detections of the same marker using weighted averaging
    
    Args:
        detections: List of detection dictionaries for the same marker class
    
    Returns:
        Single averaged detection dictionary
    """
    if len(detections) == 1:
        return detections[0]
    
    # Extract center coordinates and confidences
    centers = []
    confidences = []
    
    for det in detections:
        bbox = det.get('bbox', {})
        center_x = bbox.get('center_x', 0)
        center_y = bbox.get('center_y', 0)
        confidence = det.get('confidence', 0)
        
        centers.append([center_x, center_y])
        confidences.append(confidence)
    
    centers = np.array(centers)
    confidences = np.array(confidences)
    
    # Weight by confidence
    weights = confidences / np.sum(confidences)
    
    # Calculate weighted average
    avg_center = np.average(centers, axis=0, weights=weights)
    avg_confidence = np.average(confidences, weights=weights)
    
    # Calculate bounding box dimensions (average of all detections)
    widths = [det.get('bbox', {}).get('width', 0) for det in detections]
    heights = [det.get('bbox', {}).get('height', 0) for det in detections]
    avg_width = np.average(widths, weights=weights)
    avg_height = np.average(heights, weights=weights)
    
    # Create averaged detection
    averaged_detection = {
        'class': detections[0].get('class'),
        'class_id': detections[0].get('class_id'),
        'confidence': float(avg_confidence),
        'bbox': {
            'center_x': float(avg_center[0]),
            'center_y': float(avg_center[1]),
            'width': float(avg_width),
            'height': float(avg_height),
            'x1': float(avg_center[0] - avg_width/2),
            'y1': float(avg_center[1] - avg_height/2),
            'x2': float(avg_center[0] + avg_width/2),
            'y2': float(avg_center[1] + avg_height/2)
        }
    }
    
    print(f"Averaged {len(detections)} detections for {detections[0].get('class')} (confidence: {avg_confidence:.3f})")
    return averaged_detection


def parse_yard_marker_label(label):
    """
    Parse yard marker label to extract position and yard information
    
    Args:
        label: Yard marker label (e.g., "fl1", "nr5")
    
    Returns:
        Dictionary with parsed information
    """
    if len(label) < 3:
        return None
    
    # Extract components
    near_far = label[0]  # 'f' for far, 'n' for near
    left_right = label[1]  # 'l' for left, 'r' for right
    yard_str = label[2:]  # Yard number as string
    
    try:
        yard_number = int(yard_str)
    except ValueError:
        return None
    
    return {
        "near_far": near_far,
        "left_right": left_right,
        "yard_number": yard_number,
        "original_label": label
    }


def process_yard_marker_detections_per_frame(detection_json_path, confidence_threshold=0.7):
    """
    Process yard marker detection JSON to create correspondence points for each frame
    
    Args:
        detection_json_path: Path to yard marker detection JSON file
        confidence_threshold: Minimum confidence for detections (default: 0.7)
    
    Returns:
        Dictionary with frame-by-frame correspondence points
    """
    print(f"Processing yard marker detections per frame from: {detection_json_path}")
    
    # Load detection data
    detection_data = load_yard_marker_detections(detection_json_path)
    if not detection_data:
        return {}
    
    frames = detection_data.get('frames', [])
    print(f"Processing {len(frames)} frames for per-frame correspondence points")
    
    # Process each frame individually
    frame_correspondences = {}
    frames_with_points = 0
    last_valid_correspondence_points = []  # Track last valid frame data
    
    for i, frame_data in enumerate(frames):
        frame_number = frame_data.get('frame_number', 0)
        frame_detections = frame_data.get('detections', [])
        
        # Filter detections by confidence for this frame
        filtered_detections = filter_detections_by_confidence(frame_detections, confidence_threshold)
        
        if len(filtered_detections) < 4:
            # Not enough detections for this frame - use last valid frame data
            if last_valid_correspondence_points:
                frame_correspondences[frame_number] = last_valid_correspondence_points.copy()
                print(f"Frame {frame_number}: Insufficient points ({len(filtered_detections)}), using last valid frame data")
            else:
                frame_correspondences[frame_number] = []
                print(f"Frame {frame_number}: Insufficient points ({len(filtered_detections)}) and no previous valid data")
            continue
        
        # Group detections by marker class for this frame
        grouped_detections = group_detections_by_marker(filtered_detections)
        
        # For each marker class, use the best detection (highest confidence) instead of averaging
        best_detections = []
        for marker_class, detections in grouped_detections.items():
            # Find the detection with highest confidence
            best_detection = max(detections, key=lambda x: x.get('confidence', 0))
            best_detections.append(best_detection)
        
        # Convert to correspondence points for this frame
        correspondence_points = []
        for detection in best_detections:
            field_coords = get_field_coordinates_for_marker(detection['class'])
            if field_coords:
                last_field_coords = field_coords
                correspondence_points.append({
                    "image_point": {
                        "x": detection["bbox"]["center_x"],
                        "y": detection["bbox"]["center_y"]
                    },
                    "field_point": {
                        "x": field_coords["x"],
                        "y": field_coords["y"]
                    },
                    "yard_marker_info": {
                        "label": detection["class"],
                        "yard_line": field_coords["yard_line"],
                        "hash_side": field_coords["hash_side"],
                        "near_far": field_coords["near_far"],
                        "confidence": detection["confidence"]
                    }
                })
        
        frame_correspondences[frame_number] = correspondence_points
        
        # Update last valid correspondence points if this frame has sufficient points
        if len(correspondence_points) >= 4:
            last_valid_correspondence_points = correspondence_points.copy()
            frames_with_points += 1
        
        # Progress update every 50 frames
        if i % 50 == 0:
            progress = (i + 1) / len(frames) * 100
            print(f"Progress: {progress:.1f}% - Processed {i+1}/{len(frames)} frames ({frames_with_points} with sufficient points)")
    
    print(f"Created per-frame correspondence points: {frames_with_points}/{len(frames)} frames have sufficient points")
    
    return frame_correspondences

def process_yard_marker_detections(detection_json_path, confidence_threshold=0.7):
    """
    Process yard marker detection JSON to create correspondence points (averaged approach)
    
    Args:
        detection_json_path: Path to yard marker detection JSON file
        confidence_threshold: Minimum confidence for detections (default: 0.7)
    
    Returns:
        List of correspondence points
    """
    print(f"Processing yard marker detections from: {detection_json_path}")
    
    # Load detection data
    detection_data = load_yard_marker_detections(detection_json_path)
    if not detection_data:
        return []
    
    # Collect all detections from all frames
    all_detections = []
    for frame in detection_data.get('frames', []):
        frame_detections = frame.get('detections', [])
        all_detections.extend(frame_detections)
    
    print(f"Total detections across all frames: {len(all_detections)}")
    
    # Filter by confidence
    filtered_detections = filter_detections_by_confidence(all_detections, confidence_threshold)
    
    if len(filtered_detections) < 4:
        print(f"Insufficient detections after filtering: {len(filtered_detections)} (need at least 4)")
        return []
    
    # Group by marker class
    grouped_detections = group_detections_by_marker(filtered_detections)
    
    # Average detections for each marker class
    averaged_detections = []
    for marker_class, detections in grouped_detections.items():
        averaged = average_detections_simple(detections)
        averaged_detections.append(averaged)
    
    print(f"Final averaged detections: {len(averaged_detections)}")
    
    # Convert to correspondence points
    correspondence_points = []
    for detection in averaged_detections:
        field_coords = get_field_coordinates_for_marker(detection['class'])
        if field_coords:
            correspondence_points.append({
                "image_point": {
                    "x": detection["bbox"]["center_x"],
                    "y": detection["bbox"]["center_y"]
                },
                "field_point": {
                    "x": field_coords["x"],
                    "y": field_coords["y"]
                },
                "yard_marker_info": {
                    "label": detection["class"],
                    "yard_line": field_coords["yard_line"],
                    "hash_side": field_coords["hash_side"],
                    "near_far": field_coords["near_far"],
                    "confidence": detection["confidence"]
                }
            })
    
    print(f"Created {len(correspondence_points)} correspondence points")
    return correspondence_points


def save_correspondence_points(points, output_path):
    """
    Save correspondence points to JSON file
    
    Args:
        points: List of correspondence points
        output_path: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create the output structure
    output_data = {
        "correspondences": points,
        "metadata": {
            "total_points": len(points),
            "field_dimensions": {
                "length_yd": FIELD_LENGTH_YD,
                "width_yd": FIELD_WIDTH_YD,
                "yard_marker_distance_yd": YARD_MARKER_DIST_YD
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Correspondence points saved to: {output_path}")

def save_correspondence_points_per_frame(frame_correspondences, output_path):
    """
    Save per-frame correspondence points to JSON file
    
    Args:
        frame_correspondences: Dictionary with frame-by-frame correspondence points
        output_path: Path to output JSON file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create the output structure for per-frame data
    output_data = {
        "frame_correspondences": frame_correspondences,
        "metadata": {
            "total_frames": len(frame_correspondences),
            "frames_with_sufficient_points": sum(1 for points in frame_correspondences.values() if len(points) >= 4),
            "field_dimensions": {
                "length_yd": FIELD_LENGTH_YD,
                "width_yd": FIELD_WIDTH_YD,
                "yard_marker_distance_yd": YARD_MARKER_DIST_YD
            }
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Per-frame correspondence points saved to: {output_path}")


def main():
    """Main function for standalone execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Automated Correspondence Points Detection')
    parser.add_argument('--detection-json', type=str, required=True, 
                       help='Path to yard marker detection JSON file')
    parser.add_argument('--output', type=str, default='cache/correspondence/correspondences.json', 
                       help='Path to output JSON file')
    parser.add_argument('--confidence', type=float, default=0.7,
                       help='Minimum confidence threshold for detections (default: 0.7)')
    parser.add_argument('--min-points', type=int, default=4,
                       help='Minimum number of correspondence points required (default: 4)')
    parser.add_argument('--per-frame', action='store_true',
                       help='Generate per-frame correspondence points instead of averaged')
    
    args = parser.parse_args()
    
    try:
        if args.per_frame:
            # Process yard marker detections to create per-frame correspondence points
            frame_correspondences = process_yard_marker_detections_per_frame(args.detection_json, args.confidence)
            
            # Count frames with sufficient points
            frames_with_points = sum(1 for points in frame_correspondences.values() if len(points) >= args.min_points)
            
            if frames_with_points == 0:
                print(f"No frames with sufficient correspondence points found (need at least {args.min_points} points per frame)")
                return 1
            
            # Save per-frame points
            save_correspondence_points_per_frame(frame_correspondences, args.output)
            
            # Print summary
            print(f"\n=== PER-FRAME CORRESPONDENCE POINTS SUMMARY ===")
            print(f"Total frames processed: {len(frame_correspondences)}")
            print(f"Frames with sufficient points: {frames_with_points}")
            print(f"Success rate: {frames_with_points/len(frame_correspondences)*100:.1f}%")
            
            # Show sample frames
            sample_frames = list(frame_correspondences.keys())[:5]  # Show first 5 frames
            for frame_num in sample_frames:
                points = frame_correspondences[frame_num]
                if len(points) >= args.min_points:
                    print(f"\nFrame {frame_num} ({len(points)} points):")
                    for i, point in enumerate(points[:3]):  # Show first 3 points
                        info = point["yard_marker_info"]
                        print(f"  {i+1}. {info['label']} -> Yard {info['yard_line']}, {info['hash_side']} hash")
                        print(f"     Image: ({point['image_point']['x']:.1f}, {point['image_point']['y']:.1f})")
                        print(f"     Field: ({point['field_point']['x']:.1f}, {point['field_point']['y']:.1f}) ft")
                        print(f"     Confidence: {info['confidence']:.3f}")
                    if len(points) > 3:
                        print(f"  ... and {len(points)-3} more points")
            
        else:
            # Process yard marker detections to create correspondence points (averaged approach)
            points = process_yard_marker_detections(args.detection_json, args.confidence)
            
            # Validate points
            if len(points) < args.min_points:
                print(f"Insufficient correspondence points found: {len(points)} (need at least {args.min_points})")
                return 1
            
            # Save points
            save_correspondence_points(points, args.output)
            
            # Print summary
            print("\n=== CORRESPONDENCE POINTS SUMMARY ===")
            for i, point in enumerate(points):
                info = point["yard_marker_info"]
                print(f"{i+1}. {info['label']} -> Yard {info['yard_line']}, {info['hash_side']} hash, {info['near_far']} side")
                print(f"   Image: ({point['image_point']['x']:.1f}, {point['image_point']['y']:.1f})")
                print(f"   Field: ({point['field_point']['x']:.1f}, {point['field_point']['y']:.1f}) ft")
                print(f"   Confidence: {info['confidence']:.3f}")
                print()
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1

if __name__ == "__main__":
    exit(main())