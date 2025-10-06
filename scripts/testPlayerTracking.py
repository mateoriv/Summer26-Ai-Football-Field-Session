#!/usr/bin/env python3
"""
Test Player Tracking Script
Detects and tracks players using YOLO + DeepSORT, outputs video with bounding boxes
"""

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

def test_player_tracking(video_path, model_path="yolo_models/bestPlayerDetectorM.pt", output_path="cache/videos/test_tracking_output.mp4"):
    """
    Test player detection and tracking with video output
    
    Args:
        video_path: Path to input video file
        model_path: Path to YOLO model weights
        output_path: Path to output video file
    """
    # Load YOLO model
    model = YOLO(model_path)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}")
    print(f"Resolution: {width}x{height}")
    print(f"Processing every frame with optimizations for speed")

    # Initialize DeepSORT tracker with football-optimized parameters
    tracker = DeepSort(
        max_age=10,           # Shorter max age for faster track termination
        n_init=2,             # Fewer frames needed to confirm track
        max_iou_distance=0.3, # Stricter IoU threshold for football
        max_cosine_distance=0.1, # Stricter appearance threshold
        nn_budget=50          # Limit appearance features for speed
    )

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_number = 0
    track_colors = {}  # Store colors for each track ID
    track_classes = {}  # Store class labels for each track ID

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process every frame but with optimizations

        # Run YOLO detection with optimizations
        yolo_results = model(frame, verbose=False, conf=0.3)  # Higher confidence threshold
        detections_xywh = []
        confs = []
        detection_classes = []

        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls.cpu().item())
                conf = float(box.conf.cpu().item())
                label = model.names[cls_id]

                if label.lower() not in ["player", "referee"]:  # Include both players and refs
                    continue

                # Skip very small detections (likely false positives)
                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                width_box = x2 - x1
                height_box = y2 - y1
                
                if width_box < 20 or height_box < 20:  # Skip tiny detections
                    continue

                center_x = x1 + width_box / 2
                center_y = y1 + height_box / 2

                # Prepare for tracker
                detections_xywh.append([center_x, center_y, width_box, height_box])
                confs.append(conf)
                detection_classes.append(label)

        # Update tracker
        tracked_objects = []
        if detections_xywh:
            # Limit detections per frame for performance (keep top detections by confidence)
            max_detections = 30  # Limit to top 30 detections per frame
            if len(detections_xywh) > max_detections:
                # Sort by confidence and keep only top detections
                sorted_indices = sorted(range(len(confs)), key=lambda i: confs[i], reverse=True)
                detections_xywh = [detections_xywh[i] for i in sorted_indices[:max_detections]]
                confs = [confs[i] for i in sorted_indices[:max_detections]]
                detection_classes = [detection_classes[i] for i in sorted_indices[:max_detections]]
            
            # Convert to format expected by deep_sort_realtime
            detections_list = []
            for i, (xywh, conf, cls) in enumerate(zip(detections_xywh, confs, detection_classes)):
                # deep_sort_realtime expects ([left, top, w, h], confidence, class)
                bbox = [xywh[0] - xywh[2]/2, xywh[1] - xywh[3]/2, xywh[2], xywh[3]]
                detections_list.append((bbox, conf, cls))
            
            tracks = tracker.update_tracks(detections_list, frame=frame)
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = int(track.track_id)  # Ensure track_id is an integer
                bbox = track.to_tlwh()  # Returns [x, y, w, h]
                x, y, w, h = bbox
                
                # Convert back to x1, y1, x2, y2 for drawing
                x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
                
                # Store class information for this track
                if track_id not in track_classes:
                    # Find the class from the original detection
                    track_classes[track_id] = "player"  # Default fallback
                
                tracked_objects.append({
                    "track_id": track_id,
                    "bbox": (x1, y1, x2, y2),
                    "class": track_classes.get(track_id, "player")
                })

        # Draw bounding boxes on frame
        for obj in tracked_objects:
            track_id = obj["track_id"]
            class_name = obj["class"]
            x1, y1, x2, y2 = obj["bbox"]
            
            # Assign a consistent color for each track ID
            if track_id not in track_colors:
                # Generate a color based on track ID
                np.random.seed(track_id)
                color = tuple(map(int, np.random.randint(0, 255, 3)))
                track_colors[track_id] = color
            else:
                color = track_colors[track_id]
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID and class label
            label = f"{class_name} ID: {track_id}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(frame, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Add frame info
        info_text = f"Frame: {frame_number}/{total_frames} | Tracks: {len(tracked_objects)}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)  # Black outline

        # Write frame to output video
        out.write(frame)

        if frame_number % 50 == 0:
            print(f"Processed frame {frame_number}/{total_frames} - Active tracks: {len(tracked_objects)}")

        frame_number += 1

    # Cleanup
    cap.release()
    out.release()

    print(f"Tracking test complete. Video saved to: {output_path}")
    print(f"Total unique tracks: {len(track_colors)}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Test Player Detection & Tracking with Video Output')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='cache/videos/test_tracking_output.mp4', help='Path to output video file')
    parser.add_argument('--model', type=str, default='yolo_models/bestPlayerDetectorM.pt', help='Path to YOLO model weights')
    args = parser.parse_args()

    try:
        test_player_tracking(args.video, args.model, args.output)
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main()
