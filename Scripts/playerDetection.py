import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

def playerDetection(video_path, model_path="yolo_models/bestPlayerDetectorM.pt", output_path="cache/playerDetection/playerDetection.json"):
    """
    Detect players in a video file
    
    Args:
        video_path: Path to input video file
        model_path: Path to YOLO model weights
        output_path: Path to output JSON file
    
    Returns:
        Dictionary with detection results for all frames
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

    # Initialize results structure
    results = {
        "video_info": {
            "path": video_path,
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height
        },
        "frames": []
    }

    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        yolo_results = model(frame, verbose=False)
        detections = []

        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls.cpu().item())
                conf = float(box.conf.cpu().item())
                label = model.names[cls_id]

                if label.lower() != "player":
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                width = x2 - x1
                height = y2 - y1
                center_x = x1 + width / 2
                center_y = y1 + height / 2

                detections.append({
                    "class": label,
                    "class_id": cls_id,
                    "confidence": conf,
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
                })

        # Add detection data to results
        results["frames"].append({
            "frame_number": frame_number,
            "timestamp": frame_number / fps,
            "detections": detections
        })

        if frame_number % 50 == 0:
            print(f"Processed frame {frame_number}/{total_frames}")

        frame_number += 1

    cap.release()

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Player detection complete. Results saved to: {output_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Player Detection Module (Video)')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='cache/playerDetection/playerDetection.json', help='Path to output JSON file')
    parser.add_argument('--model', type=str, default='yolo_models/bestPlayerDetectorM.pt', help='Path to YOLO model weights')
    args = parser.parse_args()

    try:
        results = playerDetection(args.video, args.model, args.output)
        total_detections = sum(len(frame['detections']) for frame in results['frames'])
        print(f"Detected {total_detections} player objects across {len(results['frames'])} frames")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    main()