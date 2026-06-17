import cv2
import json
import os
import numpy as np
import sys
from ultralytics import YOLO

FRAME_STRIDE = 3  # process every Nth frame (~3x speedup)
BATCH_SIZE = 8    # frames per YOLO inference call

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
    model = YOLO(model_path)

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
    print(f"Frame stride: {FRAME_STRIDE} (processing ~{total_frames // FRAME_STRIDE} frames)", flush=True)

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

    def flush_batch(batch_frames, batch_frame_numbers):
        yolo_results = model(batch_frames, verbose=False)
        for fn, r in zip(batch_frame_numbers, yolo_results):
            detections = []
            for box in r.boxes:
                cls_id = int(box.cls.cpu().item())
                conf = float(box.conf.cpu().item())
                label = model.names[cls_id]

                if label.lower() != "player":
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                w = x2 - x1
                h = y2 - y1

                detections.append({
                    "class": label,
                    "class_id": cls_id,
                    "confidence": conf,
                    "bbox": {
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "width": float(w), "height": float(h),
                        "center_x": float(x1 + w / 2),
                        "center_y": float(y1 + h / 2)
                    }
                })

            results["frames"].append({
                "frame_number": fn,
                "timestamp": fn / fps,
                "detections": detections
            })

    frame_number = 0
    batch_frames = []
    batch_frame_numbers = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % FRAME_STRIDE == 0:
            batch_frames.append(frame)
            batch_frame_numbers.append(frame_number)

            if len(batch_frames) >= BATCH_SIZE:
                flush_batch(batch_frames, batch_frame_numbers)
                batch_frames = []
                batch_frame_numbers = []

        if frame_number % 10 == 0:
            print(f"Processed frame {frame_number}/{total_frames}", flush=True)

        frame_number += 1

    # flush remaining frames
    if batch_frames:
        flush_batch(batch_frames, batch_frame_numbers)

    cap.release()

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
    sys.exit(main())