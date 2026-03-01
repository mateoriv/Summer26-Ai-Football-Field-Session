import cv2
import json
import os
import numpy as np
import sys
from ultralytics import YOLO

def positionDetection(video_path, model_path="yolo_models/positionDetection.pt", output_path="cache/positionDetection/positionDetection.json", snap_frame=None, snap_detection_path=None):
    """
    Detect players and their positions in a video file
    
    Args:
        video_path: Path to input video file
        model_path: Path to YOLO model weights
        output_path: Path to output JSON file
        snap_frame: Optional frame number to process (if None, processes all frames)
        snap_detection_path: Optional path to snap detection JSON file (used to get snap frame if snap_frame is None)
    
    Returns:
        Dictionary with detection results for all frames (or just snap frame if specified)
    """
    # If snap_detection_path is provided, try to load snap frame from it
    if snap_frame is None and snap_detection_path and os.path.exists(snap_detection_path):
        try:
            with open(snap_detection_path, 'r') as f:
                snap_data = json.load(f)
            snaps = snap_data.get('snaps', [])
            if snaps and len(snaps) > 0:
                snap_frame = snaps[0].get('frame')
                print(f"Using snap frame {snap_frame} from snap detection file")
        except Exception as e:
            print(f"Warning: Could not load snap frame from {snap_detection_path}: {e}")
    
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
    
    # If snap_frame is specified, validate it
    if snap_frame is not None:
        if snap_frame < 0 or snap_frame >= total_frames:
            raise ValueError(f"Snap frame {snap_frame} is out of range (0-{total_frames-1})")
        print(f"Processing only snap frame: {snap_frame}")

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

    # If snap_frame is specified, only process that frame
    if snap_frame is not None:
        # Seek to the snap frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, snap_frame)
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {snap_frame}")
        
        # Run YOLO detection
        yolo_results = model(frame, verbose=False)
        detections = []

        for r in yolo_results:
            for box in r.boxes:
                cls_id = int(box.cls.cpu().item())
                conf = float(box.conf.cpu().item())
                label = model.names[cls_id]
                
                # Dont save refs
                if label.lower() == "ref":
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
            "frame_number": snap_frame,
            "timestamp": snap_frame / fps,
            "detections": detections
        })
        
        print(f"Processed snap frame {snap_frame}")
    else:
        # Process all frames (original behavior)
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
                    
                    # Dont save refs
                    if label.lower() == "ref":
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

            if frame_number % 5 == 0:
                print(f"Processed frame {frame_number}/{total_frames}", flush=True)

            frame_number += 1

    cap.release()

    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Position detection complete. Results saved to: {output_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Position Detection Module (Video)')
    parser.add_argument('--video', type=str, required=True, help='Path to input video file')
    parser.add_argument('--output', type=str, default='cache/positionDetection/positionDetection.json', help='Path to output JSON file')
    parser.add_argument('--model', type=str, default='yolo_models/positionDetection.pt', help='Path to YOLO model weights')
    parser.add_argument('--snap-frame', type=int, default=None, help='Frame number to process (if not provided, processes all frames)')
    parser.add_argument('--snap-detection', type=str, default=None, help='Path to snap detection JSON file (used to get snap frame if --snap-frame is not provided)')
    args = parser.parse_args()

    try:
        results = positionDetection(args.video, args.model, args.output, args.snap_frame, args.snap_detection)
        total_detections = sum(len(frame['detections']) for frame in results['frames'])
        print(f"Detected {total_detections} position objects across {len(results['frames'])} frames")
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())