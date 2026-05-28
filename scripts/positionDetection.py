import cv2
import json
import os
import sys
from ultralytics import YOLO


# Number of frames each side of the snap to detect when a snap is known.
# A window (vs. the original single snap frame) lets ByteTrack assign stable
# track_ids per player, so downstream code can majority-vote the class across
# frames -- this fixes single-frame label noise (e.g. QB missed at the exact snap)
# without retraining the model.
DEFAULT_SNAP_WINDOW = 30


def positionDetection(video_path,
                      model_path="yolo_models/positionDetection.pt",
                      output_path="cache/positionDetection/positionDetection.json",
                      snap_frame=None,
                      snap_detection_path=None,
                      window=DEFAULT_SNAP_WINDOW,
                      tracker="bytetrack.yaml"):
    """
    Detect players and their positions in a video.

    Modes:
      * snap_frame known + window > 0  -> process [snap-window, snap+window]
        frames sequentially WITH ByteTrack (each detection carries `track_id`).
      * snap_frame known + window == 0 -> single snap frame only (legacy).
      * snap_frame is None             -> process every frame WITH tracking.

    Tracking lets the matcher denoise per-frame class noise by voting across
    each track's lifetime around the snap.

    Args:
        video_path: input video.
        model_path: YOLO weights.
        output_path: output JSON.
        snap_frame: snap frame index (if known).
        snap_detection_path: optional snap_detection JSON to read snap_frame from.
        window: # frames each side of snap to detect (set 0 for legacy single-frame).
        tracker: Ultralytics tracker config (bytetrack.yaml or botsort.yaml).
    """
    # Resolve snap frame from sibling JSON if not supplied directly.
    if snap_frame is None and snap_detection_path and os.path.exists(snap_detection_path):
        try:
            with open(snap_detection_path, "r") as f:
                snap_data = json.load(f)
            snaps = snap_data.get("snaps") or []
            if snaps:
                snap_frame = snaps[0].get("frame")
                print(f"Using snap frame {snap_frame} from snap detection file")
        except Exception as e:
            print(f"Warning: could not load snap frame from {snap_detection_path}: {e}")

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

    # Determine frame range + whether to use tracking.
    if snap_frame is not None:
        if snap_frame < 0 or snap_frame >= total_frames:
            raise ValueError(f"Snap frame {snap_frame} out of range (0-{total_frames-1})")
        if window > 0:
            start = max(0, snap_frame - window)
            end = min(total_frames, snap_frame + window + 1)
            use_tracking = True
            print(f"Processing window: frames {start}-{end-1} around snap {snap_frame} with tracking")
        else:
            start, end = snap_frame, snap_frame + 1
            use_tracking = False
            print(f"Processing only snap frame {snap_frame} (window=0)")
    else:
        start, end = 0, total_frames
        use_tracking = True
        print(f"Processing all {total_frames} frames with tracking")

    results = {
        "video_info": {
            "path": video_path, "fps": fps, "total_frames": total_frames,
            "width": width, "height": height,
        },
        "frames": [],
        # Hint to downstream consumers that detections include track_id and
        # cover a contiguous window suitable for per-track aggregation.
        "tracking": {
            "enabled": bool(use_tracking),
            "tracker": tracker if use_tracking else None,
            "snap_frame": snap_frame,
            "window": window,
            "start_frame": start,
            "end_frame": end - 1,
        },
    }

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for fr in range(start, end):
        ret, frame = cap.read()
        if not ret:
            print(f"Warning: stream ended early at frame {fr}")
            break

        if use_tracking:
            # persist=True keeps ByteTrack state across calls so track_ids stay
            # stable across the window.
            yolo_results = model.track(frame, persist=True, tracker=tracker, verbose=False)
        else:
            yolo_results = model(frame, verbose=False)

        detections = []
        for r in yolo_results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls.cpu().item())
                conf = float(box.conf.cpu().item())
                label = model.names[cls_id]
                if label.lower() == "ref":
                    continue

                x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                w = x2 - x1
                h = y2 - y1
                track_id = None
                if getattr(box, "id", None) is not None:
                    try:
                        track_id = int(box.id.cpu().item())
                    except Exception:
                        track_id = None

                detections.append({
                    "class": label,
                    "class_id": cls_id,
                    "confidence": conf,
                    "track_id": track_id,
                    "bbox": {
                        "x1": float(x1), "y1": float(y1),
                        "x2": float(x2), "y2": float(y2),
                        "width": float(w), "height": float(h),
                        "center_x": float(x1 + w / 2),
                        "center_y": float(y1 + h / 2),
                    },
                })

        results["frames"].append({
            "frame_number": fr,
            "timestamp": fr / fps if fps else 0.0,
            "detections": detections,
        })

        if (fr - start) % 10 == 0:
            print(f"  ... frame {fr} ({fr-start+1}/{end-start}), {len(detections)} dets", flush=True)

    cap.release()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Position detection complete. Results saved to: {output_path}")
    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Position Detection (video) with optional ByteTrack tracking")
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="cache/positionDetection/positionDetection.json")
    parser.add_argument("--model", type=str, default="yolo_models/positionDetection.pt")
    parser.add_argument("--snap-frame", type=int, default=None)
    parser.add_argument("--snap-detection", type=str, default=None)
    parser.add_argument("--window", type=int, default=DEFAULT_SNAP_WINDOW,
                        help="frames each side of snap to process (0 = single-frame legacy)")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                        help="Ultralytics tracker config (bytetrack.yaml | botsort.yaml)")
    args = parser.parse_args()

    try:
        results = positionDetection(
            args.video, args.model, args.output,
            args.snap_frame, args.snap_detection,
            window=args.window, tracker=args.tracker,
        )
        total = sum(len(fr["detections"]) for fr in results["frames"])
        print(f"Detected {total} position objects across {len(results['frames'])} frames")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
