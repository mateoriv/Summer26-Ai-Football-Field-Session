import cv2
import json
import argparse
import os

def displayBoundingBoxesVideo(video_path, json_path, output_path=None, frame_skip=1):
    """
    Display bounding boxes from JSON on video frames and save as video.
    
    Args:
        video_path: Path to the input video file
        json_path: Path to the detection JSON file
        output_path: Path to save output video (optional)
        frame_skip: Process every Nth frame (default: 1 for all frames)
    """
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📹 Video: {total_frames} frames, {fps} FPS, {width}x{height}")

    # Load detections
    with open(json_path, "r") as f:
        data = json.load(f)

    if "frames" not in data or len(data["frames"]) == 0:
        print("No frames found in JSON")
        return

    # Create output video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"💾 Saving output to: {output_path}")

    # Create frame lookup dictionary for faster access
    frame_data = {frame["frame_number"]: frame for frame in data["frames"]}
    
    frame_count = 0
    processed_frames = 0
    
    print("🔍 Processing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Only process every frame_skip frames
        if frame_count % frame_skip == 0:
            # Get detections for this frame
            detections = frame_data.get(frame_count, {}).get("detections", [])
            
            # Draw bounding boxes
            for det in detections:
                bbox = det["bbox"]
                x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
                label = det["class"]
                conf = det["confidence"]

                # Draw rectangle (green for yard markers)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Put label
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
            
            # Add frame info
            info_text = f"Frame: {frame_count}/{total_frames} | Detections: {len(detections)}"
            cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Yard Marker Detections", frame)
            
            # Write to output video if specified
            if out:
                out.write(frame)
            
            processed_frames += 1
            
            # Progress update
            if processed_frames % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📊 Progress: {frame_count}/{total_frames} frames ({progress:.1f}%) - {len(detections)} detections")
            
            # Wait for key press (ESC to exit, SPACE to pause)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key
                break
            elif key == ord(' '):  # SPACE key
                cv2.waitKey(0)  # Wait for any key to continue
        
        frame_count += 1

    # Clean up
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"✅ Processed {processed_frames} frames with detections")

def displayBoundingBoxesImage(image_path, json_path):
    """
    Display bounding boxes from JSON on the given image.
    
    Args:
        image_path: Path to the input PNG image
        json_path: Path to the detection JSON file
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Load detections
    with open(json_path, "r") as f:
        data = json.load(f)

    # Assume we want the first (or only) frame in JSON
    if "frames" not in data or len(data["frames"]) == 0:
        print("No frames found in JSON")
        return

    detections = data["frames"][0].get("detections", [])

    # Draw bounding boxes
    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = int(bbox["x1"]), int(bbox["y1"]), int(bbox["x2"]), int(bbox["y2"])
        label = det["class"]
        conf = det["confidence"]

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Put label
        text = f"{label} {conf:.2f}"
        cv2.putText(image, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    # Show image
    cv2.imshow("Detections", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Display bounding boxes from JSON on video or image")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--image", type=str, help="Path to input PNG image")
    parser.add_argument("--json", type=str, required=True, help="Path to detection JSON file")
    parser.add_argument("--output", type=str, help="Path to save output video (optional)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (default: 1)")
    
    args = parser.parse_args()

    if args.video:
        displayBoundingBoxesVideo(args.video, args.json, args.output, args.frame_skip)
    elif args.image:
        displayBoundingBoxesImage(args.image, args.json)
    else:
        print("Error: Please specify either --video or --image")
        parser.print_help()


if __name__ == "__main__":
    main()