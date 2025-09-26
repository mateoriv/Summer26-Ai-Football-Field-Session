#!/usr/bin/env python3
"""
Real-time Object Detection using Webcam and YOLO Nano
Demonstration script for showing friends how real-time object detection works
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time

def load_yolo_model(model_path="yolov8x.pt"):
    """
    Load YOLO model
    """
    try:
        model = YOLO(model_path)
        print(f"✅ Loaded YOLO model: {model_path}")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def draw_detections(frame, results, confidence_threshold=0.5):
    """
    Draw bounding boxes and labels on the frame
    """
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Get box coordinates and confidence
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # Only draw if confidence is above threshold
                if conf > confidence_threshold:
                    # Get class name
                    class_name = result.names[cls]
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    
                    # Draw label with confidence
                    label = f"{class_name}: {conf:.2f}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    
                    # Draw label background
                    cv2.rectangle(frame, (int(x1), int(y1) - label_size[1] - 10), 
                                (int(x1) + label_size[0], int(y1)), (0, 255, 0), -1)
                    
                    # Draw label text
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def add_info_overlay(frame, fps, detections_count):
    """
    Add FPS and detection count overlay
    """
    # Add FPS counter
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add detection count
    cv2.putText(frame, f"Detections: {detections_count}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add instructions
    cv2.putText(frame, "Press 'q' to quit, 's' to save screenshot", (10, frame.shape[0] - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    """
    Main function for real-time object detection
    """
    print("🎥 Real-time Object Detection Demo")
    print("=" * 40)
    
    # Load YOLO model
    model = load_yolo_model("yolov8n.pt")
    if model is None:
        print("❌ Failed to load YOLO model. Exiting...")
        return
    
    # Initialize webcam - try different camera indices
    cap = None
    for camera_index in [0, 1, 2]:  # Try different camera indices
        cap = cv2.VideoCapture(camera_index)
        if cap.isOpened():
            print(f"✅ Webcam initialized (camera index: {camera_index})")
            break
        else:
            cap.release()
    
    if cap is None or not cap.isOpened():
        print("❌ Error: Could not open any webcam")
        print("💡 Make sure to grant camera permissions in System Preferences")
        return
    
    print("📹 Starting real-time detection...")
    print("💡 Press 'q' to quit, 's' to save screenshot")
    print("-" * 40)
    
    # FPS calculation variables
    fps_counter = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Screenshot counter
    screenshot_count = 0
    
    try:
        while True:
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Run YOLO detection
            results = model(frame, verbose=False)
            
            # Count detections
            detections_count = 0
            if results[0].boxes is not None:
                detections_count = len(results[0].boxes)
            
            # Draw detections on frame
            frame = draw_detections(frame, results, confidence_threshold=0.5)
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:  # Update FPS every 30 frames
                current_time = time.time()
                current_fps = 30 / (current_time - fps_start_time)
                fps_start_time = current_time
            
            # Add info overlay
            frame = add_info_overlay(frame, current_fps, detections_count)
            
            # Display frame
            cv2.imshow("Real-time Object Detection", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("👋 Quitting...")
                break
            elif key == ord('s'):
                # Save screenshot
                screenshot_count += 1
                filename = f"detection_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Screenshot saved: {filename}")
            elif key == ord('h'):
                # Show help
                print("\n🎮 Controls:")
                print("  'q' - Quit")
                print("  's' - Save screenshot")
                print("  'h' - Show this help")
                print("  'c' - Change confidence threshold")
                print("-" * 40)
    
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
    
    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    main()
