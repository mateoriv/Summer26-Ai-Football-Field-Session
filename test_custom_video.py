#!/usr/bin/env python3
"""
Test script to demonstrate custom video widget with bounding boxes
This shows a much better approach than trying to overlay on QMediaPlayer
"""

import sys
import json
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QSlider
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPainter, QPen, QBrush, QColor, QPixmap, QImage

class CustomVideoWidget(QWidget):
    """Custom video widget that can display video frames and bounding boxes"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setStyleSheet("background-color: black; border: 2px solid #333;")
        
        # Video properties
        self.video_path = None
        self.cap = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30
        
        # Detection data
        self.detection_data = None
        self.show_boxes = True
        
        # Playback control
        self.is_playing = False
        self.timer = QTimer()
        self.timer.timeout.connect(self.next_frame)
        
        print("🎬 CustomVideoWidget created")
    
    def load_video(self, video_path):
        """Load a video file"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"❌ Could not open video: {video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        
        print(f"✅ Loaded video: {self.total_frames} frames at {self.fps} FPS")
        return True
    
    def load_detection_data(self, detection_file):
        """Load detection data from JSON file"""
        try:
            with open(detection_file, 'r') as f:
                self.detection_data = json.load(f)
            print(f"✅ Loaded detection data: {len(self.detection_data.get('frames', []))} frames")
        except Exception as e:
            print(f"❌ Error loading detection data: {e}")
    
    def next_frame(self):
        """Move to next frame"""
        if not self.cap:
            return
        
        self.current_frame += 1
        if self.current_frame >= self.total_frames:
            self.current_frame = 0  # Loop back to start
        
        self.update()  # Trigger repaint
    
    def toggle_playback(self):
        """Toggle play/pause"""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            print("⏸️ Paused")
        else:
            self.timer.start(int(1000 / self.fps))  # Convert FPS to milliseconds
            self.is_playing = True
            print("▶️ Playing")
    
    def toggle_boxes(self):
        """Toggle bounding box display"""
        self.show_boxes = not self.show_boxes
        print(f"📦 Bounding boxes: {'ON' if self.show_boxes else 'OFF'}")
        self.update()
    
    def paintEvent(self, event):
        """Custom paint event to draw video frame and bounding boxes"""
        painter = QPainter(self)
        
        # Get current video frame
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            ret, frame = self.cap.read()
            
            if ret:
                # Convert OpenCV frame to QPixmap
                height, width, channel = frame.shape
                bytes_per_line = 3 * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
                pixmap = QPixmap.fromImage(q_image)
                
                # Scale to fit widget
                scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                painter.drawPixmap(0, 0, scaled_pixmap)
                
                # Draw bounding boxes if enabled
                if self.show_boxes and self.detection_data:
                    self.draw_bounding_boxes(painter, scaled_pixmap.size())
            else:
                # Draw black background if no frame
                painter.fillRect(self.rect(), QColor(0, 0, 0))
                painter.setPen(QPen(QColor(255, 255, 255), 2))
                painter.drawText(self.rect(), Qt.AlignCenter, f"Frame {self.current_frame}/{self.total_frames}")
        else:
            # Draw black background
            painter.fillRect(self.rect(), QColor(0, 0, 0))
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawText(self.rect(), Qt.AlignCenter, "No Video Loaded")
    
    def draw_bounding_boxes(self, painter, video_size):
        """Draw bounding boxes on the video"""
        if not self.detection_data or 'frames' not in self.detection_data:
            return
        
        # Find detections for current frame
        current_detections = self.get_detections_for_frame(self.current_frame)
        
        if current_detections:
            print(f"🎨 Drawing {len(current_detections)} bounding boxes for frame {self.current_frame}")
            
            # Calculate scaling factors
            scale_x = video_size.width() / 1280  # Assume original video is 1280x720
            scale_y = video_size.height() / 720
            
            for detection in current_detections:
                self.draw_single_bbox(painter, detection, scale_x, scale_y)
        else:
            print(f"🎨 No detections for frame {self.current_frame}")
    
    def get_detections_for_frame(self, frame_number):
        """Get detections for a specific frame"""
        if not self.detection_data or 'frames' not in self.detection_data:
            return []
        
        # Find the closest frame in the data
        closest_frame = None
        min_diff = float('inf')
        
        for frame_data in self.detection_data['frames']:
            frame_idx = frame_data.get('frame_number', 0)
            diff = abs(frame_idx - frame_number)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame_data
        
        if closest_frame and min_diff < 15:
            return closest_frame.get('detections', [])
        
        return []
    
    def draw_single_bbox(self, painter, detection, scale_x, scale_y):
        """Draw a single bounding box"""
        bbox = detection.get('bbox', {})
        if not bbox or 'x1' not in bbox:
            return
        
        # Extract coordinates
        x1 = bbox.get('x1', 0)
        y1 = bbox.get('y1', 0)
        x2 = bbox.get('x2', 0)
        y2 = bbox.get('y2', 0)
        
        # Scale coordinates
        scaled_x = int(x1 * scale_x)
        scaled_y = int(y1 * scale_y)
        scaled_w = int((x2 - x1) * scale_x)
        scaled_h = int((y2 - y1) * scale_y)
        
        # Draw bounding box
        painter.setPen(QPen(QColor(0, 255, 0), 3))
        painter.setBrush(QBrush(QColor(0, 255, 0, 50)))
        painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
        
        # Draw label
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'player')
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(scaled_x, scaled_y - 5, f"{class_name} {confidence:.2f}")
        
        print(f"🎯 Drawing {class_name} box: ({scaled_x}, {scaled_y}) {scaled_w}x{scaled_h} conf={confidence:.2f}")

class TestWindow(QMainWindow):
    """Test window for custom video widget"""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Custom Video Widget Test")
        self.setGeometry(100, 100, 800, 600)
        
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        
        # Create custom video widget
        self.video_widget = CustomVideoWidget()
        layout.addWidget(self.video_widget)
        
        # Create controls
        controls_layout = QVBoxLayout()
        
        # Play/Pause button
        self.play_button = QPushButton("▶ Play")
        self.play_button.clicked.connect(self.toggle_playback)
        controls_layout.addWidget(self.play_button)
        
        # Bounding box toggle
        self.bbox_button = QPushButton("📦 Toggle Boxes")
        self.bbox_button.clicked.connect(self.toggle_boxes)
        controls_layout.addWidget(self.bbox_button)
        
        # Frame slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.valueChanged.connect(self.seek_frame)
        controls_layout.addWidget(self.frame_slider)
        
        layout.addLayout(controls_layout)
        central_widget.setLayout(layout)
        
        # Load test video and detection data
        self.load_test_data()
    
    def load_test_data(self):
        """Load test video and detection data"""
        # Try to load a test video
        test_video = "cache/videos/test_tracking_clip2.mp4"
        if self.video_widget.load_video(test_video):
            self.frame_slider.setRange(0, self.video_widget.total_frames - 1)
            print("✅ Test video loaded")
        
        # Try to load detection data
        detection_file = "cache/processed_videos/Wide - Clip 002_detection.json"
        self.video_widget.load_detection_data(detection_file)
    
    def toggle_playback(self):
        """Toggle video playback"""
        self.video_widget.toggle_playback()
        self.play_button.setText("⏸ Pause" if self.video_widget.is_playing else "▶ Play")
    
    def toggle_boxes(self):
        """Toggle bounding boxes"""
        self.video_widget.toggle_boxes()
        self.bbox_button.setText("📦 Hide Boxes" if self.video_widget.show_boxes else "📦 Show Boxes")
    
    def seek_frame(self, frame):
        """Seek to specific frame"""
        self.video_widget.current_frame = frame
        self.video_widget.update()

def main():
    """Main function to run the test"""
    app = QApplication(sys.argv)
    
    window = TestWindow()
    window.show()
    
    print("🎬 Custom Video Widget Test")
    print("This demonstrates a much better approach than QMediaPlayer overlays!")
    print("Features:")
    print("- Direct video frame rendering")
    print("- Bounding box overlay")
    print("- Full control over playback")
    print("- No overlay issues!")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
