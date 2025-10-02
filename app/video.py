from PySide6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QSlider, QLabel, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QTime, QTimer, QRectF
from PySide6.QtGui import QPainter, QPen, QFont, QColor, QBrush, QImage, QPixmap
import json
import os
import cv2
import numpy as np

class CustomVideoWidget(QWidget):
    """Custom video widget that can draw both video and bounding boxes"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.detection_data = None
        self.yard_marker_data = None  # Add yard marker data
        self.current_frame = 0
        self.show_boxes = True
        self.overlay_items = []
        self.cap = None  # Initialize cap attribute
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False
        self.parent_window = parent  # Store parent reference
        
        # Timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        # Don't start timer automatically - only when playing
        
        print(f"🔧 CustomVideoWidget created")
    
    def set_detection_data(self, data):
        """Set the detection data for bounding boxes"""
        self.detection_data = data
        if data is None:
            print(f"🚫 Detection data cleared - no bounding boxes will be shown")
        else:
            print(f"🎯 Detection data set in custom video widget")
        if data and 'frames' in data:
            print(f"   Total frames: {len(data['frames'])}")
        self.update()
    
    def set_show_boxes(self, show):
        """Set whether to show bounding boxes"""
        self.show_boxes = show
        print(f"🔧 Custom video show_boxes set to: {show}")
        self.update()
    
    def update_frame(self):
        """Update current frame and redraw"""
        if self.cap and self.cap.isOpened() and self.is_playing:
            # Move to next frame
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0  # Loop back to start
            
            # Update parent's progress slider and time label
            if hasattr(self, 'parent_window') and self.parent_window:
                self.update_parent_controls()
            
            # Trigger repaint
            self.update()
    
    def update_parent_controls(self):
        """Update parent's progress slider and time label"""
        if not hasattr(self, 'parent_window') or not self.parent_window:
            return
            
        parent = self.parent_window
        
        # Update progress slider
        if hasattr(parent, 'progress_slider') and hasattr(parent, 'custom_video_total_frames'):
            progress = int((self.current_frame / parent.custom_video_total_frames) * 100)
            parent.progress_slider.setValue(progress)
        
        # Update time label
        if hasattr(parent, 'time_label'):
            fps = self.fps if self.fps > 0 else 30.0
            current_time = self.current_frame / fps
            total_time = self.total_frames / fps
            parent.time_label.setText(f"{int(current_time//60):02d}:{int(current_time%60):02d} / {int(total_time//60):02d}:{int(total_time%60):02d}")
    
    def paintEvent(self, event):
        """Custom paint event to draw video frame and bounding boxes"""
        painter = QPainter(self)
        
        # Get current video frame
        if self.cap and self.cap.isOpened():
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
                
                # Center the video in the widget
                x_offset = (self.width() - scaled_pixmap.width()) // 2
                y_offset = (self.height() - scaled_pixmap.height()) // 2
                painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
                
                # Draw player bounding boxes if enabled
                if self.show_boxes and self.detection_data:
                    self.draw_bounding_boxes(painter, scaled_pixmap.size(), x_offset, y_offset)
                
                # Draw yard marker bounding boxes if enabled
                if hasattr(self.parent_window, 'show_yard_marker_boxes') and self.parent_window.show_yard_marker_boxes and self.yard_marker_data:
                    self.draw_yard_marker_boxes(painter, scaled_pixmap.size(), x_offset, y_offset)
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
    
    def draw_bounding_boxes(self, painter, video_size, x_offset=0, y_offset=0):
        """Draw bounding boxes on the video"""
        if not self.detection_data or 'frames' not in self.detection_data:
            return
        
        # Get detections for current frame
        current_detections = self.get_detections_for_frame(self.current_frame)
        
        if current_detections:
            print(f"🎨 Drawing {len(current_detections)} bounding boxes for frame {self.current_frame}")
            
            # Get actual video resolution from detection data or infer from coordinates
            video_width = 1280  # Default fallback
            video_height = 720
            
            if self.detection_data and 'video_info' in self.detection_data:
                video_info = self.detection_data['video_info']
                if 'width' in video_info and 'height' in video_info:
                    video_width = video_info['width']
                    video_height = video_info['height']
                    print(f"🎯 Using video resolution from detection data: {video_width}x{video_height}")
                else:
                    # Try to infer resolution from bounding box coordinates
                    max_x = max_y = 0
                    for frame in self.detection_data.get('frames', []):
                        for det in frame.get('detections', []):
                            bbox = det.get('bbox', {})
                            max_x = max(max_x, bbox.get('x2', 0))
                            max_y = max(max_y, bbox.get('y2', 0))
                    
                    if max_x > 1280:  # Likely 1920x1080
                        video_width = 1920
                        video_height = 1080
                        print(f"🔍 Inferred video resolution from coordinates: {video_width}x{video_height} (max coords: {max_x:.0f}, {max_y:.0f})")
                    else:
                        print(f"⚠️ Using default resolution: {video_width}x{video_height}")
            
            # Calculate scaling factors based on actual video resolution
            scale_x = video_size.width() / video_width
            scale_y = video_size.height() / video_height
            
            for detection in current_detections:
                self.draw_single_bbox(painter, detection, scale_x, scale_y, x_offset, y_offset)
    
    def get_detections_for_frame(self, frame_number):
        """Get detections for a specific frame number"""
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
    
    def draw_single_bbox(self, painter, detection, scale_x, scale_y, x_offset=0, y_offset=0):
        """Draw a single bounding box"""
        bbox = detection.get('bbox', {})
        if not bbox or 'x1' not in bbox:
            return
        
        # Extract coordinates
        x1 = bbox.get('x1', 0)
        y1 = bbox.get('y1', 0)
        x2 = bbox.get('x2', 0)
        y2 = bbox.get('y2', 0)
        
        # Scale coordinates using provided scaling factors
        scaled_x = int(x1 * scale_x) + x_offset
        scaled_y = int(y1 * scale_y) + y_offset
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
    
    def draw_yard_marker_boxes(self, painter, video_size, x_offset=0, y_offset=0):
        """Draw yard marker bounding boxes on the video"""
        if not self.yard_marker_data or 'frames' not in self.yard_marker_data:
            return
        
        # Get yard marker detections for current frame
        current_detections = self.get_yard_marker_detections_for_frame(self.current_frame)
        
        if current_detections:
            print(f"🏈 Drawing {len(current_detections)} yard marker boxes for frame {self.current_frame}")
            
            # Get actual video resolution from detection data
            video_width = 1280  # Default fallback
            video_height = 720
            
            if self.yard_marker_data and 'video_info' in self.yard_marker_data:
                video_info = self.yard_marker_data['video_info']
                if 'width' in video_info and 'height' in video_info:
                    video_width = video_info['width']
                    video_height = video_info['height']
            
            # Calculate scaling factors
            scale_x = video_size.width() / video_width
            scale_y = video_size.height() / video_height
            
            for detection in current_detections:
                self.draw_single_yard_marker_bbox(painter, detection, scale_x, scale_y, x_offset, y_offset)
        else:
            print(f"🏈 No yard marker detections for frame {self.current_frame}")
    
    def get_yard_marker_detections_for_frame(self, frame_number):
        """Get yard marker detections for a specific frame number"""
        if not self.yard_marker_data or 'frames' not in self.yard_marker_data:
            return []
        
        # Find the closest frame in the data
        closest_frame = None
        min_diff = float('inf')
        
        for frame_data in self.yard_marker_data['frames']:
            frame_idx = frame_data.get('frame_number', 0)
            diff = abs(frame_idx - frame_number)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame_data
        
        if closest_frame and min_diff < 15:
            return closest_frame.get('detections', [])
        
        return []
    
    def draw_single_yard_marker_bbox(self, painter, detection, scale_x, scale_y, x_offset=0, y_offset=0):
        """Draw a single yard marker bounding box"""
        bbox = detection.get('bbox', {})
        if not bbox or 'x1' not in bbox:
            return
        
        # Extract coordinates
        x1 = bbox.get('x1', 0)
        y1 = bbox.get('y1', 0)
        x2 = bbox.get('x2', 0)
        y2 = bbox.get('y2', 0)
        
        # Scale coordinates using provided scaling factors
        scaled_x = int(x1 * scale_x) + x_offset
        scaled_y = int(y1 * scale_y) + y_offset
        scaled_w = int((x2 - x1) * scale_x)
        scaled_h = int((y2 - y1) * scale_y)
        
        # Draw yard marker bounding box (green color)
        painter.setPen(QPen(QColor(0, 255, 0), 3))
        painter.setBrush(QBrush(QColor(0, 255, 0, 50)))
        painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
        
        # Draw label
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'yard_marker')
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(scaled_x, scaled_y - 5, f"{class_name} {confidence:.2f}")
        
        print(f"🏈 Drawing {class_name} box: ({scaled_x}, {scaled_y}) {scaled_w}x{scaled_h} conf={confidence:.2f}")
    
    def test_display_boxes(self):
        """Test function"""
        print("🧪 Testing custom video widget...")
        print(f"   show_boxes: {self.show_boxes}")
        print(f"   detection_data: {self.detection_data is not None}")
        print(f"   current_frame: {self.current_frame}")
        self.update()
    
    def force_update(self):
        """Force update"""
        print("🔧 Custom video force update called")
        self.update()
    
    def toggle_playback(self):
        """Toggle play/pause for custom video widget"""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
            print("⏸️ Custom video paused")
        else:
            self.timer.start(33)  # ~30 FPS
            self.is_playing = True
            print("▶️ Custom video playing")
    
    def load_video(self, video_path):
        """Load a video file into the custom video widget"""
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

class SimpleOverlayWidget(QWidget):
    """Simple overlay widget using QLabel for testing"""
    def __init__(self, parent):
        super().__init__(parent)
        self.setStyleSheet("background: transparent; border: none;")
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.detection_data = None
        self.current_frame = 0
        self.show_boxes = True
        self.overlay_items = []
        
        # Timer to update frame position
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_frame_from_video)
        self.update_timer.start(50)  # Update every 50ms
        
        print(f"🔧 SimpleOverlayWidget created")
    
    def set_detection_data(self, data):
        """Set the detection data for bounding boxes"""
        self.detection_data = data
        print(f"🎯 Detection data set in simple overlay widget")
        print(f"   Data keys: {list(data.keys()) if data else 'None'}")
        if data and 'frames' in data:
            print(f"   Total frames: {len(data['frames'])}")
            if data['frames']:
                first_frame = data['frames'][0]
                print(f"   First frame detections: {len(first_frame.get('detections', []))}")
        
        # Force a repaint to show bounding boxes immediately
        self.update()
    
    def set_show_boxes(self, show):
        """Set whether to show bounding boxes"""
        self.show_boxes = show
        print(f"🔧 Simple overlay show_boxes set to: {show}")
        if show:
            self.update_overlay()
        else:
            self.clear_overlay()
    
    def update_overlay(self):
        """Update the overlay with current frame data"""
        if not self.show_boxes or not self.detection_data:
            return
        
        # Clear existing items
        self.clear_overlay()
        
        # Get detections for current frame
        current_detections = self.get_detections_for_frame(self.current_frame)
        
        if current_detections:
            print(f"🎨 Drawing {len(current_detections)} bounding boxes for frame {self.current_frame}")
            for detection in current_detections:
                self.draw_bounding_box_overlay(detection)
        else:
            print(f"🎨 No detections for frame {self.current_frame}")
    
    def get_detections_for_frame(self, frame_number):
        """Get detections for a specific frame number"""
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
    
    def draw_bounding_box_overlay(self, detection):
        """Draw a single bounding box using QLabel"""
        from PySide6.QtWidgets import QLabel
        
        bbox = detection.get('bbox', {})
        if not bbox or 'x1' not in bbox:
            return
        
        # Extract coordinates from bbox object
        x1 = bbox.get('x1', 0)
        y1 = bbox.get('y1', 0)
        x2 = bbox.get('x2', 0)
        y2 = bbox.get('y2', 0)
        
        # Calculate width and height
        w = x2 - x1
        h = y2 - y1
        
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'player')
        
        # Scale bounding box to widget size
        widget_rect = self.rect()
        
        # Get actual video resolution from detection data or infer from coordinates
        video_width = 1280  # Default fallback
        video_height = 720
        
        if self.detection_data and 'video_info' in self.detection_data:
            video_info = self.detection_data['video_info']
            if 'width' in video_info and 'height' in video_info:
                video_width = video_info['width']
                video_height = video_info['height']
            else:
                # Try to infer resolution from bounding box coordinates
                max_x = max_y = 0
                for frame in self.detection_data.get('frames', []):
                    for det in frame.get('detections', []):
                        bbox = det.get('bbox', {})
                        max_x = max(max_x, bbox.get('x2', 0))
                        max_y = max(max_y, bbox.get('y2', 0))
                
                if max_x > 1280:  # Likely 1920x1080
                    video_width = 1920
                    video_height = 1080
        
        scale_x = widget_rect.width() / video_width
        scale_y = widget_rect.height() / video_height
        
        scaled_x = int(x1 * scale_x)
        scaled_y = int(y1 * scale_y)
        scaled_w = int(w * scale_x)
        scaled_h = int(h * scale_y)
        
        # Create bounding box label
        bbox_label = QLabel(f"{class_name}\n{confidence:.2f}", self)
        bbox_label.setStyleSheet("""
            background-color: rgba(0, 255, 0, 100);
            border: 2px solid green;
            color: white;
            font-weight: bold;
            font-size: 10px;
        """)
        bbox_label.setGeometry(scaled_x, scaled_y, scaled_w, scaled_h)
        bbox_label.show()
        bbox_label.raise_()
        
        self.overlay_items.append(bbox_label)
    
    def clear_overlay(self):
        """Clear all overlay items"""
        for item in self.overlay_items:
            item.deleteLater()
        self.overlay_items.clear()
    
    def update_frame_position(self, frame_number):
        """Update the current frame number and refresh overlay"""
        self.current_frame = frame_number
        if self.show_boxes:
            self.update_overlay()
    
    def update_frame_from_video(self):
        """Update frame position based on video playback"""
        # Try to get the current frame from the video player
        try:
            # Find the video player in the parent hierarchy
            parent_widget = self.parent()
            while parent_widget and not hasattr(parent_widget, 'player'):
                parent_widget = parent_widget.parent()
            
            if parent_widget and hasattr(parent_widget, 'player'):
                player = parent_widget.player
                if player and player.position() > 0:
                    # Calculate frame number from video position
                    fps = 30  # Assume 30 fps
                    position_ms = player.position()
                    frame_number = int((position_ms / 1000.0) * fps)
                    
                    if frame_number != self.current_frame:
                        self.current_frame = frame_number
                        if self.show_boxes and self.detection_data:
                            self.update_overlay()
        except Exception as e:
            # Silently handle errors to avoid spam
            pass
    
    def test_display_boxes(self):
        """Test function to force display bounding boxes for debugging"""
        print("🧪 Testing simple overlay bounding box display...")
        print(f"   show_boxes: {self.show_boxes}")
        print(f"   detection_data: {self.detection_data is not None}")
        print(f"   current_frame: {self.current_frame}")
        print(f"   Simple overlay size: {self.width()}x{self.height()}")
        print(f"   Simple overlay visible: {self.isVisible()}")
        
        # Add a test box
        test_box = self.add_test_box()
        print(f"🧪 Simple overlay test box created: {test_box.isVisible()}")
        
        # Force update
        if self.show_boxes and self.detection_data:
            self.update_overlay()
    
    def force_update(self):
        """Force update the overlay display"""
        print(f"🔧 Simple overlay force update called")
        if self.show_boxes and self.detection_data:
            self.update_overlay()
        self.update()
    
    def add_test_box(self):
        """Add a simple test box"""
        from PySide6.QtWidgets import QLabel
        test_label = QLabel("SIMPLE OVERLAY TEST", self)
        test_label.setStyleSheet("background-color: red; color: white; font-size: 16px; font-weight: bold; border: 2px solid blue;")
        test_label.setGeometry(50, 50, 250, 80)
        test_label.show()
        test_label.raise_()
        print(f"🔧 Simple test box added")
        return test_label

class BoundingBoxGraphicsView(QGraphicsView):
    """Graphics view that overlays bounding boxes on top of video"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.detection_data = None
        self.current_frame = 0
        self.show_boxes = True
        self.bounding_box_items = []
        
        # Create graphics scene
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, 1280, 720)  # Initial scene size (will be updated when detection data is loaded)
        self.setScene(self.scene)
        
        # Set up the view
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet("background: transparent;")
        print(f"🔧 BoundingBoxGraphicsView initialized with scene size: {self.scene.sceneRect()}")
        
        # Timer to update frame position for bounding box sync
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_frame_position)
        self.update_timer.start(50)  # Update every 50ms for smoother updates
    
    def resizeEvent(self, event):
        """Handle resize events to update scene size"""
        super().resizeEvent(event)
        # Update scene size to match widget size
        self.scene.setSceneRect(0, 0, self.width(), self.height())
        print(f"🔧 BoundingBoxGraphicsView resized to {self.width()}x{self.height()}")
        print(f"🔧 Scene rect updated to: {self.scene.sceneRect()}")
    
    def set_detection_data(self, data):
        """Set the detection data for bounding boxes"""
        self.detection_data = data
        print(f"🎯 Detection data set in video widget")
        print(f"   Data keys: {list(data.keys()) if data else 'None'}")
        if data and 'frames' in data:
            print(f"   Total frames: {len(data['frames'])}")
            if data['frames']:
                first_frame = data['frames'][0]
                print(f"   First frame detections: {len(first_frame.get('detections', []))}")
                # Show frame number range
                frame_numbers = [frame.get('frame_number', 0) for frame in data['frames'][:10]]  # First 10 frames
                print(f"   Sample frame numbers: {frame_numbers}")
                if len(data['frames']) > 10:
                    last_frame = data['frames'][-1]
                    
        # Update scene size based on video resolution if available
        if data and 'video_info' in data:
            video_info = data['video_info']
            if 'width' in video_info and 'height' in video_info:
                video_width = video_info['width']
                video_height = video_info['height']
                self.scene.setSceneRect(0, 0, video_width, video_height)
                print(f"🔧 Updated scene size to video resolution: {video_width}x{video_height}")
                    
        # Print last frame info if available
        if data and 'frames' in data and data['frames'] and len(data['frames']) > 10:
            last_frame = data['frames'][-1]
            print(f"   Last frame number: {last_frame.get('frame_number', 0)}")
        
        # Force a repaint to show bounding boxes immediately
        self.update()
    
    def update_frame_position(self):
        """Update current frame position based on video playback"""
        if self.parent_window and hasattr(self.parent_window, 'player'):
            # Calculate frame number from video position
            position_ms = self.parent_window.player.position()
            fps = 30  # Assume 30 FPS, could be made dynamic
            new_frame = int((position_ms / 1000.0) * fps)
            
            # Only update if frame changed
            if new_frame != self.current_frame:
                self.current_frame = new_frame
                if self.show_boxes and self.detection_data:
                    self.update_bounding_boxes()
    
    def update_bounding_boxes(self):
        """Update bounding boxes in the graphics scene"""
        # Clear existing bounding box items
        for item in self.bounding_box_items:
            self.scene.removeItem(item)
        self.bounding_box_items.clear()
        
        if not self.show_boxes or not self.detection_data:
            return
        
        # Add test box to verify graphics system is working
        test_rect = QGraphicsRectItem(20, 20, 200, 100)
        test_rect.setPen(QPen(QColor(255, 0, 0), 6))
        test_rect.setBrush(QBrush(QColor(255, 0, 0, 50)))  # Semi-transparent red
        self.scene.addItem(test_rect)
        self.bounding_box_items.append(test_rect)
        
        # Add test text
        test_text = QGraphicsTextItem(f"BBox Test - Frame {self.current_frame}")
        test_text.setPos(30, 60)
        test_text.setDefaultTextColor(QColor(255, 255, 255))
        font = QFont("Arial", 12, QFont.Bold)
        test_text.setFont(font)
        self.scene.addItem(test_text)
        self.bounding_box_items.append(test_text)
        
        # Find detections for current frame
        current_detections = self.get_detections_for_frame(self.current_frame)
        
        if current_detections:
            print(f"🎨 Drawing {len(current_detections)} bounding boxes for frame {self.current_frame}")
            # Draw bounding boxes
            for detection in current_detections:
                self.draw_bounding_box_graphics(detection)
        else:
            # Always show when no detections are found (remove the 30-frame limit)
            print(f"🎨 No detections for frame {self.current_frame}")
            cyan_rect = QGraphicsRectItem(250, 50, 150, 80)
            cyan_rect.setPen(QPen(QColor(0, 255, 255), 4))
            cyan_rect.setBrush(QBrush(QColor(0, 255, 255, 50)))
            self.scene.addItem(cyan_rect)
            self.bounding_box_items.append(cyan_rect)
            
            no_det_text = QGraphicsTextItem("No detections")
            no_det_text.setPos(260, 90)
            no_det_text.setDefaultTextColor(QColor(0, 0, 0))
            self.scene.addItem(no_det_text)
            self.bounding_box_items.append(no_det_text)
    
    def get_detections_for_frame(self, frame_number):
        """Get detections for a specific frame number"""
        if not self.detection_data or 'frames' not in self.detection_data:
            print(f"🔍 Frame {frame_number}: No detection data available")
            return []
        
        # Find the closest frame in the data
        closest_frame = None
        min_diff = float('inf')
        
        for frame_data in self.detection_data['frames']:
            frame_idx = frame_data.get('frame_number', 0)  # Use 'frame_number' instead of 'frame'
            diff = abs(frame_idx - frame_number)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame_data
        
        print(f"🔍 Frame {frame_number}: Closest frame is {closest_frame.get('frame_number', 0) if closest_frame else 'None'}, diff: {min_diff}")
        
        if closest_frame and min_diff < 15:  # Increased tolerance to 15 frames
            detections = closest_frame.get('detections', [])
            print(f"🔍 Frame {frame_number}: Found {len(detections)} detections (closest frame: {closest_frame.get('frame_number', 0)}, diff: {min_diff})")
            return detections
        else:
            print(f"🔍 Frame {frame_number}: No close frame found (min_diff: {min_diff})")
        
        return []
    
    def draw_bounding_box_graphics(self, detection):
        """Draw a single bounding box using graphics items"""
        bbox = detection.get('bbox', {})
        if not bbox or 'x1' not in bbox:
            return
        
        # Extract coordinates from bbox object
        x1 = bbox.get('x1', 0)
        y1 = bbox.get('y1', 0)
        x2 = bbox.get('x2', 0)
        y2 = bbox.get('y2', 0)
        
        # Calculate width and height
        w = x2 - x1
        h = y2 - y1
        
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'player')
        track_id = detection.get('track_id', '')
        
        # Scale bounding box to widget size
        widget_rect = self.rect()
        
        # Get actual video resolution from detection data or infer from coordinates
        video_width = 1280  # Default fallback
        video_height = 720
        
        if self.detection_data and 'video_info' in self.detection_data:
            video_info = self.detection_data['video_info']
            if 'width' in video_info and 'height' in video_info:
                video_width = video_info['width']
                video_height = video_info['height']
                print(f"🎯 BoundingBoxGraphicsView using video resolution: {video_width}x{video_height}")
            else:
                # Try to infer resolution from bounding box coordinates
                max_x = max_y = 0
                for frame in self.detection_data.get('frames', []):
                    for det in frame.get('detections', []):
                        bbox = det.get('bbox', {})
                        max_x = max(max_x, bbox.get('x2', 0))
                        max_y = max(max_y, bbox.get('y2', 0))
                
                if max_x > 1280:  # Likely 1920x1080
                    video_width = 1920
                    video_height = 1080
                    print(f"🔍 BoundingBoxGraphicsView inferred video resolution: {video_width}x{video_height}")
        
        # Calculate scaling factors
        scale_x = widget_rect.width() / video_width
        scale_y = widget_rect.height() / video_height
        
        scaled_x = int(x1 * scale_x)
        scaled_y = int(y1 * scale_y)
        scaled_w = int(w * scale_x)
        scaled_h = int(h * scale_y)
        
        # Create bounding box rectangle
        bbox_rect = QGraphicsRectItem(scaled_x, scaled_y, scaled_w, scaled_h)
        bbox_rect.setPen(QPen(QColor(0, 255, 0), 4))  # Thick green box
        bbox_rect.setBrush(QBrush(QColor(0, 255, 0, 30)))  # Semi-transparent green fill
        
        # Add a test box that's guaranteed to be visible (top-left corner)
        test_rect = QGraphicsRectItem(10, 10, 100, 50)
        test_rect.setPen(QPen(QColor(255, 0, 0), 3))  # Red test box
        test_rect.setBrush(QBrush(QColor(255, 0, 0, 50)))
        self.scene.addItem(test_rect)
        self.bounding_box_items.append(test_rect)
        self.scene.addItem(bbox_rect)
        self.bounding_box_items.append(bbox_rect)
        
        # Create white outline
        outline_rect = QGraphicsRectItem(scaled_x, scaled_y, scaled_w, scaled_h)
        outline_rect.setPen(QPen(QColor(255, 255, 255), 2))  # White outline
        outline_rect.setBrush(QBrush(Qt.NoBrush))  # No fill
        self.scene.addItem(outline_rect)
        self.bounding_box_items.append(outline_rect)
        
        # Create label text
        label_text = f"{class_name}"
        if track_id:
            label_text += f" ID:{track_id}"
        label_text += f" ({confidence:.2f})"
        
        text_item = QGraphicsTextItem(label_text)
        text_item.setPos(scaled_x, scaled_y - 25)
        text_item.setDefaultTextColor(QColor(255, 255, 255))
        font = QFont("Arial", 10, QFont.Bold)
        text_item.setFont(font)
        self.scene.addItem(text_item)
        self.bounding_box_items.append(text_item)
    
    def set_show_boxes(self, show):
        """Toggle bounding box display"""
        self.show_boxes = show
        print(f"🎨 Bounding boxes {'ON' if show else 'OFF'}")
        if show and self.detection_data:
            self.update_bounding_boxes()
        elif not show:
            # Clear all bounding box items
            for item in self.bounding_box_items:
                self.scene.removeItem(item)
            self.bounding_box_items.clear()
    
    def test_display_boxes(self):
        """Test function to force display bounding boxes for debugging"""
        print("🧪 Testing bounding box display...")
        print(f"   show_boxes: {self.show_boxes}")
        print(f"   detection_data: {self.detection_data is not None}")
        print(f"   current_frame: {self.current_frame}")
        print(f"   Graphics view size: {self.width()}x{self.height()}")
        print(f"   Graphics view visible: {self.isVisible()}")
        print(f"   Scene rect: {self.scene.sceneRect()}")
        print(f"   Scene items before: {len(self.scene.items())}")
        
        # Test the simple overlay instead
        try:
            # Find the simple overlay in the parent
            parent_widget = self.parent()
            while parent_widget and not hasattr(parent_widget, 'simple_overlay'):
                parent_widget = parent_widget.parent()
            
            if parent_widget and hasattr(parent_widget, 'simple_overlay'):
                print(f"🧪 Found simple overlay, testing it...")
                test_box = parent_widget.simple_overlay.add_test_box()
                print(f"🧪 Simple overlay test box created: {test_box.isVisible()}")
            else:
                print(f"🧪 Could not find simple overlay in parent hierarchy")
        except Exception as e:
            print(f"🧪 Simple overlay test failed: {e}")
        
        # Clear existing items
        for item in self.bounding_box_items:
            self.scene.removeItem(item)
        self.bounding_box_items.clear()
        
        # Add a simple test box at a fixed position
        test_rect = QGraphicsRectItem(50, 50, 200, 100)
        test_rect.setPen(QPen(QColor(255, 0, 0), 5))  # Thick red box
        test_rect.setBrush(QBrush(QColor(255, 0, 0, 100)))  # Semi-transparent red
        self.scene.addItem(test_rect)
        self.bounding_box_items.append(test_rect)
        print(f"🧪 Test rect added: {test_rect.rect()}")
        print(f"🧪 Test rect visible: {test_rect.isVisible()}")
        print(f"🧪 Test rect pos: {test_rect.pos()}")
        
        # Add text label
        test_text = QGraphicsTextItem("TEST BOX")
        test_text.setPos(60, 80)
        test_text.setDefaultTextColor(QColor(255, 255, 255))
        self.scene.addItem(test_text)
        self.bounding_box_items.append(test_text)
        print(f"🧪 Test text added: {test_text.pos()}")
        print(f"🧪 Test text visible: {test_text.isVisible()}")
        
        # Force scene update
        self.scene.update()
        self.viewport().update()
        print(f"🧪 Scene updated, items count: {len(self.scene.items())}")
        
        # Try a different approach - add a simple rectangle at the scene origin
        simple_rect = QGraphicsRectItem(0, 0, 100, 50)
        simple_rect.setPen(QPen(QColor(0, 255, 0), 3))  # Green box
        simple_rect.setBrush(QBrush(QColor(0, 255, 0, 50)))
        self.scene.addItem(simple_rect)
        self.bounding_box_items.append(simple_rect)
        print(f"🧪 Simple rect added at origin: {simple_rect.rect()}")
        
        # Try setting the scene rect to include our items
        self.scene.setSceneRect(0, 0, 300, 200)
        print(f"🧪 Scene rect set to: {self.scene.sceneRect()}")
        
        # Force a complete refresh
        self.scene.invalidate()
        self.update()
        print(f"🧪 Complete refresh forced")
        
        # Try a completely different approach - create a simple QLabel overlay
        try:
            from PySide6.QtWidgets import QLabel
            test_label = QLabel("OVERLAY TEST", self)
            test_label.setStyleSheet("background-color: red; color: white; font-size: 20px; font-weight: bold;")
            test_label.setGeometry(10, 10, 200, 50)
            test_label.show()
            test_label.raise_()
            print(f"🧪 QLabel overlay created and shown")
            print(f"🧪 QLabel visible: {test_label.isVisible()}")
            print(f"🧪 QLabel geometry: {test_label.geometry()}")
            
            # Try to reparent the QLabel to the main window too
            try:
                main_window = self.parent()
                while main_window and not hasattr(main_window, 'centralWidget'):
                    main_window = main_window.parent()
                if main_window:
                    print(f"🧪 Reparenting QLabel to main window: {main_window}")
                    test_label.setParent(main_window)
                    test_label.setGeometry(200, 200, 300, 100)  # Different position
                    test_label.show()
                    test_label.raise_()
                    print(f"🧪 QLabel reparented and shown")
                    print(f"🧪 QLabel visible after reparent: {test_label.isVisible()}")
                    print(f"🧪 QLabel geometry after reparent: {test_label.geometry()}")
            except Exception as e:
                print(f"🧪 QLabel reparenting failed: {e}")
                
        except Exception as e:
            print(f"🧪 QLabel overlay failed: {e}")
        
        # Try to force the graphics view to repaint
        self.repaint()
        self.update()
        self.scene.update()
        print(f"🧪 Forced repaint and update")
        
        # Try making the graphics view temporarily opaque to see if it's there
        self.setStyleSheet("background-color: rgba(255, 0, 0, 100); border: 2px solid blue;")
        print(f"🧪 Graphics view made temporarily opaque with red background")
        
        # Also test the simple overlay
        try:
            print(f"🧪 Testing simple overlay...")
            test_box = self.simple_overlay.add_test_box()
            print(f"🧪 Simple overlay test box created: {test_box.isVisible()}")
        except Exception as e:
            print(f"🧪 Simple overlay test failed: {e}")
        
        print("🧪 Test box added at position (50, 50)")
        print(f"🧪 Graphics view size: {self.width()}x{self.height()}")
        print(f"🧪 Scene rect: {self.scene.sceneRect()}")
        print(f"🧪 Scene items count: {len(self.scene.items())}")
        
        # Force the graphics view to be visible and on top
        self.show()
        self.raise_()
        self.activateWindow()
        print(f"🧪 Graphics view visible after force: {self.isVisible()}")
        print(f"🧪 Graphics view geometry: {self.geometry()}")
        print(f"🧪 Graphics view scene: {self.scene()}")
        print(f"🧪 Graphics view viewport: {self.viewport()}")
        print(f"🧪 Graphics view viewport visible: {self.viewport().isVisible()}")
        print(f"🧪 Graphics view viewport geometry: {self.viewport().geometry()}")
        
        if self.detection_data and 'frames' in self.detection_data:
            # Try to show boxes from frame 0
            test_detections = self.get_detections_for_frame(0)
            print(f"   Test detections for frame 0: {len(test_detections)}")
            
        # Force update
        if self.show_boxes:
            self.update_bounding_boxes()
    
    def force_update(self):
        """Force update the widget display"""
        if self.show_boxes and self.detection_data:
            self.update_bounding_boxes()

def create_video_title_bar(dock):
    """Create a custom title bar for the video dock widget"""
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
    from PySide6.QtGui import QFont
    
    title_bar = QWidget()
    title_bar.setFixedHeight(30)
    title_bar.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
            border-bottom: 1px solid #555555;
        }
        QLabel {
            color: white;
            font-weight: bold;
        }
        QPushButton {
            background-color: transparent;
            border: none;
            color: white;
            padding: 4px;
            border-radius: 3px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #404040;
        }
        QPushButton:pressed {
            background-color: #505050;
        }
    """)
    
    layout = QHBoxLayout()
    layout.setContentsMargins(8, 4, 8, 4)
    layout.setSpacing(8)
    
    # Left spacer to center the title
    left_spacer = QWidget()
    left_spacer.setFixedWidth(20)  # Space for close button on the right
    layout.addWidget(left_spacer)
    
    # Title label (centered)
    title_label = QLabel("Video")
    title_label.setFont(QFont("Arial", 10, QFont.Bold))
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)
    
    # Right spacer to balance the left spacer
    right_spacer = QWidget()
    right_spacer.setFixedWidth(20)  # Space for buttons on the right
    layout.addWidget(right_spacer)
    
    # Player bounding boxes checkbox
    bbox_checkbox = QPushButton("📦")
    bbox_checkbox.setFixedSize(20, 20)
    bbox_checkbox.setCheckable(True)
    bbox_checkbox.setChecked(True)  # Bounding boxes visible by default
    bbox_checkbox.setToolTip("Toggle Player Bounding Boxes")
    bbox_checkbox.setStyleSheet("""
        QPushButton {
            background-color: transparent;
            border: none;
            color: white;
            padding: 4px;
            border-radius: 3px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #404040;
        }
        QPushButton:pressed {
            background-color: #505050;
        }
        QPushButton:checked {
            background-color: #0078d4;
        }
    """)
    bbox_checkbox.clicked.connect(lambda: toggle_bounding_boxes(dock.parent(), bbox_checkbox))
    layout.addWidget(bbox_checkbox)
    
    # Yard marker bounding boxes checkbox
    yard_marker_checkbox = QPushButton("🏈")
    yard_marker_checkbox.setFixedSize(20, 20)
    yard_marker_checkbox.setCheckable(True)
    yard_marker_checkbox.setChecked(False)  # Yard markers off by default
    yard_marker_checkbox.setToolTip("Toggle Yard Marker Bounding Boxes")
    yard_marker_checkbox.setStyleSheet("""
        QPushButton {
            background-color: transparent;
            border: none;
            color: white;
            padding: 4px;
            border-radius: 3px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #404040;
        }
        QPushButton:pressed {
            background-color: #505050;
        }
        QPushButton:checked {
            background-color: #28a745;
        }
    """)
    yard_marker_checkbox.clicked.connect(lambda: toggle_yard_marker_boxes(dock.parent(), yard_marker_checkbox))
    layout.addWidget(yard_marker_checkbox)
    
    # Close button (X)
    close_btn = QPushButton("✕")
    close_btn.setFixedSize(20, 20)
    close_btn.setToolTip("Close")
    close_btn.clicked.connect(dock.close)
    layout.addWidget(close_btn)
    
    title_bar.setLayout(layout)
    return title_bar

def create_video_dock(parent):
    """Create the video dock widget with custom video widget"""
    dock = QDockWidget("Video", parent)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
    
    # Set custom title bar
    dock.setTitleBarWidget(create_video_title_bar(dock))
    
    # Main container widget
    main_widget = QWidget()
    main_layout = QVBoxLayout()
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    # Create custom video widget (replaces QMediaPlayer approach)
    custom_video = CustomVideoWidget(parent)
    custom_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    main_layout.addWidget(custom_video, 4)
    
    # Store reference for later use
    parent.custom_video = custom_video

    # Controls container
    controls_widget = QWidget()
    controls_widget.setStyleSheet("""
        QWidget {
            background-color: #2b2b2b;
            border-top: 1px solid #555555;
        }
    """)
    controls_layout = QHBoxLayout()
    controls_layout.setContentsMargins(10, 8, 10, 8)
    controls_layout.setSpacing(15)

    # Play/Pause button
    parent.play_button = QPushButton("▶")
    parent.play_button.setFixedSize(40, 30)
    parent.play_button.setStyleSheet("""
        QPushButton {
            background-color: #404040;
            border: 1px solid #555555;
            color: white;
            padding: 6px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #505050;
        }
        QPushButton:pressed {
            background-color: #606060;
        }
    """)
    parent.play_button.clicked.connect(lambda: toggle_custom_video_playback(parent))
    controls_layout.addWidget(parent.play_button)

    # Time label
    parent.time_label = QLabel("00:00 / 00:00")
    parent.time_label.setFixedWidth(100)
    parent.time_label.setStyleSheet("""
        QLabel {
            color: white;
            font-weight: bold;
            font-size: 12px;
        }
    """)
    controls_layout.addWidget(parent.time_label)

    # Progress slider
    parent.progress_slider = QSlider(Qt.Horizontal)
    parent.progress_slider.setRange(0, 0)
    parent.progress_slider.setStyleSheet("""
        QSlider::groove:horizontal {
            border: 1px solid #555555;
            height: 6px;
            background: #404040;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #606060;
            border: 1px solid #555555;
            width: 16px;
            margin: -6px 0;
            border-radius: 8px;
        }
        QSlider::handle:horizontal:hover {
            background: #707070;
        }
        QSlider::handle:horizontal:pressed {
            background: #808080;
        }
        QSlider::sub-page:horizontal {
            background: #606060;
            border-radius: 3px;
        }
    """)
    parent.progress_slider.sliderMoved.connect(lambda position: seek_custom_video(parent, position))
    parent.progress_slider.sliderPressed.connect(lambda: pause_custom_video_for_drag(parent))
    parent.progress_slider.sliderReleased.connect(lambda: resume_custom_video_after_drag(parent))
    controls_layout.addWidget(parent.progress_slider, 1)

    # Volume label
    volume_label = QLabel("●")
    volume_label.setStyleSheet("""
        QLabel {
            color: white;
            font-size: 12px;
            font-weight: bold;
        }
    """)
    controls_layout.addWidget(volume_label)

    # Volume slider (simplified for custom video)
    parent.volume_slider = QSlider(Qt.Horizontal)
    parent.volume_slider.setRange(0, 100)
    parent.volume_slider.setValue(50)
    parent.volume_slider.setFixedWidth(80)
    parent.volume_slider.setStyleSheet("""
        QSlider::groove:horizontal {
            border: 1px solid #555555;
            height: 4px;
            background: #404040;
            border-radius: 2px;
        }
        QSlider::handle:horizontal {
            background: #606060;
            border: 1px solid #555555;
            width: 12px;
            margin: -5px 0;
            border-radius: 6px;
        }
        QSlider::handle:horizontal:hover {
            background: #707070;
        }
        QSlider::handle:horizontal:pressed {
            background: #808080;
        }
        QSlider::sub-page:horizontal {
            background: #606060;
            border-radius: 2px;
        }
    """)
    # Volume control is simplified for custom video widget
    parent.volume_slider.valueChanged.connect(lambda volume: print(f"Volume set to: {volume}%"))
    controls_layout.addWidget(parent.volume_slider)

    controls_widget.setLayout(controls_layout)
    main_layout.addWidget(controls_widget, 1)

    main_widget.setLayout(main_layout)
    dock.setWidget(main_widget)

    # Initialize bounding box state
    parent.show_bounding_boxes = True

    return dock

def toggle_custom_video_playback(parent):
    """Toggle play/pause for custom video widget"""
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        custom_video.toggle_playback()
        
        # Update button text
        if custom_video.is_playing:
            parent.play_button.setText("⏸")
        else:
            parent.play_button.setText("▶")

def seek_custom_video(parent, position):
    """Seek to specific frame in custom video"""
    if hasattr(parent, 'custom_video') and hasattr(parent, 'custom_video_total_frames'):
        custom_video = parent.custom_video
        total_frames = parent.custom_video_total_frames
        
        # Calculate frame and clamp to valid range
        frame = int((position / 100.0) * total_frames)
        frame = max(0, min(frame, total_frames - 1))  # Clamp to valid range
        
        custom_video.current_frame = frame
        custom_video.update()
        
        # Update time label
        fps = custom_video.fps if custom_video.fps > 0 else 30.0
        current_time = frame / fps
        total_time = total_frames / fps
        parent.time_label.setText(f"{int(current_time//60):02d}:{int(current_time%60):02d} / {int(total_time//60):02d}:{int(total_time%60):02d}")

def pause_custom_video_for_drag(parent):
    """Pause custom video when dragging slider"""
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        if custom_video.is_playing:
            custom_video.timer.stop()
            custom_video.is_playing = False
            parent.was_playing_before_drag = True
        else:
            parent.was_playing_before_drag = False

def resume_custom_video_after_drag(parent):
    """Resume custom video after dragging slider"""
    if hasattr(parent, 'custom_video') and hasattr(parent, 'was_playing_before_drag') and parent.was_playing_before_drag:
        custom_video = parent.custom_video
        custom_video.timer.start(33)  # ~30 FPS
        custom_video.is_playing = True
    parent.was_playing_before_drag = False

def load_video_for_custom_widget(parent, video_path):
    """Load a video file into the custom video widget"""
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        if custom_video.load_video(video_path):
            # Update progress slider range (0-100 for percentage)
            parent.progress_slider.setRange(0, 100)
            parent.custom_video_total_frames = custom_video.total_frames
            
            # Update time label
            total_time = custom_video.total_frames / custom_video.fps
            parent.time_label.setText(f"00:00 / {int(total_time//60):02d}:{int(total_time%60):02d}")
            
            print(f"✅ Video loaded: {custom_video.total_frames} frames at {custom_video.fps} FPS")
            return True
        else:
            print(f"❌ Failed to load video: {video_path}")
            return False
    return False

def set_current_video_path(parent, video_path):
    """Set the current video path for bounding box data loading"""
    parent.current_video_path = video_path
    print(f"Current video path set to: {video_path}")
    
    # Load video into custom widget
    load_video_for_custom_widget(parent, video_path)
    
    # Try to load detection data for this video
    if parent.show_bounding_boxes:
        load_detection_data_for_custom_video(parent)

def toggle_playback(parent):
    if parent.player.playbackState() == QMediaPlayer.PlayingState:
        parent.player.pause()
    else:
        parent.player.play()

def set_position(parent, position):
    parent.player.setPosition(position)

def set_volume(parent, volume):
    parent.audio_output.setVolume(volume / 100.0)

def update_position(parent, position):
    if not parent.progress_slider.isSliderDown():
        parent.progress_slider.setValue(position)
    
    # Update time label
    current_time = QTime(0, 0, 0, 0).addMSecs(position)
    duration = parent.player.duration()
    total_time = QTime(0, 0, 0, 0).addMSecs(duration) if duration > 0 else QTime(0, 0, 0, 0)
    
    current_format = "mm:ss" if duration < 3600000 else "hh:mm:ss"
    total_format = "mm:ss" if duration < 3600000 else "hh:mm:ss"
    
    parent.time_label.setText(f"{current_time.toString(current_format)} / {total_time.toString(total_format)}")

def update_duration(parent, duration):
    parent.progress_slider.setRange(0, duration)

def update_play_button(parent, state):
    if state == QMediaPlayer.PlayingState:
        parent.play_button.setText("⏸")
    else:
        parent.play_button.setText("▶")

def pause_for_drag(parent):
    """Pause video and remember if it was playing before drag"""
    parent.was_playing_before_drag = (parent.player.playbackState() == QMediaPlayer.PlayingState)
    parent.player.pause()

def resume_after_drag(parent):
    """Resume playback only if it was playing before drag started"""
    if hasattr(parent, 'was_playing_before_drag') and parent.was_playing_before_drag:
        parent.player.play()
    parent.was_playing_before_drag = False

def toggle_bounding_boxes(parent, button):
    """Toggle bounding box visibility on the custom video widget"""
    if not hasattr(parent, 'show_bounding_boxes'):
        parent.show_bounding_boxes = True
    
    parent.show_bounding_boxes = button.isChecked()
    
    # Update custom video widget
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        custom_video.set_show_boxes(parent.show_bounding_boxes)
        
        if parent.show_bounding_boxes:
            print("📦 Bounding boxes: ON")
            # Try to load detection data if available
            load_detection_data_for_custom_video(parent)
            
            # Test the display
            print("🔧 Testing custom video bounding box display...")
            custom_video.test_display_boxes()
            custom_video.force_update()
            print("🔧 Test complete!")
        else:
            print("📦 Bounding boxes: OFF")
    else:
        print("❌ No custom video widget found!")

def toggle_yard_marker_boxes(parent, button):
    """Toggle yard marker bounding box visibility on the custom video widget"""
    if not hasattr(parent, 'show_yard_marker_boxes'):
        parent.show_yard_marker_boxes = False
    
    parent.show_yard_marker_boxes = button.isChecked()
    
    # Update custom video widget
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        
        if parent.show_yard_marker_boxes:
            print("🏈 Yard marker boxes: ON")
            # Try to load yard marker detection data if available
            load_yard_marker_data_for_custom_video(parent)
            
            # Test the display
            print("🔧 Testing custom video yard marker display...")
            custom_video.test_display_boxes()
            custom_video.force_update()
            print("🔧 Test complete!")
        else:
            print("🏈 Yard marker boxes: OFF")
            # Clear yard marker data when disabled
            if hasattr(custom_video, 'yard_marker_data'):
                custom_video.yard_marker_data = None
                custom_video.update()
    else:
        print("❌ No custom video widget found!")

def find_video_widget(parent):
    """Find the BoundingBoxGraphicsView in the parent's dock widgets"""
    print(f"🔍 Searching for video widget in {len(parent.findChildren(QDockWidget))} dock widgets")
    
    for dock in parent.findChildren(QDockWidget):
        print(f"🔍 Checking dock: {dock.windowTitle()}")
        if dock.widget():
            # Look for BoundingBoxGraphicsView in the dock's widget hierarchy
            graphics_view = dock.widget().findChild(BoundingBoxGraphicsView)
            if graphics_view:
                print(f"🔍 Found BoundingBoxGraphicsView in dock: {dock.windowTitle()}")
                return graphics_view
            
            # Also look for SimpleOverlayWidget
            simple_overlay = dock.widget().findChild(SimpleOverlayWidget)
            if simple_overlay:
                print(f"🔍 Found SimpleOverlayWidget in dock: {dock.windowTitle()}")
                return simple_overlay
    
    print(f"🔍 No video widget found in dock widgets")
    return None

def load_detection_data_for_current_video(parent):
    """Load detection data for the currently playing video"""
    if not hasattr(parent, 'current_video_path') or not parent.current_video_path:
        print("No current video path available")
        return
    
    # Try to find detection JSON file
    video_name = os.path.splitext(os.path.basename(parent.current_video_path))[0]
    
    # Get the project root directory (two levels up from app/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detection_file = os.path.join(project_root, "cache",  parent.current_folder, "players", f"{video_name}_detection.json")
    
    print(f"Looking for detection file: {detection_file}")
    print(f"File exists: {os.path.exists(detection_file)}")
    
    if os.path.exists(detection_file):
        try:
            with open(detection_file, 'r') as f:
                detection_data = json.load(f)
            
            # Set detection data in video widget
            video_widget = find_video_widget(parent)
            if video_widget:
                video_widget.set_detection_data(detection_data)
                print(f"✅ Loaded detection data from: {detection_file}")
                print(f"   Total frames with detections: {len(detection_data.get('frames', []))}")
            else:
                print("❌ Could not find video widget to set detection data")
        except Exception as e:
            print(f"❌ Error loading detection data: {e}")
    else:
        print(f"❌ Detection file not found: {detection_file}")
        print("   Process the video first to generate detection data")

def load_detection_data_for_custom_video(parent):
    """Load detection data for custom video widget"""
    if not hasattr(parent, 'current_video_path') or not parent.current_video_path:
        print("❌ No current video path set")
        return
    
    # Construct detection file path
    video_path = parent.current_video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Get the project root directory (two levels up from app/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detection_file = os.path.join(project_root, "cache",  os.path.basename(parent.current_folder), "players", f"{video_name}_detection.json")
    
    print(f"🔍 Looking for detection file: {detection_file}")
    
    if os.path.exists(detection_file):
        try:
            with open(detection_file, 'r') as f:
                detection_data = json.load(f)
            
            print(f"✅ Loaded detection data: {len(detection_data.get('frames', []))} frames")
            
            # Set detection data in custom video widget
            if hasattr(parent, 'custom_video'):
                parent.custom_video.set_detection_data(detection_data)
                parent.custom_video.set_show_boxes(True)  # Enable boxes when data is available
                print("🎯 Detection data set in custom video widget")
            else:
                print("❌ No custom video widget found")
                
        except Exception as e:
            print(f"❌ Error loading detection data: {e}")
            # Clear detection data on error
            if hasattr(parent, 'custom_video'):
                parent.custom_video.set_detection_data(None)
                parent.custom_video.set_show_boxes(False)
    else:
        print(f"❌ Detection file not found: {detection_file}")
        print("   Process the video first to generate detection data")
        
        # Clear detection data when no file exists
        if hasattr(parent, 'custom_video'):
            parent.custom_video.set_detection_data(None)
            parent.custom_video.set_show_boxes(False)
            print("🚫 No detection data - bounding boxes disabled")

def load_yard_marker_data_for_custom_video(parent):
    """Load yard marker detection data for custom video widget"""
    if not hasattr(parent, 'current_video_path') or not parent.current_video_path:
        print("❌ No current video path set")
        return
    
    # Construct yard marker detection file path
    video_path = parent.current_video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Get the project root directory (two levels up from app/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yard_marker_file = os.path.join(project_root, "cache",  os.path.basename(parent.current_folder), "yard_markers", f"{video_name}_yard_markers.json")
    
    print(f"🔍 Looking for yard marker detection file: {yard_marker_file}")
    
    if os.path.exists(yard_marker_file):
        try:
            with open(yard_marker_file, 'r') as f:
                yard_marker_data = json.load(f)
            
            print(f"✅ Loaded yard marker detection data: {len(yard_marker_data.get('frames', []))} frames")
            
            # Set yard marker data in custom video widget
            if hasattr(parent, 'custom_video'):
                parent.custom_video.yard_marker_data = yard_marker_data
                print("🏈 Yard marker data set in custom video widget")
            else:
                print("❌ No custom video widget found")
                
        except Exception as e:
            print(f"❌ Error loading yard marker detection data: {e}")
            # Clear yard marker data on error
            if hasattr(parent, 'custom_video'):
                parent.custom_video.yard_marker_data = None
    else:
        print(f"❌ Yard marker detection file not found: {yard_marker_file}")
        print("   Process the video first to generate yard marker detection data")
        
        # Clear yard marker data when no file exists
        if hasattr(parent, 'custom_video'):
            parent.custom_video.yard_marker_data = None
            print("🚫 No yard marker data - yard marker boxes disabled")