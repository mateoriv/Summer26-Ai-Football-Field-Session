from PySide6.QtWidgets import (QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
                               QPushButton, QSlider, QLabel, QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsRectItem, QGraphicsTextItem)
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtCore import Qt, QTime, QTimer, QRectF, QObject, QEvent
from PySide6.QtGui import QPainter, QPen, QFont, QColor, QBrush, QImage, QPixmap

import json
import os
import cv2
import numpy as np
import pandas as pd
from fileAccess import get_cache_dir

# Define colors for specific position labels
POSITION_COLORS = {
    'qb': QColor(255, 255, 0),              # Yellow
    'oline': QColor(0, 0, 255),             # Blue
    'running_back': QColor(255, 165, 0),    # Orange
    'wide_receiver': QColor(0, 255, 255),   # Cyan
    'tight_end': QColor(255, 0, 255),       # Magenta 
    'defense': QColor(128, 128, 128),       # Gray 
    'player': QColor(255, 0, 0),            # Red (Generic Player Fallback)
    'yard_marker': QColor(0, 255, 127)      # Spring Green (Yard Marker)
}
YARD_MARKERS = ["f5", "fl1", "fl2", "fl3", "fl4", "fr1", "fr2", "fr3", "fr4", "n5", "nl1", "nl2", "nl3", "nl4", "nr1", "nr2", "nr3", "nr4"]
    


class SnapMarkerSlider(QWidget):
    """Custom slider widget that displays snap markers on the timeline"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.slider = QSlider(Qt.Horizontal, self)
        self.snap_frames = []  # List of snap frame numbers
        self.total_frames = 0
        
        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.slider)
        self.setLayout(layout)
        
        # Set up slider style
        self.slider.setStyleSheet("""
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
    
    def set_snap_frames(self, snap_frames, total_frames):
        """Set snap frames to display"""
        self.snap_frames = snap_frames
        self.total_frames = total_frames
        self.update()
    
    def paintEvent(self, event):
        """Override paint event to draw snap markers"""
        super().paintEvent(event)
        
        if not self.snap_frames or self.total_frames == 0:
            return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Get slider geometry
        slider_rect = self.slider.geometry()
        slider_width = slider_rect.width()
        slider_height = slider_rect.height()
        slider_y = slider_rect.y()
        slider_x = slider_rect.x()
        
        # Draw snap markers as white dashes (vertical lines extending above and below slider)
        dash_height = 8  # Height of dash above/below slider
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        for snap_frame in self.snap_frames:
            # Calculate x position based on frame number
            x = slider_x + int((snap_frame / self.total_frames) * slider_width)
            # Draw vertical dash extending above and below the slider
            # Top dash (above slider)
            painter.drawLine(x, slider_y - dash_height, x, slider_y)
            # Bottom dash (below slider)
            painter.drawLine(x, slider_y + slider_height, x, slider_y + slider_height + dash_height)


class CustomVideoWidget(QWidget):
    """
    Custom video widget that uses OpenCV to display frames and QPainter
    to draw overlay graphics (bounding boxes) directly on the video frame.
    
    The frame retrieval and drawing logic is unified to handle different types
    of detection data (players and yard markers) efficiently.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.position_detection_data = None 
        self.yard_marker_data = None
        self.current_frame = 0
        self.show_boxes = False
        self.show_yard_marker_boxes = False
        self.show_offense_selection = False
        self.offense_selection_frame = None
        self.offense_selection_points = []
        self.offense_selection_classes = []
        self.overlay_items = []
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.is_playing = False
        self.parent_window = parent
        
        # Timer for frame updates (~30 FPS playback)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        
    def set_detection_data(self, data):
        """
        Set the detection data for player bounding boxes. 
        """
        self.position_detection_data = data
        if data is None:
            print("Player detection data cleared - no bounding boxes will be shown")
        if data and 'frames' in data:
            print(f"   Total player detection frames: {len(data['frames'])}")
        self.update()
    
    def set_show_boxes(self, show):
        """Set whether to show player/position bounding boxes."""
        self.show_boxes = show
        self.update()
    
    def set_show_yard_marker_boxes(self, show):
        """Set whether to show yard marker bounding boxes."""
        self.show_yard_marker_boxes = show
        self.update()

    def set_offense_selection(self, frame_number, points, classes=None):
        """Store which frame and points correspond to the selected 11 offensive players."""
        self.offense_selection_frame = frame_number
        self.offense_selection_points = points or []
        self.offense_selection_classes = classes or []
        self.update()

    def set_show_offense_selection(self, show):
        """Toggle highlighting of the selected 11 offensive players at the snap frame."""
        self.show_offense_selection = show
        self.update()
    
    def update_frame(self):
        """Update current frame and redraw."""
        if self.cap and self.cap.isOpened() and self.is_playing:
            self.current_frame += 1
            if self.current_frame >= self.total_frames:
                self.current_frame = 0  # Loop back to start
            
            if self.parent_window:
                self.update_parent_controls()
                # Update virtual field with current frame
                self.update_virtual_field()
            
            self.update()
    
    def update_virtual_field(self):
        """Update the virtual field with current frame"""
        if not hasattr(self, 'parent_window') or not self.parent_window:
            return
        
        parent = self.parent_window
        if hasattr(parent, 'virtual_field'):
            from virtualField import update_virtual_field_with_video_frame
            update_virtual_field_with_video_frame(parent, self.current_frame)
    
    def update_parent_controls(self):
        """Update parent's progress slider and time label."""
        if not hasattr(self, 'parent_window') or not self.parent_window:
            return
            
        parent = self.parent_window
        
        # Update progress slider
        if hasattr(parent, 'progress_slider') and hasattr(parent, 'custom_video_total_frames'):
            progress = int((self.current_frame / parent.custom_video_total_frames) * 100)
            parent.progress_slider.slider.setValue(progress)
        
        # Update time label
        if hasattr(parent, 'time_label'):
            fps = self.fps if self.fps > 0 else 30.0
            current_time = self.current_frame / fps
            total_time = self.total_frames / fps
            parent.time_label.setText(f"{int(current_time//60):02d}:{int(current_time%60):02d} / {int(total_time//60):02d}:{int(total_time%60):02d}")
    
    def paintEvent(self, event):
        """Custom paint event to draw video frame and all enabled bounding boxes."""
        painter = QPainter(self)
        
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

                # Define data sources to draw
                video_data_sources = []
                if self.show_boxes and self.position_detection_data:
                    video_data_sources.append(self.position_detection_data)
                
                if self.show_yard_marker_boxes and self.yard_marker_data:
                    video_data_sources.append(self.yard_marker_data)

                for data in video_data_sources:
                    current_detections = self._get_detections_for_frame(self.current_frame, data)
                    
                    if current_detections:
                        # Determine resolution for scaling
                        video_width = 1280
                        video_height = 720
                        if 'video_info' in data:
                            video_info = data['video_info']
                            video_width = video_info.get('width', video_width)
                            video_height = video_info.get('height', video_height)
                        
                        # Calculate scaling factors
                        scale_x = scaled_pixmap.width() / video_width
                        scale_y = scaled_pixmap.height() / video_height
                        
                        for detection in current_detections:
                            self._draw_single_bbox(painter, detection, scale_x, scale_y, x_offset, y_offset)
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

    
    def _get_detections_for_frame(self, frame_number, data_source):
        """
        [UNIFIED] Get detections for a specific frame number from a given data source (internal helper).
        
        data_source: self.position_detection_data or self.yard_marker_data
        """
        if not data_source or 'frames' not in data_source:
            return []
        
        # Find the closest frame in the data
        closest_frame = None
        min_diff = float('inf')
        
        for frame_data in data_source['frames']:
            frame_idx = frame_data.get('frame_number', 0)
            diff = abs(frame_idx - frame_number)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame_data
        
        # In offense selection mode the positions file only has the snap frame,
        # so show that frame's detections everywhere in the video.
        tolerance = 100000 if getattr(self, 'show_offense_selection', False) else 15
        if closest_frame and min_diff < tolerance:
            return closest_frame.get('detections', [])

        return []
    
    def _draw_single_bbox(self, painter, detection, scale_x, scale_y, x_offset=0, y_offset=0):
        """
        Draw a single bounding box with dynamic styling based on class (internal helper).
        Uses POSITION_COLORS for dynamic coloring.
        """
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
        
        # Define colors based on class name using the mapping
        class_name = detection.get('class', 'player')
        if class_name in YARD_MARKERS:
            class_name = "yard_marker"

        # Default box color
        box_color = POSITION_COLORS.get(class_name.lower(), POSITION_COLORS['player'])

        # When offense selection mode is on, color boxes by team role
        if getattr(self, "show_offense_selection", False) and class_name != 'yard_marker':
            if class_name == 'defense':
                box_color = POSITION_COLORS['defense']             # gray
            elif class_name == 'qb':
                box_color = POSITION_COLORS['qb']                  # yellow
            elif class_name == 'running_back':
                box_color = POSITION_COLORS['running_back']        # orange
            elif class_name in ('wide_receiver', 'tight_end'):
                box_color = POSITION_COLORS['wide_receiver']       # cyan
            else:
                box_color = POSITION_COLORS['oline']               # blue for other offense
       
        # Draw bounding box
        painter.setPen(QPen(box_color, 3))
        # Use a semi-transparent brush
        painter.setBrush(QBrush(QColor(box_color.red(), box_color.green(), box_color.blue(), 50))) 
        painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
        
        # # Draw label if yard marker
        if class_name == "yard_marker":
            class_name = detection.get('class', 'player')
            confidence = detection.get('confidence', 0.0)
            
            # Use contrasting text color (e.g., black or white based on background)
            # Using white text for general visibility against darker video content
            painter.setPen(QPen(QColor(255, 255, 255), 1))
            
            # Draw a semi-transparent background rectangle for the text label
            text_label = f"{class_name} {confidence:.2f}"
            font_metrics = painter.fontMetrics()
            text_width = font_metrics.horizontalAdvance(text_label)
            text_height = font_metrics.height()
            
            # Draw background rectangle for the text
            text_bg_rect = QRectF(scaled_x, scaled_y - text_height - 5, text_width + 6, text_height + 2)
            painter.fillRect(text_bg_rect, QBrush(QColor(0, 0, 0, 150))) # Dark semi-transparent background
            
            # Draw the text on top
            painter.drawText(scaled_x + 3, scaled_y - 8, text_label)
        
    def force_update(self):
        self.update()
    
    def toggle_playback(self):
        """Toggle play/pause for custom video widget"""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
        else:
            # Calculate timer interval based on actual video FPS
            fps = self.fps if self.fps > 0 else 30.0
            interval_ms = int(1000.0 / fps)  # Convert FPS to milliseconds
            self.timer.start(interval_ms)
            self.is_playing = True
    
    def load_video(self, video_path):
        """Load a video file into the custom video widget"""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            print(f"Could not open video: {video_path}")
            return False
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame = 0
        
        print(f"Loaded video: {self.total_frames} frames at {self.fps} FPS")
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
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_frame_from_video)
        self.update_timer.start(50)
        
    
    def set_detection_data(self, data):
        self.detection_data = data
        self.update()
    
    def set_show_boxes(self, show):
        self.show_boxes = show
        if show:
            self.update_overlay()
        else:
            self.clear_overlay()
    
    def update_overlay(self):
        if not self.show_boxes or not self.detection_data:
            return
        
        self.clear_overlay()
        current_detections = self.get_detections_for_frame(self.current_frame)
        
        if current_detections:
            for detection in current_detections:
                self.draw_bounding_box_overlay(detection)
    
    def get_detections_for_frame(self, frame_number):
        if not self.detection_data or 'frames' not in self.detection_data:
            return []
        
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
        from PySide6.QtWidgets import QLabel
        
        bbox = detection.get('bbox', {})
        if not bbox or 'x1' not in bbox:
            return
        
        x1 = bbox.get('x1', 0)
        y1 = bbox.get('y1', 0)
        x2 = bbox.get('x2', 0)
        y2 = bbox.get('y2', 0)
        
        w = x2 - x1
        h = y2 - y1
        
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'player')
        widget_rect = self.rect()
        
        video_width = 1280
        video_height = 720
        
        if self.detection_data and 'video_info' in self.detection_data:
            video_info = self.detection_data['video_info']
            if 'width' in video_info and 'height' in video_info:
                video_width = video_info['width']
                video_height = video_info['height']
            else:
                max_x = max_y = 0
                for frame in self.detection_data.get('frames', []):
                    for det in frame.get('detections', []):
                        bbox = det.get('bbox', {})
                        max_x = max(max_x, bbox.get('x2', 0))
                        max_y = max(max_y, bbox.get('y2', 0))
                
                if max_x > 1280:
                    video_width = 1920
                    video_height = 1080
        
        scale_x = widget_rect.width() / video_width
        scale_y = widget_rect.height() / video_height
        
        scaled_x = int(x1 * scale_x)
        scaled_y = int(y1 * scale_y)
        scaled_w = int(w * scale_x)
        scaled_h = int(h * scale_y)
        
        box_color = POSITION_COLORS.get(class_name.lower(), POSITION_COLORS['player'])
        
        bbox_label = QLabel(f"{class_name}\n{confidence:.2f}", self)
        bbox_label.setStyleSheet(f"""
            background-color: rgba({box_color.red()}, {box_color.green()}, {box_color.blue()}, 100);
            border: 2px solid rgb({box_color.red()}, {box_color.green()}, {box_color.blue()});
            color: white;
            font-weight: bold;
            font-size: 10px;
        """)
        bbox_label.setGeometry(scaled_x, scaled_y, scaled_w, scaled_h)
        bbox_label.show()
        bbox_label.raise_()
        
        self.overlay_items.append(bbox_label)
    
    def clear_overlay(self):
        for item in self.overlay_items:
            item.deleteLater()
        self.overlay_items.clear()
    
    def update_frame_position(self, frame_number):
        self.current_frame = frame_number
        if self.show_boxes:
            self.update_overlay()
    
    def update_frame_from_video(self):
        try:
            parent_widget = self.parent()
            while parent_widget and not hasattr(parent_widget, 'player'):
                parent_widget = parent_widget.parent()
            
            if parent_widget and hasattr(parent_widget, 'player'):
                player = parent_widget.player
                if player and player.position() > 0:
                    fps = 30
                    position_ms = player.position()
                    frame_number = int((position_ms / 1000.0) * fps)
                    
                    if frame_number != self.current_frame:
                        self.current_frame = frame_number
                        if self.show_boxes and self.detection_data:
                            self.update_overlay()
        except Exception:
            pass
    
    def force_update(self):
        if self.show_boxes and self.detection_data:
            self.update_overlay()
        self.update()


class BoundingBoxGraphicsView(QGraphicsView):
    """Graphics view that overlays bounding boxes on top of video"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.detection_data = None
        self.current_frame = 0
        self.show_boxes = True
        self.bounding_box_items = []
        
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, 1280, 720)
        self.setScene(self.scene)
        
        self.setRenderHint(QPainter.Antialiasing)
        self.setStyleSheet("background: transparent;")
        
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_frame_position)
        self.update_timer.start(50)
    
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.scene.setSceneRect(0, 0, self.width(), self.height())
    
    def set_detection_data(self, data):
        self.detection_data = data
                    
        if data and 'video_info' in data:
            video_info = data['video_info']
            if 'width' in video_info and 'height' in video_info:
                video_width = video_info['width']
                video_height = video_info['height']
                self.scene.setSceneRect(0, 0, video_width, video_height)
                    
        self.update()
    
    def update_frame_position(self):
        if self.parent_window and hasattr(self.parent_window, 'player'):
            position_ms = self.parent_window.player.position()
            fps = 30
            new_frame = int((position_ms / 1000.0) * fps)
            
            if new_frame != self.current_frame:
                self.current_frame = new_frame
                if self.show_boxes and self.detection_data:
                    self.update_bounding_boxes()
    
    def update_bounding_boxes(self):
        for item in self.bounding_box_items:
            self.scene.removeItem(item)
        self.bounding_box_items.clear()
        
        if not self.show_boxes or not self.detection_data:
            return
        
        current_detections = self.get_detections_for_frame(self.current_frame)
        
        if current_detections:
            for detection in current_detections:
                self.draw_bounding_box_graphics(detection)
        else:
            print(f"No detections for frame {self.current_frame}")
    
    def get_detections_for_frame(self, frame_number):
        if not self.detection_data or 'frames' not in self.detection_data:
            return []
        
        closest_frame = None
        min_diff = float('inf')
        
        for frame_data in self.detection_data['frames']:
            frame_idx = frame_data.get('frame_number', 0)
            diff = abs(frame_idx - frame_number)
            if diff < min_diff:
                min_diff = diff
                closest_frame = frame_data
        
        if closest_frame and min_diff < 15:
            detections = closest_frame.get('detections', [])
            return detections
        
        return []
    
    def draw_bounding_box_graphics(self, detection):
        bbox = detection.get('bbox', {})
        if not bbox or 'x1' not in bbox:
            return
        
        x1 = bbox.get('x1', 0)
        y1 = bbox.get('y1', 0)
        x2 = bbox.get('x2', 0)
        y2 = bbox.get('y2', 0)
        
        w = x2 - x1
        h = y2 - y1
        
        confidence = detection.get('confidence', 0.0)
        class_name = detection.get('class', 'player')
        track_id = detection.get('track_id', '')
        
        widget_rect = self.rect()
        
        video_width = 1280
        video_height = 720
        
        if self.detection_data and 'video_info' in self.detection_data:
            video_info = self.detection_data['video_info']
            if 'width' in video_info and 'height' in video_info:
                video_width = video_info['width']
                video_height = video_info['height']
            else:
                max_x = max_y = 0
                for frame in self.detection_data.get('frames', []):
                    for det in frame.get('detections', []):
                        bbox = det.get('bbox', {})
                        max_x = max(max_x, bbox.get('x2', 0))
                        max_y = max(max_y, bbox.get('y2', 0))
                
                if max_x > 1280:
                    video_width = 1920
                    video_height = 1080
        
        scale_x = widget_rect.width() / video_width
        scale_y = widget_rect.height() / video_height
        
        scaled_x = int(x1 * scale_x)
        scaled_y = int(y1 * scale_y)
        scaled_w = int(w * scale_x)
        scaled_h = int(h * scale_y)
        
        box_color = POSITION_COLORS.get(class_name.lower(), POSITION_COLORS['player'])
        
        # Create bounding box rectangle
        bbox_rect = QGraphicsRectItem(scaled_x, scaled_y, scaled_w, scaled_h)
        bbox_rect.setPen(QPen(box_color, 4))
        bbox_rect.setBrush(QBrush(QColor(box_color.red(), box_color.green(), box_color.blue(), 30)))
        
        self.scene.addItem(bbox_rect)
        self.bounding_box_items.append(bbox_rect)
        
        # Create white outline
        outline_rect = QGraphicsRectItem(scaled_x, scaled_y, scaled_w, scaled_h)
        outline_rect.setPen(QPen(QColor(255, 255, 255), 2))
        outline_rect.setBrush(QBrush(Qt.NoBrush))
        self.scene.addItem(outline_rect)
        self.bounding_box_items.append(outline_rect)
        
        # Create label text
        label_text = f"{class_name}"
        if track_id:
            label_text += f" ID:{track_id}"
        label_text += f" ({confidence:.2f})"
        
        text_item = QGraphicsTextItem(label_text)
        text_item.setPos(scaled_x, scaled_y - 25)
        text_item.setDefaultTextColor(QColor(255, 255, 255))  # White text
        font = QFont("Arial", 10, QFont.Bold)
        text_item.setFont(font)
        self.scene.addItem(text_item)
        self.bounding_box_items.append(text_item)
    
    def set_show_boxes(self, show):
        self.show_boxes = show
        print(f"Bounding boxes {'ON' if show else 'OFF'}")
        if show and self.detection_data:
            self.update_bounding_boxes()
        elif not show:
            for item in self.bounding_box_items:
                self.scene.removeItem(item)
            self.bounding_box_items.clear()
    
    def force_update(self):
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
    
    left_spacer = QWidget()
    left_spacer.setFixedWidth(20)
    layout.addWidget(left_spacer)
    
    title_label = QLabel("Video")
    title_label.setFont(QFont("Arial", 10, QFont.Bold))
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)
    
    right_spacer = QWidget()
    right_spacer.setFixedWidth(20)
    layout.addWidget(right_spacer)
    
    # Player bounding boxes checkbox
    bbox_checkbox = QPushButton("Players")
    bbox_checkbox.setFixedSize(50, 20)
    bbox_checkbox.setCheckable(True)
    bbox_checkbox.setChecked(False)
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
    yard_marker_checkbox = QPushButton("Yard")
    yard_marker_checkbox.setFixedSize(50, 20)
    yard_marker_checkbox.setCheckable(True)
    yard_marker_checkbox.setChecked(False)
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
    yard_marker_checkbox.clicked.connect(lambda: toggle_bounding_boxes(dock.parent(), yard_marker_checkbox))
    layout.addWidget(yard_marker_checkbox)

    # Offense selection highlight toggle
    offense_button = QPushButton("Static Offense")
    offense_button.setFixedSize(90, 20)
    offense_button.setCheckable(True)
    offense_button.setChecked(False)
    offense_button.setToolTip("Highlight the 11 offensive players used for training at the snap frame")
    offense_button.setStyleSheet("""
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
            background-color: #ffa500;
        }
    """)
    offense_button.clicked.connect(lambda: toggle_offense_selection(dock.parent(), offense_button))
    layout.addWidget(offense_button)

    # Legend toggle button
    legend_checkbox = QPushButton("Legend")
    legend_checkbox.setFixedSize(50, 20)
    legend_checkbox.setCheckable(True)
    legend_checkbox.setChecked(False)
    legend_checkbox.setToolTip("Toggle Color to Position Legend ")
    legend_checkbox.setStyleSheet("""
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
            background-color: #8A2BE2;
        }
    """)
    legend_checkbox.clicked.connect(lambda: toggle_legend(dock.parent(), legend_checkbox))
    layout.addWidget(legend_checkbox)


    title_bar.setLayout(layout)
    return title_bar

def create_video_dock(parent):
    """Create the video dock widget with custom video widget"""
    dock = QDockWidget("Video", parent)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(QDockWidget.DockWidgetMovable)
    
    dock.setTitleBarWidget(create_video_title_bar(dock))
    
    main_widget = QWidget()
    main_layout = QVBoxLayout()
    main_layout.setContentsMargins(0, 0, 0, 0)
    main_layout.setSpacing(0)

    custom_video = CustomVideoWidget(parent)
    custom_video.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    main_layout.addWidget(custom_video, 4)
    
    parent.custom_video = custom_video

    controls_widget = QWidget()
    controls_widget.setFixedHeight(50)  # Fixed height for controls bar
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

    # Progress slider with snap markers
    parent.progress_slider = SnapMarkerSlider(parent)
    parent.progress_slider.slider.setRange(0, 0)
    parent.progress_slider.slider.sliderMoved.connect(lambda position: seek_custom_video(parent, position))
    parent.progress_slider.slider.sliderPressed.connect(lambda: pause_custom_video_for_drag(parent))
    parent.progress_slider.slider.sliderReleased.connect(lambda: resume_custom_video_after_drag(parent))
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
    parent.volume_slider.valueChanged.connect(lambda volume: print(f"Volume set to: {volume}%"))
    controls_layout.addWidget(parent.volume_slider)

    controls_widget.setLayout(controls_layout)
    main_layout.addWidget(controls_widget)  # No stretch factor - fixed height

    main_widget.setLayout(main_layout)
    dock.setWidget(main_widget)

    parent.show_bounding_boxes = False
    parent.show_yard_marker_boxes = False
    
    # Sync the video widget's internal state with the parent's button states
    custom_video.set_show_boxes(False)
    custom_video.set_show_yard_marker_boxes(False)

    return dock

def toggle_custom_video_playback(parent):
    """Toggle play/pause for custom video widget"""
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        custom_video.toggle_playback()
        
        if custom_video.is_playing:
            parent.play_button.setText("⏸")
        else:
            parent.play_button.setText("▶")

def seek_custom_video(parent, position):
    """Seek to specific frame in custom video"""
    if hasattr(parent, 'custom_video') and hasattr(parent, 'custom_video_total_frames'):
        custom_video = parent.custom_video
        total_frames = parent.custom_video_total_frames
        
        frame = int((position / 100.0) * total_frames)
        frame = max(0, min(frame, total_frames - 1))
        
        custom_video.current_frame = frame
        custom_video.update()
        
        # Update virtual field with current frame
        custom_video.update_virtual_field()
        
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
        # Calculate timer interval based on actual video FPS
        fps = custom_video.fps if custom_video.fps > 0 else 30.0
        interval_ms = int(1000.0 / fps)  # Convert FPS to milliseconds
        custom_video.timer.start(interval_ms)
        custom_video.is_playing = True
    parent.was_playing_before_drag = False

def load_video_for_custom_widget(parent, video_path):
    """Load a video file into the custom video widget"""
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        if custom_video.load_video(video_path):
            parent.progress_slider.slider.setRange(0, 100)
            parent.progress_slider.slider.setValue(0)  # Reset slider to zero when switching videos
            
            parent.custom_video_total_frames = custom_video.total_frames
            # Load snap detection data if available
            load_snap_detection_data(parent, video_path)
            
            total_time = custom_video.total_frames / custom_video.fps
            parent.time_label.setText(f"00:00 / {int(total_time//60):02d}:{int(total_time%60):02d}")
            
            print(f"Video loaded: {custom_video.total_frames} frames at {custom_video.fps} FPS")
            return True
        else:
            print(f"Failed to load video: {video_path}")
            return False
    return False

def load_snap_detection_data(parent, video_path):
    """Load snap detection data and display markers on timeline"""
    if not hasattr(parent, 'current_folder') or not hasattr(parent, 'progress_slider'):
        return
    
    try:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        current_folder_name = os.path.basename(parent.current_folder)
        # Use shared cache directory function
        base_cache_dir = get_cache_dir()
        snap_file = os.path.join(base_cache_dir, current_folder_name, "snap_detection", f"{video_name}_snap_detection.json")
        
        if os.path.exists(snap_file):
            with open(snap_file, 'r') as f:
                snap_data = json.load(f)
            
            snaps = snap_data.get('snaps', [])
            snap_frames = [snap['frame'] for snap in snaps]
            
            # Get total frames from video
            if hasattr(parent, 'custom_video_total_frames'):
                total_frames = parent.custom_video_total_frames
                parent.progress_slider.set_snap_frames(snap_frames, total_frames)
                print(f"Loaded {len(snap_frames)} snap markers")
        else:
            # Clear snap markers if file doesn't exist
            parent.progress_slider.set_snap_frames([], 0)
    except Exception as e:
        print(f"Error loading snap detection data: {e}")

def set_current_video_path(parent, video_path):
    """Set the current video path for bounding box data loading"""
    parent.current_video_path = video_path
    
    load_video_for_custom_widget(parent, video_path)
    
    load_and_set_detection_data(parent, "players") 
    load_and_set_detection_data(parent, "yard_markers")
    
    # Load snap detection data for timeline markers
    load_snap_detection_data(parent, video_path)
    
    # Load homography data and offense positions for virtual field
    if hasattr(parent, 'current_folder') and hasattr(parent, 'virtual_field'):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        from virtualField import load_homography_data_for_virtual_field, load_offense_positions_for_virtual_field
        homography_loaded = load_homography_data_for_virtual_field(parent, video_name, parent.current_folder)
        # If homography data not found, ensure virtual field is cleared
        if not homography_loaded:
            if hasattr(parent, 'virtual_field'):
                parent.virtual_field.current_frame = 0
                parent.virtual_field.update()
        # Load offense positions (clears to None if not available for this clip)
        load_offense_positions_for_virtual_field(parent, video_name, parent.current_folder)
    
    # Sync the video widget's internal state with the parent's button states
    if hasattr(parent, 'custom_video'):
        parent.custom_video.set_show_boxes(parent.show_bounding_boxes)
        parent.custom_video.set_show_yard_marker_boxes(parent.show_yard_marker_boxes)
        # If offense selection highlight is enabled, recompute it for the new video
        if getattr(parent.custom_video, "show_offense_selection", False):
            _compute_offense_selection_for_current_video(parent)


def _compute_offense_selection_for_current_video(parent):
    """
    Load the 11 offensive player points for the current video from the
    precomputed offense_positions.csv built by CNN/build_offense_positions_dataset.py.
    """
    try:
        if not hasattr(parent, "current_video_path") or not parent.current_video_path:
            print("[OFFENSE SELECTION] No current video path set.")
            return False
        if not hasattr(parent, "current_folder") or not parent.current_folder:
            print("[OFFENSE SELECTION] No current folder set.")
            return False

        video_name = os.path.splitext(os.path.basename(parent.current_video_path))[0]
        folder_name = os.path.basename(parent.current_folder.rstrip("/\\"))
        base_cache_dir = get_cache_dir()

        offense_csv_path = os.path.join(base_cache_dir, folder_name, "offense_positions.csv")
        if not os.path.exists(offense_csv_path):
            print(f"[OFFENSE SELECTION] offense_positions.csv not found at {offense_csv_path}")
            return False

        df = pd.read_csv(offense_csv_path)
        if "clip_name" not in df.columns:
            print("[OFFENSE SELECTION] 'clip_name' column missing in offense_positions.csv")
            return False

        # Find the row for this video
        row = df.loc[df["clip_name"] == video_name]
        if row.empty:
            print(f"[OFFENSE SELECTION] No offense row found for clip_name '{video_name}'")
            return False

        row = row.iloc[0]
        points = []
        for i in range(1, 12):
            x_col = f"ox{i}"
            y_col = f"oy{i}"
            if x_col in row and y_col in row:
                x_val = row[x_col]
                y_val = row[y_col]
                if pd.notna(x_val) and pd.notna(y_val):
                    points.append((float(x_val), float(y_val)))

        if len(points) < 1:
            print(f"[OFFENSE SELECTION] No valid offense points for '{video_name}' in offense_positions.csv")
            return False

        if hasattr(parent, "custom_video"):
            classes = _get_offense_point_classes(video_name, folder_name, base_cache_dir, points)
            parent.custom_video.set_offense_selection(0, points, classes)
            print(f"[OFFENSE SELECTION] Loaded offense selection for {video_name} from offense_positions.csv")
            return True

        return False
    except Exception as e:
        print(f"[OFFENSE SELECTION] Error loading offense selection from CSV: {e}")
        return False


def _get_offense_point_classes(video_name, folder_name, base_cache_dir, offense_points):
    """Return a list of class names (e.g. 'qb', 'defense', 'player') for each offense point.

    Matches each (ox, oy) pixel coordinate to the nearest detection at the snap frame
    using the positions JSON produced by the position detection pipeline.
    """
    try:
        snap_file = os.path.join(base_cache_dir, folder_name, "snap_detection",
                                 f"{video_name}_snap_detection.json")
        if not os.path.exists(snap_file):
            return []
        with open(snap_file, 'r') as f:
            snap_data = json.load(f)
        snaps = snap_data.get('snaps', [])
        if not snaps:
            return []
        snap_frame = snaps[0].get('frame')

        positions_file = os.path.join(base_cache_dir, folder_name, "positions",
                                      f"{video_name}_position.json")
        if not os.path.exists(positions_file):
            return []
        with open(positions_file, 'r') as f:
            positions_data = json.load(f)

        snap_detections = []
        for fr in positions_data.get('frames', []):
            if fr.get('frame_number') == snap_frame:
                snap_detections = fr.get('detections', [])
                break

        if not snap_detections:
            return []

        classes = []
        for (ox, oy) in offense_points:
            best_class = 'player'
            best_dist = float('inf')
            for det in snap_detections:
                bbox = det.get('bbox', {})
                cx = bbox.get('center_x')
                cy = bbox.get('center_y')
                if cx is None or cy is None:
                    continue
                dist = (ox - cx) ** 2 + (oy - cy) ** 2
                if dist < best_dist:
                    best_dist = dist
                    best_class = det.get('class', 'player').lower()
            classes.append(best_class)

        return classes
    except Exception as e:
        print(f"[OFFENSE SELECTION] Error getting point classes: {e}")
        return []


def toggle_bounding_boxes(parent, button):
    """Toggle bounding box visibility on the custom video widget"""
    # Determine box type from button text 
    button_text = button.text().lower()
    if "player" in button_text:
        box_type = "players" 
    elif "yard" in button_text:
        box_type = "yard_markers"
    else:
        print(f"Unknown button type: {button_text}")
        return
    
    # Set up state variables based on box type
    if box_type == "players":
        state_attr = "show_bounding_boxes"
        display_name = "Player Bounding boxes"
        data_type = "players"
        setter_method = "set_show_boxes"
    elif box_type == "yard_markers":
        state_attr = "show_yard_marker_boxes"
        display_name = "Yard marker boxes"
        data_type = "yard_markers"
        setter_method = "set_show_yard_marker_boxes"
    else:
        print(f"Unknown box type: {box_type}")
        return
    
    # Initialize state if it doesn't exist
    if not hasattr(parent, state_attr):
        setattr(parent, state_attr, False)
    
    # Update state from button
    setattr(parent, state_attr, button.isChecked())
    is_enabled = getattr(parent, state_attr)
    
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        
        # Set the widget's internal state
        getattr(custom_video, setter_method)(is_enabled)
        
        if is_enabled:
            print(f"{display_name}: ON")
            load_and_set_detection_data(parent, data_type)
            if box_type == "players":
                custom_video.force_update()
        else:
            print(f"{display_name}: OFF")
    else:
        print("No custom video widget found!")


def _ensure_qb_labeled(positions_data):
    """QB fallback: when no QB was detected, the deepest backfield player gets labeled QB."""
    for frame_data in positions_data.get('frames', []):
        detections = frame_data.get('detections', [])

        off_dets = [d for d in detections
                    if d.get('class', '').lower() not in ('defense', 'ref', 'yard_marker')
                    and 'bbox' in d
                    and d['bbox'].get('center_x') is not None
                    and d['bbox'].get('center_y') is not None]
        def_xs = [d['bbox']['center_x'] for d in detections
                  if d.get('class', '').lower() == 'defense'
                  and d.get('bbox', {}).get('center_x') is not None]

        if len(off_dets) < 3:
            continue

        if any(d.get('class', '').lower() == 'qb' for d in detections):
            continue

        off_xs = [d['bbox']['center_x'] for d in off_dets]
        off_ys = [d['bbox']['center_y'] for d in off_dets]

        if def_xs:
            def_med_x = sorted(def_xs)[len(def_xs) // 2]
            left_count = sum(1 for x in off_xs if x < def_med_x)
            offense_side = "left" if left_count >= len(off_xs) / 2 else "right"
        else:
            offense_side = "left"

        # Exclude split-wide outliers (large Y deviation) from QB candidacy
        mean_y = sum(off_ys) / len(off_ys)
        std_y = (sum((y - mean_y) ** 2 for y in off_ys) / len(off_ys)) ** 0.5
        interior = ([d for d in off_dets if std_y == 0 or abs(d['bbox']['center_y'] - mean_y) <= 1.5 * std_y]
                    or off_dets)

        qb_det = (min(interior, key=lambda d: d['bbox']['center_x']) if offense_side == "left"
                  else max(interior, key=lambda d: d['bbox']['center_x']))
        qb_det['class'] = 'qb'
        print(f"[OFFENSE SELECTION] QB fallback: assigned qb to player at "
              f"({qb_det['bbox']['center_x']:.0f}, {qb_det['bbox']['center_y']:.0f})")


def _label_running_backs(positions_data):
    """Label backfield players near the QB as running_back (orange).

    Operates after _ensure_qb_labeled so QB is always present.
    Candidates must be interior (not split-wide) and within 200 px of the QB.
    Capped at 2 RBs per frame.
    """
    RB_MAX_DIST_PX = 200
    MAX_RB = 1
    WR_CLASSES = {'wide_receiver', 'tight_end'}

    for frame_data in positions_data.get('frames', []):
        dets = frame_data.get('detections', [])

        qb = next((d for d in dets
                   if d.get('class', '').lower() == 'qb'
                   and d.get('bbox', {}).get('center_x') is not None), None)
        if qb is None:
            continue

        off_dets = [d for d in dets
                    if d.get('class', '').lower() not in ('defense', 'ref', 'yard_marker', *WR_CLASSES)
                    and 'bbox' in d
                    and d['bbox'].get('center_x') is not None
                    and d['bbox'].get('center_y') is not None]

        if len(off_dets) < 3:
            continue

        # Exclude split-wide outliers from RB candidacy
        off_ys = [d['bbox']['center_y'] for d in off_dets]
        mean_y = sum(off_ys) / len(off_ys)
        std_y = (sum((y - mean_y) ** 2 for y in off_ys) / len(off_ys)) ** 0.5
        interior = [d for d in off_dets
                    if d is not qb
                    and (std_y == 0 or abs(d['bbox']['center_y'] - mean_y) <= 1.5 * std_y)]

        qb_cx = qb['bbox']['center_x']
        qb_cy = qb['bbox']['center_y']
        threshold_sq = RB_MAX_DIST_PX ** 2

        candidates = sorted(
            [d for d in interior
             if (d['bbox']['center_x'] - qb_cx) ** 2 + (d['bbox']['center_y'] - qb_cy) ** 2 <= threshold_sq],
            key=lambda d: (d['bbox']['center_x'] - qb_cx) ** 2 + (d['bbox']['center_y'] - qb_cy) ** 2
        )

        for d in candidates[:MAX_RB]:
            d['class'] = 'running_back'
            print(f"[RB] Labeled running_back at "
                  f"({d['bbox']['center_x']:.0f}, {d['bbox']['center_y']:.0f})")


def _resolve_qb_rb_by_height(positions_data):
    """Swap QB/RB labels if the RB has a taller bbox than the QB.

    The QB is typically more upright at the snap; the RB tends to crouch lower.
    A 10% tolerance prevents swapping when heights are nearly equal.
    """
    HEIGHT_TOLERANCE = 0.10

    for frame_data in positions_data.get('frames', []):
        dets = frame_data.get('detections', [])

        qb = next((d for d in dets if d.get('class', '').lower() == 'qb'), None)
        rb = next((d for d in dets if d.get('class', '').lower() == 'running_back'), None)

        if qb is None or rb is None:
            continue

        qb_h = qb.get('bbox', {}).get('height', 0)
        rb_h = rb.get('bbox', {}).get('height', 0)

        if qb_h <= 0 or rb_h <= 0:
            continue

        if rb_h > qb_h * (1 + HEIGHT_TOLERANCE):
            qb['class'] = 'running_back'
            rb['class'] = 'qb'
            print(f"[QB/RB] Swapped — RB bbox taller ({rb_h:.0f}px) than QB ({qb_h:.0f}px)")


def _load_position_data_for_offense_mode(parent):
    """Swap the video widget's detection data to the position-labeled JSON.

    positionDetection.pt outputs qb/defense/oline/etc. classes; this replaces
    the generic player-detection data so bounding boxes show team colors.
    Runs _ensure_qb_labeled so the backfield player is always highlighted yellow
    even when the model misses the QB classification.
    Also enables box visibility so the user sees the result immediately.
    """
    if not hasattr(parent, 'current_video_path') or not parent.current_video_path:
        return
    video_name = os.path.splitext(os.path.basename(parent.current_video_path))[0]
    folder_name = os.path.basename(getattr(parent, 'current_folder', '').rstrip('/\\'))
    base_cache_dir = get_cache_dir()
    positions_file = os.path.join(base_cache_dir, folder_name, "positions",
                                  f"{video_name}_position.json")
    if not os.path.exists(positions_file):
        print(f"[OFFENSE SELECTION] Positions file not found: {positions_file}")
        return
    PLAYER_POS_MATCH_PX = 150  # max center distance to pair a player bbox with a position label

    try:
        with open(positions_file, 'r') as f:
            data = json.load(f)

        # --- Merge: replace position-model bboxes with tight player-detector bboxes ---
        # Iterate over player detections (accurate locations); for each one find the
        # nearest position detection within PLAYER_POS_MATCH_PX and inherit its label.
        # Player detections with no nearby position label are excluded (sideline/crowd).
        player_file = os.path.join(base_cache_dir, folder_name, "players",
                                   f"{video_name}_detection.json")
        if os.path.exists(player_file):
            try:
                with open(player_file, 'r') as f:
                    player_data = json.load(f)

                # Index player detections by frame number
                player_index = {}
                for fr in player_data.get('frames', []):
                    fnum = fr.get('frame_number', 0)
                    player_index[fnum] = fr.get('detections', [])

                threshold_sq = PLAYER_POS_MATCH_PX ** 2

                for frame_data in data.get('frames', []):
                    fnum = frame_data.get('frame_number', 0)

                    # Find player detections for this frame (or nearest within 5)
                    if fnum in player_index:
                        pl_dets = player_index[fnum]
                    elif player_index:
                        closest = min(player_index.keys(), key=lambda k: abs(k - fnum))
                        pl_dets = player_index[closest] if abs(closest - fnum) <= 5 else []
                    else:
                        pl_dets = []

                    pos_dets = frame_data.get('detections', [])

                    merged = []
                    for pl in pl_dets:
                        pl_bbox = pl.get('bbox', {})
                        plcx = pl_bbox.get('center_x')
                        plcy = pl_bbox.get('center_y')
                        if plcx is None or plcy is None:
                            continue

                        # Find nearest position detection by center distance
                        best_dist, best_pos = float('inf'), None
                        for pos in pos_dets:
                            pos_bbox = pos.get('bbox', {})
                            pcx = pos_bbox.get('center_x')
                            pcy = pos_bbox.get('center_y')
                            if pcx is None or pcy is None:
                                continue
                            dist = (plcx - pcx) ** 2 + (plcy - pcy) ** 2
                            if dist < best_dist:
                                best_dist = dist
                                best_pos = pos

                        if best_pos is not None and best_dist < threshold_sq:
                            merged.append({
                                'class': best_pos['class'],
                                'confidence': best_pos.get('confidence', 1.0),
                                'bbox': pl_bbox,
                            })

                    if merged:
                        frame_data['detections'] = merged

                print(f"[OFFENSE SELECTION] Merged player-detector bboxes with position labels")
            except Exception as merge_err:
                print(f"[OFFENSE SELECTION] Warning: player bbox merge failed: {merge_err}")

        # Cap wide receivers at 3 per frame, keeping the most split-wide (largest Y deviation).
        # Extras are relabeled 'oline' so they still show but don't inflate the WR count.
        WR_CLASSES = {'wide_receiver', 'tight_end'}
        MAX_WR = 5
        for frame_data in data.get('frames', []):
            dets = frame_data.get('detections', [])
            wr_dets = [d for d in dets if d.get('class', '').lower() in WR_CLASSES]
            if len(wr_dets) > MAX_WR:
                off_ys = [d['bbox']['center_y'] for d in dets
                          if d.get('class', '').lower() not in ('defense', 'ref', 'yard_marker')
                          and d['bbox'].get('center_y') is not None]
                if off_ys:
                    mean_y = sum(off_ys) / len(off_ys)
                    wr_dets.sort(key=lambda d: abs(d['bbox']['center_y'] - mean_y), reverse=True)
                    for d in wr_dets[MAX_WR:]:
                        d['class'] = 'oline'

        _ensure_qb_labeled(data)
        _label_running_backs(data)
        _resolve_qb_rb_by_height(data)
        if hasattr(parent, 'custom_video'):
            parent.custom_video.set_detection_data(data)
            parent.custom_video.set_show_boxes(True)
        print(f"[OFFENSE SELECTION] Loaded position-labeled data for offense mode")
    except Exception as e:
        print(f"[OFFENSE SELECTION] Error loading position data: {e}")


def _restore_player_data_after_offense_mode(parent):
    """Restore the generic player-detection data after offense mode is turned off."""
    load_and_set_detection_data(parent, "players")
    if hasattr(parent, 'custom_video'):
        parent.custom_video.set_show_boxes(getattr(parent, 'show_bounding_boxes', False))


def _update_virtual_field_labels(parent):
    """Populate virtual_field.offense_label_points from the corrected position data.

    Builds a list of (center_x, center_y, class_name) tuples in image space so
    the virtual field can nearest-neighbour match each dot's original_bbox to the
    correct position class.
    """
    if not hasattr(parent, 'virtual_field'):
        return
    vf = parent.virtual_field
    pos_data = getattr(parent.custom_video, 'position_detection_data', None) if hasattr(parent, 'custom_video') else None
    if pos_data is None:
        vf.offense_label_points = []
        return

    label_pts = []
    for frame_data in pos_data.get('frames', []):
        for det in frame_data.get('detections', []):
            bbox = det.get('bbox', {})
            cx = bbox.get('center_x')
            cy = bbox.get('center_y')
            cls = det.get('class', 'player').lower()
            if cx is not None and cy is not None:
                label_pts.append((cx, cy, cls))

    vf.offense_label_points = label_pts


def toggle_offense_selection(parent, button):
    """Toggle highlight of the 11 offensive players at the snap frame."""
    from PySide6.QtWidgets import QMessageBox
    if not hasattr(parent, "custom_video"):
        return

    if button.isChecked():
        ok = _compute_offense_selection_for_current_video(parent)
        if not ok:
            button.setChecked(False)
            video_name = ""
            if hasattr(parent, "current_video_path") and parent.current_video_path:
                video_name = os.path.splitext(os.path.basename(parent.current_video_path))[0]
            msg = QMessageBox(parent)
            msg.setWindowTitle("No Offense Data")
            msg.setText(f"No offense data found for '{video_name}'.")
            msg.setInformativeText("Run the full processing pipeline on this clip first (all 7 steps must complete, including Static Process).")
            msg.setIcon(QMessageBox.Information)
            msg.exec()
            return
        parent.custom_video.set_show_offense_selection(True)
        # Use position-labeled data so QB/defense/oline class names are present
        _load_position_data_for_offense_mode(parent)
        # Populate virtual field label points from the now-corrected position data
        _update_virtual_field_labels(parent)
    else:
        parent.custom_video.set_show_offense_selection(False)
        # Restore generic player detection data
        _restore_player_data_after_offense_mode(parent)
        if hasattr(parent, 'virtual_field'):
            parent.virtual_field.offense_label_points = []

    # Sync virtual field team coloring
    if hasattr(parent, 'virtual_field'):
        parent.virtual_field.offense_selection_mode = button.isChecked()
        parent.virtual_field.update()

def toggle_yard_marker_boxes(parent, button):
    """Toggle yard marker bounding box visibility - wrapper for unified function"""
    toggle_bounding_boxes(parent, button, "yard_markers")

def toggle_legend(parent, button=None):
    """ Shows a legend that tells user relationship beteween readable positions and their colors"""

    # If legend exists and this function was called, time to close and delete it
    if hasattr(parent, "legend_widget") and parent.legend_widget is not None:
        if parent.legend_widget.isVisible():
            parent.legend_widget.close()
            parent.legend_widget.deleteLater()
            parent.legend_widget = None
            if button:
                button.setChecked(False)
            return

    # Otherwise build the legend
    legend = QWidget(parent)
    parent.legend_widget = legend

    # Make legend frameless + movable + not in taskbar
    legend.setWindowFlags(
        Qt.Tool
        | Qt.FramelessWindowHint
        | Qt.WindowStaysOnTopHint
    )

    # Add manual dragging
    def mousePressEvent(event):
        legend.drag_pos = event.globalPos() - legend.frameGeometry().topLeft()
        event.accept()

    def mouseMoveEvent(event):
        if event.buttons() & Qt.LeftButton:
            legend.move(event.globalPos() - legend.drag_pos)
            event.accept()

    legend.mousePressEvent = mousePressEvent
    legend.mouseMoveEvent = mouseMoveEvent

    # Make sure matches theme
    is_dark = getattr(parent, "is_dark", True)
    legend_bg = "#222" if is_dark else "#ffffff"
    legend_text = "white" if is_dark else "black"

    legend.setStyleSheet(
        f"""
        QWidget {{
            background-color: {legend_bg};
            color: {legend_text};
            border: none;
            border-radius: 6px;
        }}
        QLabel {{
            color: {legend_text};
        }}
        """
    )

    # Layout
    layout = QVBoxLayout(legend)
    layout.setContentsMargins(10, 10, 10, 10)

    # Populate items
    for pos, color in POSITION_COLORS.items():
        row = QHBoxLayout()

        color_box = QLabel()
        color_box.setFixedSize(20, 20)
        color_box.setStyleSheet(
            f"""
            background-color: rgba({color.red()}, {color.green()}, {color.blue()}, 255);
            border: 1px solid {'white' if is_dark else 'black'};
            """
        )

        text_label = QLabel(pos)
        text_label.setStyleSheet("font-size: 14px; padding-left: 6px;")

        row.addWidget(color_box)
        row.addWidget(text_label)
        row.addStretch()
        layout.addLayout(row)

    legend.adjustSize()
    legend.show()

    parent.position_legend()


def load_and_set_detection_data(parent, data_type):
    """
    [UNIFIED] Load detection data for a specified data type ("players" or "yard_markers") 
    and set it in the video widget.
    """
    if not hasattr(parent, 'current_video_path') or not parent.current_video_path:
        print("No current video path set")
        return
    
    video_path = parent.current_video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Logic for determining the folder name based on data_type
    if data_type == "players":
        data_folder_name = "players"
        file_suffix = "_detection"
    elif data_type == "yard_markers":
        data_folder_name = "yard_markers"
        file_suffix = "_yard_markers"
    else:
        print(f"Unknown data type: {data_type}")
        return

    current_folder_name = getattr(parent, 'current_folder', 'default_folder')
    
    # Use shared cache directory function
    base_cache_dir = get_cache_dir()
    detection_file = os.path.join(base_cache_dir, os.path.basename(current_folder_name), data_folder_name, f"{video_name}{file_suffix}.json")
    
    if os.path.exists(detection_file):
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            print(f"Loaded {data_type} data from {os.path.basename(detection_file)}: {len(data.get('frames', []))} frames")
            
            if hasattr(parent, 'custom_video'):
                if data_type == "players":
                    parent.custom_video.set_detection_data(data)
                elif data_type == "yard_markers":
                    parent.custom_video.yard_marker_data = data
                
            else:
                print("No custom video widget found")
                
        except Exception as e:
            print(f"Error loading {data_type} data: {e}")
            if hasattr(parent, 'custom_video'):
                if data_type == "players":
                    parent.custom_video.set_detection_data(None)
                elif data_type == "yard_markers":
                    parent.custom_video.yard_marker_data = None
    else:
        print(f"{data_type.capitalize()} file not found: {detection_file}")
        if hasattr(parent, 'custom_video'):
            if data_type == "players":
                parent.custom_video.set_detection_data(None)
                parent.custom_video.set_show_boxes(False)
            elif data_type == "yard_markers":
                parent.custom_video.yard_marker_data = None

def load_and_set_detection_data_fallback(parent):
    """
    Loads generic player detection data if position data is missing, 
    ensuring the video can still display *some* boxes.
    """
    video_path = parent.current_video_path
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    current_folder_name = getattr(parent, 'current_folder', 'default_folder')
    
    # Use shared cache directory function
    base_cache_dir = get_cache_dir()
    detection_file = os.path.join(base_cache_dir, os.path.basename(current_folder_name), "players", f"{video_name}_detection.json")
    
    if os.path.exists(detection_file):
        try:
            with open(detection_file, 'r') as f:
                data = json.load(f)
            
            print(f"Loaded FALLBACK generic player data from {os.path.basename(detection_file)}.")
            if hasattr(parent, 'custom_video'):
                parent.custom_video.set_detection_data(data)
                
        except Exception as e:
            print(f"Error loading FALLBACK player data: {e}")
            if hasattr(parent, 'custom_video'):
                parent.custom_video.set_detection_data(None)
    else:
        print(f"FALLBACK generic player file not found: {detection_file}")
        if hasattr(parent, 'custom_video'):
            parent.custom_video.set_detection_data(None)

def pause_for_drag(parent):
    if hasattr(parent, 'player'):
        parent.was_playing_before_drag = (parent.player.playbackState() == QMediaPlayer.PlayingState)
        parent.player.pause()

def resume_after_drag(parent):
    if hasattr(parent, 'player') and hasattr(parent, 'was_playing_before_drag') and parent.was_playing_before_drag:
        parent.player.play()
    parent.was_playing_before_drag = False

def find_video_widget(parent):
    for dock in parent.findChildren(QDockWidget):
        if dock.widget():
            graphics_view = dock.widget().findChild(BoundingBoxGraphicsView)
            if graphics_view:
                return graphics_view
            
            simple_overlay = dock.widget().findChild(SimpleOverlayWidget)
            if simple_overlay:
                return simple_overlay
    
    print(f"No video widget found in dock widgets")
    return None