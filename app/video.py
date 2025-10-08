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
    """
    Custom video widget that uses OpenCV to display frames and QPainter
    to draw overlay graphics (bounding boxes) directly on the video frame.
    
    The frame retrieval and drawing logic is unified to handle different types
    of detection data (players and yard markers) efficiently.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: black;")
        self.detection_data = None
        self.yard_marker_data = None
        self.current_frame = 0
        self.show_boxes = False
        self.show_yard_marker_boxes = False
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
        """Set the detection data for player bounding boxes."""
        self.detection_data = data
        if data is None:
            print("Detection data cleared - no player bounding boxes will be shown")
        if data and 'frames' in data:
            print(f"   Total player frames: {len(data['frames'])}")
        self.update()
    
    def set_show_boxes(self, show):
        """Set whether to show player bounding boxes."""
        self.show_boxes = show
        self.update()
    
    def set_show_yard_marker_boxes(self, show):
        """Set whether to show yard marker bounding boxes."""
        self.show_yard_marker_boxes = show
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
            parent.progress_slider.setValue(progress)
        
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
                if self.show_boxes and self.detection_data:
                    video_data_sources.append(self.detection_data)
                
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
        
        data_source: self.detection_data or self.yard_marker_data
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
        
        # Use a tolerance of 15 frames
        if closest_frame and min_diff < 15:
            return closest_frame.get('detections', [])
        
        return []
    
    def _draw_single_bbox(self, painter, detection, scale_x, scale_y, x_offset=0, y_offset=0):
        """
        Draw a single bounding box with dynamic styling based on class (internal helper).
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
        
        # Define colors based on class name
        class_name = detection.get('class', 'player')
        
        if class_name == 'player':
            box_color = QColor(0, 255, 0) # Green for players
        elif class_name == 'yard_marker':
            box_color = QColor(0, 255, 255) # Cyan for yard markers
        else:
            box_color = QColor(255, 255, 0) # Yellow fallback
        
        # Draw bounding box
        painter.setPen(QPen(box_color, 3))
        # Use a semi-transparent brush
        painter.setBrush(QBrush(QColor(box_color.red(), box_color.green(), box_color.blue(), 50))) 
        painter.drawRect(scaled_x, scaled_y, scaled_w, scaled_h)
        
        # Draw label
        confidence = detection.get('confidence', 0.0)
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.drawText(scaled_x, scaled_y - 5, f"{class_name} {confidence:.2f}")
        
    def force_update(self):
        self.update()
    
    def toggle_playback(self):
        """Toggle play/pause for custom video widget"""
        if self.is_playing:
            self.timer.stop()
            self.is_playing = False
        else:
            self.timer.start(33)  # ~30 FPS
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
        
        # Create bounding box rectangle
        bbox_rect = QGraphicsRectItem(scaled_x, scaled_y, scaled_w, scaled_h)
        bbox_rect.setPen(QPen(QColor(0, 255, 0), 4))
        bbox_rect.setBrush(QBrush(QColor(0, 255, 0, 30)))
        
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
        text_item.setDefaultTextColor(QColor(255, 255, 255))
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
    parent.volume_slider.valueChanged.connect(lambda volume: print(f"Volume set to: {volume}%"))
    controls_layout.addWidget(parent.volume_slider)

    controls_widget.setLayout(controls_layout)
    main_layout.addWidget(controls_widget, 1)

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
        custom_video.timer.start(33)
        custom_video.is_playing = True
    parent.was_playing_before_drag = False

def load_video_for_custom_widget(parent, video_path):
    """Load a video file into the custom video widget"""
    if hasattr(parent, 'custom_video'):
        custom_video = parent.custom_video
        if custom_video.load_video(video_path):
            parent.progress_slider.setRange(0, 100)
            parent.custom_video_total_frames = custom_video.total_frames
            
            total_time = custom_video.total_frames / custom_video.fps
            parent.time_label.setText(f"00:00 / {int(total_time//60):02d}:{int(total_time%60):02d}")
            
            print(f"Video loaded: {custom_video.total_frames} frames at {custom_video.fps} FPS")
            return True
        else:
            print(f"Failed to load video: {video_path}")
            return False
    return False

def set_current_video_path(parent, video_path):
    """Set the current video path for bounding box data loading"""
    parent.current_video_path = video_path
    
    load_video_for_custom_widget(parent, video_path)
    
    # Always load both player and yard marker detection data when switching videos
    # The visibility is controlled by the toggle buttons, but data is always loaded
    load_and_set_detection_data(parent, "players")
    load_and_set_detection_data(parent, "yard_markers")
    
    # Load homography data for virtual field
    if hasattr(parent, 'current_folder') and hasattr(parent, 'virtual_field'):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        from virtualField import load_homography_data_for_virtual_field
        load_homography_data_for_virtual_field(parent, video_name, parent.current_folder)
    
    # Sync the video widget's internal state with the parent's button states
    if hasattr(parent, 'custom_video'):
        parent.custom_video.set_show_boxes(parent.show_bounding_boxes)
        parent.custom_video.set_show_yard_marker_boxes(parent.show_yard_marker_boxes)

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

def toggle_yard_marker_boxes(parent, button):
    """Toggle yard marker bounding box visibility - wrapper for unified function"""
    toggle_bounding_boxes(parent, button, "yard_markers")

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
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    detection_file = os.path.join(project_root, "cache", os.path.basename(current_folder_name), data_folder_name, f"{video_name}{file_suffix}.json")
    
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
