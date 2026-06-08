from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSlider
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor
import subprocess
import platform
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import os
import json
from fileAccess import get_cache_dir

# Import the color map definition from video.py for color consistency
try:
    # Attempt relative import from parent directory structure
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from video import POSITION_COLORS
except ImportError:
    # Define a fallback or generic colors if import fails
    print("WARNING: Could not import POSITION_COLORS from video.py. Using default colors.")
    POSITION_COLORS = {
        'qb': QColor(255, 165, 0),
        'defense': QColor(0, 0, 255),
        'player': QColor(0, 255, 0)
    }

# Add scripts directory to path to import field drawing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from staticProcess import identify_qb

class VirtualFieldWidget(QWidget):
    """Widget that displays a static football field with player dots"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.current_frame = 0
        self.homography_data = None
        self.offense_positions = None  # list of (nx, ny) yard coords for 11 offensive players
        self.qb_index = None
        self.field_image = None
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555555;")

        # Load field image
        self.load_field_image()
        
    def load_field_image(self):
        """Load a simple football field image as background"""
        # Create a simple field image without matplotlib to avoid layering issues
        field_width = 400
        field_height = 200
        
        # Create a green field background
        field_img = np.full((field_height, field_width, 3), [34, 139, 34], dtype=np.uint8)
        
        # Draw yard lines (every 10 yards)
        for i in range(0, field_width, field_width // 10):  # 10 sections for 100 yards
            cv2.line(field_img, (i, 0), (i, field_height), (255, 255, 255), 2)
        
        # Draw hash marks (every 5 yards)
        for i in range(field_width // 20, field_width, field_width // 20):
            cv2.line(field_img, (i, field_height // 4), (i, 3 * field_height // 4), (255, 255, 255), 1)
        
        # Convert to QPixmap
        field_rgb = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)
        h, w, ch = field_rgb.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage
        q_image = QImage(field_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.field_image = QPixmap.fromImage(q_image)
        
    def load_homography_data(self, video_name, folder_name):
        """Load homography data for the current video"""
        try:
            # Use shared cache directory function
            base_cache_dir = get_cache_dir()
            homography_file = os.path.join(base_cache_dir, os.path.basename(folder_name), "homography", f"{video_name}_normalized_positions.json")
            
            if os.path.exists(homography_file):
                with open(homography_file, 'r') as f:
                    self.homography_data = json.load(f)
                print(f"Loaded homography data: {self.homography_data.get('total_frames')} frames")
                
                # Set to first frame if homography data exists
                if self.homography_data and 'normalized_positions' in self.homography_data:
                    normalized_positions = self.homography_data['normalized_positions']
                    if normalized_positions:
                        # Find the first frame (minimum frame number)
                        frame_numbers = [int(k) for k in normalized_positions.keys() if k.isdigit()]
                        if frame_numbers:
                            first_frame = min(frame_numbers)
                            self.current_frame = first_frame
                            self.update()  # Update display to show first frame
                            print(f"Set virtual field to first frame: {first_frame}")
                
                return True
            else:
                print(f"Homography file not found: {homography_file}")
                # Clear homography data and reset frame to clear the display
                self.homography_data = None
                self.current_frame = 0
                self.update()  # Force repaint to clear player dots
                return False
        except Exception as e:
            print(f"Error loading homography data: {e}")
            # Clear homography data and reset frame to clear the display
            self.homography_data = None
            self.current_frame = 0
            self.update()  # Force repaint to clear player dots
            return False
    
    def load_offense_positions(self, video_name, folder_name):
        """Load the 11 offensive player field positions from offense_positions.csv."""
        try:
            import pandas as pd
            base_cache_dir = get_cache_dir()
            csv_path = os.path.join(base_cache_dir, os.path.basename(folder_name), "offense_positions.csv")

            if not os.path.exists(csv_path):
                print(f"[Virtual Field] offense_positions.csv not found: {csv_path}")
                self.offense_positions = None
                self.qb_index = None
                self.update()
                return False

            df = pd.read_csv(csv_path)
            if "clip_name" not in df.columns:
                self.offense_positions = None
                self.qb_index = None
                self.update()
                return False

            row = df.loc[df["clip_name"] == video_name]
            if row.empty:
                print(f"[Virtual Field] No offense row for '{video_name}'")
                self.offense_positions = None
                self.qb_index = None
                self.update()
                return False

            row = row.iloc[0]
            positions = []
            for i in range(1, 12):
                nx_col, ny_col = f"nx{i}", f"ny{i}"
                if nx_col in row and ny_col in row:
                    nx, ny = row[nx_col], row[ny_col]
                    if pd.notna(nx) and pd.notna(ny):
                        positions.append((float(nx), float(ny)))

            if positions:
                self.offense_positions = positions
                points_for_qb = [[nx, ny, 0.0, 0.0] for nx, ny in positions]
                self.qb_index = identify_qb(points_for_qb)
                print(f"[Virtual Field] Loaded {len(positions)} offense positions for {video_name}, QB index: {self.qb_index}")
                self.update()
                return True

            self.offense_positions = None
            self.qb_index = None
            self.update()
            return False
        except Exception as e:
            print(f"[Virtual Field] Error loading offense positions: {e}")
            self.offense_positions = None
            self.qb_index = None
            self.update()
            return False

    def set_current_frame(self, frame_number):
        """Set the current frame and update the display"""
        self.current_frame = frame_number
        self.update()
    
    def paintEvent(self, event):
        """Paint the field with player dots"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw field background
        if self.field_image:
            # Calculate field dimensions within the widget (accounting for aspect ratio)
            field_width = min(self.width(), int(self.height() * 100 / 53.33))
            field_height = min(self.height(), int(self.width() * 53.33 / 100))
            
            # Center the field in the widget
            field_x_offset = (self.width() - field_width) // 2
            field_y_offset = (self.height() - field_height) // 2
            
            # Scale field image to fit the calculated field dimensions
            scaled_field = self.field_image.scaled(field_width, field_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(field_x_offset, field_y_offset, scaled_field)
        
        # Calculate field layout (shared by both draw paths)
        field_width = min(self.width(), int(self.height() * 100 / 53.33))
        field_height = min(self.height(), int(self.width() * 53.33 / 100))
        field_x_offset = (self.width() - field_width) // 2
        field_y_offset = (self.height() - field_height) // 2
        field_left = field_x_offset
        field_right = field_x_offset + field_width
        field_top = field_y_offset
        field_bottom = field_y_offset + field_height

        def draw_dot(nx, ny, color):
            wx = int(field_x_offset + (nx * field_width / 100))
            wy = int(field_y_offset + field_height - (ny * field_height / 53.33))
            if field_left <= wx <= field_right and field_top <= wy <= field_bottom:
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(QColor(255, 255, 255), 3))
                painter.drawEllipse(wx - 8, wy - 8, 16, 16)

        if self.homography_data and 'normalized_positions' in self.homography_data:
            normalized_positions = self.homography_data['normalized_positions']
            frame_key = str(self.current_frame)
            if frame_key in normalized_positions:
                players = normalized_positions[frame_key]

                # Collect field x positions to find LOS and defense side
                xs = []
                for p in players:
                    x = p.get('normalized_position', {}).get('x')
                    if x is not None:
                        xs.append(float(x))

                defense_side = None  # 'above' or 'below' median x
                if len(xs) >= 2:
                    los_x = float(np.median(xs))
                    farthest_dist = 0
                    for x in xs:
                        dist = abs(x - los_x)
                        if dist > farthest_dist:
                            farthest_dist = dist
                            defense_side = 'above' if x > los_x else 'below'

                for player in players:
                    normalized_pos = player.get('normalized_position', {})
                    x = normalized_pos.get('x', 0)
                    y = normalized_pos.get('y', 0)
                    if defense_side is not None:
                        px = float(x)
                        los_x_val = float(np.median(xs))
                        is_defense = (defense_side == 'above' and px > los_x_val) or \
                                     (defense_side == 'below' and px < los_x_val)
                        dot_color = POSITION_COLORS['defense'] if is_defense else POSITION_COLORS['player']
                    else:
                        dot_color = POSITION_COLORS['player']
                    draw_dot(x, y, dot_color)


        
        # Draw frame number
        painter.setPen(QPen(QColor(0, 0, 0)))  # Black text
        painter.drawText(10, 20, f"Frame: {self.current_frame}")
        
        painter.end()

def draw_field(ax, correspondence_points=None):
    """Draw a college football field to scale (yards) - from drawPlayers.py"""
    import matplotlib.patches as patches
    
    # Field constants (yards)
    FIELD_LENGTH = 120.0                # 120 yards (100 + 2 endzones)
    FIELD_WIDTH = 160.0 / 3.0           # 160 ft -> yards (160/3 ~= 53.3333)
    HASH_DIST_FT = 40.0                 # hash marks are 40 ft from sideline
    HASH_NEAR_YD = HASH_DIST_FT / 3.0   # in yards (~13.3333)
    HASH_TOP_YD = FIELD_WIDTH - HASH_NEAR_YD
    HASH_LEN = 0.5

    # Base rectangle
    field = patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, linewidth=2,
                              edgecolor='black', facecolor='green', zorder=0)
    ax.add_patch(field)

    # Yard lines every 5 yards (thinner) and every 10 (thicker)
    for x in range(10, int(FIELD_LENGTH), 5):
        lw = 2 if x % 10 == 0 else 1
        ax.plot([x, x], [0, FIELD_WIDTH], color='white', linewidth=lw, zorder=1)

    # Hash marks (every yard between 10 and 110 except multiples of 5)
    for x in range(11, 110):
        if x % 5 == 0:
            continue
        ax.plot([x, x], [HASH_NEAR_YD - HASH_LEN / 2, HASH_NEAR_YD + HASH_LEN / 2],
                color='white', linewidth=1, zorder=2)
        ax.plot([x, x], [HASH_TOP_YD - HASH_LEN / 2, HASH_TOP_YD + HASH_LEN / 2],
                color='white', linewidth=1, zorder=2)

    # End zones
    ez1 = patches.Rectangle((0, 0), 10, FIELD_WIDTH, linewidth=1,
                            edgecolor='white', facecolor='darkblue', alpha=0.6, zorder=1)
    ez2 = patches.Rectangle((110, 0), 10, FIELD_WIDTH, linewidth=1,
                            edgecolor='white', facecolor='darkred', alpha=0.6, zorder=1)
    ax.add_patch(ez1); ax.add_patch(ez2)

    # Yard numbers (every 10) - correct football field numbering
    for x in range(20, 110, 10):
        yard_number = x - 10  # Convert to actual yard number
        if yard_number <= 50:
            # First half: 10, 20, 30, 40, 50
            display_number = yard_number
        else:
            # Second half: 40, 30, 20, 10 (counting down from 50)
            display_number = 100 - yard_number
        
        ax.text(x, 9, str(display_number), color='white', fontsize=10, ha='center', va='center', zorder=3)
        ax.text(x, FIELD_WIDTH-9, str(display_number), color='white', fontsize=10, ha='center', va='center', rotation=180, zorder=3)

    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw yard marker dots if correspondence points are provided
    if correspondence_points:
        draw_yard_marker_dots(ax, correspondence_points)

def draw_yard_marker_dots(ax, correspondence_points=None):
    """
    Draw white dots on the field to show yard marker positions
    
    Args:
        ax: Matplotlib axes object
        correspondence_points: List of correspondence points with field coordinates
    """
    if not correspondence_points:
        return
    
    # Field dimensions (in yards for plotting)
    for point in correspondence_points:
        field_coords = point.get('field_point', {})
        marker_info = point.get('yard_marker_info', {})
        
        # Convert feet to yards for plotting
        x_yards = field_coords.get('x', 0) / 3.0  # Convert feet to yards
        y_yards = field_coords.get('y', 0) / 3.0  # Convert feet to yards
        
        # Draw white dot
        ax.plot(x_yards, y_yards, 'wo', markersize=8, markeredgecolor='black', 
                markeredgewidth=1, zorder=10)
        
        # Add label
        label = marker_info.get('label', '')
        ax.text(x_yards, y_yards + 2, label, color='white', fontsize=8, 
                ha='center', va='bottom', zorder=11, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

def update_field_with_correspondence_points(parent, correspondence_file, frame_number=0):
    """
    Update the virtual field to show yard marker dots from correspondence points
    
    Args:
        parent: Main window parent
        correspondence_file: Path to correspondence points JSON file
        frame_number: Frame number to display (for per-frame correspondence points)
    """
    import json
    
    try:
        if os.path.exists(correspondence_file):
            with open(correspondence_file, 'r') as f:
                correspondence_data = json.load(f)
            
            # Check if this is per-frame data or single correspondence points
            if 'frame_correspondences' in correspondence_data:
                # Per-frame correspondence points
                frame_correspondences = correspondence_data.get('frame_correspondences', {})
                correspondence_points = frame_correspondences.get(str(frame_number), [])
                
                if not correspondence_points:
                    # Try to find the closest frame with correspondence points
                    available_frames = [int(f) for f in frame_correspondences.keys() if frame_correspondences[f]]
                    if available_frames:
                        closest_frame = min(available_frames, key=lambda x: abs(x - frame_number))
                        correspondence_points = frame_correspondences.get(str(closest_frame), [])
                        print(f"Using correspondence points from frame {closest_frame} (requested: {frame_number})")
            else:
                # Single correspondence points (legacy format)
                correspondence_points = correspondence_data.get('correspondences', [])
            
            # Clear and redraw field with yard marker dots
            if hasattr(parent, 'field_axes'):
                parent.field_axes.clear()
                draw_field(parent.field_axes, correspondence_points)
                parent.field_canvas.draw()
                
                # Update title to show frame number if using per-frame data
                if 'frame_correspondences' in correspondence_data:
                    parent.field_axes.set_title(f"Virtual Field - Frame {frame_number} ({len(correspondence_points)} points)", 
                                              color='white', fontsize=12)
                
        else:
            print(f"Correspondence file not found: {correspondence_file}")
            
    except Exception as e:
        print(f"Error updating field with correspondence points: {e}")

def load_correspondence_video(parent):
    """Load the correspondence points video for playback"""
    try:
        # Try to find the correspondence video file
        cache_dir = "cache"
        video_files = []
        
        if os.path.exists(cache_dir):
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith("_correspondence_video.mp4"):
                        video_files.append(os.path.join(root, file))
        
        if video_files:
            # Get the most recent video file
            latest_video = max(video_files, key=os.path.getmtime)
            
            # Load video
            if hasattr(parent, 'video_cap') and parent.video_cap:
                parent.video_cap.release()
            
            parent.video_cap = cv2.VideoCapture(latest_video)
            if parent.video_cap.isOpened():
                total_frames = int(parent.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                parent.frame_slider.setMaximum(total_frames - 1)
                parent.frame_slider.setValue(0)
                parent.frame_label.setText(f"0 / {total_frames}")
                
                # Load first frame
                show_video_frame(parent, 0)
                
                print(f"Loaded correspondence points video: {latest_video}")
                return True
            else:
                print("Failed to open video file")
                return False
        else:
            print("No correspondence points video found. Please run the processing pipeline first.")
            return False
            
    except Exception as e:
        print(f"Error loading correspondence points video: {e}")
        return False

def show_video_frame(parent, frame_number):
    """Display a specific video frame"""
    if not hasattr(parent, 'video_cap') or not parent.video_cap:
        return
    
    try:
        parent.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = parent.video_cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Clear the field and show video frame
            parent.field_axes.clear()
            parent.field_axes.imshow(frame_rgb)
            parent.field_axes.axis('off')
            parent.field_canvas.draw()
            
            # Update frame counter
            total_frames = int(parent.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            parent.frame_label.setText(f"{frame_number} / {total_frames}")
            
    except Exception as e:
        print(f"Error displaying video frame: {e}")

def toggle_video_playback(parent):
    """Toggle video play/pause"""
    if not hasattr(parent, 'video_cap') or not parent.video_cap:
        # Try to load video first
        if load_correspondence_video(parent):
            parent.video_timer.start(33)  # ~30 FPS
            parent.play_button.setText("⏸️ Pause")
        return
    
    if parent.video_timer.isActive():
        parent.video_timer.stop()
        parent.play_button.setText("▶️ Play")
    else:
        parent.video_timer.start(33)  # ~30 FPS
        parent.play_button.setText("⏸️ Pause")

def next_video_frame(parent):
    """Advance to next video frame"""
    if not hasattr(parent, 'video_cap') or not parent.video_cap:
        return
    
    current_frame = int(parent.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(parent.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if current_frame < total_frames - 1:
        show_video_frame(parent, current_frame + 1)
        parent.frame_slider.setValue(current_frame + 1)
    else:
        # End of video - pause
        parent.video_timer.stop()
        parent.play_button.setText("▶️ Play")

def seek_video_frame(parent, frame_number):
    """Seek to a specific video frame"""
    if not hasattr(parent, 'video_cap') or not parent.video_cap:
        return
    
    show_video_frame(parent, frame_number)

def toggle_scoreboard(parent, button):
    """Toggle scoreboard visibility and resize field accordingly"""
    if hasattr(parent, 'scoreboard_widget'):
        if button.isChecked():
            parent.scoreboard_widget.show()
            # Scoreboard visible - use normal size
            if hasattr(parent, 'field_figure'):
                parent.field_figure.set_size_inches(16, 10)
                parent.field_canvas.draw()
        else:
            parent.scoreboard_widget.hide()
            # Scoreboard hidden - make field larger
            if hasattr(parent, 'field_figure'):
                parent.field_figure.set_size_inches(20, 12)  # Much larger when scoreboard is hidden
                parent.field_canvas.draw()

def create_dock_title_bar(dock, parent):
    """Create a custom title bar for the dock widget with scoreboard toggle"""
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel
    from PySide6.QtCore import Qt
    
    title_bar = QWidget()
    title_bar.setStyleSheet("""
        QWidget {
            background-color: #3c3c3c;
            border: none;
            padding: 2px;
        }
    """)
    
    layout = QHBoxLayout()
    layout.setContentsMargins(5, 2, 5, 2)
    layout.setSpacing(5)
    
    # Title label
    title_label = QLabel(dock.windowTitle())
    title_label.setStyleSheet("color: white; font-weight: bold;")
    layout.addWidget(title_label)
    
    layout.addStretch()
    
    # Scoreboard toggle button
    scoreboard_btn = QPushButton("📊")
    scoreboard_btn.setCheckable(True)
    scoreboard_btn.setChecked(False)  # Start with scoreboard hidden
    scoreboard_btn.setToolTip("Toggle Scoreboard")
    scoreboard_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 3px;
            padding: 4px 8px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:checked {
            background-color: #2196F3;
        }
    """)
    scoreboard_btn.clicked.connect(lambda: toggle_scoreboard(parent, scoreboard_btn))
    layout.addWidget(scoreboard_btn)
    
    # Close button
    close_btn = QPushButton("✕")
    close_btn.setToolTip("Close Dock")
    close_btn.setStyleSheet("""
        QPushButton {
            background-color: #f44336;
            color: white;
            border: none;
            border-radius: 3px;
            padding: 4px 8px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #d32f2f;
        }
    """)
    close_btn.clicked.connect(dock.close)
    layout.addWidget(close_btn)
    
    title_bar.setLayout(layout)
    return title_bar

def create_virtual_field_dock(parent):
    """Create a simplified virtual field dock with static field image and player dots"""
    dock = QDockWidget("Virtual Field", parent)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
    
    # Set custom title bar with scoreboard toggle
    dock.setTitleBarWidget(create_dock_title_bar(dock, parent))
    
    # Main widget
    main_widget = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(5, 5, 5, 5)
    
    # Create scoreboard section
    scoreboard_widget = create_scoreboard(parent)
    scoreboard_widget.hide()  # Start with scoreboard hidden
    layout.addWidget(scoreboard_widget)
    
    # Create the virtual field widget
    virtual_field = VirtualFieldWidget(parent)
    layout.addWidget(virtual_field)
    
    # Store reference for updates
    parent.virtual_field = virtual_field
    
    main_widget.setLayout(layout)
    dock.setWidget(main_widget)
    
    return dock

def update_virtual_field_with_video_frame(parent, frame_number):
    """Update the virtual field to show the current video frame's player positions"""
    if hasattr(parent, 'virtual_field'):
        parent.virtual_field.set_current_frame(frame_number)

def load_homography_data_for_virtual_field(parent, video_name, folder_name):
    """Load homography data for the virtual field"""
    if hasattr(parent, 'virtual_field'):
        return parent.virtual_field.load_homography_data(video_name, folder_name)
    return False

def load_offense_positions_for_virtual_field(parent, video_name, folder_name):
    """Load offensive player positions for the virtual field"""
    if hasattr(parent, 'virtual_field'):
        return parent.virtual_field.load_offense_positions(video_name, folder_name)
    return False

def create_scoreboard(parent):
    """Create a scoreboard widget with orange football scoreboard design"""
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
    from PySide6.QtCore import Qt
    
    scoreboard_widget = QFrame()
    scoreboard_widget.setStyleSheet("""
        QFrame {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #FF6B35, stop:1 #E55A2B);
            border: 3px solid #FF8C42;
            border-radius: 12px;
            padding: 15px;
        }
    """)
    
    layout = QHBoxLayout()
    layout.setSpacing(30)
    layout.setContentsMargins(20, 15, 20, 15)
    
    # Home team section
    home_layout = QVBoxLayout()
    home_layout.setAlignment(Qt.AlignCenter)
    home_team_label = QLabel("HOME")
    home_team_label.setStyleSheet("""
        color: white; 
        font-weight: bold; 
        font-size: 16px;
        background-color: rgba(0,0,0,0.3);
        padding: 5px 10px;
        border-radius: 5px;
    """)
    home_score_label = QLabel("0")
    home_score_label.setStyleSheet("""
        color: white; 
        font-size: 36px; 
        font-weight: bold;
        background-color: rgba(0,0,0,0.4);
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.3);
    """)
    home_layout.addWidget(home_team_label)
    home_layout.addWidget(home_score_label)
    layout.addLayout(home_layout)
    
    # Game info section
    game_layout = QVBoxLayout()
    game_layout.setAlignment(Qt.AlignCenter)
    quarter_label = QLabel("Q1")
    quarter_label.setStyleSheet("""
        color: white; 
        font-size: 20px; 
        font-weight: bold;
        background-color: rgba(0,0,0,0.3);
        padding: 8px 15px;
        border-radius: 6px;
    """)
    time_label = QLabel("15:00")
    time_label.setStyleSheet("""
        color: white; 
        font-size: 24px; 
        font-weight: bold;
        background-color: rgba(0,0,0,0.4);
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.3);
    """)
    game_layout.addWidget(quarter_label)
    game_layout.addWidget(time_label)
    layout.addLayout(game_layout)
    
    # Away team section
    away_layout = QVBoxLayout()
    away_layout.setAlignment(Qt.AlignCenter)
    away_team_label = QLabel("AWAY")
    away_team_label.setStyleSheet("""
        color: white; 
        font-weight: bold; 
        font-size: 16px;
        background-color: rgba(0,0,0,0.3);
        padding: 5px 10px;
        border-radius: 5px;
    """)
    away_score_label = QLabel("0")
    away_score_label.setStyleSheet("""
        color: white; 
        font-size: 36px; 
        font-weight: bold;
        background-color: rgba(0,0,0,0.4);
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.3);
    """)
    away_layout.addWidget(away_team_label)
    away_layout.addWidget(away_score_label)
    layout.addLayout(away_layout)
    
    scoreboard_widget.setLayout(layout)
    
    # Store reference for toggling
    parent.scoreboard_widget = scoreboard_widget
    
    return scoreboard_widget