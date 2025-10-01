from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import os

# Add Scripts directory to path to import field drawing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Scripts'))

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
    FIELD_LENGTH = 120.0  # 120 yards
    FIELD_WIDTH = 160.0 / 3.0  # ~53.33 yards
    
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

def update_field_with_correspondence_points(parent, correspondence_file):
    """
    Update the virtual field to show yard marker dots from correspondence points
    
    Args:
        parent: Main window parent
        correspondence_file: Path to correspondence points JSON file
    """
    import json
    
    try:
        if os.path.exists(correspondence_file):
            with open(correspondence_file, 'r') as f:
                correspondence_data = json.load(f)
            
            correspondence_points = correspondence_data.get('correspondences', [])
            
            # Clear and redraw field with yard marker dots
            if hasattr(parent, 'field_axes'):
                parent.field_axes.clear()
                draw_field(parent.field_axes, correspondence_points)
                parent.field_canvas.draw()
                
                print(f"✅ Updated virtual field with {len(correspondence_points)} yard marker dots")
        else:
            print(f"❌ Correspondence file not found: {correspondence_file}")
            
    except Exception as e:
        print(f"❌ Error updating field with correspondence points: {e}")

def toggle_scoreboard(parent, button):
    """Toggle scoreboard visibility"""
    if hasattr(parent, 'scoreboard_widget'):
        if button.isChecked():
            parent.scoreboard_widget.show()
        else:
            parent.scoreboard_widget.hide()

def create_dock_title_bar(dock, parent):
    """Create a custom title bar for the dock widget with scoreboard toggle"""
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
    from PySide6.QtCore import Qt
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
    left_spacer.setFixedWidth(40)  # Space for buttons on the right
    layout.addWidget(left_spacer)
    
    # Title label (centered)
    title_label = QLabel("Virtual Field")
    title_label.setFont(QFont("Arial", 10, QFont.Bold))
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)
    
    # Right spacer to balance the left spacer
    right_spacer = QWidget()
    right_spacer.setFixedWidth(40)  # Space for buttons on the right
    layout.addWidget(right_spacer)
    
    # Scoreboard toggle button (small icon)
    scoreboard_btn = QPushButton("📊")
    scoreboard_btn.setFixedSize(20, 20)
    scoreboard_btn.setCheckable(True)
    scoreboard_btn.setChecked(True)  # Scoreboard visible by default
    scoreboard_btn.setToolTip("Toggle Scoreboard")
    scoreboard_btn.clicked.connect(lambda: toggle_scoreboard(parent, scoreboard_btn))
    layout.addWidget(scoreboard_btn)
    
    # Close button (X)
    close_btn = QPushButton("✕")
    close_btn.setFixedSize(20, 20)
    close_btn.setToolTip("Close")
    close_btn.clicked.connect(dock.close)
    layout.addWidget(close_btn)
    
    title_bar.setLayout(layout)
    return title_bar

def create_virtual_field_dock(parent):
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
    layout.addWidget(scoreboard_widget)
    
    # Create matplotlib figure
    fig = Figure(figsize=(8, 4), facecolor='#2b2b2b')
    canvas = FigureCanvas(fig)
    canvas.setStyleSheet("background-color: #2b2b2b;")
    
    # Create axes with dark background
    ax = fig.add_subplot(111, facecolor='#2b2b2b')
    
    # Draw the football field
    draw_field(ax)
    
    # Store references for later updates
    parent.field_figure = fig
    parent.field_axes = ax
    parent.field_canvas = canvas
    
    layout.addWidget(canvas)
    main_widget.setLayout(layout)
    dock.setWidget(main_widget)
    
    return dock

def create_scoreboard(parent):
    """Create a scoreboard widget with empty data fields"""
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFont
    
    scoreboard = QWidget()
    scoreboard.setFixedHeight(170)
    scoreboard.setStyleSheet("""
        QWidget {
            background-color: #1a1a1a;
            border: 2px solid #444444;
            border-radius: 8px;
        }
        QLabel {
            color: white;
            font-weight: bold;
        }
    """)
    
    # Main layout
    main_layout = QHBoxLayout()
    main_layout.setContentsMargins(15, 10, 15, 10)
    main_layout.setSpacing(20)
    
    # Left team section
    left_team_layout = QVBoxLayout()
    left_team_layout.setAlignment(Qt.AlignCenter)
    
    left_team_name = QLabel("HOME")
    left_team_name.setFont(QFont("Arial", 14, QFont.Bold))
    left_team_name.setAlignment(Qt.AlignCenter)
    left_team_name.setMinimumWidth(80)
    left_team_layout.addWidget(left_team_name)
    
    left_score = QLabel("0")
    left_score.setFont(QFont("Arial", 24, QFont.Bold))
    left_score.setAlignment(Qt.AlignCenter)
    left_score.setStyleSheet("color: #ff6600;")
    left_team_layout.addWidget(left_score)
    
    left_timeouts = QLabel("T.O.L: 3")
    left_timeouts.setFont(QFont("Arial", 10))
    left_timeouts.setAlignment(Qt.AlignCenter)
    left_team_layout.addWidget(left_timeouts)
    
    main_layout.addLayout(left_team_layout)
    
    # Center section (clock and game info)
    center_layout = QVBoxLayout()
    center_layout.setAlignment(Qt.AlignCenter)
    
    # Game clock
    game_clock = QLabel("15:00")
    game_clock.setFont(QFont("Arial", 20, QFont.Bold))
    game_clock.setAlignment(Qt.AlignCenter)
    game_clock.setStyleSheet("color: #ff6600;")
    center_layout.addWidget(game_clock)
    
    # Game status
    game_status = QLabel("1st Quarter")
    game_status.setFont(QFont("Arial", 12))
    game_status.setAlignment(Qt.AlignCenter)
    game_status.setMinimumWidth(100)
    center_layout.addWidget(game_status)
    
    # Down and distance
    down_distance = QLabel("1st & 10")
    down_distance.setFont(QFont("Arial", 14, QFont.Bold))
    down_distance.setAlignment(Qt.AlignCenter)
    down_distance.setStyleSheet("color: #ff6600;")
    center_layout.addWidget(down_distance)
    
    # Ball position
    ball_position = QLabel("Ball on: 25")
    ball_position.setFont(QFont("Arial", 10))
    ball_position.setAlignment(Qt.AlignCenter)
    center_layout.addWidget(ball_position)
    
    main_layout.addLayout(center_layout)
    
    # Right team section
    right_team_layout = QVBoxLayout()
    right_team_layout.setAlignment(Qt.AlignCenter)
    
    right_team_name = QLabel("AWAY")
    right_team_name.setFont(QFont("Arial", 14, QFont.Bold))
    right_team_name.setAlignment(Qt.AlignCenter)
    right_team_name.setMinimumWidth(80)
    right_team_layout.addWidget(right_team_name)
    
    right_score = QLabel("0")
    right_score.setFont(QFont("Arial", 24, QFont.Bold))
    right_score.setAlignment(Qt.AlignCenter)
    right_score.setStyleSheet("color: #ff6600;")
    right_team_layout.addWidget(right_score)
    
    right_timeouts = QLabel("T.O.L: 3")
    right_timeouts.setFont(QFont("Arial", 10))
    right_timeouts.setAlignment(Qt.AlignCenter)
    right_team_layout.addWidget(right_timeouts)
    
    main_layout.addLayout(right_team_layout)
    
    # Store references for later updates
    parent.scoreboard_widget = scoreboard
    parent.left_team_name = left_team_name
    parent.left_score = left_score
    parent.left_timeouts = left_timeouts
    parent.right_team_name = right_team_name
    parent.right_score = right_score
    parent.right_timeouts = right_timeouts
    parent.game_clock = game_clock
    parent.game_status = game_status
    parent.down_distance = down_distance
    parent.ball_position = ball_position
    
    scoreboard.setLayout(main_layout)
    return scoreboard