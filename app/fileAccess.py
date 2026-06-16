from PySide6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QTreeView,
    QFileSystemModel, QMenu
)
from PySide6.QtCore import Qt, QDir, QFileInfo
import os
import re
import sys
from pathlib import Path
import pandas as pd

# Video types the app can open (tree view, auto-load, click-to-play).
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv')


def natural_key(name):
    """Sort key putting 'Clip 2' before 'Clip 10' (numeric runs compared as
    numbers, the rest case-insensitively) -- the order the label sheet expects."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]

def _get_exe_dir():
    """Return the directory that contains (or should contain) bundled resources.

    - PyInstaller onefile : sys._MEIPASS  (temp extraction folder)
    - Other frozen builds : working directory at startup
    - Development         : project root derived from this file's location
    """
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    if getattr(sys, "frozen", False):
        # Generic frozen-app fallback: bundled assets typically live in the
        # current working directory at startup.
        return os.getcwd()
    # Development: navigate up from app/fileAccess.py → project root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_cache_dir():
    """Return a *persistent* cache directory that survives across runs.

    For compiled builds the cache lives next to the executable on disk
    (not in the temp extraction folder, which is wiped on exit).
    For development it lives under the project root.
    """
    if hasattr(sys, "_MEIPASS") or getattr(sys, "frozen", False):
        cache_dir = os.path.join(os.path.dirname(sys.executable), "cache")
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cache_dir = os.path.join(project_root, "cache")

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_project_root():
    """Return the root directory where bundled assets (scripts/, yolo_models/, …) live."""
    return _get_exe_dir()

def create_file_title_bar(dock):
    """Create a custom title bar for the file access dock widget"""
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
    
    # Title label (centered)
    title_label = QLabel("File Access")
    title_label.setFont(QFont("Arial", 10, QFont.Bold))
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)

    title_bar.setLayout(layout)
    return title_bar

def create_file_dock(parent):
    dock = QDockWidget("File Access", parent)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(QDockWidget.DockWidgetMovable)
    
    # Set custom title bar
    dock.setTitleBarWidget(create_file_title_bar(dock))
    
    # Main widget
    main_widget = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)

    # Tree view for navigation
    parent.tree_model = QFileSystemModel()

    # Show directories AND video files -- a folder may hold the clips directly,
    # with no subfolders to click. Non-video files stay hidden.
    parent.tree_model.setFilter(QDir.AllDirs | QDir.Files | QDir.NoDotAndDotDot)
    parent.tree_model.setNameFilters([f"*{ext}" for ext in VIDEO_EXTS])
    parent.tree_model.setNameFilterDisables(False)  # hide non-matching files (don't grey them)
    
    parent.tree_view = QTreeView()
    parent.tree_view.setModel(parent.tree_model)
    # Don't set a root index initially - will be set when a folder is loaded
    parent.tree_view.setHeaderHidden(True)
    parent.tree_view.setVisible(False)
    
    # Disable expansion triangles and folder expansion
    parent.tree_view.setRootIsDecorated(False)
    parent.tree_view.setItemsExpandable(False)
    # Initialize with empty state - no folder loaded
    initialize_empty_tree_view(parent)
    
    # Single click loads the folder content but doesn't change tree view
    parent.tree_view.clicked.connect(lambda index: on_tree_clicked(parent, index))
    
    # Double click expands/collapses folders in the tree view
    parent.tree_view.doubleClicked.connect(lambda index: on_tree_double_clicked(parent, index))
    

    
    # Hide all columns except Name
    parent.tree_view.setColumnHidden(1, True)
    parent.tree_view.setColumnHidden(2, True)
    parent.tree_view.setColumnHidden(3, True)
    
    layout.addWidget(parent.tree_view)
    main_widget.setLayout(layout)
    dock.setWidget(main_widget)

    # Add methods to parent
    parent.load_folder = lambda folder_path: load_folder(parent, folder_path, change_view=True)
    parent.open_video_file = lambda video_path: open_video_file(parent, video_path)

    return dock

def initialize_empty_tree_view(parent):
    """Initialize tree view with empty state when no folder is loaded"""
    # Hide the tree view when no folder is loaded
    parent.tree_view.setVisible(False)

def on_tree_clicked(parent, index):
    """Handle single click on tree view items - load folder content but don't change tree view"""
    path = parent.tree_model.filePath(index)

    # A video file: open exactly that clip. If its folder isn't the loaded one
    # yet, load the folder context (CSV, cache) around the clicked clip.
    if os.path.isfile(path):
        if path.lower().endswith(VIDEO_EXTS):
            folder = os.path.dirname(path)
            if getattr(parent, "current_folder", None) != folder:
                load_folder(parent, folder, change_view=False,
                            preferred_video=os.path.basename(path))
            elif hasattr(parent, 'open_video_file'):
                parent.open_video_file(path)
        return

    if has_mp4(path):
        load_folder(parent, path, change_view=False)
    else:
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.warning(parent, "No MP4 Files", "This folder does not contain any MP4 files.")

def has_mp4(folder_path: str) -> bool:
    """Return True if the folder contains at least one .mp4 file."""
    if not os.path.isdir(folder_path):
        return False
    for f in os.listdir(folder_path):
        if f.lower().endswith(".mp4") and os.path.isfile(os.path.join(folder_path, f)):
            return True
    return False

def on_tree_double_clicked(parent, index):
    """Handle double click on tree view items - load folder content"""
    # Since we only show directories and expansion is disabled, 
    # double-click should do the same as single for now
    on_tree_clicked(parent, index)

def load_folder(parent, folder_path, change_view=False, preferred_video=None):
    parent.current_folder = folder_path

    # Change the tree view only if explicitly requested (from Open Folder button)
    if change_view:
        # Set the root path for the model first
        parent.tree_model.setRootPath(folder_path)
        tree_index = parent.tree_model.index(folder_path)
        parent.tree_view.setRootIndex(tree_index)

        # Make the tree view visible when a folder is loaded
        parent.tree_view.setVisible(True)

    # Only auto-load if the folder actually contains video files
    # This prevents loading parent directories that don't have videos
    if has_mp4(folder_path):
        auto_load_folder_content(parent, folder_path, preferred_video=preferred_video)
    else:
        print(f"[INFO] Folder does not contain MP4 files, skipping auto-load: {folder_path}")

def auto_load_folder_content(parent, folder_path, preferred_video=None):
    """Load the folder's CSV and a video: `preferred_video` (the clicked clip)
    when given, else the first clip in natural order."""
    try:
        # Ensure folder_path is absolute and exists
        folder_path = os.path.abspath(folder_path)
        if not os.path.isdir(folder_path):
            print(f"Error: Folder does not exist: {folder_path}")
            return
        
        # Get folder name and cache directory path
        folder_name = os.path.basename(folder_path.rstrip('/\\'))
        
        # Use shared cache directory function
        base_cache_dir = get_cache_dir()
        cache_dir = os.path.join(base_cache_dir, folder_name)
        
        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)
        
        # Find all CSV files in the cache directory
        csv_files = []
        if os.path.exists(cache_dir):
            csv_files = [
                f for f in os.listdir(cache_dir)
                if f.lower().endswith('.csv') and os.path.isfile(os.path.join(cache_dir, f))
            ]
        
        # Find all video files in the video folder, in natural clip order
        # ('Clip 2' before 'Clip 10') so the CSV rows and the auto-loaded
        # first video follow the label sheet's order.
        video_files = sorted(
            (f for f in os.listdir(folder_path)
             if f.lower().endswith(VIDEO_EXTS)
             and os.path.isfile(os.path.join(folder_path, f))),
            key=natural_key)
        
        # Create CSV with video titles if none exists and videos are present
        if not csv_files and video_files:
            csv_path = create_video_based_csv(cache_dir, video_files, folder_name)
            if csv_path:
                csv_files = [os.path.basename(csv_path)]
                print(f"Created new CSV with video titles: {csv_path}")
        
       # Load primary metadata CSV if available (from cache directory).
        # Prefer the folder's *_data.csv (labels, clip info), not training CSVs
        # like offense_positions.csv.
        if csv_files and hasattr(parent, 'load_csv_file'):
            preferred_name = f"{folder_name}_data.csv"
            chosen = None
            if preferred_name in csv_files:
                chosen = preferred_name
            else:
                # Fallback: first CSV in sorted order
                chosen = sorted(csv_files)[0]

            first_csv = os.path.join(cache_dir, chosen)
            if os.path.exists(first_csv):
                parent.load_csv_file(first_csv)
                print(f"Loaded CSV: {os.path.abspath(first_csv)}")
            else:
                print(f"Warning: CSV file not found: {first_csv}")
        
        # Load and play a video: the clicked one when given, else the first.
        if video_files and hasattr(parent, 'open_video_file'):
            chosen_video = (preferred_video
                            if preferred_video in video_files else video_files[0])
            parent.open_video_file(os.path.join(folder_path, chosen_video))
            print(f"Playing video: {chosen_video}")
            

    except Exception as e:
        print(f"Error auto-loading folder content: {e}")
        import traceback
        traceback.print_exc()

def create_video_based_csv(output_dir, video_files, folder_name=None):
    """Create a CSV file with video clip names as the first column"""
    try:
        # Get folder name for CSV filename if not provided
        if folder_name is None:
            folder_name = os.path.basename(output_dir.rstrip('/\\'))
        
        csv_filename = f"{folder_name}_data.csv"
        csv_path = os.path.join(output_dir, csv_filename)
        
        # Check if CSV already exists (shouldn't, but just in case)
        if os.path.exists(csv_path):
            return csv_path
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Create CSV with video clip names as the first column, in natural
        # clip order ('Clip 2' before 'Clip 10') -- the label sheet's order.
        video_names = sorted((os.path.splitext(video)[0] for video in video_files),
                             key=natural_key)
        
        # Create default data structure with video names as the first column
        default_data = {
            'CLIP NAME': video_names,
            'HASH': "",
            'YARD LINE': 0,
            'PERSONNEL' : 0,
            'BACKFIELD' : "",
            'FIB/FSL' : "",
            'FRONT COUNT': "",
            'FRONT STRENGTH': "",
            'FRONT RELIABLE': "",
            'OFF FORM' : "",
            'TEMPLATE FORM': "",
            'TEMPLATE SCORE': "",
            'FORM VARIATION': "",
            'QB ALIGN': "",
            'SET': "",
            'WR SPLITS': "",
        }
        
        # Create DataFrame and save as CSV
        df = pd.DataFrame(default_data)
        df.to_csv(csv_path, index=False)
        
        print(f"Created CSV with {len(video_files)} video entries")
        return csv_path
        
    except Exception as e:
        print(f"Error creating video-based CSV: {e}")
        return None

def open_video_file(parent, video_path):
    """Open and play a video file using custom video widget"""
    # Set current video path for bounding box data loading
    from video import set_current_video_path
    set_current_video_path(parent, video_path)
    
    # The set_current_video_path function now handles loading the video into the custom widget
    # and setting up the progress slider and time label
    
    # Update button text to show it's ready to play
    parent.play_button.setText("▶")
    print(f"Video loaded: {video_path}")

