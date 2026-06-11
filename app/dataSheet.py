from PySide6.QtWidgets import QDockWidget, QTableView, QVBoxLayout, QWidget, QHeaderView, QAbstractItemView, QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex
from PySide6.QtGui import QFont
import pandas as pd
import os
import re

class CSVTableModel(QAbstractTableModel):
    def __init__(self, data=None, parent=None):
        super().__init__(parent)
        self._data = data if data is not None else pd.DataFrame()
        self.video_clip_column = None
        self.video_time_column = None
        self.empty_message = "No data available."
        self.has_unsaved_changes = False

    def rowCount(self, parent=QModelIndex()):
        if self._data.empty:
            return 1  # Show one row for empty message
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        if self._data.empty:
            return 1  # Show one column for empty message
        return len(self._data.columns)

    def flags(self, index):
        base = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if not self._data.empty:
            col = self._data.columns[index.column()].upper()
            if col not in ('CLIP NAME',):
                return base | Qt.ItemIsEditable
        return base

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole and not self._data.empty:
            self._data.iloc[index.row(), index.column()] = value
            self.dataChanged.emit(index, index, [role])
            self.has_unsaved_changes = True
            return True
        return False

    def save_to_csv(self, path):
        self._data.to_csv(path, index=False)
        self.has_unsaved_changes = False

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            if self._data.empty:
                # Show empty message in first cell
                if index.row() == 0 and index.column() == 0:
                    return self.empty_message
                return None
            return str(self._data.iloc[index.row(), index.column()])

        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            if self._data.empty:
                return "Data" if section == 0 else None
            return str(self._data.columns[section])
        return None

    def load_csv(self, csv_path):
        try:
            self.beginResetModel()
            self._data = pd.read_csv(csv_path)

            # Try to auto-detect video clip and time columns
            clip_col = None
            for col in self._data.columns:
                col_lower = col.lower()
                if 'clip' in col_lower or 'video' in col_lower:
                    self.video_clip_column = col
                    clip_col = col
                if 'time' in col_lower or 'timestamp' in col_lower:
                    self.video_time_column = col

            # Sort rows by the clip name column using natural/numeric order
            if clip_col and not self._data.empty:
                def _nat_key(val):
                    return [int(c) if c.isdigit() else c.lower()
                            for c in re.split(r'(\d+)', str(val))]
                self._data = self._data.iloc[
                    self._data[clip_col].map(_nat_key).argsort()
                ].reset_index(drop=True)

            self.endResetModel()
            self.has_unsaved_changes = False
            return True
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return False

    def get_video_info(self, row):
        """Get video file path and timestamp for the given row"""
        if self._data.empty or row >= len(self._data):
            return None, None
        
        video_file = None
        timestamp = None
        
        # Get video file from detected column or first column that looks like a file path
        if self.video_clip_column:
            video_file = self._data.iloc[row][self.video_clip_column]
        else:
            for col in self._data.columns:
                value = str(self._data.iloc[row][col])
                if any(ext in value.lower() for ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']):
                    video_file = value
                    break
        
        # Get timestamp if available
        if self.video_time_column:
            timestamp = self._data.iloc[row][self.video_time_column]
        
        return video_file, timestamp


def create_data_sheet_dock(parent):
    dock = QDockWidget("Data Sheet", parent)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(QDockWidget.DockWidgetMovable)

    # Set custom title bar with process button
    dock.setTitleBarWidget(create_data_sheet_title_bar(dock, parent))

    # Main widget
    main_widget = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    
    # Create table view
    parent.tableView = QTableView()
    parent.csv_model = CSVTableModel()
    parent.tableView.setModel(parent.csv_model)
    
    # Configure table view
    parent.tableView.setSelectionBehavior(QAbstractItemView.SelectRows)
    parent.tableView.setSelectionMode(QAbstractItemView.SingleSelection)
    parent.tableView.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
    parent.tableView.setAlternatingRowColors(True)
    
    # Connect selection change to play corresponding video
    parent.tableView.selectionModel().selectionChanged.connect(
        lambda: on_row_selected(parent)
    )

    # Enable Save button when edits are made, disable when fresh data loads
    parent.csv_model.dataChanged.connect(
        lambda *_: parent.save_stats_btn.setEnabled(parent.csv_model.has_unsaved_changes)
    )
    parent.csv_model.modelReset.connect(
        lambda: parent.save_stats_btn.setEnabled(False)
    )
    
    layout.addWidget(parent.tableView)
    main_widget.setLayout(layout)
    dock.setWidget(main_widget)
    
    # Add method to parent
    parent.load_csv_file = lambda csv_path: load_csv_file(parent, csv_path)
    
    return dock

def load_csv_file(parent, csv_path):
    """Load a CSV file into the data sheet"""
    if parent.csv_model.load_csv(csv_path):
        parent.tableView.resizeColumnsToContents()
        parent.current_csv_path = csv_path  # Store current CSV path for refresh
        return True
    return False

def on_row_selected(parent):
    """Handle row selection to play corresponding video clip"""
    selected_indexes = parent.tableView.selectionModel().selectedRows()
    if not selected_indexes:
        return
    
    row = selected_indexes[0].row()
    video_file, timestamp = parent.csv_model.get_video_info(row)

    if not video_file.lower().endswith(".mp4"):
            video_file += ".mp4"

    if video_file:
        # Construct full path if the video file is relative
        if not os.path.isabs(video_file) and hasattr(parent, 'current_folder'):
            video_file = os.path.join(parent.current_folder, video_file)
        
        if os.path.exists(video_file):
            play_video_clip(parent, video_file, timestamp)
        else:
            print(f"Video file not found: {video_file}")

def play_video_clip(parent, video_path, timestamp=None):
    """Play video at specific timestamp if available using custom video widget"""
    # Set current video path for bounding box data loading
    from video import set_current_video_path
    set_current_video_path(parent, video_path)
    
    # The set_current_video_path function now handles loading the video into the custom widget
    # and setting up the progress slider and time label
    
    # Seek to timestamp if provided (for custom video widget)
    if timestamp and hasattr(parent, 'custom_video'):
        try:
            # Handle different timestamp formats
            if isinstance(timestamp, (int, float)):
                # Assume seconds
                frame_number = int(timestamp * 30)  # Assume 30 FPS
            elif isinstance(timestamp, str):
                # Handle time strings like "00:01:30" or "1m30s"
                if ':' in timestamp:
                    parts = timestamp.split(':')
                    if len(parts) == 3:  # HH:MM:SS
                        hours, minutes, seconds = map(float, parts)
                        frame_number = int((hours * 3600 + minutes * 60 + seconds) * 30)
                    elif len(parts) == 2:  # MM:SS
                        minutes, seconds = map(float, parts)
                        frame_number = int((minutes * 60 + seconds) * 30)
                    else:
                        frame_number = 0
                else:
                    frame_number = 0
            else:
                frame_number = 0
                
            # Set the frame in the custom video widget
            parent.custom_video.current_frame = frame_number
            parent.custom_video.update()
            
            # Update virtual field with current frame
            parent.custom_video.update_virtual_field()
            
            # Update progress slider
            if hasattr(parent, 'custom_video_total_frames'):
                progress = int((frame_number / parent.custom_video_total_frames) * 100)
                parent.progress_slider.slider.setValue(progress)
                
        except (ValueError, TypeError):
            frame_number = 0
    
    # Update button text to show it's ready to play
    parent.play_button.setText("▶")

def create_data_sheet_title_bar(dock, parent):
    """Create a custom title bar for the data sheet dock widget with process button"""
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
    
    # Export CSV button (left of title)
    export_csv_btn = QPushButton("Export CSV")
    export_csv_btn.setFixedSize(70, 20)
    export_csv_btn.setStyleSheet("""
        QPushButton {
            background-color: #6c757d;
            border: none;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 10px;
        }
        QPushButton:hover {
            background-color: #5a6268;
        }
        QPushButton:pressed {
            background-color: #545b62;
        }
    """)
    export_csv_btn.setToolTip("Export CSV to Selected Folder")
    export_csv_btn.clicked.connect(lambda: export_csv_to_folder(parent))
    layout.addWidget(export_csv_btn)
    
    # Left spacer to center the title
    left_spacer = QWidget()
    left_spacer.setFixedWidth(60)  # Space for buttons on the right
    layout.addWidget(left_spacer)
    
    # Title label (centered)
    title_label = QLabel("Data Sheet")
    title_label.setFont(QFont("Arial", 10, QFont.Bold))
    title_label.setAlignment(Qt.AlignCenter)
    layout.addWidget(title_label)
    
    # Right spacer to balance the left spacer
    right_spacer = QWidget()
    right_spacer.setFixedWidth(60)  # Space for buttons on the right
    layout.addWidget(right_spacer)
    
    # Save stats button
    save_stats_btn = QPushButton("Save")
    save_stats_btn.setFixedSize(50, 20)
    save_stats_btn.setEnabled(False)
    save_stats_btn.setToolTip("Save stat edits to CSV")
    save_stats_btn.setStyleSheet("""
        QPushButton {
            background-color: #28a745;
            border: none;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 10px;
        }
        QPushButton:hover {
            background-color: #218838;
        }
        QPushButton:pressed {
            background-color: #1e7e34;
        }
        QPushButton:disabled {
            background-color: #444444;
            color: #888888;
        }
    """)
    save_stats_btn.clicked.connect(lambda: save_stats_edits(parent))
    parent.save_stats_btn = save_stats_btn
    layout.addWidget(save_stats_btn)

    # Process button
    process_btn = QPushButton("Process")
    process_btn.setFixedSize(60, 20)
    process_btn.setStyleSheet("""
        QPushButton {
            background-color: #0078d4;
            border: none;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 10px;
        }
        QPushButton:hover {
            background-color: #106ebe;
        }
        QPushButton:pressed {
            background-color: #005a9e;
        }
    """)
    process_btn.setToolTip("Process Selected Video")
    process_btn.clicked.connect(lambda: process_selected_video(parent))
    layout.addWidget(process_btn)
    
    # Batch Process button
    batch_process_btn = QPushButton("Batch")
    batch_process_btn.setFixedSize(60, 20)
    batch_process_btn.setStyleSheet("""
        QPushButton {
            background-color: #28a745;
            border: none;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-weight: bold;
            font-size: 10px;
        }
        QPushButton:hover {
            background-color: #218838;
        }
        QPushButton:pressed {
            background-color: #1e7e34;
        }
    """)
    batch_process_btn.setToolTip("Batch Process All Videos in Folder")
    batch_process_btn.clicked.connect(lambda: batch_process_videos(parent))
    layout.addWidget(batch_process_btn)
    
    title_bar.setLayout(layout)
    return title_bar

def process_selected_video(parent):
    """Process the currently selected video file"""
    import os
    from PySide6.QtWidgets import QMessageBox
    from processingDialog import ProcessingDialog
    
    # Stop video playback if it's currently playing
    if hasattr(parent, 'custom_video') and parent.custom_video.is_playing:
        parent.custom_video.toggle_playback()  # This will stop playback
        parent.play_button.setText("▶")  # Update button to show play state
        print("Video playback stopped before processing")
    
    # Get the currently selected row
    selected_indexes = parent.tableView.selectionModel().selectedRows()
    if not selected_indexes:
        QMessageBox.warning(parent, "No Selection", "Please select a video row to process.")
        return
    
    # Get the video file path from the selected row
    try:
        video_file, timestamp = parent.csv_model.get_video_info(selected_indexes[0].row())
        
        # Construct full path to video file (same logic as play_video_clip)
        if not os.path.isabs(video_file) and hasattr(parent, 'current_folder'):
            video_path = os.path.join(parent.current_folder, video_file)
        else:
            video_path = video_file

        if not video_path.lower().endswith(".mp4"):
            video_path += ".mp4"

        print(f"Processing video: {video_path}")
        
        if not os.path.exists(video_path):
            QMessageBox.warning(parent, "File Not Found", f"Video file not found: {video_path}")
            return
        
        # Show the modal processing dialog
        # Use cache directory outside of app folder (in project root)
        dialog = ProcessingDialog(parent, video_path, parent.current_folder, cache_dir="cache")
        dialog.exec()
            
    except Exception as e:
        QMessageBox.critical(parent, "Error", f"An error occurred: {str(e)}")
        print(f"Error in process_selected_video: {str(e)}")

def batch_process_videos(parent):
    """Open batch processing dialog for all videos in current folder"""
    # Stop video playback if it's currently playing
    if hasattr(parent, 'custom_video') and parent.custom_video.is_playing:
        parent.custom_video.toggle_playback()  # This will stop playback
        parent.play_button.setText("▶")  # Update button to show play state
        print("Video playback stopped before batch processing")
    
    from batchProcessingDialog import BatchProcessingDialog
    dialog = BatchProcessingDialog(parent)
    dialog.exec()

def save_stats_edits(parent):
    """Save manually edited stats back to the CSV file on disk"""
    if not hasattr(parent, 'current_csv_path') or not parent.current_csv_path:
        QMessageBox.warning(parent, "No File", "No CSV file is currently loaded.")
        return
    try:
        parent.csv_model.save_to_csv(parent.current_csv_path)
        parent.save_stats_btn.setEnabled(False)
        print(f"Stats saved to: {parent.current_csv_path}")
    except Exception as e:
        QMessageBox.critical(parent, "Save Failed", f"Could not save CSV:\n{str(e)}")
        print(f"Error saving stats: {str(e)}")

def export_csv_to_folder(parent):
    """Export the current CSV data to a selected folder"""
    # Check if there's data to export
    if not hasattr(parent, 'csv_model') or parent.csv_model._data.empty:
        QMessageBox.warning(parent, "No Data", "No data available to export. Please load a video folder first.")
        return
    
    # Get the default folder from file access tree view root path
    default_folder = ""
    if hasattr(parent, 'tree_model'):
        root_path = parent.tree_model.rootPath()
        if root_path and os.path.exists(root_path):
            default_folder = root_path
    # Fallback to current_folder if tree_model root path is not available
    if not default_folder and hasattr(parent, 'current_folder') and parent.current_folder:
        default_folder = parent.current_folder
    
    # Open folder selection dialog with default folder
    folder = QFileDialog.getExistingDirectory(parent, "Select Folder to Export CSV", default_folder)
    if not folder:
        return  # User cancelled
    
    try:
        # Get the current CSV filename or use a default name
        if hasattr(parent, 'current_csv_path') and parent.current_csv_path:
            csv_filename = os.path.basename(parent.current_csv_path)
        else:
            # Generate a default filename based on current folder or timestamp
            if hasattr(parent, 'current_folder') and parent.current_folder:
                folder_name = os.path.basename(parent.current_folder.rstrip('/\\'))
                csv_filename = f"{folder_name}_data.csv"
            else:
                import time
                csv_filename = f"exported_data_{int(time.time())}.csv"
        
        # Construct full path
        csv_path = os.path.join(folder, csv_filename)
        
        # Check if file already exists
        if os.path.exists(csv_path):
            reply = QMessageBox.question(
                parent, 
                "File Exists", 
                f"The file '{csv_filename}' already exists in this folder. Do you want to overwrite it?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
        
        # Export the data
        parent.csv_model._data.to_csv(csv_path, index=False)
        
        QMessageBox.information(
            parent, 
            "Export Successful", 
            f"CSV file exported successfully to:\n{csv_path}"
        )
        print(f"CSV exported to: {csv_path}")
        
    except Exception as e:
        QMessageBox.critical(
            parent, 
            "Export Failed", 
            f"An error occurred while exporting the CSV:\n{str(e)}"
        )
        print(f"Error exporting CSV: {str(e)}")
