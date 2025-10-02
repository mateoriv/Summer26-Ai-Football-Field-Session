from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, 
    QDockWidget, QWidget, QVBoxLayout, QLabel, QStyleFactory

)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
import sys
import os
import json
import darkdetect
from video import create_video_dock
from fileAccess import create_file_dock
from dataSheet import create_data_sheet_dock
from virtualField import create_virtual_field_dock
from palette import get_light_palette, get_dark_palette

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Hudl AI Analysis")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.resize(1200, 800)
        
        # Store current folder path
        self.current_folder = ""
        
        # Settings file path - user-specific in home directory
        from pathlib import Path
        self.settings_file = str(Path.home() / ".hudl_ai_settings.json")

        # --- Menu Bar ---
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("File")

        open_folder_action = QAction("Open Folder", self)
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        
        # Add Set Default Folder action
        set_default_folder_action = QAction("Set Default Folder", self)
        set_default_folder_action.triggered.connect(self.set_default_folder)
        file_menu.addAction(set_default_folder_action)
        
        # Add Clear Default Folder action
        clear_default_folder_action = QAction("Clear Default Folder", self)
        clear_default_folder_action.triggered.connect(self.clear_default_folder)
        file_menu.addAction(clear_default_folder_action)
        
        file_menu.addSeparator()
    
        export_action = QAction("Export", self)
        export_action.triggered.connect(self.export_data)
        file_menu.addAction(export_action)

        close_action = QAction("Close", self)
        close_action.triggered.connect(self.close)
        file_menu.addAction(close_action)

        # Window Menu
        window_menu = menu_bar.addMenu("Window")

        # --- Dock Widgets ---
        self.video_dock = create_video_dock(self)
        self.file_dock = create_file_dock(self)
        self.data_dock = create_data_sheet_dock(self)
        self.virtual_dock = create_virtual_field_dock(self)

        # Add docks in 2x2 grid layout
        self.addDockWidget(Qt.TopDockWidgetArea, self.video_dock)
        self.addDockWidget(Qt.TopDockWidgetArea, self.file_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.data_dock)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.virtual_dock)

        # Create 2x2 grid layout
        self.splitDockWidget(self.video_dock, self.file_dock, Qt.Horizontal)
        self.splitDockWidget(self.data_dock, self.virtual_dock, Qt.Horizontal)
        
        # Set equal sizes for all dock widgets
        self.set_equal_dock_sizes()

        # Add dock visibility toggles to Window menu
        window_menu.addAction(self.video_dock.toggleViewAction())
        window_menu.addAction(self.file_dock.toggleViewAction())
        window_menu.addAction(self.data_dock.toggleViewAction())
        window_menu.addAction(self.virtual_dock.toggleViewAction())
        
        # Add scoreboard toggle action
        scoreboard_action = QAction("Toggle Scoreboard", self)
        scoreboard_action.triggered.connect(self.toggle_scoreboard)
        window_menu.addAction(scoreboard_action)
        
        # Load settings and default folder after docks are created
        self.load_settings()

        # Add ability to change palette theme
        window_menu = menu_bar.addMenu("Theme")
        window_menu.addAction("Light", lambda: app.setPalette(get_light_palette()))
        window_menu.addAction("Dark", lambda: app.setPalette(get_dark_palette()))
        window_menu.addAction("System", lambda: apply_system_theme(app))

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            # Set as current folder
            self.current_folder = folder
            # Call the file access method to load the folder
            from fileAccess import load_folder
            load_folder(self, folder, change_view=True)

    def open_video(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self, "Open Video", self.current_folder or "", "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv)"
        )
        if video_file:
            # Call the video method to open the file
            if hasattr(self, 'open_video_file'):
                self.open_video_file(video_file)

    def export_data(self):
        print("Exporting data... (stub)")
    
    def toggle_scoreboard(self):
        """Toggle scoreboard visibility in the virtual field dock"""
        if hasattr(self, 'scoreboard_widget'):
            if self.scoreboard_widget.isVisible():
                self.scoreboard_widget.hide()
            else:
                self.scoreboard_widget.show()
    
    def set_equal_dock_sizes(self):
        """Set all dock widgets to equal sizes in a 2x2 grid"""
        # Get the main window size
        main_size = self.size()
        half_width = main_size.width() // 2
        half_height = main_size.height() // 2
        
        # Resize all dock widgets to equal sizes
        self.video_dock.resize(half_width, half_height)
        self.file_dock.resize(half_width, half_height)
        self.data_dock.resize(half_width, half_height)
        self.virtual_dock.resize(half_width, half_height)
        
        # Ensure the layout is properly applied
        self.resizeDocks([self.video_dock, self.file_dock, self.data_dock, self.virtual_dock], 
                        [half_width, half_width, half_width, half_width], Qt.Horizontal)
        self.resizeDocks([self.video_dock, self.data_dock, self.file_dock, self.virtual_dock], 
                        [half_height, half_height, half_height, half_height], Qt.Vertical)
    
    def load_settings(self):
        """Load application settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    default_folder = settings.get('default_folder', '')
                    if default_folder and os.path.exists(default_folder):
                        self.current_folder = default_folder
                        print(f"📁 Loaded default folder: {default_folder}")
                        # Auto-load the default folder
                        self.load_default_folder()
                    else:
                        print("📁 No valid default folder found")
            else:
                print("📁 No settings file found")
        except Exception as e:
            print(f"❌ Error loading settings: {e}")
    
    def save_settings(self):
        """Save application settings to file"""
        try:
            settings = {
                'default_folder': self.current_folder
            }
            print(f"🔧 Saving to file: {self.settings_file}")
            print(f"🔧 Settings content: {settings}")
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
            print(f"💾 Settings saved: default_folder = {self.current_folder}")
        except Exception as e:
            print(f"❌ Error saving settings: {e}")
    
    def set_default_folder(self):
        """Set the current folder as the default folder"""
        print(f"🔍 Current folder: {self.current_folder}")
        if self.current_folder and os.path.exists(self.current_folder):
            print(f"💾 Saving settings for folder: {self.current_folder}")
            self.save_settings()
            print(f"✅ Default folder set to: {self.current_folder}")
        else:
            print("❌ No valid folder selected. Please open a folder first.")
    
    def clear_default_folder(self):
        """Clear the default folder setting"""
        self.current_folder = ""
        self.save_settings()
        print("🗑️ Default folder cleared")
    
    def load_default_folder(self):
        """Load the default folder if it exists"""
        if self.current_folder and os.path.exists(self.current_folder):
            print(f"📂 Attempting to load default folder: {self.current_folder}")
            # Import the load_folder function from fileAccess
            from fileAccess import load_folder
            try:
                load_folder(self, self.current_folder, change_view=True)
                print(f"✅ Successfully loaded default folder: {self.current_folder}")
            except Exception as e:
                print(f"❌ Error loading default folder: {e}")
        else:
            print("❌ No valid default folder to load")


def apply_system_theme(app):
    if darkdetect.isDark():
        app.setPalette(get_dark_palette())
    else:
        app.setPalette(get_light_palette())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle(QStyleFactory.create("Fusion"))
    apply_system_theme(app)
    window = MainWindow()
    window.show()
    
    # Set equal dock sizes after window is shown
    window.set_equal_dock_sizes()
    
    sys.exit(app.exec())