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
from video import create_video_dock, set_current_video_path
from fileAccess import create_file_dock
from dataSheet import create_data_sheet_dock
from virtualField import create_virtual_field_dock
from palette import get_light_palette, get_dark_palette

class MainWindow(QMainWindow):
    def __init__(self, dark_mode=False):
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
        self.set_menu_colors(dark_mode)

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
    
        # export_action = QAction("Export", self)
        # export_action.triggered.connect(self.export_data)
        # file_menu.addAction(export_action)

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
        
        
        # Load settings and default folder after docks are created
        self.load_settings()

        # Add ability to change palette theme
        theme_menu = menu_bar.addMenu("Theme")
        theme_menu.addAction("Light", lambda: self.apply_palette(get_light_palette(), False))
        theme_menu.addAction("Dark", lambda: self.apply_palette(get_dark_palette(), True))
        theme_menu.addAction("System", self.apply_system_palette)


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
                # Call the correct utility function to load the video,
                # which then handles loading the custom video widget and detection data.
                set_current_video_path(self, video_file)

    # def export_data(self):
    #     print("Exporting data... (stub)")
    
    
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
    
    def apply_palette(self, palette, dark_mode):
        """Apply palette and sync menu bar colors."""
        QApplication.instance().setPalette(palette)
        self.set_menu_colors(dark_mode)

    def apply_system_palette(self):
        """Follow OS theme and sync menu bar colors."""
        is_dark = darkdetect.isDark()
        palette = get_dark_palette() if is_dark else get_light_palette()
        self.apply_palette(palette, is_dark)

    def set_menu_colors(self, dark_mode):
        """Ensure menu bar titles contrast with current theme."""
        text_color = "white" if dark_mode else "black"
        self.menuBar().setStyleSheet(
            f"QMenuBar {{ color: {text_color}; }}\n"
            f"QMenuBar::item {{ color: {text_color}; }}\n"
            "QMenu { color: black; }\n"
            "QMenu::item:selected { background-color: #dcdcdc; }\n"
        )

    def load_settings(self):
        """Load application settings from file"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    settings = json.load(f)
                    default_folder = settings.get('default_folder', '')
                    if default_folder and os.path.exists(default_folder):
                        self.current_folder = default_folder
                        # Auto-load the default folder
                        self.load_default_folder()
        except Exception as e:
            print(f"Error loading settings: {e}")
    
    def save_settings(self):
        """Save application settings to file"""
        try:
            settings = {
                'default_folder': self.current_folder
            }
            with open(self.settings_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            print(f"Error saving settings: {e}")
    
    def set_default_folder(self):
        """Set the current folder as the default folder"""
        if self.current_folder and os.path.exists(self.current_folder):
            self.save_settings()
        else:
            print("No valid folder selected. Please open a folder first.")
    
    def clear_default_folder(self):
        """Clear the default folder setting"""
        self.current_folder = ""
        self.save_settings()
    
    def load_default_folder(self):
        """Load the default folder if it exists"""
        if self.current_folder and os.path.exists(self.current_folder):
            # Import the load_folder function from fileAccess
            from fileAccess import load_folder
            try:
                load_folder(self, self.current_folder, change_view=True)
            except Exception as e:
                print(f"Error loading default folder: {e}")


def apply_system_theme(app):
    if darkdetect.isDark():
        app.setPalette(get_dark_palette())
        return True
    else:
        app.setPalette(get_light_palette())
        return False


if __name__ == "__main__":
    # Set attribute BEFORE creating QApplication
    QApplication.setAttribute(Qt.AA_DontUseNativeDialogs, False)
    
    app = QApplication(sys.argv)
    
    # Remove Fusion style to allow native dialogs
    #app.setStyle(QStyleFactory.create("windows11"))
    
    system_dark = apply_system_theme(app)
    window = MainWindow(system_dark)
    window.show()
    
    # Set equal dock sizes after window is shown
    window.set_equal_dock_sizes()
    
    sys.exit(app.exec())