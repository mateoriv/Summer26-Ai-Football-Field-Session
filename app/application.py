from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, 
    QDockWidget, QWidget, QVBoxLayout, QLabel, QStyleFactory

)
from PySide6.QtGui import QAction
from PySide6.QtCore import Qt
import sys
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

        # --- Menu Bar ---
        menu_bar = self.menuBar()

        # File Menu
        file_menu = menu_bar.addMenu("File")

        open_folder_action = QAction("Open Folder", self)
        open_folder_action.triggered.connect(self.open_folder)
        file_menu.addAction(open_folder_action)
        
        # Add Open Video action
        open_video_action = QAction("Open Video", self)
        open_video_action.triggered.connect(self.open_video)
        file_menu.addAction(open_video_action)

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

        # Add ability to change palette theme
        window_menu = menu_bar.addMenu("Theme")
        window_menu.addAction("Light", lambda: app.setPalette(get_light_palette()))
        window_menu.addAction("Dark", lambda: app.setPalette(get_dark_palette()))
        window_menu.addAction("System", lambda: apply_system_theme(app))

    def create_dock(self, title):
        dock = QDockWidget(title, self)
        dock.setAllowedAreas(Qt.AllDockWidgetAreas)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetClosable)
        dock.setTitleBarWidget(QWidget())

        widget = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{title} content here"))
        widget.setLayout(layout)

        widget.setStyleSheet("""
            QWidget {
                border: 2px solid #888;
            }
        """)

        dock.setWidget(widget)
        return dock

    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Open Folder")
        if folder:
            # Call the file access method to load the folder
            if hasattr(self, 'load_folder'):
                self.load_folder(folder)

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