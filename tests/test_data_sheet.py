#!/usr/bin/env python3
"""
Test script to verify the data sheet header bar changes
"""

import sys
from PySide6.QtWidgets import QApplication
from app.application import MainWindow

def test_data_sheet_header():
    """Test the data sheet with increased header bar"""
    app = QApplication(sys.argv)
    window = MainWindow()
    
    # Show the window
    window.show()
    
    # Print information about the data sheet dock
    data_dock = window.data_dock
    print(f"Data Sheet Dock Title: {data_dock.windowTitle()}")
    print(f"Data Sheet Dock Features: {data_dock.features()}")
    
    # Check if custom title bar is set
    title_bar = data_dock.titleBarWidget()
    if title_bar:
        print(f"Custom Title Bar Height: {title_bar.height()}")
        print("✓ Custom title bar is active")
    else:
        print("✗ Custom title bar not found")
    
    print("\nApplication is running. Check the Data Sheet dock for the increased header bar size.")
    print("The header bar should now be 50px tall instead of the default ~25px.")
    
    return app.exec()

if __name__ == "__main__":
    sys.exit(test_data_sheet_header())
