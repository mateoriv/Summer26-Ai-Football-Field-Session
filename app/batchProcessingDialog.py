#!/usr/bin/env python3
"""
Batch Processing Dialog
Modal dialog for batch processing multiple videos with progress tracking
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QTextEdit, QPushButton, QFrame, QWidget, QFileDialog, QListWidget, QListWidgetItem, QSpinBox
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QProcess
from PySide6.QtGui import QFont, QIcon, QTextCursor
import subprocess
import os
import sys
import json
import time
from pathlib import Path
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import multiprocessing

# Required for ProcessPoolExecutor on Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

def process_single_video_standalone(video_path, video_folder, output_dir="cache"):
    """Standalone function to process a single video - used with ProcessPoolExecutor"""
    try:
        video_name = Path(video_path).stem
        
        # If output_dir is relative, make it relative to project root, not app directory
        if not os.path.isabs(output_dir):
            # Get project root (parent of app directory)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, output_dir)
        else:
            output_dir = os.path.abspath(output_dir)
        
        # Ensure video folder exists in output directory
        os.makedirs(f"{output_dir}/{video_folder}", exist_ok=True)
        os.makedirs(f"{output_dir}/{video_folder}/players", exist_ok=True)
        os.makedirs(f"{output_dir}/{video_folder}/yard_markers", exist_ok=True)
        os.makedirs(f"{output_dir}/{video_folder}/correspondence", exist_ok=True)
        os.makedirs(f"{output_dir}/{video_folder}/virtual_field", exist_ok=True)
        os.makedirs(f"{output_dir}/{video_folder}/results", exist_ok=True)
        
        # Step 1: Player Detection
        detection_output = f"{output_dir}/{video_folder}/players/{video_name}_detection.json"
        detection_cmd = [
            "python3", "scripts/playerDetection.py", 
            "--video", video_path, 
            "--output", detection_output
        ]
        
        if not _run_command_standalone(detection_cmd, "Player Detection"):
            return False
        
        # Step 2: Yard Marker Detection
        yard_marker_output = f"{output_dir}/{video_folder}/yard_markers/{video_name}_yard_markers.json"
        yard_marker_cmd = [
            "python3", "scripts/yardMarkerDetection.py",
            "--video", video_path,
            "--output", yard_marker_output
        ]
        
        if not _run_command_standalone(yard_marker_cmd, "Yard Marker Detection"):
            return False
        
        # # Step 3: Auto Correspondence Points
        # correspondence_output = f"{output_dir}/{video_folder}/correspondence/{video_name}_correspondence.json"
        # correspondence_cmd = [
        #     "python3", "scripts/autoCorrespondancePoints.py",
        #     "--detection-json", yard_marker_output,
        #     "--output", correspondence_output,
        #     "--confidence", "0.7"
        # ]
        
        # if not _run_command_standalone(correspondence_cmd, "Correspondence Points Generation"):
        #     return False
        
        # # Step 4: Homography Transformation
        # if os.path.exists(correspondence_output):
        #     homography_output = f"{output_dir}/{video_folder}/{video_name}_homography.json"
        #     homography_cmd = [
        #         "python3", "scripts/homographyTransform.py",
        #         "--input", detection_output,
        #         "--correspondence", correspondence_output,
        #         "--output", homography_output
        #     ]
            
        #     if not _run_command_standalone(homography_cmd, "Homography Transformation"):
        #         return False
        # else:
        #     print(f"No correspondence points found for {video_name}, skipping homography transformation")
        
        # # Step 5: Render Field Video
        # if os.path.exists(homography_output):
        #     field_video_output = f"{output_dir}/{video_folder}/virtual_field/{video_name}_field.mp4"
        #     render_cmd = [
        #         "python3", "scripts/renderFieldVideo.py",
        #         "--input", homography_output,
        #         "--output", field_video_output
        #     ]
            
        #     if not _run_command_standalone(render_cmd, "Field Video Rendering"):
        #         return False
        # else:
        #     print(f"No homography data found for {video_name}, skipping field video rendering")
        
        return True
        
    except Exception as e:
        print(f"Error processing {video_name}: {str(e)}")
        return False

def _run_command_standalone(cmd, step_name):
    """Standalone function to run a command - used with ProcessPoolExecutor"""
    # Get the correct working directory (project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    try:
        # Start the process with correct working directory
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=project_root,
            env=env
        )
        
        # Wait for process to complete
        return_code = process.wait()
        
        if return_code == 0:
            return True
        else:
            print(f"{step_name} failed with return code {return_code}")
            return False
            
    except Exception as e:
        print(f"Error running {step_name}: {str(e)}")
        return False

class BatchProcessingWorker(QThread):
    """Worker thread for batch processing multiple videos"""
    progress_updated = Signal(int, str)  # progress percentage, status message
    output_received = Signal(str)  # terminal output
    video_completed = Signal(str, bool)  # video name, success status
    batch_completed = Signal(dict)  # results
    batch_failed = Signal(str)  # error message
    
    def __init__(self, video_paths, output_dir="cache", max_workers=2):
        super().__init__()
        self.video_paths = video_paths
        # If output_dir is relative, make it relative to project root, not app directory
        if not os.path.isabs(output_dir):
            # Get project root (parent of app directory)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.output_dir = os.path.join(project_root, output_dir)
        else:
            self.output_dir = os.path.abspath(output_dir)
        self.max_workers = max_workers
        self.process = None
        self.is_cancelled = False
        self.current_video_index = 0
        self.total_videos = len(video_paths)
        self.completed_videos = 0
        self.failed_videos = 0
        self.results = []
        self.lock = threading.Lock()
        
    def run(self):
        """Run batch processing for all videos with parallel processing"""
        try:
            self.output_received.emit(f"Starting batch processing of {self.total_videos} videos")
            self.output_received.emit(f"Using {self.max_workers} parallel workers")
            self.output_received.emit("-" * 50)
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Use ProcessPoolExecutor for true parallel processing
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all video processing tasks
                future_to_video = {}
                for i, video_path in enumerate(self.video_paths):
                    if self.is_cancelled:
                        return
                    
                    video_name = Path(video_path).stem
                    video_folder = os.path.basename(os.path.dirname(video_path))
                    
                    # Submit the processing task
                    future = executor.submit(process_single_video_standalone, video_path, video_folder, self.output_dir)
                    future_to_video[future] = (i, video_path, video_name)
                
                # Process completed tasks as they finish
                for future in as_completed(future_to_video):
                    if self.is_cancelled:
                        return
                    
                    i, video_path, video_name = future_to_video[future]
                    
                    try:
                        success = future.result()
                        
                        with self.lock:
                            if success:
                                self.completed_videos += 1
                                self.video_completed.emit(video_name, True)
                                self.output_received.emit(f"✓ {video_name} completed successfully")
                            else:
                                self.failed_videos += 1
                                self.video_completed.emit(video_name, False)
                                self.output_received.emit(f"✗ {video_name} failed")
                            
                            # Add to results
                            self.results.append({
                                "video_path": video_path,
                                "video_name": video_name,
                                "success": success,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            # Update progress
                            total_processed = self.completed_videos + self.failed_videos
                            overall_progress = int((total_processed / self.total_videos) * 100)
                            self.progress_updated.emit(overall_progress, f"Processed {total_processed}/{self.total_videos} videos")
                            
                    except Exception as e:
                        with self.lock:
                            self.failed_videos += 1
                            self.video_completed.emit(video_name, False)
                            self.output_received.emit(f"✗ {video_name} failed with error: {str(e)}")
                            
                            self.results.append({
                                "video_path": video_path,
                                "video_name": video_name,
                                "success": False,
                                "error": str(e),
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                            })
            
            # Final progress update
            self.progress_updated.emit(100, "Batch processing completed!")
            self.output_received.emit(f"\nBatch processing completed!")
            self.output_received.emit(f"Successfully processed: {self.completed_videos}/{self.total_videos} videos")
            self.output_received.emit(f"Failed: {self.failed_videos}/{self.total_videos} videos")
            
            # Return results
            final_results = {
                "total_videos": self.total_videos,
                "completed_videos": self.completed_videos,
                "failed_videos": self.failed_videos,
                "results": self.results,
                "status": "completed"
            }
            
            # Save results to JSON
            results_file = f"{self.output_dir}/batch_results_{int(time.time())}.json"
            with open(results_file, 'w') as f:
                json.dump(final_results, f, indent=2)
            
            self.output_received.emit(f"Results saved to: {results_file}")
            self.batch_completed.emit(final_results)
            
        except Exception as e:
            error_msg = f"Error during batch processing: {str(e)}"
            self.output_received.emit(f"ERROR: {error_msg}")
            self.batch_failed.emit(error_msg)
    
    
    def cancel(self):
        """Cancel the batch processing"""
        self.is_cancelled = True
        if self.process:
            self.process.terminate()


class BatchProcessingDialog(QDialog):
    """Modal dialog for batch processing multiple videos"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.worker = None
        self.video_paths = []
        self.parent_window = parent
        self.setup_ui()
        self.load_current_folder()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Batch Process Videos")
        self.setModal(True)
        self.setFixedSize(800, 700)
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        
        # Main layout
        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header_frame = QFrame()
        header_frame.setStyleSheet("""
            QFrame {
                background-color: #2b2b2b;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        header_layout = QVBoxLayout()
        header_layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title_label = QLabel("Batch Process Videos")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: white; margin-bottom: 5px;")
        header_layout.addWidget(title_label)
        
        # Description
        desc_label = QLabel("Process all videos in the currently selected folder automatically.")
        desc_label.setFont(QFont("Arial", 10))
        desc_label.setStyleSheet("color: #cccccc;")
        desc_label.setWordWrap(True)
        header_layout.addWidget(desc_label)
        
        header_frame.setLayout(header_layout)
        layout.addWidget(header_frame)
        
      
        # Progress section
        progress_frame = QFrame()
        progress_frame.setStyleSheet("""
            QFrame {
                background-color: #1e1e1e;
                border-radius: 8px;
                padding: 15px;
            }
        """)
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(15, 15, 15, 15)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #555555;
                border-radius: 5px;
                text-align: center;
                background-color: #2b2b2b;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 3px;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("Ready to process")
        self.status_label.setFont(QFont("Arial", 10))
        self.status_label.setStyleSheet("color: #cccccc; margin-top: 5px;")
        progress_layout.addWidget(self.status_label)
        
        progress_frame.setLayout(progress_layout)
        layout.addWidget(progress_frame)
        
        # Terminal output
        terminal_frame = QFrame()
        terminal_frame.setStyleSheet("""
            QFrame {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 8px;
                padding: 10px;
            }
        """)
        terminal_layout = QVBoxLayout()
        terminal_layout.setContentsMargins(10, 10, 10, 10)
        
        # Terminal label
        terminal_label = QLabel("Processing Output:")
        terminal_label.setFont(QFont("Arial", 10, QFont.Bold))
        terminal_label.setStyleSheet("color: #cccccc; margin-bottom: 5px;")
        terminal_layout.addWidget(terminal_label)
        
        # Terminal output
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setFont(QFont("Consolas", 10))
        self.terminal_output.setStyleSheet("""
            QTextEdit {
                background-color: #0d1117;
                color: #ffffff;
                border: 1px solid #30363d;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 10px;
                line-height: 1.4;
            }
            QTextEdit:focus {
                border: 1px solid #58a6ff;
            }
        """)
        terminal_layout.addWidget(self.terminal_output)
        
        terminal_frame.setLayout(terminal_layout)
        layout.addWidget(terminal_frame)
        
        # Buttons
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self.start_button = QPushButton("Start Processing")
        self.start_button.setFixedSize(140, 30)
        self.start_button.setStyleSheet("""
            QPushButton {
                background-color: #28a745;
                border: none;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #218838;
            }
            QPushButton:pressed {
                background-color: #1e7e34;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
        """)
        self.start_button.clicked.connect(self.start_processing)
        self.start_button.setEnabled(False)
        button_layout.addWidget(self.start_button)
        
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setFixedSize(80, 30)
        self.cancel_button.setStyleSheet("""
            QPushButton {
                background-color: #dc3545;
                border: none;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #c82333;
            }
            QPushButton:pressed {
                background-color: #bd2130;
            }
        """)
        self.cancel_button.clicked.connect(self.cancel_processing)
        button_layout.addWidget(self.cancel_button)
        
        self.close_button = QPushButton("Close")
        self.close_button.setFixedSize(80, 30)
        self.close_button.setStyleSheet("""
            QPushButton {
                background-color: #6c757d;
                border: none;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #5a6268;
            }
            QPushButton:pressed {
                background-color: #545b62;
            }
        """)
        self.close_button.clicked.connect(self.accept)
        self.close_button.setVisible(False)
        button_layout.addWidget(self.close_button)
        
        layout.addLayout(button_layout)
        self.setLayout(layout)
    
    def load_current_folder(self):
        """Load videos from the currently selected folder"""
        if not hasattr(self.parent_window, 'current_folder') or not self.parent_window.current_folder:
            self.start_button.setEnabled(False)
            self.status_label.setText("No folder selected. Please open a folder first.")
            return
        
        folder = self.parent_window.current_folder
        self.folder_path = folder
        
        # Find all video files in the folder
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
        self.video_paths = []
        
        for ext in video_extensions:
            pattern = os.path.join(folder, f"*{ext}")
            self.video_paths.extend(glob.glob(pattern))
            pattern = os.path.join(folder, f"*{ext.upper()}")
            self.video_paths.extend(glob.glob(pattern))
        
        # Enable start button if videos found
        if self.video_paths:
            self.start_button.setEnabled(True)
            self.status_label.setText(f"Found {len(self.video_paths)} videos ready to process")
        else:
            self.start_button.setEnabled(False)
            self.status_label.setText("No video files found in current folder")
    
    def start_processing(self):
        """Start batch processing"""
        if not self.video_paths:
            return
        
        # Get the number of parallel workers from the spinbox
        max_workers = 100
        
        self.worker = BatchProcessingWorker(self.video_paths, max_workers=max_workers)
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.output_received.connect(self.add_output)
        self.worker.video_completed.connect(self.video_completed)
        self.worker.batch_completed.connect(self.batch_completed)
        self.worker.batch_failed.connect(self.batch_failed)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        
        self.worker.start()
    
    def update_progress(self, percentage, status):
        """Update progress bar and status"""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(status)
    
    def add_output(self, text):
        """Add text to terminal output"""
        self.terminal_output.append(text)
        # Auto-scroll to bottom
        cursor = self.terminal_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.terminal_output.setTextCursor(cursor)
    
    def video_completed(self, video_name, success):
        """Handle individual video completion"""
        status = "✓ Completed" if success else "✗ Failed"
        self.add_output(f"{status}: {video_name}")
    
    def batch_completed(self, results):
        """Handle batch processing completion"""
        self.progress_bar.setValue(100)
        self.status_label.setText("Batch processing completed!")
        
        # Show results
        self.add_output(f"\nBatch processing completed!")
        self.add_output(f"Successfully processed: {results['completed_videos']}/{results['total_videos']} videos")
        self.add_output(f"Failed: {results['failed_videos']}/{results['total_videos']} videos")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.start_button.setVisible(False)
        self.close_button.setVisible(True)
    
    def batch_failed(self, error_message):
        """Handle batch processing failure"""
        self.status_label.setText("Batch processing failed!")
        self.add_output(f"\nBatch processing failed: {error_message}")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.start_button.setVisible(False)
        self.close_button.setVisible(True)
    
    def cancel_processing(self):
        """Cancel batch processing"""
        if self.worker:
            self.worker.cancel()
            self.worker.wait()
        
        self.status_label.setText("Batch processing cancelled")
        self.add_output("\nBatch processing cancelled by user")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.start_button.setVisible(False)
        self.close_button.setVisible(True)
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        if self.worker and self.worker.isRunning():
            self.cancel_processing()
        event.accept()
