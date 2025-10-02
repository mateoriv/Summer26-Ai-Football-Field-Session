#!/usr/bin/env python3
"""
Processing Dialog
Modal dialog for video processing with progress tracking and terminal output
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QTextEdit, QPushButton, QFrame, QWidget
)
from PySide6.QtCore import Qt, QThread, QTimer, Signal, QProcess
from PySide6.QtGui import QFont, QIcon, QTextCursor
import subprocess
import os
import sys
import json
import time
from pathlib import Path

class ProcessingWorker(QThread):
    """Worker thread for processing video"""
    progress_updated = Signal(int, str)  # progress percentage, status message
    output_received = Signal(str)  # terminal output
    step_completed = Signal(str, bool)  # step name, success status
    processing_completed = Signal(dict)  # results
    processing_failed = Signal(str)  # error message
    
    def __init__(self, video_path, video_folder, output_dir="cache"):
        super().__init__()
        self.video_path = video_path
        self.video_folder =  os.path.basename(video_folder)
        self.output_dir = output_dir
        self.process = None
        self.is_cancelled = False
        self.current_step = 0  # 0: detection, 1: yard markers, 2: homography, 3: rendering
        self.video_name = None
        self.detection_output = None
        self.homography_output = None
        print(os.path.basename(self.video_folder))
        print(video_folder)
        
    def run(self):
        """Run the current step of the video processing pipeline"""
        try:
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir + "/" + self.video_folder, exist_ok=True)
            
            # Get video filename without extension
            if not self.video_name:
                self.video_name = Path(self.video_path).stem
                self.output_received.emit(f"Processing video: {self.video_path}")
                self.output_received.emit(f"Output directory: {self.output_dir}")
                self.output_received.emit("-" * 50)
            
            if self.current_step == 0:
                # Step 1: Player Detection
                self.progress_updated.emit(0, "Step 1: Initializing player detection...")
                self.output_received.emit("Step 1: Running player detection...")
                
                self.detection_output = f"{self.output_dir}/{self.video_folder}/players/{self.video_name}_detection.json"
                detection_cmd = [
                    "python3", "Scripts/playerDetection.py", 
                    "--video", self.video_path, 
                    "--output", self.detection_output
                ]
                
                if self.is_cancelled:
                    return
                    
                result = self._run_command(detection_cmd, "Player Detection", 0, 100)
                self.output_received.emit(f"DEBUG: Player Detection result: {result}")
                if result:
                    self.output_received.emit("DEBUG: About to emit step_completed signal for Player Detection")
                    self.step_completed.emit("Player Detection", True)
                    self.output_received.emit("DEBUG: step_completed signal emitted for Player Detection")
                else:
                    self.output_received.emit("DEBUG: About to emit step_completed signal for Player Detection (failed)")
                    self.step_completed.emit("Player Detection", False)
                    self.output_received.emit("DEBUG: step_completed signal emitted for Player Detection (failed)")
                    
            elif self.current_step == 1:
                # Step 2: Yard Marker Detection
                self.progress_updated.emit(0, "Step 2: Initializing yard marker detection...")
                self.output_received.emit("Step 2: Running yard marker detection...")
                
                yard_marker_output = f"{self.output_dir}/{self.video_folder}/yard_markers/{self.video_name}_yard_markers.json"
                yard_marker_cmd = [
                    "python3", "Scripts/yardMarkerDetection.py",
                    "--video", self.video_path,
                    "--output", yard_marker_output
                ]
                
                if self.is_cancelled:
                    return
                    
                result = self._run_command(yard_marker_cmd, "Yard Marker Detection", 0, 100)
                if result:
                    self.step_completed.emit("Yard Marker Detection", True)
                else:
                    self.step_completed.emit("Yard Marker Detection", False)
                    
            elif self.current_step == 2:
                # Step 3: Auto Correspondence Points
                self.progress_updated.emit(0, "Step 3: Initializing correspondence points generation...")
                self.output_received.emit("Step 3: Generating correspondence points from yard markers...")
                
                yard_marker_output = f"{self.output_dir}/{self.video_folder}/yard_markers/{self.video_name}_yard_markers.json"
                correspondence_output = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
                
                correspondence_cmd = [
                    "python3", "Scripts/autoCorrespondancePoints.py",
                    "--detection-json", yard_marker_output,
                    "--output", correspondence_output,
                    "--confidence", "0.7"
                ]
                
                if self.is_cancelled:
                    return
                    
                result = self._run_command(correspondence_cmd, "Correspondence Points Generation", 0, 100)
                if result:
                    # Update virtual field with yard marker dots
                    from virtualField import update_field_with_correspondence_points
                    update_field_with_correspondence_points(self.parent(), correspondence_output)
                    self.step_completed.emit("Correspondence Points Generation", True)
                else:
                    self.step_completed.emit("Correspondence Points Generation", False)
                    
            elif self.current_step == 3:
                # Step 4: Homography Transformation
                self.progress_updated.emit(0, "Step 4: Initializing homography transformation...")
                self.output_received.emit("Step 4: Running homography transformation...")
                
                correspondence_file = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
                
                if os.path.exists(correspondence_file):
                    self.output_received.emit("Correspondence points found, running homography transformation...")
                    self.homography_output = f"{self.output_dir}/{self.video_folder}/{self.video_name}_homography.json"
                    homography_cmd = [
                        "python3", "Scripts/homographyTransform.py",
                        "--input", self.detection_output,
                        "--correspondence", correspondence_file,
                        "--output", self.homography_output
                    ]
                    
                    if self.is_cancelled:
                        return
                        
                    result = self._run_command(homography_cmd, "Homography Transformation", 0, 100)
                    if result:
                        self.step_completed.emit("Homography Transformation", True)
                    else:
                        self.step_completed.emit("Homography Transformation", False)
                else:
                    self.output_received.emit("No correspondence points found, skipping homography transformation")
                    self.step_completed.emit("Homography Transformation", False)
                    
            elif self.current_step == 4:
                # Step 5: Render Field Video
                self.progress_updated.emit(0, "Step 5: Initializing field video rendering...")
                self.output_received.emit("Step 5: Rendering field video...")
                
                if self.homography_output and os.path.exists(self.homography_output):
                    field_video_output = f"{self.output_dir}/{self.video_folder}/virtual_field/{self.video_name}_field.mp4"
                    render_cmd = [
                        "python3", "Scripts/renderFieldVideo.py",
                        "--input", self.homography_output,
                        "--output", field_video_output
                    ]
                    
                    if self.is_cancelled:
                        return
                        
                    result = self._run_command(render_cmd, "Field Video Rendering", 0, 100)
                    if result:
                        self.step_completed.emit("Field Video Rendering", True)
                        
                        # Complete processing
                        self.progress_updated.emit(100, "Processing completed successfully!")
                        self.output_received.emit("Processing completed successfully!")
                        
                        # Return results
                        results = {
                            "video_path": self.video_path,
                            "video_name": self.video_name,
                            "detection_output": self.detection_output,
                            "homography_output": self.homography_output,
                            "field_video_output": field_video_output,
                            "status": "completed"
                        }
                        
                        # Save results to JSON
                        results_file = f"{self.output_dir}/{self.video_folder}/results/{self.video_name}_results.json"
                        with open(results_file, 'w') as f:
                            json.dump(results, f, indent=2)
                        
                        self.output_received.emit(f"Results saved to: {results_file}")
                        self.processing_completed.emit(results)
                    else:
                        self.step_completed.emit("Field Video Rendering", False)
                else:
                    self.output_received.emit("Skipping field video rendering (no homography data)")
                    self.step_completed.emit("Field Video Rendering", False)
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            self.output_received.emit(f"ERROR: {error_msg}")
            self.processing_failed.emit(error_msg)
    
    def _run_command(self, cmd, step_name, progress_start, progress_end):
        """Run a command and handle output with progress updates"""
        if self.is_cancelled:
            return False
            
        # Get the correct working directory (project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_received.emit(f"Working directory: {project_root}")
        self.output_received.emit(f"Running: {' '.join(cmd)}")
        
        try:
            # Start the process with correct working directory
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=project_root
            )
            
            # Read output line by line with progress updates
            line_count = 0
            start_time = time.time()
            
            for line in iter(self.process.stdout.readline, ''):
                if self.is_cancelled:
                    self.process.terminate()
                    return False
                    
                line = line.strip()
                if line:
                    self.output_received.emit(line)
                    line_count += 1
                    
                    # Update progress based on time elapsed and output
                    elapsed_time = time.time() - start_time
                    if line_count % 3 == 0 or elapsed_time > 1:  # Update every 3 lines or every 1 second
                        # Look for frame processing indicators in the output
                        frame_progress = 0
                        if "frame" in line.lower() or "processing" in line.lower():
                            # Try to extract frame numbers from output
                            import re
                            frame_match = re.search(r'frame\s*(\d+)', line.lower())
                            if frame_match:
                                current_frame = int(frame_match.group(1))
                                # Assume typical video has 300-600 frames (10-20 seconds at 30fps)
                                frame_progress = min(0.6, current_frame / 500)
                        
                        # Time-based progress (smoother)
                        time_progress = min(0.4, elapsed_time / 20)  # Assume max 20 seconds per step
                        
                        # Output-based progress (based on line count)
                        output_progress = min(0.3, line_count / 50)  # Assume max 50 lines of output
                        
                        # Combine all progress indicators
                        combined_progress = time_progress + output_progress + frame_progress
                        combined_progress = min(0.95, combined_progress)  # Cap at 95% until completion
                        
                        progress = progress_start + (progress_end - progress_start) * combined_progress
                        self.progress_updated.emit(int(progress), f"{step_name} in progress...")
            
            # Wait for process to complete
            return_code = self.process.wait()
            
            if return_code == 0:
                self.progress_updated.emit(progress_end, f"{step_name} completed successfully")
                self.output_received.emit(f"✅ {step_name} completed successfully")
                return True
            else:
                self.output_received.emit(f"❌ {step_name} failed with return code {return_code}")
                self.output_received.emit(f"Command: {' '.join(cmd)}")
                self.output_received.emit(f"Working directory: {project_root}")
                return False
                
        except Exception as e:
            self.output_received.emit(f"❌ Error running {step_name}: {str(e)}")
            self.output_received.emit(f"Command: {' '.join(cmd)}")
            self.output_received.emit(f"Working directory: {project_root}")
            return False
    
    def next_step(self):
        """Advance to the next processing step"""
        self.current_step += 1
        if self.current_step <= 3:
            self.start()
    
    def cancel(self):
        """Cancel the processing"""
        self.is_cancelled = True
        if self.process:
            self.process.terminate()


class ProcessingDialog(QDialog):
    """Modal dialog for video processing with progress tracking"""
    
    def __init__(self, parent, video_path, video_folder):
        super().__init__(parent)
        self.video_path = video_path
        self.video_folder = video_folder 
        self.worker = None
        self.current_step = 0
        self.step_names = ["Player Detection", "Yard Marker Detection", "Correspondence Points Generation", "Homography Transformation", "Field Video Rendering"]
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.update_progress_timer)
        self.setup_ui()
        self.start_processing()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Processing Video")
        self.setModal(True)
        self.setFixedSize(700, 600)
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
        title_label = QLabel("Processing Video")
        title_label.setFont(QFont("Arial", 14, QFont.Bold))
        title_label.setStyleSheet("color: white; margin-bottom: 5px;")
        header_layout.addWidget(title_label)
        
        # Video path
        video_label = QLabel(f"Video: {os.path.basename(self.video_path)}")
        video_label.setFont(QFont("Arial", 10))
        video_label.setStyleSheet("color: #cccccc;")
        video_label.setWordWrap(True)
        header_layout.addWidget(video_label)
        
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
        self.status_label = QLabel("Initializing...")
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
        
        self.next_button = QPushButton("Next Step")
        self.next_button.setFixedSize(100, 30)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #6c757d;
                color: #adb5bd;
            }
        """)
        self.next_button.clicked.connect(self.next_step)
        self.next_button.setEnabled(False)
        button_layout.addWidget(self.next_button)
        
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
        
        # Disable close button initially
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint)
    
    def start_processing(self):
        """Start the video processing"""
        self.worker = ProcessingWorker(self.video_path, self.video_folder)
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.output_received.connect(self.add_output)
        self.worker.step_completed.connect(self.step_completed)
        self.worker.processing_completed.connect(self.processing_completed)
        self.worker.processing_failed.connect(self.processing_failed)
        
        # Debug signal connections
        self.add_output("DEBUG: Setting up signal connections...")
        self.add_output("DEBUG: Signal connections established")
        
        self.worker.start()
        
        # Start progress timer for smoother updates
        self.progress_timer.start(500)  # Update every 500ms
    
    def update_progress(self, percentage, status):
        """Update progress bar and status"""
        self.progress_bar.setValue(percentage)
        self.status_label.setText(status)
        
        # Update dialog title to show current step
        if self.current_step < len(self.step_names):
            step_name = self.step_names[self.current_step]
            self.setWindowTitle(f"Processing Video - {step_name}")
    
    def update_progress_timer(self):
        """Timer-based progress update for smoother progress bar"""
        if self.worker and self.worker.isRunning():
            # Gradually increase progress if no updates are coming
            current_value = self.progress_bar.value()
            if current_value < 95:  # Don't go to 100% until step completes
                new_value = min(95, current_value + 1)
                self.progress_bar.setValue(new_value)
    
    def add_output(self, text):
        """Add text to terminal output"""
        self.terminal_output.append(text)
        # Auto-scroll to bottom
        cursor = self.terminal_output.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self.terminal_output.setTextCursor(cursor)
    
    def step_completed(self, step_name, success):
        """Handle step completion"""
        self.add_output(f"DEBUG: step_completed called with step_name='{step_name}', success={success}")
        
        # Stop the progress timer
        self.progress_timer.stop()
        
        if success:
            self.add_output(f"✅ {step_name} completed successfully!")
            self.add_output(f"DEBUG: Enabling Next button...")
            self.next_button.setEnabled(True)
            self.next_button.setText(f"Next: {self.get_next_step_name()}")
            self.add_output(f"DEBUG: Next button enabled: {self.next_button.isEnabled()}")
        else:
            self.add_output(f"❌ {step_name} failed!")
            self.next_button.setEnabled(False)
            self.next_button.setText("Next Step")
    
    def get_next_step_name(self):
        """Get the name of the next step"""
        if self.current_step < len(self.step_names) - 1:
            return self.step_names[self.current_step + 1]
        else:
            return "Complete"
    
    def next_step(self):
        """Advance to the next processing step"""
        if self.worker:
            self.current_step += 1
            self.next_button.setEnabled(False)
            self.next_button.setText("Next Step")
            self.progress_bar.setValue(0)  # Reset progress bar
            self.worker.next_step()
            
            # Restart progress timer for the new step
            self.progress_timer.start(500)
    
    def processing_completed(self, results):
        """Handle successful processing completion"""
        self.progress_timer.stop()  # Stop the progress timer
        self.progress_bar.setValue(100)
        self.status_label.setText("Processing completed successfully!")
        self.add_output("\n🎉 Processing completed successfully!")
        self.add_output(f"Results saved to: {results.get('field_video_output', 'N/A')}")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.next_button.setVisible(False)
        self.close_button.setVisible(True)
        
        # Re-enable close button
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
    
    def processing_failed(self, error_message):
        """Handle processing failure"""
        self.progress_timer.stop()  # Stop the progress timer
        self.status_label.setText("Processing failed!")
        self.add_output(f"\n❌ Processing failed: {error_message}")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.next_button.setVisible(False)
        self.close_button.setVisible(True)
        
        # Re-enable close button
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
    
    def cancel_processing(self):
        """Cancel the processing"""
        self.progress_timer.stop()  # Stop the progress timer
        if self.worker:
            self.worker.cancel()
            self.worker.wait()
        
        self.status_label.setText("Processing cancelled")
        self.add_output("\n⚠️ Processing cancelled by user")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.next_button.setVisible(False)
        self.close_button.setVisible(True)
        
        # Re-enable close button
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
    
    def closeEvent(self, event):
        """Handle dialog close event"""
        if self.worker and self.worker.isRunning():
            self.cancel_processing()
        event.accept()