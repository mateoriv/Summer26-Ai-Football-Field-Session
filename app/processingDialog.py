#!/usr/bin/env python3
"""
Processing Dialog
Modal dialog for video processing with progress tracking and terminal output
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QTextEdit, QPushButton, QFrame, QWidget, QMessageBox
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
    show_skip_dialog = Signal(str, int)  # step name, step index
    
    def __init__(self, video_path, video_folder, output_dir="cache"):
        super().__init__()
        self.video_path = video_path
        self.video_folder =  os.path.basename(video_folder)
        # Make output directory absolute to avoid working directory issues
        self.output_dir = os.path.abspath(output_dir)
        self.process = None
        self.is_cancelled = False
        self.current_step = 0  # 0: detection, 1: yard markers, 2: homography, 3: rendering
        self.video_name = None
        self.detection_output = None
        self.homography_output = None
        
        # Frame tracking for accurate progress
        self.total_frames = 0
        self.current_frame = 0
        self.frames_processed = 0
        self.bootup_start_time = None
        self.bootup_duration = 10  # Expected bootup time in seconds
        
        # User choice dialog variables
        self.user_choice_needed = False
        self.pending_step_name = None
        self.pending_step_index = None
        self.user_choice_result = None
        
    def check_step_completed(self, step_index):
        """Check if a processing step has already been completed"""
        if step_index == 0:  # Player Detection
            detection_file = f"{self.output_dir}/{self.video_folder}/players/{self.video_name}_detection.json"
            return os.path.exists(detection_file)
        elif step_index == 1:  # Yard Marker Detection
            yard_marker_file = f"{self.output_dir}/{self.video_folder}/yard_markers/{self.video_name}_yard_markers.json"
            return os.path.exists(yard_marker_file)
        elif step_index == 2:  # Correspondence Points Generation
            correspondence_file = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
            return os.path.exists(correspondence_file)
        elif step_index == 3:  # Homography Transformation
            homography_file = f"{self.output_dir}/{self.video_folder}/homography/{self.video_name}_normalized_positions.json"
            return os.path.exists(homography_file)
        elif step_index == 4:  # Field Video Rendering
            field_video_file = f"{self.output_dir}/{self.video_folder}/field_video/{self.video_name}_field_video.mp4"
            return os.path.exists(field_video_file)
        return False
    
    def ask_user_skip_step(self, step_name, step_index):
        """Ask user if they want to skip a completed step or re-run it"""
        # Emit signal to main thread to show dialog
        self.user_choice_needed = True
        self.pending_step_name = step_name
        self.pending_step_index = step_index
        self.user_choice_result = None
        
        # Emit signal to show dialog in main thread
        self.show_skip_dialog.emit(step_name, step_index)
        
        # Wait for user choice (with timeout to prevent infinite wait)
        timeout_count = 0
        while self.user_choice_needed and timeout_count < 1000:  # 10 second timeout
            time.sleep(0.01)
            timeout_count += 1
        
        if timeout_count >= 1000:
            return "cancel"  # Timeout - cancel processing
        
        return self.user_choice_result
    
    def set_user_choice(self, choice):
        """Set the user's choice from the dialog"""
        self.user_choice_result = choice
        self.user_choice_needed = False
        
    def get_video_frame_count(self):
        """Get the total number of frames in the video"""
        try:
            import cv2
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                return total_frames
        except Exception as e:
            print(f"Error getting video frame count: {e}")
        return 0
        
    def run(self):
        """Run the current step of the video processing pipeline"""
        try:
            # Get total frame count for accurate progress tracking
            self.total_frames = self.get_video_frame_count()
            if self.total_frames > 0:
                self.output_received.emit(f"Video has {self.total_frames} frames")
            else:
                self.output_received.emit("Could not determine video frame count, using estimated progress")
            
            # Ensure output directory exists
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(self.output_dir + "/" + self.video_folder, exist_ok=True)
            
            # Get video filename without extension
            if not self.video_name:
                self.video_name = Path(self.video_path).stem
                if not self.video_name:
                    # Fallback: extract from filename manually
                    self.video_name = os.path.splitext(os.path.basename(self.video_path))[0]
                self.output_received.emit(f"Processing video: {self.video_path}")
                self.output_received.emit(f"Video name extracted: {self.video_name}")
                self.output_received.emit(f"Output directory: {self.output_dir}")
                self.output_received.emit("-" * 50)

                
            self.detection_output = f"{self.output_dir}/{self.video_folder}/players/{self.video_name}_detection.json"
            yard_marker_output = f"{self.output_dir}/{self.video_folder}/yard_markers/{self.video_name}_yard_markers.json"
            correspondence_output = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
            self.homography_output = f"{self.output_dir}/{self.video_folder}/homography/{self.video_name}_normalized_positions.json"
            field_video_output = f"{self.output_dir}/{self.video_folder}/field_video/{self.video_name}_field_video.mp4"
        
            if self.current_step == 0:
                # Step 1: Player Detection
                step_name = "Player Detection"
                self.output_received.emit(f"Step 1: Checking {step_name}...")
                
                # Check if step is already completed
                if self.check_step_completed(0):
                    self.output_received.emit(f"✓ {step_name} already completed!")
                    user_choice = self.ask_user_skip_step(step_name, 0)
                    
                    if user_choice == "cancel":
                        self.processing_failed.emit("Processing cancelled by user")
                        return
                    elif user_choice == "skip":
                        self.output_received.emit(f"Skipping {step_name} - using existing results")
                        self.step_completed.emit(step_name, True)
                        return
                    else:  # rerun
                        self.output_received.emit(f"Re-running {step_name}...")
                
                self.progress_updated.emit(0, "Step 1: Initializing player detection...")
                self.output_received.emit("Step 1: Running player detection...")
                
                
                detection_cmd = [
                    "python", "scripts/playerDetection.py", 
                    "--video", self.video_path, 
                    "--output", self.detection_output
                ]
                
                if self.is_cancelled:
                    return
                    
                result = self._run_command(detection_cmd, "Player Detection", 0, 100)
                if result:
                    self.step_completed.emit("Player Detection", True)
                else:
                    self.step_completed.emit("Player Detection", False)
                    
            elif self.current_step == 1:
                # Step 2: Yard Marker Detection
                step_name = "Yard Marker Detection"
                self.output_received.emit(f"Step 2: Checking {step_name}...")
                
                # Check if step is already completed
                if self.check_step_completed(1):
                    self.output_received.emit(f"✓ {step_name} already completed!")
                    user_choice = self.ask_user_skip_step(step_name, 1)
                    
                    if user_choice == "cancel":
                        self.processing_failed.emit("Processing cancelled by user")
                        return
                    elif user_choice == "skip":
                        self.output_received.emit(f"Skipping {step_name} - using existing results")
                        self.step_completed.emit(step_name, True)
                        return
                    else:  # rerun
                        self.output_received.emit(f"Re-running {step_name}...")
                
                self.progress_updated.emit(0, "Step 2: Initializing yard marker detection...")
                self.output_received.emit("Step 2: Running yard marker detection...")
                
                
                yard_marker_cmd = [
                    "python", "scripts/yardMarkerDetection.py",
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
                step_name = "Correspondence Points Generation"
                self.output_received.emit(f"Step 3: Checking {step_name}...")
                
                # Check if step is already completed
                if self.check_step_completed(2):
                    self.output_received.emit(f"✓ {step_name} already completed!")
                    user_choice = self.ask_user_skip_step(step_name, 2)
                    
                    if user_choice == "cancel":
                        self.processing_failed.emit("Processing cancelled by user")
                        return
                    elif user_choice == "skip":
                        self.output_received.emit(f"Skipping {step_name} - using existing results")
                        # Update virtual field with existing correspondence points
                        correspondence_output = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
                        from virtualField import update_field_with_correspondence_points
                        update_field_with_correspondence_points(self.parent(), correspondence_output, frame_number=0)
                        self.step_completed.emit(step_name, True)
                        return
                    else:  # rerun
                        self.output_received.emit(f"Re-running {step_name}...")
                
                self.progress_updated.emit(0, "Step 3: Initializing correspondence points generation...")
                self.output_received.emit("Step 3: Generating correspondence points from yard markers...")
                

                
                correspondence_cmd = [
                    "python", "scripts/autoCorrespondancePoints.py",
                    "--detection-json", yard_marker_output,
                    "--output", correspondence_output,
                    "--confidence", "0.7",
                    "--per-frame"
                ]
                
                if self.is_cancelled:
                    return
                    
                result = self._run_command(correspondence_cmd, "Correspondence Points Generation", 0, 100)
                if result:
                    # Update virtual field with yard marker dots (show frame 0 by default)
                    from virtualField import update_field_with_correspondence_points
                    update_field_with_correspondence_points(self.parent(), correspondence_output, frame_number=0)
                    self.step_completed.emit("Correspondence Points Generation", True)
                else:
                    self.step_completed.emit("Correspondence Points Generation", False)
                    
            elif self.current_step == 3:
                # Step 4: Homography Transformation
                step_name = "Homography Transformation"
                self.output_received.emit(f"Step 4: Checking {step_name}...")
                
                # Check if step is already completed
                if self.check_step_completed(3):
                    self.output_received.emit(f"✓ {step_name} already completed!")
                    user_choice = self.ask_user_skip_step(step_name, 3)
                    
                    if user_choice == "cancel":
                        self.processing_failed.emit("Processing cancelled by user")
                        return
                    elif user_choice == "skip":
                        self.output_received.emit(f"Skipping {step_name} - using existing results")
                        self.homography_output = f"{self.output_dir}/{self.video_folder}/homography/{self.video_name}_homography.json"
                        self.step_completed.emit(step_name, True)
                        return
                    else:  # rerun
                        self.output_received.emit(f"Re-running {step_name}...")
                
                self.progress_updated.emit(0, "Step 4: Initializing per-frame homography transformation...")
                self.output_received.emit("Step 4: Running per-frame homography transformation...")
                
                correspondence_file = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
                
                if os.path.exists(correspondence_file):
                    self.output_received.emit("Correspondence points found, running per-frame homography transformation...")
                    homography_cmd = [
                        "python", "scripts/perFrameHomographyTransform.py",
                        "--player-detections", self.detection_output,
                        "--correspondence-points", correspondence_file,
                        "--output", self.homography_output
                    ]
                
                    if self.is_cancelled:
                        return
                        
                    result = self._run_command(homography_cmd, "Homography Transformation", 0, 100)
                    print(f"DEBUG: Homography result={result}")
                    if result:
                        print("DEBUG: Emitting step_completed signal for Homography Transformation (True)")
                        self.step_completed.emit("Homography Transformation", True)
                    else:
                        print("DEBUG: Emitting step_completed signal for Homography Transformation (False)")
                        self.step_completed.emit("Homography Transformation", False)
                else:
                    self.output_received.emit("No correspondence points found, skipping homography transformation")
                    self.step_completed.emit("Homography Transformation", False)
                    
            
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
            # Start the process with correct working directory and unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True,
                cwd=project_root,
                env=env
            )

            # Simple output reading with progress updates
            line_count = 0
            start_time = time.time()
            last_update_time = 0

            while self.process.poll() is None:  # Process is still running
                if self.is_cancelled:
                    self.process.terminate()
                    return False
    
                # Non-blocking output reading with select
                import select
                try:
                    if select.select([self.process.stdout], [], [], 0.01)[0]:  # 10ms timeout
                        line = self.process.stdout.readline()
                        if line:
                            line = line.strip()
                            if line:
                                self.output_received.emit(line)
                                line_count += 1
                                
                                # Look for frame processing indicators in the output
                                if "frame" in line.lower() or "processing" in line.lower():
                                    # Try to extract frame numbers from output
                                    import re
                                    # Look for patterns like "Processed frame 150/1000" or "frame 150"
                                    frame_match = re.search(r'frame\s*(\d+)(?:/(\d+))?', line.lower())
                                    if frame_match:
                                        self.current_frame = int(frame_match.group(1))
                                         # If we start getting frame data, we're past bootup
                                        if self.bootup_start_time is None:
                                            self.bootup_start_time = time.time() - start_time
                except:
                    # If readline fails, continue with timeout
                    pass
                
                # Update frame tracking
                self.frames_processed = self.current_frame
        
                # Update progress every 0.1 seconds regardless of output
                elapsed_time = time.time() - start_time
                if elapsed_time - last_update_time >= 0.1:
                    last_update_time = elapsed_time
    
                    # Calculate progress based on phase
                    if self.bootup_start_time is not None:
                        # Frame processing phase: 25% to 100% based on actual frames
                        if self.total_frames > 0 and self.current_frame > 0:
                            frame_progress = self.current_frame / self.total_frames
                            # 25% + (75% * frame_progress) = 25% to 100%
                            progress = (0.25 + 0.75 * frame_progress) * 100
                            
                            # Show detailed progress every 10 frames to avoid spam
                            if self.current_frame % 10 == 0:
                                self.output_received.emit(f"Processing frame {self.current_frame}/{self.total_frames} ({frame_progress*100:.1f}%)")
                    else:
                        # Bootup phase: 0% to 25% over 10 seconds
                        bootup_progress = min(1, elapsed_time / 30.0)
                        progress =  bootup_progress * 25
                        
                        # if elapsed_time > 10:  # Show bootup progress after 2 seconds
                        #     self.output_received.emit(f"Initializing {step_name}... ({bootup_progress*100:.1f}%)")
                    
                    self.progress_updated.emit(int(progress), f"{step_name} in progress...")
            
            # Wait for process to complete
            return_code = self.process.wait()
            
            if return_code == 0:
                self.progress_updated.emit(progress_end, f"{step_name} completed successfully")
                self.output_received.emit(f"{step_name} completed successfully")
                return True
            else:
                self.output_received.emit(f"{step_name} failed with return code {return_code}")
                self.output_received.emit(f"Command: {' '.join(cmd)}")
                self.output_received.emit(f"Working directory: {project_root}")
                return False
                
        except Exception as e:
            self.output_received.emit(f"Error running {step_name}: {str(e)}")
            self.output_received.emit(f"Command: {' '.join(cmd)}")
            self.output_received.emit(f"Working directory: {project_root}")
            return False
    
    def next_step(self):
        """Advance to the next processing step"""
        self.current_step += 1
        # Reset frame tracking for new step
        self.current_frame = 0
        self.frames_processed = 0
        self.bootup_start_time = None
        if self.current_step <= 5:
            self.start()
    
    def get_progress_info(self):
        """Get current progress information"""
        if self.total_frames > 0 and self.frames_processed > 0:
            progress_percent = (self.frames_processed / self.total_frames) * 100
            return f"Frame {self.frames_processed}/{self.total_frames} ({progress_percent:.1f}%)"
        return "Processing..."
    
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
        self.worker.show_skip_dialog.connect(self.show_skip_dialog)
        
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
            # Only update if we have frame data and are past bootup
            if (hasattr(self.worker, 'bootup_start_time') and 
                self.worker.bootup_start_time is not None and
                hasattr(self.worker, 'total_frames') and 
                hasattr(self.worker, 'current_frame') and
                self.worker.total_frames > 0 and 
                self.worker.current_frame > 0):
                
                # Gradually increase progress if no updates are coming (only during frame processing)
                current_value = self.progress_bar.value()
                if current_value < 95:  # Don't go to 100% until step completes
                    new_value = min(95, current_value + 1)
                    self.progress_bar.setValue(new_value)
            # If we're still in bootup phase, do nothing - let main logic handle it
    
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
            self.add_output(f"{step_name} completed successfully!")
            self.next_button.setEnabled(True)
            self.next_button.setText(f"Next: {self.get_next_step_name()}")
        else:
            self.add_output(f"{step_name} failed!")
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
        self.add_output("\nProcessing completed successfully!")
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
        self.add_output(f"\nProcessing failed: {error_message}")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.next_button.setVisible(False)
        self.close_button.setVisible(True)
        
        # Re-enable close button
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
    
    def show_skip_dialog(self, step_name, step_index):
        """Show dialog asking user to skip or re-run a completed step"""
        msg = QMessageBox()
        msg.setWindowTitle("Step Already Completed")
        msg.setText(f"'{step_name}' has already been completed.")
        msg.setInformativeText("Would you like to skip this step and use the existing results, or re-run it?")
        
        skip_button = msg.addButton("Skip (Use Existing)", QMessageBox.AcceptRole)
        rerun_button = msg.addButton("Re-run Step", QMessageBox.RejectRole)
        cancel_button = msg.addButton("Cancel", QMessageBox.DestructiveRole)
        
        msg.setDefaultButton(skip_button)
        result = msg.exec()
        
        if msg.clickedButton() == skip_button:
            self.worker.set_user_choice("skip")
        elif msg.clickedButton() == rerun_button:
            self.worker.set_user_choice("rerun")
        else:
            self.worker.set_user_choice("cancel")
    
    def cancel_processing(self):
        """Cancel the processing"""
        self.progress_timer.stop()  # Stop the progress timer
        if self.worker:
            self.worker.cancel()
            self.worker.wait()
        
        self.status_label.setText("Processing cancelled")
        self.add_output("\nProcessing cancelled by user")
        
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