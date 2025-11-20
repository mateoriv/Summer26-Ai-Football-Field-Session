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
from video import load_snap_detection_data
import subprocess
import os
import sys
import json
import time
from pathlib import Path

def get_project_root():
    """Return the project root, accounting for PyInstaller one-file extraction."""
    if hasattr(sys, "_MEIPASS"):
        return sys._MEIPASS
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_resource_path(*relative_parts):
    """Build an absolute path rooted at the project directory."""
    return os.path.join(get_project_root(), *relative_parts)


def get_python_executable():
    """Get the correct Python executable for the current platform"""
    if sys.platform.startswith('win'):
        # On Windows, try 'python' first, then 'python3'
        for cmd in ['python', 'python3']:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    return cmd
            except FileNotFoundError:
                continue
        return 'python'  # Fallback
    else:
        # On Unix-like systems, try 'python3' first, then 'python'
        for cmd in ['python3', 'python']:
            try:
                result = subprocess.run([cmd, '--version'], capture_output=True, text=True)
                if result.returncode == 0:
                    return cmd
            except FileNotFoundError:
                continue
        return 'python3'  # Fallback

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
        # If output_dir is relative, make it relative to project root, not app directory
        if not os.path.isabs(output_dir):
            self.output_dir = os.path.join(get_project_root(), output_dir)
        else:
            self.output_dir = os.path.abspath(output_dir)
        print(f"Output directory: {self.output_dir}")
        self.process = None
        self.is_cancelled = False
        self.current_step = 0
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
        elif step_index == 1: # Position Detection
            position_file = f"{self.output_dir}/{self.video_folder}/positions/{self.video_name}_position.json"
            return os.path.exists(position_file)
        elif step_index == 2:  # Snap Detection
            snap_file = f"{self.output_dir}/{self.video_folder}/snap_detection/{self.video_name}_snap_detection.json"
            return os.path.exists(snap_file)
        elif step_index == 3:  # Yard Marker Detection
            yard_marker_file = f"{self.output_dir}/{self.video_folder}/yard_markers/{self.video_name}_yard_markers.json"
            return os.path.exists(yard_marker_file)
        elif step_index == 4:  # Correspondence Points Generation
            correspondence_file = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
            return os.path.exists(correspondence_file)
        elif step_index == 5:  # Homography Transformation
            homography_file = f"{self.output_dir}/{self.video_folder}/homography/{self.video_name}_normalized_positions.json"
            return os.path.exists(homography_file)
        elif step_index == 6:  # Static Process
            # Static process doesn't create a file, so we check if homography exists (prerequisite)
            homography_file = f"{self.output_dir}/{self.video_folder}/homography/{self.video_name}_normalized_positions.json"
            snap_file = f"{self.output_dir}/{self.video_folder}/snap_detection/{self.video_name}_snap_detection.json"
            return os.path.exists(homography_file) and os.path.exists(snap_file)
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
            self.position_output = f"{self.output_dir}/{self.video_folder}/positions/{self.video_name}_position.json"
            self.snap_output = f"{self.output_dir}/{self.video_folder}/snap_detection/{self.video_name}_snap_detection.json"
            yard_marker_output = f"{self.output_dir}/{self.video_folder}/yard_markers/{self.video_name}_yard_markers.json"
            correspondence_output = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
            self.homography_output = f"{self.output_dir}/{self.video_folder}/homography/{self.video_name}_normalized_positions.json"
        
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
                    get_python_executable(), get_resource_path("scripts", "playerDetection.py"),
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
                # Step 2: Position Detection
                step_name = "Position Detection"
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
                
                self.progress_updated.emit(0, "Step 2: Initializing position detection...")
                self.output_received.emit("Step 2: Running position detection...")
                
                
                position_cmd = [
                    get_python_executable(), "scripts/positionDetection.py",
                    "--video", self.video_path, 
                    "--output", self.position_output
                ]
                
                if self.is_cancelled:
                    return
                    
                result = self._run_command(position_cmd, "Position Detection", 0, 100)
                if result:
                    self.step_completed.emit("Position Detection", True)
                else:
                    self.step_completed.emit("Position Detection", False)
                    
            elif self.current_step == 2:
                # Step 3: Snap Detection
                step_name = "Snap Detection"
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
                        self.step_completed.emit(step_name, True)
                        return
                    else:  # rerun
                        self.output_received.emit(f"Re-running {step_name}...")
                
                self.progress_updated.emit(0, "Step 3: Initializing snap detection...")
                self.output_received.emit("Step 3: Running snap detection...")
                
                snap_cmd = [
                    get_python_executable(), get_resource_path("scripts", "snapDetection.py"),
                    "--player-detections", self.detection_output,
                    "--output", self.snap_output
                ]
                
                if self.is_cancelled:
                    return
                    
                result = self._run_command(snap_cmd, "Snap Detection", 0, 100)
                if result:
                    self.step_completed.emit("Snap Detection", True)
                else:
                    self.step_completed.emit("Snap Detection", False)
                    
            elif self.current_step == 3:
                # Step 4: Yard Marker Detection
                step_name = "Yard Marker Detection"
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
                        self.step_completed.emit(step_name, True)
                        return
                    else:  # rerun
                        self.output_received.emit(f"Re-running {step_name}...")
                
                self.progress_updated.emit(0, "Step 4: Initializing yard marker detection...")
                self.output_received.emit("Step 4: Running yard marker detection...")
                
                
                yard_marker_cmd = [
                    get_python_executable(), get_resource_path("scripts", "yardMarkerDetection.py"),
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
                    
            elif self.current_step == 4:
                # Step 5: Auto Correspondence Points
                step_name = "Correspondence Points Generation"
                self.output_received.emit(f"Step 5: Checking {step_name}...")
                
                # Check if step is already completed
                if self.check_step_completed(4):
                    self.output_received.emit(f"✓ {step_name} already completed!")
                    user_choice = self.ask_user_skip_step(step_name, 4)
                    
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
                
                self.progress_updated.emit(0, "Step 5: Initializing correspondence points generation...")
                self.output_received.emit("Step 5: Generating correspondence points from yard markers...")
                

                
                correspondence_cmd = [
                    get_python_executable(), get_resource_path("scripts", "autoCorrespondancePoints.py"),
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
                    
            elif self.current_step == 5:
                # Step 6: Homography Transformation
                step_name = "Homography Transformation"
                self.output_received.emit(f"Step 6: Checking {step_name}...")
                
                # Check if step is already completed
                if self.check_step_completed(5):
                    self.output_received.emit(f"✓ {step_name} already completed!")
                    user_choice = self.ask_user_skip_step(step_name, 5)
                    
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
                
                self.progress_updated.emit(0, "Step 6: Initializing per-frame homography transformation...")
                self.output_received.emit("Step 6: Running per-frame homography transformation...")
                
                correspondence_file = f"{self.output_dir}/{self.video_folder}/correspondence/{self.video_name}_correspondence.json"
                
                if os.path.exists(correspondence_file):
                    self.output_received.emit("Correspondence points found, running per-frame homography transformation...")
                    homography_cmd = [
                        get_python_executable(), "scripts/perFrameHomographyTransform.py",
                        "--position-detections", self.position_output,
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
                    
            elif self.current_step == 6:
                # Step 7: Static Process
                step_name = "Static Process"
                self.output_received.emit(f"Step 7: Checking {step_name}...")
                
                # Check prerequisites
                snap_file = f"{self.output_dir}/{self.video_folder}/snap_detection/{self.video_name}_snap_detection.json"
                homography_file = f"{self.output_dir}/{self.video_folder}/homography/{self.video_name}_normalized_positions.json"
                
                if not os.path.exists(snap_file):
                    self.output_received.emit("ERROR: Snap detection file not found. Cannot run static process.")
                    self.step_completed.emit(step_name, False)
                    return
                
                if not os.path.exists(homography_file):
                    self.output_received.emit("ERROR: Homography file not found. Cannot run static process.")
                    self.step_completed.emit(step_name, False)
                    return
                
                # Check if step is already completed (optional - static process can be re-run)
                if self.check_step_completed(6):
                    self.output_received.emit(f"✓ {step_name} prerequisites met!")
                    user_choice = self.ask_user_skip_step(step_name, 6)
                    
                    if user_choice == "cancel":
                        self.processing_failed.emit("Processing cancelled by user")
                        return
                    elif user_choice == "skip":
                        self.output_received.emit(f"Skipping {step_name}")
                        self.step_completed.emit(step_name, True)
                        return
                    else:  # rerun
                        self.output_received.emit(f"Re-running {step_name}...")
                
                self.progress_updated.emit(0, "Step 7: Loading snap detection data...")
                self.output_received.emit("Step 7: Running static process...")
                
                # Load snap detection to get snap frame number
                try:
                    
                    # Run static process
                    static_process_cmd = [
                        get_python_executable(), get_resource_path("CNN", "staticProcess.py"),
                        "--video-name", self.video_name,
                        "--folder-name", self.video_folder,
                        "--cache-dir", "cache"
                    ]
                    
                    if self.is_cancelled:
                        return
                    
                    result = self._run_command(static_process_cmd, step_name, 0, 100)
                    if result:
                        self.step_completed.emit(step_name, True)
                    else:
                        self.step_completed.emit(step_name, False)
                        
                except Exception as e:
                    self.output_received.emit(f"ERROR: Failed to run static process: {str(e)}")
                    import traceback
                    self.output_received.emit(traceback.format_exc())
                    self.step_completed.emit(step_name, False)
                    
            
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
            self.output_received.emit(f"ERROR: {error_msg}")
            self.processing_failed.emit(error_msg)
    
    def _run_command(self, cmd, step_name, progress_start, progress_end):
        """Run a command and handle output with progress updates"""
        if self.is_cancelled:
            return False
            
        # Get the correct working directory (project root)
        project_root = get_project_root()
        self.output_received.emit(f"Working directory: {project_root}")
        self.output_received.emit(f"Running: {' '.join(cmd)}")
        
        try:
            # Start the process with correct working directory and unbuffered output
            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
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
            
            # Cross-platform non-blocking output reading using threading
            import threading
            import queue
            
            # Create queues for output lines (separate for stdout and stderr)
            self.output_queue = queue.Queue()
            self.error_queue = queue.Queue()
            self.output_thread = None
            self.error_thread = None
            
            # Start output reading threads
            self.output_thread = threading.Thread(target=self._read_output_thread, args=(self.process.stdout, self.output_queue), daemon=True)
            self.error_thread = threading.Thread(target=self._read_output_thread, args=(self.process.stderr, self.error_queue), daemon=True)
            self.output_thread.start()
            self.error_thread.start()

            while self.process.poll() is None:  # Process is still running
                if self.is_cancelled:
                    self.process.terminate()
                    return False
    
                # Process any available output from the queue (one line per iteration)
                try:
                    line = self.output_queue.get_nowait()
                    line = line.strip()
                    print(line)
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
                except queue.Empty:
                    # No output available, continue
                    pass
                
                # Process any error output from stderr
                try:
                    error_line = self.error_queue.get_nowait()
                    error_line = error_line.strip()
                    if error_line:
                        # Emit error lines with ERROR prefix to make them visible
                        self.output_received.emit(f"ERROR: {error_line}")
                        print(f"ERROR: {error_line}")
                except queue.Empty:
                    # No error output available, continue
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
                            
                    else:
                        # Bootup phase: 0% to 25% over 10 seconds
                        bootup_progress = min(1, elapsed_time / 30.0)
                        progress =  bootup_progress * 25
                        
                        # if elapsed_time > 10:  # Show bootup progress after 2 seconds
                        #     self.output_received.emit(f"Initializing {step_name}... ({bootup_progress*100:.1f}%)")
                    
                    self.progress_updated.emit(int(progress), f"{step_name} in progress...")
            
            # Wait for process to complete
            return_code = self.process.wait()
            
            # Wait for output threads to finish and read any remaining output
            if self.output_thread and self.output_thread.is_alive():
                self.output_thread.join(timeout=2.0)
            if self.error_thread and self.error_thread.is_alive():
                self.error_thread.join(timeout=2.0)
            
            # Read any remaining output from queues
            remaining_output = []
            while True:
                try:
                    line = self.output_queue.get_nowait()
                    if line.strip():
                        remaining_output.append(line.strip())
                except queue.Empty:
                    break
            
            # Read any remaining error output from stderr
            remaining_errors = []
            while True:
                try:
                    error_line = self.error_queue.get_nowait()
                    if error_line.strip():
                        remaining_errors.append(error_line.strip())
                except queue.Empty:
                    break
            
            # Emit remaining output
            for line in remaining_output:
                self.output_received.emit(line)
            
            # Emit remaining errors with ERROR prefix
            for error_line in remaining_errors:
                self.output_received.emit(f"ERROR: {error_line}")
            
            if return_code == 0:
                self.progress_updated.emit(progress_end, f"{step_name} completed successfully")
                self.output_received.emit(f"{step_name} completed successfully")
                return True
            else:
                self.output_received.emit("")
                self.output_received.emit("=" * 60)
                self.output_received.emit(f"ERROR: {step_name} failed with return code {return_code}")
                self.output_received.emit(f"Command: {' '.join(cmd)}")
                self.output_received.emit(f"Working directory: {project_root}")
                if remaining_errors:
                    self.output_received.emit("")
                    self.output_received.emit("Error output:")
                    for error_line in remaining_errors:
                        self.output_received.emit(f"  {error_line}")
                self.output_received.emit("=" * 60)
                return False
                
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.output_received.emit("")
            self.output_received.emit("=" * 60)
            self.output_received.emit(f"ERROR: Exception occurred while running {step_name}")
            self.output_received.emit(f"Exception: {str(e)}")
            self.output_received.emit(f"Command: {' '.join(cmd)}")
            self.output_received.emit(f"Working directory: {project_root}")
            self.output_received.emit("")
            self.output_received.emit("Traceback:")
            for line in error_traceback.splitlines():
                self.output_received.emit(f"  {line}")
            self.output_received.emit("=" * 60)
            return False
    
    def next_step(self):
        """Advance to the next processing step"""
        self.current_step += 1
        # Reset frame tracking for new step
        self.current_frame = 0
        self.frames_processed = 0
        self.bootup_start_time = None
        if self.current_step <= 6:
            self.start()
    
    def get_progress_info(self):
        """Get current progress information"""
        if self.total_frames > 0 and self.frames_processed > 0:
            progress_percent = (self.frames_processed / self.total_frames) * 100
            return f"Frame {self.frames_processed}/{self.total_frames} ({progress_percent:.1f}%)"
        return "Processing..."
    
    def _read_output_thread(self, stream, output_queue):
        """Background thread to read subprocess output from a stream"""
        try:
            while self.process and (self.process.poll() is None or stream):
                line = stream.readline()
                if line:
                    output_queue.put(line)
                elif self.process.poll() is not None:
                    # Process has finished, try to read any remaining data
                    remaining = stream.read()
                    if remaining:
                        for line in remaining.splitlines(keepends=True):
                            output_queue.put(line)
                    break
                else:
                    time.sleep(0.01)
        except Exception as e:
            # If there's an error reading, try to put it in the queue
            try:
                output_queue.put(f"Error reading output: {str(e)}\n")
            except:
                pass

    def cancel(self):
        """Cancel the processing"""
        self.is_cancelled = True
        if self.process:
            self.process.terminate()


class ProcessingDialog(QDialog):
    """Modal dialog for video processing with progress tracking"""
    
    def __init__(self, parent, video_path, video_folder, cache_dir="cache"):
        super().__init__(parent)
        self.video_path = video_path
        self.video_folder = video_folder 
        self.cache_dir = cache_dir
        self.worker = None
        self.current_step = 0
        self.step_names = [
            "Player Detection", 
            "Position Detection",
            "Snap Detection", 
            "Yard Marker Detection", 
            "Correspondence Points Generation", 
            "Homography Transformation", 
            "Static Process"
        ]
        self.progress_timer = QTimer()
    
        self.setup_ui()
        self.start_processing()
    
    def setup_ui(self):
        """Setup the dialog UI"""
        self.setWindowTitle("Processing Video")
        self.setModal(True)
        # Make dialog resizable with minimum size
        self.setMinimumSize(700, 600)
        self.resize(900, 700)  # Default size, but user can resize
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)
        
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
        # Set minimum height for the terminal output
        self.terminal_output.setMinimumHeight(200)
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
        self.next_button.setFixedHeight(30)
        self.next_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border: none;
                color: white;
                border-radius: 4px;
                font-weight: bold;
                padding: 5px 15px; 
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
        self.worker = ProcessingWorker(self.video_path, self.video_folder, self.cache_dir)
        
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
            
            # If snap detection completed, reload snap markers in the video player
            if step_name == "Snap Detection":
                parent_window = self.parent()
                if parent_window and hasattr(parent_window, 'current_video_path') and parent_window.current_video_path:
                    # Add a small delay to ensure file is written
                    QTimer.singleShot(500, lambda: self.reload_snap_markers(parent_window))
            
            # Check if this was the final step
            if self.current_step >= len(self.step_names) - 1:
                # Final step completed - automatically finish processing
                self.add_output("All processing steps completed!")
                self.processing_completed({"homography_output": "Processing completed successfully"})
            else:
                # Not the final step - show next button
                self.next_button.setEnabled(True)
                self.next_button.setText(f"Next: {self.get_next_step_name()}")
        else:
            self.add_output(f"{step_name} failed!")
            self.next_button.setEnabled(False)
            self.next_button.setText("Next Step")
    
    def reload_snap_markers(self, parent_window):
        """Reload snap detection markers after snap detection completes"""
        try:
            if hasattr(parent_window, 'current_video_path') and parent_window.current_video_path:
                load_snap_detection_data(parent_window, parent_window.current_video_path)
                self.add_output("Snap markers updated on timeline")
        except Exception as e:
            self.add_output(f"Error reloading snap markers: {e}")
    
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
        self.add_output(f"Results saved to: {results.get('homography_output', 'N/A')}")
        
        # Reload data sheet and virtual field with updated data
        self.reload_data_sheet_and_virtual_field()
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.next_button.setVisible(False)
        self.close_button.setVisible(True)
        
        # Re-enable close button
        self.setWindowFlags(Qt.Dialog | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
    
    def reload_data_sheet_and_virtual_field(self):
        """Reload the data sheet CSV and update virtual field with new homography data"""
        parent_window = self.parent()
        if not parent_window:
            return
        
        try:
            # Reload CSV file if it exists
            if hasattr(parent_window, 'current_csv_path') and parent_window.current_csv_path:
                if os.path.exists(parent_window.current_csv_path):
                    self.add_output(f"Reloading data sheet: {parent_window.current_csv_path}")
                    if hasattr(parent_window, 'load_csv_file'):
                        parent_window.load_csv_file(parent_window.current_csv_path)
                        self.add_output("Data sheet reloaded successfully")
                else:
                    # Try to find CSV in cache directory
                    if hasattr(parent_window, 'current_folder') and parent_window.current_folder:
                        folder_name = os.path.basename(parent_window.current_folder.rstrip('/\\'))
                        csv_path = os.path.join(get_project_root(), "cache", folder_name, f"{folder_name}_data.csv")
                        if os.path.exists(csv_path) and hasattr(parent_window, 'load_csv_file'):
                            self.add_output(f"Reloading data sheet from cache: {csv_path}")
                            parent_window.load_csv_file(csv_path)
                            self.add_output("Data sheet reloaded successfully")
            
            # Reload homography data for virtual field
            if hasattr(parent_window, 'current_video_path') and parent_window.current_video_path:
                video_name = os.path.splitext(os.path.basename(parent_window.current_video_path))[0]
                if hasattr(parent_window, 'current_folder') and parent_window.current_folder:
                    self.add_output(f"Reloading virtual field with updated homography data for: {video_name}")
                    from virtualField import load_homography_data_for_virtual_field
                    if load_homography_data_for_virtual_field(parent_window, video_name, parent_window.current_folder):
                        # Update virtual field display with current frame
                        if hasattr(parent_window, 'custom_video') and hasattr(parent_window.custom_video, 'current_frame'):
                            from virtualField import update_virtual_field_with_video_frame
                            update_virtual_field_with_video_frame(parent_window, parent_window.custom_video.current_frame)
                            self.add_output("Virtual field updated with new homography data")
                    else:
                        self.add_output("Warning: Could not reload homography data for virtual field")
        except Exception as e:
            self.add_output(f"Error reloading data sheet/virtual field: {e}")
            import traceback
            self.add_output(traceback.format_exc())
    
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
