#!/usr/bin/env python3
"""
Batch Processing Dialog
Modal dialog for batch processing multiple videos with progress tracking
"""

from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, 
    QTextEdit, QPushButton, QFrame, QWidget, QFileDialog, QListWidget, QListWidgetItem, QSpinBox, QCheckBox
)
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QFont, QIcon, QTextCursor
import subprocess
import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
import glob
import threading
import multiprocessing

# Set up logging for batch processing
def setup_batch_logging():
    """Set up logging for batch processing operations"""
    log_dir = Path.home() / ".hudl_ai_logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"batch_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger for batch processing
    logger = logging.getLogger('batch_processing')
    logger.setLevel(logging.DEBUG)
    
    # Avoid adding handlers multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    return logger, log_file

# Initialize logger
_batch_logger, _batch_log_file = setup_batch_logging()

def get_project_root():
    """Return project root, handling PyInstaller one-file extractions."""
    # Import here to avoid circular imports
    from fileAccess import get_project_root as get_root
    return get_root()

def get_cache_dir():
    """Get the cache directory path."""
    # Import here to avoid circular imports
    from fileAccess import get_cache_dir as get_cache
    return get_cache()

def get_resource_path(*parts):
    """Build an absolute path rooted at the project directory or _MEIPASS when compiled."""
    if hasattr(sys, "_MEIPASS"):
        # Running as compiled executable - resources are in _MEIPASS
        return os.path.join(sys._MEIPASS, *parts)
    else:
        # Running in development mode
        return os.path.join(get_project_root(), *parts)


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

# Required for ProcessPoolExecutor on Windows
if __name__ == '__main__':
    multiprocessing.freeze_support()

class ProcessingCancelled(Exception):
    """Raised when batch processing is cancelled by the user."""
    pass

def is_video_processed(video_path, video_folder, output_dir="cache"):
    """Check if a video has already been fully processed.
    
    A video is considered processed if all required files exist:
    - correspondence/{video_name}_correspondence.json
    - homography/{video_name}_normalized_positions.json
    - players/{video_name}_detection.json
    - snap_detection/{video_name}_snap_detection.json
    - yard_markers/{video_name}_yard_markers.json
    """
    try:
        video_name = Path(video_path).stem
        
        if not os.path.isabs(output_dir):
            # Use persistent cache directory instead of temp PyInstaller folder
            output_dir = get_cache_dir()
        else:
            output_dir = os.path.abspath(output_dir)
        
        if video_folder:
            video_folder = os.path.basename(video_folder.rstrip("/\\"))
        else:
            video_folder = Path(video_path).parent.name
        
        base_dir = os.path.join(output_dir, video_folder)
        
        # Check for all required files
        required_files = [
            os.path.join(base_dir, "correspondence", f"{video_name}_correspondence.json"),
            os.path.join(base_dir, "homography", f"{video_name}_normalized_positions.json"),
            os.path.join(base_dir, "players", f"{video_name}_detection.json"),
            os.path.join(base_dir, "snap_detection", f"{video_name}_snap_detection.json"),
            os.path.join(base_dir, "yard_markers", f"{video_name}_yard_markers.json"),
        ]
        
        # All files must exist for video to be considered processed
        return all(os.path.exists(f) for f in required_files)
    except Exception as e:
        _batch_logger.error(f"Error checking if video is processed: {e}", exc_info=True)
        return False


def process_single_video_standalone(video_path, video_folder, output_dir="cache"):
    """Standalone function kept for future multi-processing support."""
    return process_single_video(
        video_path,
        video_folder,
        output_dir=output_dir,
        output_callback=lambda msg: print(msg, flush=True)
    )


def process_single_video(video_path, video_folder, output_dir="cache", output_callback=None, status_callback=None, cancel_check=None):
    """Process a single video sequentially, mirroring ProcessingDialog steps."""
    emit_output = output_callback or (lambda msg: print(msg, flush=True))
    emit_status = status_callback or (lambda msg: emit_output(msg))
    
    try:
        video_name = Path(video_path).stem
        _batch_logger.info(f"Starting processing for video: {video_name} (path: {video_path})")
        
        if not os.path.isabs(output_dir):
            # Use persistent cache directory instead of temp PyInstaller folder
            output_dir = get_cache_dir()
        else:
            output_dir = os.path.abspath(output_dir)
        
        if video_folder:
            video_folder = os.path.basename(video_folder.rstrip("/\\"))
        else:
            video_folder = Path(video_path).parent.name
        base_dir = os.path.join(output_dir, video_folder)
        
        directories = [
            base_dir,
            os.path.join(base_dir, "players"),
            os.path.join(base_dir, "snap_detection"),
            os.path.join(base_dir, "yard_markers"),
            os.path.join(base_dir, "correspondence"),
            os.path.join(base_dir, "homography"),
          
        ]
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
        
        detection_output = os.path.join(base_dir, "players", f"{video_name}_detection.json")
        position_output = os.path.join(base_dir, "positions", f"{video_name}_position.json")
        snap_output = os.path.join(base_dir, "snap_detection", f"{video_name}_snap_detection.json")
        yard_marker_output = os.path.join(base_dir, "yard_markers", f"{video_name}_yard_markers.json")
        correspondence_output = os.path.join(base_dir, "correspondence", f"{video_name}_correspondence.json")
        homography_output = os.path.join(base_dir, "homography", f"{video_name}_normalized_positions.json")
        
        print(f"Snap output: {snap_output}")
        print(f"Yard marker output: {yard_marker_output}")
        print(f"Correspondence output: {correspondence_output}")
        print(f"Homography output: {homography_output}")
        
        steps = [
            {
                "name": "Player Detection",
                "cmd": [
                    get_python_executable(), get_resource_path("scripts", "playerDetection.py"),
                    "--video", video_path,
                    "--output", detection_output
                ]
            },
            {
                "name": "Position Detection",
                "cmd": [
                    get_python_executable(), get_resource_path("scripts", "positionDetection.py"),
                    "--video", video_path,
                    "--output", position_output
                ]
            },
            {
                "name": "Snap Detection",
                "cmd": [
                    get_python_executable(), get_resource_path("scripts", "snapDetection.py"),
                    "--player-detections", detection_output,
                    "--output", snap_output
                ]
            },
            {
                "name": "Yard Marker Detection",
                "cmd": [
                    get_python_executable(), get_resource_path("scripts", "yardMarkerDetection.py"),
                    "--video", video_path,
                    "--output", yard_marker_output
                ]
            },
            {
                "name": "Correspondence Points Generation",
                "cmd": [
                    get_python_executable(), get_resource_path("scripts", "autoCorrespondancePoints.py"),
                    "--detection-json", yard_marker_output,
                    "--output", correspondence_output,
                    "--confidence", "0.7",
                    "--per-frame"
                ]
            },
            {
                "name": "Homography Transformation",
                "cmd": [
                    get_python_executable(), get_resource_path("scripts", "perFrameHomographyTransform.py"),
                    "--position-detections", detection_output,
                    "--correspondence-points", correspondence_output,
                    "--output", homography_output
                ],
                "prereq": [correspondence_output, detection_output]
            },
            {
                "name": "Static Process",
                "cmd": [
                    get_python_executable(), get_resource_path("CNN", "staticProcess.py"),
                    "--video-name", video_name,
                    "--folder-name", video_folder,
                    "--cache-dir", output_dir
                ],
                "prereq": [snap_output, homography_output]
            }
        ]
        
        emit_output(f"Starting processing for {video_name}")
        _batch_logger.debug(f"Processing {video_name}: Output directory: {output_dir}, Video folder: {video_folder}")
        for step in steps:
            if cancel_check and cancel_check():
                raise ProcessingCancelled("Processing cancelled before step start")
            
            step_name = step["name"]
            _batch_logger.info(f"Processing {video_name}: Starting step '{step_name}'")
            emit_status(f"{step_name} in progress")
            
            prereq = step.get("prereq")
            if prereq:
                missing = []
                if isinstance(prereq, list):
                    missing = [p for p in prereq if not os.path.exists(p)]
                else:
                    if not os.path.exists(prereq):
                        missing = [prereq]
                if missing:
                    _batch_logger.warning(f"Processing {video_name}: Missing prerequisites for {step_name}: {missing}")
                    emit_output(f"Missing prerequisites for {step_name}: {missing}")
                    return False
            
            success = _run_command_standalone(
                step["cmd"],
                step_name,
                output_callback=lambda msg, step_name=step_name: emit_output(f"{step_name}: {msg}"),
                cancel_check=cancel_check
            )
            
            if not success:
                _batch_logger.error(f"Processing {video_name}: Step '{step_name}' failed")
                emit_output(f"{step_name} failed for {video_name}")
                return False
            
            _batch_logger.info(f"Processing {video_name}: Step '{step_name}' completed successfully")
            emit_output(f"✓ {step_name} completed for {video_name}")
        
        _batch_logger.info(f"Processing {video_name}: All steps completed successfully")
        emit_output(f"All steps completed for {video_name}")
        return True
    
    except ProcessingCancelled:
        _batch_logger.warning(f"Processing cancelled for video: {video_path}")
        raise
    except Exception as e:
        _batch_logger.error(f"Error processing video {video_path}: {str(e)}", exc_info=True)
        emit_output(f"Error processing {video_path}: {str(e)}")
        return False

def _run_command_standalone(cmd, step_name, output_callback=None, cancel_check=None):
    """Run a command and stream output, optionally supporting cancellation."""
    # Get working directory (use _MEIPASS when compiled)
    if hasattr(sys, "_MEIPASS"):
        working_dir = sys._MEIPASS
    else:
        working_dir = get_project_root()
    emit_output = output_callback or (lambda msg: print(msg, flush=True))
    
    _batch_logger.debug(f"Running command for '{step_name}': {' '.join(cmd)}")
    _batch_logger.debug(f"Working directory: {working_dir}")
    
    try:
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True,
            cwd=working_dir,
            env=env
        )
        
        while True:
            if cancel_check and cancel_check():
                process.terminate()
                raise ProcessingCancelled(f"{step_name} cancelled")
            
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.05)
                continue
            
            emit_output(line.strip())
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code == 0:
            _batch_logger.debug(f"Command for '{step_name}' completed successfully (return code: 0)")
            return True
        
        _batch_logger.warning(f"Command for '{step_name}' failed with return code {return_code}")
        emit_output(f"{step_name} failed with return code {return_code}")
        return False
    
    except ProcessingCancelled:
        _batch_logger.warning(f"Command for '{step_name}' was cancelled")
        raise
    except Exception as e:
        _batch_logger.error(f"Exception running command for '{step_name}': {str(e)}", exc_info=True)
        emit_output(f"Error running {step_name}: {str(e)}")
        return False

class BatchProcessingWorker(QThread):
    """Worker thread for batch processing multiple videos"""
    progress_updated = Signal(int, str)  # progress percentage, status message
    output_received = Signal(str)  # terminal output
    video_completed = Signal(str, bool)  # video name, success status
    batch_completed = Signal(dict)  # results
    batch_failed = Signal(str)  # error message
    batch_cancelled = Signal()  # batch processing cancelled
    
    def __init__(self, video_paths, output_dir="cache", max_workers=2):
        super().__init__()
        self.video_paths = video_paths
        # Use persistent cache directory instead of temp PyInstaller folder
        if not os.path.isabs(output_dir):
            self.output_dir = get_cache_dir()
        else:
            self.output_dir = os.path.abspath(output_dir)
        self.max_workers = max_workers
        self.is_cancelled = False
        self.current_video_index = 0
        self.total_videos = len(video_paths)
        self.completed_videos = 0
        self.failed_videos = 0
        self.results = []
        self.lock = threading.Lock()
        
    def cancel(self):
        """Cancel the batch processing"""
        if not self.is_cancelled:
            self.is_cancelled = True
            self.output_received.emit("Cancelling batch processing...")
            _batch_logger.info("Batch processing worker cancelled by user")
    
    def is_cancelled_check(self):
        """Check if processing should be cancelled"""
        return self.is_cancelled
        
    def run(self):
        """Run batch processing sequentially (infrastructure ready for future parallelism)."""
        try:
            _batch_logger.info(f"Starting batch processing: {self.total_videos} videos, output_dir: {self.output_dir}")
            self.output_received.emit(f"Starting batch processing of {self.total_videos} videos")
            self.output_received.emit("Running in sequential mode (multi-worker ready)")
            self.output_received.emit("-" * 50)
            
            if self.total_videos == 0:
                _batch_logger.warning("Batch processing started with no videos to process")
                self.progress_updated.emit(100, "No videos to process")
                final_results = {
                    "total_videos": 0,
                    "completed_videos": 0,
                    "failed_videos": 0,
                    "results": [],
                    "status": "completed"
                }
                self.batch_completed.emit(final_results)
                return
            
            os.makedirs(self.output_dir, exist_ok=True)
            
            for index, video_path in enumerate(self.video_paths, start=1):
                if self.is_cancelled:
                    raise ProcessingCancelled("Processing cancelled before next clip")
                
                video_name = Path(video_path).stem or f"video_{index}"
                video_folder = os.path.basename(os.path.dirname(video_path)) or ""
                clip_label = f"[Clip {index}/{self.total_videos}] {video_name}"
                
                _batch_logger.info(f"Processing video {index}/{self.total_videos}: {video_name} (path: {video_path})")
                self.output_received.emit(f"{clip_label} - starting")
                
                def clip_output(message, prefix=clip_label):
                    self.output_received.emit(f"{prefix}: {message}")
                
                def clip_status(step_message, current_index=index, current_name=video_name):
                    processed_so_far = self.completed_videos + self.failed_videos
                    progress_percent = int((processed_so_far / self.total_videos) * 100) if self.total_videos else 0
                    status = f"Clip {current_index}/{self.total_videos} - {current_name}: {step_message}"
                    self.progress_updated.emit(progress_percent, status)
                
                try:
                    success = process_single_video(
                        video_path,
                        video_folder,
                        output_dir=self.output_dir,
                        output_callback=clip_output,
                        status_callback=clip_status,
                        cancel_check=self.is_cancelled_check
                    )
                except ProcessingCancelled:
                    _batch_logger.warning(f"Batch processing cancelled by user at video {index}/{self.total_videos}: {video_name}")
                    self.output_received.emit("Batch processing cancelled by user")
                    self.batch_cancelled.emit()
                    return
                except Exception as e:
                    _batch_logger.error(f"Unexpected error processing video {video_name}: {str(e)}", exc_info=True)
                    clip_output(f"Unexpected error: {e}")
                    success = False
                
                with self.lock:
                    if success:
                        self.completed_videos += 1
                        _batch_logger.info(f"Video {index}/{self.total_videos} completed successfully: {video_name}")
                        self.video_completed.emit(video_name, True)
                        clip_output("Completed successfully")
                    else:
                        self.failed_videos += 1
                        _batch_logger.warning(f"Video {index}/{self.total_videos} failed: {video_name}")
                        self.video_completed.emit(video_name, False)
                        clip_output("Failed")
                    
                    result_entry = {
                        "video_path": video_path,
                        "video_name": video_name,
                        "success": success,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    if not success:
                        result_entry["error"] = "Processing failed"
                    self.results.append(result_entry)
                    
                    total_processed = self.completed_videos + self.failed_videos
                    overall_progress = int((total_processed / self.total_videos) * 100) if self.total_videos else 100
                    self.progress_updated.emit(overall_progress, f"Processed {total_processed}/{self.total_videos} clips")
            
            self.progress_updated.emit(100, "Batch processing completed!")
            self.output_received.emit("\nBatch processing completed!")
            self.output_received.emit(f"Successfully processed: {self.completed_videos}/{self.total_videos} videos")
            self.output_received.emit(f"Failed: {self.failed_videos}/{self.total_videos} videos")
            
            _batch_logger.info(f"Batch processing completed: {self.completed_videos} succeeded, {self.failed_videos} failed out of {self.total_videos} total")
            
            final_results = {
                "total_videos": self.total_videos,
                "completed_videos": self.completed_videos,
                "failed_videos": self.failed_videos,
                "results": self.results,
                "status": "completed"
            }
            
            results_file = f"{self.output_dir}/batch_results_{int(time.time())}.json"
            try:
                with open(results_file, 'w') as f:
                    json.dump(final_results, f, indent=2)
                _batch_logger.info(f"Batch processing results saved to: {results_file}")
            except Exception as e:
                _batch_logger.error(f"Failed to save batch processing results to {results_file}: {e}", exc_info=True)
            
            self.output_received.emit(f"Results saved to: {results_file}")
            self.batch_completed.emit(final_results)
            
        except ProcessingCancelled:
            _batch_logger.warning("Batch processing cancelled by user")
            self.output_received.emit("Batch processing cancelled by user")
            self.batch_cancelled.emit()
        except Exception as e:
            error_msg = f"Error during batch processing: {str(e)}"
            _batch_logger.error(f"Batch processing failed: {error_msg}", exc_info=True)
            self.output_received.emit(f"ERROR: {error_msg}")
            self.batch_failed.emit(error_msg)


class BatchProcessingDialog(QDialog):
    """Modal dialog for batch processing multiple videos"""
    
    def __init__(self, parent):
        super().__init__(parent)
        self.worker = None
        self.video_paths = []
        self.parent_window = parent
        self.skipped_videos_count = 0
        _batch_logger.info(f"BatchProcessingDialog initialized. Log file: {_batch_log_file}")
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
        
        # Skip processed checkbox (placed close to buttons)
        self.skip_processed_checkbox = QCheckBox("Skip processed videos")
        self.skip_processed_checkbox.setStyleSheet("""
            QCheckBox {
                color: #000000;
                font-size: 11px;
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
                border: 2px solid #555555;
                border-radius: 3px;
                background-color: #2b2b2b;
            }
            QCheckBox::indicator:checked {
                background-color: #0078d4;
                border-color: #0078d4;
            }
            QCheckBox::indicator:hover {
                border-color: #0078d4;
            }
        """)
        self.skip_processed_checkbox.setToolTip("Skip videos that have already been fully processed")
        button_layout.addWidget(self.skip_processed_checkbox)
        
        # Add small spacing between checkbox and buttons
        button_layout.addSpacing(10)
        
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
        found_videos = []
        seen = set()
        
        for ext in video_extensions:
            for pattern in (os.path.join(folder, f"*{ext}"), os.path.join(folder, f"*{ext.upper()}")):
                for video_path in glob.glob(pattern):
                    if video_path not in seen:
                        seen.add(video_path)
                        found_videos.append(video_path)
        
        self.video_paths = found_videos
        _batch_logger.info(f"Found {len(self.video_paths)} unique videos in {folder}")
        _batch_logger.debug(f"Video paths: {self.video_paths}")
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
            _batch_logger.warning("Attempted to start batch processing with no videos")
            return
        
        _batch_logger.info(f"Starting batch processing for {len(self.video_paths)} videos")
        
        # Filter out already processed videos if checkbox is checked
        videos_to_process = self.video_paths.copy()
        skipped_count = 0
        
        if self.skip_processed_checkbox.isChecked():
            processed_videos = []
            unprocessed_videos = []
            
            # Get video folder from parent window
            video_folder = self.parent_window.current_folder if hasattr(self.parent_window, 'current_folder') else None
            
            for video_path in self.video_paths:
                if is_video_processed(video_path, video_folder):
                    processed_videos.append(video_path)
                    skipped_count += 1
                else:
                    unprocessed_videos.append(video_path)
            
            videos_to_process = unprocessed_videos
            
            if skipped_count > 0:
                _batch_logger.info(f"Skipping {skipped_count} already processed video(s)")
                self.add_output(f"Skipping {skipped_count} already processed video(s):")
                for video_path in processed_videos:
                    video_name = Path(video_path).stem
                    self.add_output(f"  - {video_name}")
                    _batch_logger.debug(f"Skipping already processed video: {video_name}")
                self.add_output("")
        
        if not videos_to_process:
            _batch_logger.info("All videos have already been processed. Nothing to do.")
            self.status_label.setText("All videos have already been processed!")
            self.add_output("All videos have already been processed. Nothing to do.")
            self.start_button.setEnabled(True)
            return
        
        # Store skipped count for later display
        self.skipped_videos_count = skipped_count
        
        # Update status to show how many videos will be processed
        if skipped_count > 0:
            self.status_label.setText(f"Processing {len(videos_to_process)} videos ({skipped_count} skipped)")
        else:
            self.status_label.setText(f"Processing {len(videos_to_process)} videos")
        
        # Get the number of parallel workers from the spinbox
        max_workers = 15
        
        _batch_logger.info(f"Creating batch processing worker for {len(videos_to_process)} videos (max_workers: {max_workers})")
        self.worker = BatchProcessingWorker(videos_to_process, max_workers=max_workers)
        
        # Connect signals
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.output_received.connect(self.add_output)
        self.worker.video_completed.connect(self.video_completed)
        self.worker.batch_completed.connect(self.batch_completed)
        self.worker.batch_failed.connect(self.batch_failed)
        self.worker.batch_cancelled.connect(self.batch_cancelled)
        
        # Update UI
        self.start_button.setEnabled(False)
        self.cancel_button.setVisible(True)
        
        _batch_logger.info("Starting batch processing worker thread")
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
        _batch_logger.info(f"Batch processing completed: {results['completed_videos']} succeeded, {results['failed_videos']} failed out of {results['total_videos']} total")
        self.progress_bar.setValue(100)
        self.status_label.setText("Batch processing completed!")
        
        # Show results
        self.add_output(f"\nBatch processing completed!")
        if self.skipped_videos_count > 0:
            self.add_output(f"Skipped (already processed): {self.skipped_videos_count} videos")
        self.add_output(f"Successfully processed: {results['completed_videos']}/{results['total_videos']} videos")
        self.add_output(f"Failed: {results['failed_videos']}/{results['total_videos']} videos")
        
        # Reload data sheet CSV
        self.reload_data_sheet_csv()
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.start_button.setVisible(False)
        self.close_button.setVisible(True)
    
    def reload_data_sheet_csv(self):
        """Reload the data sheet CSV after batch processing"""
        if not self.parent_window:
            return
        
        try:
            # Try to reload from current CSV path first
            if hasattr(self.parent_window, 'current_csv_path') and self.parent_window.current_csv_path:
                if os.path.exists(self.parent_window.current_csv_path):
                    self.add_output(f"\nReloading data sheet: {self.parent_window.current_csv_path}")
                    if hasattr(self.parent_window, 'load_csv_file'):
                        self.parent_window.load_csv_file(self.parent_window.current_csv_path)
                        self.add_output("Data sheet reloaded successfully")
                        return
            
            # Try to find CSV in cache directory based on current folder
            if hasattr(self.parent_window, 'current_folder') and self.parent_window.current_folder:
                folder_name = os.path.basename(self.parent_window.current_folder.rstrip('/\\'))
                # Use shared cache directory function
                base_cache_dir = get_cache_dir()
                csv_path = os.path.join(base_cache_dir, folder_name, f"{folder_name}_data.csv")
                
                if os.path.exists(csv_path) and hasattr(self.parent_window, 'load_csv_file'):
                    self.add_output(f"\nReloading data sheet from cache: {csv_path}")
                    self.parent_window.load_csv_file(csv_path)
                    self.add_output("Data sheet reloaded successfully")
                else:
                    self.add_output(f"\nNote: CSV file not found at {csv_path}")
        except Exception as e:
            self.add_output(f"\nWarning: Could not reload data sheet: {str(e)}")
            _batch_logger.error(f"Error reloading data sheet after batch processing: {e}", exc_info=True)
    
    def batch_failed(self, error_message):
        """Handle batch processing failure"""
        _batch_logger.error(f"Batch processing failed: {error_message}")
        self.status_label.setText("Batch processing failed!")
        self.add_output(f"\nBatch processing failed: {error_message}")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.start_button.setVisible(False)
        self.close_button.setVisible(True)
    
    def batch_cancelled(self):
        """Handle batch processing cancellation"""
        _batch_logger.info("Batch processing cancelled by user")
        self.status_label.setText("Batch processing cancelled")
        self.add_output(f"\nBatch processing cancelled by user")
        
        # Show close button and hide other buttons
        self.cancel_button.setVisible(False)
        self.start_button.setVisible(False)
        self.close_button.setVisible(True)
    
    def cancel_processing(self):
        """Cancel batch processing"""
        if self.worker and self.worker.isRunning():
            _batch_logger.info("User requested cancellation of batch processing")
            self.worker.cancel()
            # Don't wait here - let the worker handle the cancellation
            # The batch_cancelled signal will handle the UI updates
        else:
            # If no worker or worker not running, just update UI
            _batch_logger.info("Batch processing cancelled (no active worker)")
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
            # Give the worker a moment to cancel gracefully
            if not self.worker.wait(3000):  # 3 second timeout
                self.worker.terminate()  # Force terminate if needed
        event.accept()
