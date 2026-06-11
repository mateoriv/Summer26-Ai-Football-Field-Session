#!/usr/bin/env python3
"""
Script execution utilities shared by the single-video and batch processing
dialogs.

These helpers locate the project root / cache directory and build the command
used to run the pipeline scripts as subprocesses, handling both development
mode and PyInstaller one-file bundles.

This module intentionally avoids importing Qt (or any heavy dependency) at
import time so it can be imported and unit-tested in isolation. The fileAccess
import inside ``get_project_root``/``get_cache_dir`` is deferred to call time.
"""

import os
import subprocess
import sys


def get_project_root():
    """Return the project root, accounting for PyInstaller one-file extraction."""
    # Imported lazily to avoid pulling in Qt (via fileAccess) at module load.
    from fileAccess import get_project_root as get_root
    return get_root()


def get_cache_dir():
    """Get the cache directory path."""
    # Imported lazily to avoid pulling in Qt (via fileAccess) at module load.
    from fileAccess import get_cache_dir as get_cache
    return get_cache()


def get_resource_path(*relative_parts):
    """Build an absolute path rooted at the project directory or _MEIPASS when compiled."""
    if hasattr(sys, "_MEIPASS"):
        # Running as compiled executable - resources are in _MEIPASS
        return os.path.join(sys._MEIPASS, *relative_parts)
    else:
        # Running in development mode
        return os.path.join(get_project_root(), *relative_parts)


def get_python_executable():
    """Get the correct Python executable for the current platform.

    When running as a PyInstaller executable, returns sys.executable.
    When running in development, returns the system Python executable.
    """
    # If running as PyInstaller bundle, use sys.executable
    if hasattr(sys, "_MEIPASS") or getattr(sys, 'frozen', False):
        return sys.executable

    # Running in development mode - use system Python
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


def build_script_command(script_path, *args):
    """Build a command to run a Python script, handling PyInstaller bundles correctly.

    Args:
        script_path: Path to the Python script
        *args: Additional command-line arguments for the script

    Returns:
        Tuple of (command_list, env_dict, script_path) for PyInstaller mode
        Or (command_list, None, None) for development mode.
    """
    python_exe = get_python_executable()

    if hasattr(sys, "_MEIPASS") or getattr(sys, 'frozen', False):
        # For PyInstaller: use environment variable to signal script execution mode
        # The application.py will check this and run the script instead of launching GUI
        env = os.environ.copy()
        env['PYINSTALLER_RUN_SCRIPT'] = script_path

        # Build sys.argv for the script (pipe-separated for safety)
        argv_items = [os.path.basename(script_path)] + [str(arg) for arg in args]
        argv_str = '|'.join(argv_items)
        env['PYINSTALLER_SCRIPT_ARGV'] = argv_str

        # Return command (just the executable), env dict, and script path
        # The caller's _run_command method will need to use the env
        return ([python_exe], env, script_path)
    else:
        # Development mode: normal execution
        return ([python_exe, script_path] + list(args), None, None)
