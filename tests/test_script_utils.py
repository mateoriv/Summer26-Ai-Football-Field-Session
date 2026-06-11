"""Tests for app/scriptUtils -- the script-command builder shared by the
single-video and batch processing dialogs (extracted from a copy-paste).

This module is intentionally Qt-free so it can be imported and tested without a
running event loop.
"""

import os
import sys

import scriptUtils


def test_get_python_executable_returns_a_real_interpreter_name():
    exe = scriptUtils.get_python_executable()
    assert isinstance(exe, str)
    # In dev mode it is one of these; under PyInstaller it is sys.executable.
    assert exe in ("python", "python3") or exe == sys.executable


def test_build_script_command_dev_mode(monkeypatch):
    # Ensure we exercise the development branch (not the PyInstaller branch).
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(sys, "frozen", False, raising=False)

    cmd, env, script = scriptUtils.build_script_command(
        "/path/to/script.py", "--arg", "value", 7,
    )

    assert env is None
    assert script is None
    # [python, script_path, *args] with args stringified by the caller chain.
    assert cmd[0] == scriptUtils.get_python_executable()
    assert cmd[1] == "/path/to/script.py"
    assert cmd[2:] == ["--arg", "value", 7]


def test_build_script_command_pyinstaller_mode(monkeypatch):
    # Simulate a frozen PyInstaller bundle.
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", "/tmp/_MEI", raising=False)

    cmd, env, script = scriptUtils.build_script_command(
        "/bundle/scripts/playerDetection.py", "--video", "clip.mp4",
    )

    assert cmd == [sys.executable]
    assert script == "/bundle/scripts/playerDetection.py"
    assert env is not None
    assert env["PYINSTALLER_RUN_SCRIPT"] == "/bundle/scripts/playerDetection.py"
    # argv is pipe-joined: basename then the args.
    assert env["PYINSTALLER_SCRIPT_ARGV"] == "playerDetection.py|--video|clip.mp4"


def test_get_resource_path_joins_under_meipass(monkeypatch):
    monkeypatch.setattr(sys, "_MEIPASS", os.path.join("/tmp", "_MEI"), raising=False)
    p = scriptUtils.get_resource_path("scripts", "staticProcess.py")
    assert p == os.path.join("/tmp", "_MEI", "scripts", "staticProcess.py")
