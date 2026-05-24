#!/usr/bin/env python3
"""
Formation labeling tool (Phase 1.1 of FORMATION_TRAINING_PLAN.md).

Iterates clips in a cache folder, renders the 11 offense + 11 defense players
at the snap frame on a normalized field, and lets a human assign:
  - offense base formation + variation tags (multi-select)
  - defense front + coverage shell

Each save appends a row to:
  cache/<folder>/offense_positions.csv
  cache/<folder>/defense_positions.csv

Usage:
  python modelTraining/labeling_tool.py --cache cache --folder "Testing Footage"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)


# ---------------------------------------------------------------------------
# Taxonomy (inlined for Phase 1.1; extract to formations/taxonomy.json in Phase 2)
# ---------------------------------------------------------------------------

OFFENSE_BASE = [
    "DALLAS", "TREY", "EMPTY", "SLOT", "TRIPS", "DENVER", "DETROIT", "OTHER",
]
OFFENSE_TAGS = ["Y OFF", "U OFF", "TITE", "WING", "WG", "OPEN"]

DEFENSE_FRONT = ["3-4", "4-3", "BEAR", "46", "EVEN", "ODD", "NICKEL", "DIME", "OTHER"]
DEFENSE_COVERAGE = ["C0", "C1", "C2", "C3", "C4", "C2-MAN", "C-MATCH", "OTHER"]


# ---------------------------------------------------------------------------
# Field geometry (matches app/virtualField.py conventions: 100 x 53.33 yd, Y up)
# ---------------------------------------------------------------------------

FIELD_LENGTH_YD = 100.0
FIELD_WIDTH_YD = 53.33

DEFENSE_LABELS = {"defense", "def", "d"}
REF_LABELS = {"ref", "referee", "official"}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class ClipData:
    name: str
    snap_frame: Optional[int]
    offense_pts: list[tuple[float, float, str]] = field(default_factory=list)
    defense_pts: list[tuple[float, float, str]] = field(default_factory=list)
    note: str = ""  # human-readable status (why a clip is empty, etc.)


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _snap_frame_for(snap_path: str) -> Optional[int]:
    data = _read_json(snap_path)
    if not data:
        return None
    snaps = data.get("snaps") or []
    if snaps:
        return snaps[0].get("frame")
    return data.get("frame")


def _split_offense_defense(
    players: list[dict],
) -> tuple[list[tuple[float, float, str]], list[tuple[float, float, str]]]:
    offense: list[tuple[float, float, str]] = []
    defense: list[tuple[float, float, str]] = []
    for p in players:
        label = str(p.get("object_label", "")).strip().lower()
        if label in REF_LABELS:
            continue
        pos = p.get("normalized_position") or {}
        x = pos.get("x")
        y = pos.get("y")
        if x is None or y is None:
            continue
        entry = (float(x), float(y), label)
        if label in DEFENSE_LABELS:
            defense.append(entry)
        else:
            offense.append(entry)
    return offense, defense


def load_clips(cache_dir: str, folder_name: str) -> list[ClipData]:
    """Scan cache/<folder>/snap_detection/ and assemble a ClipData per video."""
    base = Path(cache_dir) / folder_name
    snap_dir = base / "snap_detection"
    hom_dir = base / "homography"

    if not snap_dir.is_dir():
        return []

    clips: list[ClipData] = []
    for snap_file in sorted(snap_dir.glob("*_snap_detection.json")):
        stem = snap_file.name.replace("_snap_detection.json", "")
        snap_frame = _snap_frame_for(str(snap_file))

        hom_path = hom_dir / f"{stem}_normalized_positions.json"
        hom = _read_json(str(hom_path)) if hom_path.exists() else None

        offense: list[tuple[float, float, str]] = []
        defense: list[tuple[float, float, str]] = []
        note = ""
        if hom is None:
            note = "no homography file"
        elif snap_frame is None:
            note = "no snap frame in snap_detection.json"
        else:
            frames = hom.get("normalized_positions") or {}
            players = frames.get(str(snap_frame), [])
            if not players:
                note = f"no players at snap frame {snap_frame}"
            else:
                offense, defense = _split_offense_defense(players)

        clips.append(ClipData(
            name=stem,
            snap_frame=snap_frame,
            offense_pts=offense,
            defense_pts=defense,
            note=note,
        ))
    return clips


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def _feature_header(prefix: str = "n") -> list[str]:
    out: list[str] = []
    for i in range(1, 12):
        out.extend([f"{prefix}x{i}", f"{prefix}y{i}"])
    return out


OFFENSE_CSV_HEADER = ["clip_name", "off_base", "off_tags"] + _feature_header("n")
DEFENSE_CSV_HEADER = ["clip_name", "def_front", "def_coverage"] + _feature_header("n")


def _pad_to_11(pts: list[tuple[float, float, str]]) -> list[tuple[float, float]]:
    """Return 11 (x, y) pairs, padding with (0, 0) if fewer were detected."""
    xy = [(x, y) for (x, y, _label) in pts[:11]]
    while len(xy) < 11:
        xy.append((0.0, 0.0))
    return xy


def _read_labeled_clip_names(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    seen: set[str] = set()
    try:
        with csv_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row.get("clip_name")
                if name:
                    seen.add(name)
    except Exception:
        pass
    return seen


def _append_row(csv_path: Path, header: list[str], row: list) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(header)
        writer.writerow(row)


def save_offense_row(csv_path: Path, clip: ClipData, base: str, tags: list[str]) -> None:
    pts = _pad_to_11(clip.offense_pts)
    row = [clip.name, base, ",".join(tags)]
    for x, y in pts:
        row.extend([f"{x:.6f}", f"{y:.6f}"])
    _append_row(csv_path, OFFENSE_CSV_HEADER, row)


def save_defense_row(csv_path: Path, clip: ClipData, front: str, coverage: str) -> None:
    pts = _pad_to_11(clip.defense_pts)
    row = [clip.name, front, coverage]
    for x, y in pts:
        row.extend([f"{x:.6f}", f"{y:.6f}"])
    _append_row(csv_path, DEFENSE_CSV_HEADER, row)


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

class FieldCanvas(QWidget):
    """Renders a normalized field with offense (orange) and defense (blue) dots."""

    OFFENSE_COLOR = QColor(255, 165, 0)
    DEFENSE_COLOR = QColor(60, 120, 255)
    QB_COLOR = QColor(255, 230, 0)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(800, 450)
        self.setStyleSheet("background-color: #1f1f1f;")
        self._clip: Optional[ClipData] = None

    def show_clip(self, clip: Optional[ClipData]) -> None:
        self._clip = clip
        self.update()

    def _field_rect(self) -> tuple[int, int, int, int]:
        """Return (x_offset, y_offset, width_px, height_px) for the field area."""
        margin = 24
        avail_w = max(self.width() - 2 * margin, 100)
        avail_h = max(self.height() - 2 * margin, 50)
        aspect = FIELD_LENGTH_YD / FIELD_WIDTH_YD
        if avail_w / avail_h > aspect:
            h = avail_h
            w = int(h * aspect)
        else:
            w = avail_w
            h = int(w / aspect)
        x = (self.width() - w) // 2
        y = (self.height() - h) // 2
        return x, y, w, h

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        fx, fy, fw, fh = self._field_rect()

        # Field background
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(QBrush(QColor(34, 110, 34)))
        painter.drawRect(fx, fy, fw, fh)

        # Yard lines every 10 yards
        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        for yd in range(10, int(FIELD_LENGTH_YD), 10):
            x = fx + int(yd * fw / FIELD_LENGTH_YD)
            painter.drawLine(x, fy, x, fy + fh)

        # Midline accent
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        mid_x = fx + fw // 2
        painter.drawLine(mid_x, fy, mid_x, fy + fh)

        if self._clip is None:
            painter.setPen(QPen(QColor(220, 220, 220)))
            painter.drawText(fx + 12, fy + 24, "Select a clip from the list")
            return

        # Player dots
        for (x, y, label) in self._clip.offense_pts[:11]:
            color = self.QB_COLOR if label in ("qb", "quarterback") else self.OFFENSE_COLOR
            self._draw_dot(painter, fx, fy, fw, fh, x, y, color)
        for (x, y, label) in self._clip.defense_pts[:11]:
            self._draw_dot(painter, fx, fy, fw, fh, x, y, self.DEFENSE_COLOR)

        # Header text
        painter.setPen(QPen(QColor(230, 230, 230)))
        info = f"{self._clip.name}  |  snap={self._clip.snap_frame}  |  off={len(self._clip.offense_pts)}  def={len(self._clip.defense_pts)}"
        if self._clip.note:
            info += f"  |  ⚠ {self._clip.note}"
        painter.drawText(fx, fy - 8, info)

    def _draw_dot(
        self,
        painter: QPainter,
        fx: int, fy: int, fw: int, fh: int,
        x_yd: float, y_yd: float,
        color: QColor,
    ) -> None:
        # Clip into field bounds defensively
        if not (0.0 <= x_yd <= FIELD_LENGTH_YD and 0.0 <= y_yd <= FIELD_WIDTH_YD):
            return
        px = fx + int(x_yd * fw / FIELD_LENGTH_YD)
        py = fy + fh - int(y_yd * fh / FIELD_WIDTH_YD)
        painter.setPen(QPen(QColor(255, 255, 255), 2))
        painter.setBrush(QBrush(color))
        painter.drawEllipse(px - 7, py - 7, 14, 14)


class LabelPanel(QWidget):
    """Right-side label entry form."""

    def __init__(self, parent=None):
        super().__init__(parent)

        self.off_base = QComboBox()
        self.off_base.addItems(OFFENSE_BASE)

        self.off_tag_boxes: dict[str, QCheckBox] = {}
        tags_box = QGroupBox("Offense tags")
        tags_layout = QHBoxLayout(tags_box)
        for tag in OFFENSE_TAGS:
            cb = QCheckBox(tag)
            self.off_tag_boxes[tag] = cb
            tags_layout.addWidget(cb)
        tags_layout.addStretch(1)

        self.def_front = QComboBox()
        self.def_front.addItems(DEFENSE_FRONT)
        self.def_coverage = QComboBox()
        self.def_coverage.addItems(DEFENSE_COVERAGE)

        off_box = QGroupBox("Offense")
        off_form = QFormLayout(off_box)
        off_form.addRow("Base formation:", self.off_base)

        def_box = QGroupBox("Defense")
        def_form = QFormLayout(def_box)
        def_form.addRow("Front:", self.def_front)
        def_form.addRow("Coverage:", self.def_coverage)

        self.save_btn = QPushButton("Save && next  (⌘S)")
        self.save_btn.setShortcut("Ctrl+S")
        self.skip_btn = QPushButton("Skip  (→)")
        self.skip_btn.setShortcut(Qt.Key_Right)

        btn_row = QHBoxLayout()
        btn_row.addWidget(self.skip_btn)
        btn_row.addWidget(self.save_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(off_box)
        layout.addWidget(tags_box)
        layout.addWidget(def_box)
        layout.addStretch(1)
        layout.addLayout(btn_row)

    def selected_tags(self) -> list[str]:
        return [t for t, cb in self.off_tag_boxes.items() if cb.isChecked()]

    def reset(self) -> None:
        self.off_base.setCurrentIndex(0)
        for cb in self.off_tag_boxes.values():
            cb.setChecked(False)
        self.def_front.setCurrentIndex(0)
        self.def_coverage.setCurrentIndex(0)


class LabelingWindow(QMainWindow):
    LABELED_MARK = "✓ "

    def __init__(self, cache_dir: str, folder_name: str):
        super().__init__()
        self.cache_dir = cache_dir
        self.folder_name = folder_name
        self.offense_csv = Path(cache_dir) / folder_name / "offense_positions.csv"
        self.defense_csv = Path(cache_dir) / folder_name / "defense_positions.csv"
        self.labeled: set[str] = (
            _read_labeled_clip_names(self.offense_csv)
            | _read_labeled_clip_names(self.defense_csv)
        )

        self.clips = load_clips(cache_dir, folder_name)
        if not self.clips:
            QMessageBox.critical(
                self, "No clips",
                f"No snap_detection JSON found under {cache_dir}/{folder_name}/snap_detection/.\n"
                "Run the pipeline first.",
            )
            sys.exit(1)

        self.setWindowTitle(f"Formation Labeling — {folder_name}")
        self.resize(1300, 720)

        # UI
        self.clip_list = QListWidget()
        for clip in self.clips:
            self._add_clip_item(clip)
        self.clip_list.currentRowChanged.connect(self._on_select)

        self.canvas = FieldCanvas()
        self.panel = LabelPanel()
        self.panel.save_btn.clicked.connect(self._on_save)
        self.panel.skip_btn.clicked.connect(self._advance)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("Clips"))
        left_layout.addWidget(self.clip_list)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.addWidget(self.panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        self.setCentralWidget(splitter)

        bar = QStatusBar()
        self.setStatusBar(bar)
        self._refresh_status()

        self.clip_list.setCurrentRow(self._first_unlabeled_row())

    def _add_clip_item(self, clip: ClipData) -> None:
        marker = self.LABELED_MARK if clip.name in self.labeled else "  "
        item = QListWidgetItem(f"{marker}{clip.name}")
        if clip.note:
            item.setForeground(QColor(200, 120, 120))
        self.clip_list.addItem(item)

    def _refresh_item(self, row: int) -> None:
        clip = self.clips[row]
        marker = self.LABELED_MARK if clip.name in self.labeled else "  "
        self.clip_list.item(row).setText(f"{marker}{clip.name}")

    def _first_unlabeled_row(self) -> int:
        for i, c in enumerate(self.clips):
            if c.name not in self.labeled:
                return i
        return 0

    def _refresh_status(self) -> None:
        self.statusBar().showMessage(
            f"{len(self.labeled)} / {len(self.clips)} labeled  |  writing to {self.offense_csv}"
        )

    def _on_select(self, row: int) -> None:
        if 0 <= row < len(self.clips):
            self.canvas.show_clip(self.clips[row])

    def _on_save(self) -> None:
        row = self.clip_list.currentRow()
        if row < 0:
            return
        clip = self.clips[row]
        base = self.panel.off_base.currentText()
        tags = self.panel.selected_tags()
        front = self.panel.def_front.currentText()
        coverage = self.panel.def_coverage.currentText()

        try:
            save_offense_row(self.offense_csv, clip, base, tags)
            save_defense_row(self.defense_csv, clip, front, coverage)
        except Exception as e:
            QMessageBox.warning(self, "Save failed", str(e))
            return

        self.labeled.add(clip.name)
        self._refresh_item(row)
        self._refresh_status()
        self.panel.reset()
        self._advance()

    def _advance(self) -> None:
        row = self.clip_list.currentRow()
        next_row = row + 1
        if next_row < len(self.clips):
            self.clip_list.setCurrentRow(next_row)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", default="cache", help="Cache root (default: cache)")
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder name under cache/ to label (e.g. 'Testing Footage')",
    )
    args = parser.parse_args()

    cache_dir = os.path.abspath(args.cache)
    if not os.path.isdir(os.path.join(cache_dir, args.folder)):
        print(
            f"[ERROR] Folder not found: {os.path.join(cache_dir, args.folder)}",
            file=sys.stderr,
        )
        sys.exit(2)

    app = QApplication(sys.argv)
    win = LabelingWindow(cache_dir, args.folder)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
