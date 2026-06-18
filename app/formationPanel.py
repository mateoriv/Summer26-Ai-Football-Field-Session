"""
Formation choice panel shown to the right of the virtual field.

The coach sees, for the current clip:
  * the SYSTEM's pick, marked with its confidence  (e.g. "DETROIT  system (0.32)")
  * the top-3 recommended formations as one-tap buttons
  * a dropdown with all 17 formations (the override long tail)
  * a PICTURE of whichever formation the mouse is over, so the coach compares
    the diagram against the field by eye -- no playbook on the side needed

Picking is never automatic: the system's guess is highlighted as the fast
default, but the coach must tap Confirm. Confirm persists a verified label
(verified_store) and records agreed-vs-overrode, which is the live accuracy
meter + clean training truth.
"""
import csv
import os
import sys

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QFrame,
    QListWidget, QListWidgetItem,
)
from PySide6.QtCore import Qt, Signal, QSize
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont

from fileAccess import get_cache_dir

# Template coords + verified store live in formations/.
_FORM_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "formations")
if _FORM_DIR not in sys.path:
    sys.path.append(_FORM_DIR)

TEMPLATE_CSV = os.path.join(_FORM_DIR, "offense_formation_coordinates_17.csv")
ROLE_ORDER = ["Q", "T", "W", "S", "X", "Y", "Z", "U"]
# 5 OL on the LOS (depth 0), ~1.25 yd splits -- matches template_matcher.OL_TEMPLATE.
OL_POINTS = [(-2.5, 0.0), (-1.25, 0.0), (0.0, 0.0), (1.25, 0.0), (2.5, 0.0)]


def pretty(name):
    """template key -> coach-readable label: 'trey_y_off' -> 'TREY Y OFF'."""
    return (name or "").upper().replace("_", " ").strip()


def load_templates(csv_path=TEMPLATE_CSV):
    """name -> list of (kind, x, y); kind in {ol, qb, back, recv}."""
    out = {}
    try:
        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                name = (row.get("formation") or "").strip()
                if not name:
                    continue
                pts = [("ol", x, y) for x, y in OL_POINTS]
                for r in ROLE_ORDER:
                    xs = (row.get(f"{r}_x") or "").strip()
                    ys = (row.get(f"{r}_y") or "").strip()
                    if xs == "" or ys == "":
                        continue
                    kind = "qb" if r == "Q" else ("back" if r == "T" else "recv")
                    pts.append((kind, float(xs), float(ys)))
                out[name] = pts
    except OSError:
        pass
    return out


def render_formation(points, w=240, h=190):
    """Draw a formation diagram (top-down, LOS horizontal) to a QPixmap."""
    pm = QPixmap(w, h)
    pm.fill(QColor(28, 96, 40))  # field green
    if not points:
        return pm
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)

    # Fixed yard window so every formation renders at the same scale/orientation
    # as its neighbours (lateral x in [-13,13], depth y in [-6,1.5], LOS at y=0).
    xmin, xmax, ymin, ymax = -13.0, 13.0, -6.0, 1.5
    m = 14

    def to_px(x, y):
        px = m + (x - xmin) / (xmax - xmin) * (w - 2 * m)
        py = m + (ymax - y) / (ymax - ymin) * (h - 2 * m)
        return int(px), int(py)

    # LOS line
    lx0, ly = to_px(xmin, 0.0)
    lx1, _ = to_px(xmax, 0.0)
    p.setPen(QPen(QColor(245, 230, 90), 2, Qt.DashLine))
    p.drawLine(lx0, ly, lx1, ly)

    QB = QColor(255, 200, 0)
    BACK = QColor(255, 140, 0)
    RECV = QColor(225, 55, 55)
    OL = QColor(235, 235, 235)
    for kind, x, y in points:
        cx, cy = to_px(x, y)
        if kind == "ol":
            p.setBrush(QBrush(OL)); p.setPen(QPen(QColor(60, 60, 60), 1))
            p.drawRect(cx - 5, cy - 5, 10, 10)
        else:
            color = QB if kind == "qb" else (BACK if kind == "back" else RECV)
            p.setBrush(QBrush(color)); p.setPen(QPen(QColor(255, 255, 255), 2))
            p.drawEllipse(cx - 6, cy - 6, 12, 12)
            if kind == "qb":
                p.setFont(QFont("Arial", 8, QFont.Bold))
                p.setPen(QPen(QColor(0, 0, 0)))
                p.drawText(cx + 8, cy + 4, "QB")
    p.end()
    return pm


class HoverButton(QPushButton):
    """QPushButton that emits `hovered` when the mouse enters it."""
    hovered = Signal(str)

    def __init__(self, key, text, parent=None):
        super().__init__(text, parent)
        self._key = key

    def enterEvent(self, event):
        self.hovered.emit(self._key)
        super().enterEvent(event)


class FormationPanel(QWidget):
    """Right-side panel: system pick (marked) + top-3 + dropdown + preview."""
    formation_confirmed = Signal(str)  # chosen label (pretty form)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.templates = load_templates()
        self.system_pick = None        # template key
        self.system_conf = None
        self.selected_key = None       # currently chosen template key
        self.video_name = None
        self.folder_name = None
        self.setFixedWidth(270)
        self.setStyleSheet("background-color: #2b2b2b;")
        self._build_ui()
        self.set_choices(None, None, [], None, None)

    # ---------- UI ----------
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        header = QLabel("FORMATION")
        header.setStyleSheet("color:#dddddd; font-weight:bold; font-size:12px;")
        lay.addWidget(header)

        # System pick, marked with confidence
        self.system_label = QLabel()
        self.system_label.setWordWrap(True)
        self.system_label.setStyleSheet(
            "color:#00c8ff; font-weight:bold; font-size:12px;"
            "border:1px solid #00688c; border-radius:4px; padding:5px;")
        lay.addWidget(self.system_label)

        rec = QLabel("Recommended")
        rec.setStyleSheet("color:#9a9a9a; font-size:10px;")
        lay.addWidget(rec)
        self.rec_buttons = []
        for _ in range(3):
            b = HoverButton("", "", self)
            b.setStyleSheet(self._btn_style())
            b.hovered.connect(self._on_hover)
            b.clicked.connect(lambda _=False, btn=b: self._select(btn._key))
            self.rec_buttons.append(b)
            lay.addWidget(b)

        alll = QLabel("All formations")
        alll.setStyleSheet("color:#9a9a9a; font-size:10px;")
        lay.addWidget(alll)
        self.combo = QComboBox()
        self.combo.setStyleSheet(
            "QComboBox{color:#eee; background:#3a3a3a; padding:4px; font-size:11px;}")
        for key in sorted(self.templates):
            self.combo.addItem(pretty(key), key)
        self.combo.highlighted.connect(  # fires on hover over a dropdown item
            lambda i: self._on_hover(self.combo.itemData(i)))
        self.combo.activated.connect(
            lambda i: self._select(self.combo.itemData(i)))
        lay.addWidget(self.combo)

        # Preview picture of the hovered/selected formation
        self.preview = QLabel()
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setFixedHeight(190)
        self.preview.setStyleSheet("border:1px solid #555;")
        lay.addWidget(self.preview)
        self.preview_caption = QLabel("")
        self.preview_caption.setAlignment(Qt.AlignCenter)
        self.preview_caption.setStyleSheet("color:#ddd; font-size:11px; font-weight:bold;")
        lay.addWidget(self.preview_caption)

        self.confirm_btn = QPushButton("Confirm selection")
        self.confirm_btn.setStyleSheet(
            "QPushButton{background:#2e7d32; color:white; font-weight:bold;"
            "padding:7px; border-radius:4px;} QPushButton:hover{background:#388e3c;}"
            "QPushButton:disabled{background:#444; color:#888;}")
        self.confirm_btn.clicked.connect(self._on_confirm)
        lay.addWidget(self.confirm_btn)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#bbb; font-size:10px;")
        lay.addWidget(self.status)

        # Adjustment history -- the human-in-the-loop record lives here in the
        # board (next to Confirm), not as a separate overlay on the field.
        hist_header = QLabel("History")
        hist_header.setStyleSheet("color:#9a9a9a; font-size:10px;")
        lay.addWidget(hist_header)
        self.history_list = QListWidget()
        self.history_list.setFixedHeight(120)
        self.history_list.setStyleSheet(
            "QListWidget{color:#ccc; background:#222; border:1px solid #444;"
            "font-size:10px;} QListWidget::item{padding:2px;}")
        lay.addWidget(self.history_list)
        lay.addStretch()

    def _refresh_history(self):
        """Reload this clip's confirmation history into the in-panel list,
        newest first. Each row reads as the correction story: agreed vs override."""
        self.history_list.clear()
        if not self.video_name:
            return
        try:
            import verified_store
            hist = verified_store.load_formation_history(
                self.video_name, self.folder_name, get_cache_dir())
        except Exception as e:
            self.history_list.addItem(f"(history unavailable: {e})")
            return
        if not hist:
            item = QListWidgetItem("— no confirmations yet —")
            item.setForeground(QColor("#888"))
            self.history_list.addItem(item)
            return
        for row in reversed(hist):  # newest first
            chosen = row.get("chosen_formation", "?")
            sys_pick = row.get("system_pick", "")
            agreed = row.get("agreed")
            if not isinstance(agreed, bool):
                agreed = str(agreed).strip().lower() == "true"
            if agreed:
                text = f"✓ {chosen}  (agreed with AI)"
                color = QColor("#7dd77d")
            else:
                text = f"✎ {chosen}" + (f"   (AI: {sys_pick})" if sys_pick else "  (override)")
                color = QColor("#ffcf6b")
            item = QListWidgetItem(text)
            item.setForeground(color)
            self.history_list.addItem(item)

    def _btn_style(self, marked=False):
        if marked:  # the system pick: amber border so the coach sees the AI choice
            return ("QPushButton{color:#fff; background:#3a3a3a; text-align:left;"
                    "padding:6px; border:2px solid #ffb300; border-radius:4px;}"
                    "QPushButton:hover{background:#454545;}")
        return ("QPushButton{color:#eee; background:#3a3a3a; text-align:left;"
                "padding:6px; border:1px solid #555; border-radius:4px;}"
                "QPushButton:hover{background:#454545;}")

    # ---------- data in ----------
    def set_choices(self, system_pick, confidence, ranking, video_name, folder_name):
        """ranking: list of template keys, best first. system_pick: template key."""
        self.system_pick = system_pick
        self.system_conf = confidence
        self.video_name = video_name
        self.folder_name = folder_name
        self.selected_key = system_pick

        if system_pick:
            conf = f" ({float(confidence):.2f})" if confidence is not None else ""
            self.system_label.setText(f"{pretty(system_pick)}\nsystem chose{conf}")
        else:
            self.system_label.setText("no system read for this clip")

        # top-3 (dedup, keep order; pad from nothing if short)
        seen, top = set(), []
        for k in ranking:
            if k and k not in seen and k in self.templates:
                seen.add(k); top.append(k)
            if len(top) == 3:
                break
        for i, b in enumerate(self.rec_buttons):
            if i < len(top):
                k = top[i]
                is_sys = (k == system_pick)
                b.setText(("★ " if is_sys else "") + pretty(k) + ("  · system" if is_sys else ""))
                b._key = k
                b.setStyleSheet(self._btn_style(marked=is_sys))
                b.setEnabled(True); b.show()
            else:
                b.hide()

        # sync dropdown + preview to the system pick
        if system_pick:
            idx = self.combo.findData(system_pick)
            if idx >= 0:
                self.combo.setCurrentIndex(idx)
            self._show_preview(system_pick)
        else:
            self.preview.clear(); self.preview_caption.setText("")
        self.confirm_btn.setEnabled(bool(system_pick) or bool(self.templates))
        self.status.setText("")
        self._refresh_history()

    # ---------- interactions ----------
    def _on_hover(self, key):
        if key:
            self._show_preview(key)

    def _select(self, key):
        if not key:
            return
        self.selected_key = key
        self._show_preview(key)
        idx = self.combo.findData(key)
        if idx >= 0 and self.combo.currentIndex() != idx:
            self.combo.setCurrentIndex(idx)

    def _show_preview(self, key):
        pts = self.templates.get(key)
        self.preview.setPixmap(render_formation(pts, self.preview.width() or 240, 188))
        tag = ""
        if key == self.system_pick:
            tag = "  (system pick)"
        elif self.selected_key == key:
            tag = "  (selected)"
        self.preview_caption.setText(pretty(key) + tag)

    def _on_confirm(self):
        key = self.selected_key
        if not key or not self.video_name:
            self.status.setText("nothing to confirm")
            return
        chosen = pretty(key)
        agreed = (key == self.system_pick)
        sys_pick = pretty(self.system_pick) if self.system_pick else ""
        try:
            import verified_store
            base_cache = get_cache_dir()
            prev = (verified_store.load_verified(
                self.video_name, self.folder_name, base_cache) or {}).get("formation") or {}
            prev_formation = prev.get("formation", "")
            verified_store.save_formation_verification(
                self.video_name, self.folder_name, base_cache,
                {"formation": chosen, "system_pick": sys_pick,
                 "system_confidence": self.system_conf, "agreed": bool(agreed)})
            verified_store.append_formation_training_label(self.folder_name, base_cache, {
                "folder": self.folder_name, "clip": self.video_name,
                "chosen_formation": chosen,
                "system_pick": sys_pick,
                "system_confidence": "" if self.system_conf is None else self.system_conf,
                "agreed": bool(agreed), "verified_at": ""})
            # Append-only audit trail of every confirmation/correction for this clip.
            hist = verified_store.load_formation_history(
                self.video_name, self.folder_name, base_cache)
            verified_store.append_formation_history(self.folder_name, base_cache, {
                "folder": self.folder_name, "clip": self.video_name,
                "seq": len(hist),  # save_* already appended this confirmation
                "chosen_formation": chosen,
                "previous_formation": prev_formation,
                "system_pick": sys_pick,
                "system_confidence": "" if self.system_conf is None else self.system_conf,
                "agreed": bool(agreed),
                "verified_at": (hist[-1].get("verified_at", "") if hist else "")})
        except Exception as e:
            self.status.setText(f"save failed: {e}")
            return
        if agreed:
            self.status.setText(f"✓ Saved — agreed with AI ({chosen})")
        else:
            sp = pretty(self.system_pick) if self.system_pick else "—"
            self.status.setText(f"✓ Saved — overrode AI (was {sp})  →  {chosen}")
        self._refresh_history()
        self.formation_confirmed.emit(chosen)
