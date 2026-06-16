from PySide6.QtWidgets import QDockWidget, QWidget, QVBoxLayout, QLabel, QPushButton, QHBoxLayout, QSlider
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QBrush, QColor, QFont
import subprocess
import platform
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import sys
import os
import json
from fileAccess import get_cache_dir

# Import the color map definition from video.py for color consistency
try:
    # Attempt relative import from parent directory structure
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    from video import POSITION_COLORS
except ImportError:
    # Define a fallback or generic colors if import fails
    print("WARNING: Could not import POSITION_COLORS from video.py. Using default colors.")
    POSITION_COLORS = {
        'qb': QColor(255, 165, 0),
        'defense': QColor(0, 0, 255),
        'player': QColor(0, 255, 0)
    }

# Add scripts directory to path to import field drawing functions
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

class VirtualFieldWidget(QWidget):
    """Widget that displays a static football field with player dots"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent_window = parent
        self.current_frame = 0
        self.homography_data = None
        self.field_image = None
        # Pre-snap formation snapshot (classed players in field yards + LOS),
        # written by formations/line_count_classifier.save_snapshot. When set,
        # the field renders which team is attacking vs defending, the line of
        # scrimmage, and the attack direction -- the clear pre-snap picture.
        self.formation_snapshot = None
        # Predicted formation name (template matcher) for the info panel,
        # pushed by video.set_formation_info_for_virtual_field.
        self.formation_name = ""
        self.formation_confidence = None
        self.offense_selection_mode = False
        self.offense_label_points = []  # list of (center_x, center_y, class_name) in image space
        self.setMinimumSize(400, 300)
        self.setStyleSheet("background-color: #2b2b2b; border: 1px solid #555555;")
        
        # Load field image
        self.load_field_image()
        
    def load_field_image(self):
        """Load a simple football field image as background"""
        # Create a simple field image without matplotlib to avoid layering issues
        field_width = 400
        field_height = 200
        
        # Create a green field background
        field_img = np.full((field_height, field_width, 3), [34, 139, 34], dtype=np.uint8)
        
        # Draw yard lines (every 10 yards)
        for i in range(0, field_width, field_width // 10):  # 10 sections for 100 yards
            cv2.line(field_img, (i, 0), (i, field_height), (255, 255, 255), 2)
        
        # Draw hash marks (every 5 yards)
        for i in range(field_width // 20, field_width, field_width // 20):
            cv2.line(field_img, (i, field_height // 4), (i, 3 * field_height // 4), (255, 255, 255), 1)
        
        # Convert to QPixmap
        field_rgb = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)
        h, w, ch = field_rgb.shape
        bytes_per_line = ch * w
        from PySide6.QtGui import QImage
        q_image = QImage(field_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.field_image = QPixmap.fromImage(q_image)
        
    def load_homography_data(self, video_name, folder_name):
        """Load homography data for the current video"""
        try:
            # Use shared cache directory function
            base_cache_dir = get_cache_dir()
            homography_file = os.path.join(base_cache_dir, os.path.basename(folder_name), "homography", f"{video_name}_normalized_positions.json")
            
            if os.path.exists(homography_file):
                with open(homography_file, 'r') as f:
                    self.homography_data = json.load(f)
                print(f"Loaded homography data: {self.homography_data.get('total_frames')} frames")
                # Stable whole-clip team (jersey colour) is trusted only when
                # assignTeamColors flagged the split reliable; otherwise we fall
                # back to the snap-window role match rather than show wrong teams.
                meta = self.homography_data.get("team_color_meta") or {}
                self._team_reliable = bool(meta.get("reliable"))
                if "team_color_meta" in self.homography_data:
                    print(f"Team colours: reliable={self._team_reliable} "
                          f"(sep={meta.get('separation')})")
                # The live dots are all labeled generic 'player'; load the
                # position detector's per-frame ROLES so each live dot can be
                # colored by its real team (matched by pixel bbox).
                self._load_role_index(video_name, folder_name)
                
                # Set to first frame if homography data exists
                if self.homography_data and 'normalized_positions' in self.homography_data:
                    normalized_positions = self.homography_data['normalized_positions']
                    if normalized_positions:
                        # Find the first frame (minimum frame number)
                        frame_numbers = [int(k) for k in normalized_positions.keys() if k.isdigit()]
                        if frame_numbers:
                            first_frame = min(frame_numbers)
                            self.current_frame = first_frame
                            self.update()  # Update display to show first frame
                            print(f"Set virtual field to first frame: {first_frame}")
                
                return True
            else:
                print(f"Homography file not found: {homography_file}")
                # Clear homography data and reset frame to clear the display
                self.homography_data = None
                self.current_frame = 0
                self.update()  # Force repaint to clear player dots
                return False
        except Exception as e:
            print(f"Error loading homography data: {e}")
            # Clear homography data and reset frame to clear the display
            self.homography_data = None
            self.current_frame = 0
            self.update()  # Force repaint to clear player dots
            return False
    
    def set_current_frame(self, frame_number):
        """Set the current frame and update the display"""
        self.current_frame = frame_number
        self.update()

    def _load_role_index(self, video_name, folder_name):
        """Per-frame (center_x, center_y, role) from the position detector, so a
        live homography dot (all labeled generic 'player') can inherit its real
        team by matching pixel bbox centers. Live data only -- no snapshot."""
        self._role_by_frame = {}
        try:
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'formations'))
            import line_count_classifier as lcc
            pos_p = os.path.join(get_cache_dir(), os.path.basename(folder_name),
                                 "positions", f"{video_name}_position.json")
            if not os.path.exists(pos_p):
                return
            with open(pos_p) as f:
                pdata = json.load(f)
            for fr in pdata.get("frames", []):
                n = fr.get("frame_number")
                if n is None:
                    continue
                lst = []
                for d in fr.get("detections", []):
                    b = d.get("bbox") or {}
                    if b.get("center_x") is None:
                        continue
                    lst.append((b["center_x"], b["center_y"], lcc.normalize_class(d.get("class"))))
                self._role_by_frame[int(n)] = lst
            print(f"Loaded role index for live team coloring: {len(self._role_by_frame)} frames")
        except Exception as e:
            print(f"Role index load failed (live dots stay one color): {e}")

    def _role_for_dot(self, frame, original_bbox):
        """Nearest position-detector role to this live dot's pixel center."""
        roles = getattr(self, "_role_by_frame", {}).get(int(frame))
        if not roles or not original_bbox:
            return None
        cx, cy = original_bbox.get("center_x"), original_bbox.get("center_y")
        if cx is None or cy is None:
            return None
        best, best_d = None, 60.0 * 60.0  # match within ~60 px
        for rx, ry, role in roles:
            dd = (rx - cx) ** 2 + (ry - cy) ** 2
            if dd < best_d:
                best_d, best = dd, role
        return best

    def load_formation_snapshot(self, video_name, folder_name):
        """Load the pre-snap formation snapshot for this clip (or clear it)."""
        try:
            base_cache_dir = get_cache_dir()
            snap_file = os.path.join(base_cache_dir, os.path.basename(folder_name),
                                     "formation", f"{video_name}_formation.json")
            if os.path.exists(snap_file):
                with open(snap_file, "r") as f:
                    self.formation_snapshot = json.load(f)
                print(f"Loaded formation snapshot: front={self.formation_snapshot.get('on_line_count')}, "
                      f"strength={self.formation_snapshot.get('strength')}")
                self.update()
                return True
            self.formation_snapshot = None
            self.update()
            return False
        except Exception as e:
            print(f"Error loading formation snapshot: {e}")
            self.formation_snapshot = None
            self.update()
            return False

    def _live_formation_snapshot(self, video_name, folder_basename):
        """Run the line-count classifier on the cached snap and return a snapshot
        dict (offense/defense players in field yards + LOS) for rendering, or
        None if the clip can't be read. No video processing -- cache only."""
        try:
            base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
            for p in (os.path.join(base, "formations"), os.path.join(base, "scripts")):
                if p not in sys.path:
                    sys.path.append(p)
            import line_count_classifier as lcc
            res = lcc.recognize_from_cache(video_name, folder_basename, get_cache_dir())
            if not res or res.get("on_line_count") is None:
                return None
            return res  # already carries players[], los, on_line_count, strength, reliable
        except Exception as e:
            print(f"Live formation snapshot failed: {e}")
            return None

    def _field_rect(self):
        """Field draw rectangle inside the widget (x_off, y_off, w, h)."""
        field_width = min(self.width(), int(self.height() * 100 / 53.33))
        field_height = min(self.height(), int(self.width() * 53.33 / 100))
        x_off = (self.width() - field_width) // 2
        y_off = (self.height() - field_height) // 2
        return x_off, y_off, field_width, field_height

    def set_formation_info(self, name=None, confidence=None):
        """Predicted formation name for the info panel (template matcher)."""
        self.formation_name = name or ""
        self.formation_confidence = confidence
        self.update()

    def _to_widget(self, fx, fy):
        """Field yards (120 x 53.33, origin bottom-left) -> widget pixels."""
        x_off, y_off, fw, fh = self._field_rect()
        wx = int(x_off + (fx / 120.0) * fw)
        wy = int(y_off + fh - (fy / (160.0 / 3.0)) * fh)
        return wx, wy

    def _near_snap(self):
        """True when the current frame sits in the pre-snap window, where the
        classed formation snapshot is the truthful picture to show."""
        snap = self.formation_snapshot
        if not snap:
            return False
        sf = snap.get("snap_frame")
        if sf is None or not self.homography_data:
            return True  # static snapshot is all we have
        return abs(int(self.current_frame) - int(sf)) <= 20

    def _paint_los(self, painter):
        """Line of scrimmage + attack-direction arrow from the snapshot."""
        snap = self.formation_snapshot
        if not snap:
            return
        los = snap.get("los") or {}
        los_x = los.get("x_yd")
        if los_x is None:
            return
        attack_dir = int(los.get("attack_dir_x", snap.get("attack_dir_x", 1)) or 1)
        x_off, y_off, fw, fh = self._field_rect()
        lx, _ = self._to_widget(los_x, 0)
        painter.setPen(QPen(QColor(255, 255, 0), 2, Qt.DashLine))
        painter.drawLine(lx, y_off, lx, y_off + fh)
        # Attack-direction arrow from the LOS toward the defense.
        arrow_len = max(18, fw // 12)
        ax1 = lx + attack_dir * arrow_len
        ay = y_off + 16
        painter.setPen(QPen(QColor(220, 40, 40), 3))
        painter.drawLine(lx, ay, ax1, ay)
        painter.drawLine(ax1, ay, ax1 - attack_dir * 7, ay - 5)
        painter.drawLine(ax1, ay, ax1 - attack_dir * 7, ay + 5)

    def _paint_snapshot_players(self, painter):
        """The classed pre-snap picture: offense red / defense blue / QB gold,
        thick rings on the line-of-scrimmage front, QB + Center labeled."""
        snap = self.formation_snapshot
        x_off, y_off, fw, fh = self._field_rect()
        OFFENSE = QColor(220, 40, 40)
        DEFENSE = QColor(40, 90, 220)
        QB_GOLD = QColor(255, 200, 0)

        labels = []
        for p in snap.get("players", []):
            fx, fy = p.get("x"), p.get("y")
            if fx is None or fy is None:
                continue
            wx, wy = self._to_widget(fx, fy)
            if not (x_off <= wx <= x_off + fw and y_off <= wy <= y_off + fh):
                continue
            is_qb = (p.get("grp") == "qb" or p.get("pos") == "qb" or p.get("role") == "qb")
            color = QB_GOLD if is_qb else (OFFENSE if p.get("team") == "offense" else DEFENSE)
            painter.setBrush(QBrush(color))
            # Players ON the line of scrimmage get a thick white ring so the
            # "front count" is visually countable; others a thin ring. A
            # human-verified QB gets a thick gold ring.
            on_line = (p.get("pos") == "line")
            if p.get("vqb"):
                painter.setPen(QPen(QB_GOLD, 3))
            else:
                painter.setPen(QPen(QColor(255, 255, 255), 3 if on_line else 1))
            r = 8 if on_line else 7
            painter.drawEllipse(wx - r, wy - r, 2 * r, 2 * r)
            if is_qb:
                labels.append((wx, wy, "QB", QB_GOLD))
            elif p.get("ctr"):
                labels.append((wx, wy, "C", QColor(255, 255, 255)))
        # Labels last so dots never cover them.
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        for wx, wy, text, color in labels:
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(wx + 10, wy + 5, text)
            painter.setPen(QPen(color, 1))
            painter.drawText(wx + 9, wy + 4, text)

    def _paint_info_panel(self, painter):
        """Top-left panel: the formation read at a glance -- predicted name,
        front/box counts, strength, receivers per side, QB/C/team status."""
        snap = self.formation_snapshot or {}
        lines = []  # (text, color)
        WHITE = QColor(255, 255, 255)
        GREY = QColor(190, 190, 190)
        CYAN = QColor(0, 200, 255)
        AMBER = QColor(255, 191, 0)

        if self.formation_name:
            text = f"FORMATION: {self.formation_name.upper().replace('_', ' ')}"
            if self.formation_confidence is not None:
                try:
                    text += f"  ({float(self.formation_confidence):.2f})"
                except (TypeError, ValueError):
                    pass
            lines.append((text, CYAN))

        if snap.get("on_line_count") is not None:
            reliable = snap.get("reliable")
            bucket = snap.get("bucket") or "--"
            strength = snap.get("strength") or "--"
            flag = "" if reliable else "  (?)"
            lines.append((f"READ: {bucket}  ·  STRENGTH {strength}{flag}",
                          WHITE if reliable else AMBER))
            lines.append((f"FRONT: {snap.get('on_line_count')} on line  ·  "
                          f"BOX {snap.get('box_count', '--')}", WHITE))
            rl, rr = snap.get("recv_left"), snap.get("recv_right")
            if rl is not None:
                lines.append((f"RECEIVERS: {rl} left  /  {rr} right", WHITE))
            qb_ok = "✓" if snap.get("qb_recovered") else "--"
            c_ok = "✓" if snap.get("center_found") else "--"
            n_off = snap.get("n_offense", "--")
            n_def = snap.get("n_defense", "--")
            if snap.get("qb_verified"):
                lines.append((f"QB ✓ VERIFIED  ·  C {c_ok}  ·  OFF {n_off}  ·  DEF {n_def}",
                              QColor(255, 200, 0)))
            else:
                lines.append((f"QB {qb_ok}  ·  C {c_ok}  ·  OFF {n_off}  ·  DEF {n_def}", GREY))
            src = snap.get("team_source")
            if src:
                lines.append((f"TEAMS: {'jersey colour' if src == 'jersey_color' else 'detector class'}",
                              GREY))

        if not lines:
            return
        x_off, y_off, _, _ = self._field_rect()
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        fm = painter.fontMetrics()
        tw = max(fm.horizontalAdvance(t) for t, _ in lines)
        th = fm.height()
        pad = 8
        painter.fillRect(x_off + 6, y_off + 6, tw + 2 * pad, th * len(lines) + 2 * pad,
                         QColor(0, 0, 0, 180))
        for i, (text, color) in enumerate(lines):
            painter.setPen(QPen(color, 1))
            painter.drawText(x_off + 6 + pad, y_off + 6 + pad + fm.ascent() + i * th, text)

    def _paint_team_legend(self, painter):
        x_off, y_off, fw, fh = self._field_rect()
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        painter.setPen(QPen(QColor(220, 40, 40)))
        painter.drawText(x_off + 8, y_off + fh - 8, "● OFFENSE (attacking)")
        painter.setPen(QPen(QColor(40, 90, 220)))
        painter.drawText(x_off + 158, y_off + fh - 8, "● DEFENSE")
        painter.setPen(QPen(QColor(255, 200, 0)))
        painter.drawText(x_off + 238, y_off + fh - 8, "● QB")
    
    def paintEvent(self, event):
        """Paint the field with player dots"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw field background
        if self.field_image:
            # Calculate field dimensions within the widget (accounting for aspect ratio)
            field_width = min(self.width(), int(self.height() * 100 / 53.33))
            field_height = min(self.height(), int(self.width() * 53.33 / 100))
            
            # Center the field in the widget
            field_x_offset = (self.width() - field_width) // 2
            field_y_offset = (self.height() - field_height) // 2
            
            # Scale field image to fit the calculated field dimensions
            scaled_field = self.field_image.scaled(field_width, field_height, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            painter.drawPixmap(field_x_offset, field_y_offset, scaled_field)
        
        # Team colors so the per-frame tracking reads as offense vs defense + QB
        # (the field FOLLOWS the video frame by frame; coloring is by team, not
        # by the per-class palette). QB highlighted gold.
        OFFENSE = QColor(220, 40, 40)
        DEFENSE = QColor(40, 90, 220)
        QB_GOLD = QColor(255, 200, 0)
        OFFENSE_LABELS = {"oline", "qb", "running_back", "wide_receiver", "tight_end"}

        def team_color(label):
            if label == "qb":
                return QB_GOLD
            if label in OFFENSE_LABELS:
                return OFFENSE
            if label == "defense":
                return DEFENSE
            return QColor(150, 150, 150)  # ref / unknown

        # In the pre-snap window the classed formation snapshot (offense vs
        # defense, who is on the line, QB + Center) is the truthful picture --
        # show it instead of the raw live dots.
        near_snap = self._near_snap() and bool((self.formation_snapshot or {}).get("players"))
        if near_snap:
            self._paint_snapshot_players(painter)

        # Draw player dots if we have homography data
        if not near_snap and self.homography_data and 'normalized_positions' in self.homography_data:
            normalized_positions = self.homography_data['normalized_positions']
            frame_key = str(self.current_frame)
            
            if frame_key in normalized_positions:
                players = normalized_positions[frame_key]
                
                # Draw each player as a dot
                for i, player in enumerate(players):
                    # Get coordinates from normalized_position
                    normalized_pos = player.get('normalized_position', {})
                    x = normalized_pos.get('x', 0)
                    y = normalized_pos.get('y', 0)
                    
                    # Get object label (e.g., 'qb', 'defense')
                    object_label = player.get('object_label', 'player').lower()
                    
                    # Convert field coordinates to widget coordinates
                    # Field is 100 yards long and 53.33 yards wide
                    # (0,0) is bottom left corner of the field
                    
                    # Calculate field dimensions within the widget (accounting for aspect ratio)
                    field_width = min(self.width(), int(self.height() * 100 / 53.33))
                    field_height = min(self.height(), int(self.width() * 53.33 / 100))
                    
                    # Center the field in the widget
                    field_x_offset = (self.width() - field_width) // 2
                    field_y_offset = (self.height() - field_height) // 2
                    
                    # Map X: 0-100 yards to field_width pixels
                    widget_x = int(field_x_offset + (x * field_width / 100))
                    
                    # Map Y: 0-53.33 yards to field_height pixels (flip Y so 0,0 is bottom)
                    widget_y = int(field_y_offset + field_height - (y * field_height / 53.33))
                    
                    # Only draw if coordinates are within field bounds
                    field_left = field_x_offset
                    field_right = field_x_offset + field_width
                    field_top = field_y_offset
                    field_bottom = field_y_offset + field_height
                    
                    if field_left <= widget_x <= field_right and field_top <= widget_y <= field_bottom:
                        role = self._role_for_dot(self.current_frame, player.get('original_bbox'))
                        if getattr(self, 'offense_selection_mode', False):
                            # Offense-selection mode: color dots by the role
                            # class of the nearest labeled offense point.
                            resolved_label = object_label
                            label_pts = getattr(self, 'offense_label_points', [])
                            if label_pts:
                                orig_bbox = player.get('original_bbox', {})
                                ocx = orig_bbox.get('center_x')
                                ocy = orig_bbox.get('center_y')
                                if ocx is not None and ocy is not None:
                                    best_cls, _ = min(
                                        ((cls, (ocx - px) ** 2 + (ocy - py) ** 2)
                                         for px, py, cls in label_pts),
                                        key=lambda t: t[1]
                                    )
                                    resolved_label = best_cls
                            if resolved_label == 'defense':
                                dot_color = POSITION_COLORS['defense']
                            elif resolved_label == 'qb':
                                dot_color = POSITION_COLORS['qb']
                            elif resolved_label == 'wide_receiver':
                                dot_color = POSITION_COLORS['wide_receiver']
                            else:
                                dot_color = POSITION_COLORS['oline']
                        else:
                            # Stable whole-clip team from assignTeamColors (jersey
                            # colour, written into normalized_positions) -- trusted
                            # only when the split was flagged reliable. QB shown gold
                            # where the role detector saw it (snap window).
                            team = player.get('team') if getattr(self, '_team_reliable', False) else None
                            if role == 'qb':
                                dot_color = QB_GOLD
                            elif team == 'offense':
                                dot_color = OFFENSE
                            elif team == 'defense':
                                dot_color = DEFENSE
                            else:
                                dot_color = team_color(role or object_label)


                        # Draw player dot - make it more visible
                        painter.setBrush(QBrush(dot_color))
                        painter.setPen(QPen(QColor(255, 255, 255), 3))  # Thicker white border
                        painter.drawEllipse(widget_x - 8, widget_y - 8, 16, 16)  # Larger dot

                        # The QB stays labeled while the play runs.
                        if role == 'qb':
                            painter.setFont(QFont("Arial", 9, QFont.Bold))
                            painter.setPen(QPen(QColor(0, 0, 0), 1))
                            painter.drawText(widget_x + 11, widget_y + 5, "QB")
                            painter.setPen(QPen(QB_GOLD, 1))
                            painter.drawText(widget_x + 10, widget_y + 4, "QB")

        # Always-on overlays: line of scrimmage + attack arrow, the formation
        # info panel (top-left), and the team-colour legend.
        self._paint_los(painter)
        self._paint_info_panel(painter)
        self._paint_team_legend(painter)

        # Draw frame number (bottom-right corner, clear of panel and legend)
        x_off, y_off, fw, fh = self._field_rect()
        painter.setFont(QFont("Arial", 9))
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(x_off + fw - 90, y_off + fh - 8, f"Frame: {self.current_frame}")

        painter.end()

def draw_field(ax, correspondence_points=None):
    """Draw a college football field to scale (yards) - from drawPlayers.py"""
    import matplotlib.patches as patches
    
    # Field constants (yards)
    FIELD_LENGTH = 120.0                # 120 yards (100 + 2 endzones)
    FIELD_WIDTH = 160.0 / 3.0           # 160 ft -> yards (160/3 ~= 53.3333)
    HASH_DIST_FT = 40.0                 # hash marks are 40 ft from sideline
    HASH_NEAR_YD = HASH_DIST_FT / 3.0   # in yards (~13.3333)
    HASH_TOP_YD = FIELD_WIDTH - HASH_NEAR_YD
    HASH_LEN = 0.5

    # Base rectangle
    field = patches.Rectangle((0, 0), FIELD_LENGTH, FIELD_WIDTH, linewidth=2,
                              edgecolor='black', facecolor='green', zorder=0)
    ax.add_patch(field)

    # Yard lines every 5 yards (thinner) and every 10 (thicker)
    for x in range(10, int(FIELD_LENGTH), 5):
        lw = 2 if x % 10 == 0 else 1
        ax.plot([x, x], [0, FIELD_WIDTH], color='white', linewidth=lw, zorder=1)

    # Hash marks (every yard between 10 and 110 except multiples of 5)
    for x in range(11, 110):
        if x % 5 == 0:
            continue
        ax.plot([x, x], [HASH_NEAR_YD - HASH_LEN / 2, HASH_NEAR_YD + HASH_LEN / 2],
                color='white', linewidth=1, zorder=2)
        ax.plot([x, x], [HASH_TOP_YD - HASH_LEN / 2, HASH_TOP_YD + HASH_LEN / 2],
                color='white', linewidth=1, zorder=2)

    # End zones
    ez1 = patches.Rectangle((0, 0), 10, FIELD_WIDTH, linewidth=1,
                            edgecolor='white', facecolor='darkblue', alpha=0.6, zorder=1)
    ez2 = patches.Rectangle((110, 0), 10, FIELD_WIDTH, linewidth=1,
                            edgecolor='white', facecolor='darkred', alpha=0.6, zorder=1)
    ax.add_patch(ez1); ax.add_patch(ez2)

    # Yard numbers (every 10) - correct football field numbering
    for x in range(20, 110, 10):
        yard_number = x - 10  # Convert to actual yard number
        if yard_number <= 50:
            # First half: 10, 20, 30, 40, 50
            display_number = yard_number
        else:
            # Second half: 40, 30, 20, 10 (counting down from 50)
            display_number = 100 - yard_number
        
        ax.text(x, 9, str(display_number), color='white', fontsize=10, ha='center', va='center', zorder=3)
        ax.text(x, FIELD_WIDTH-9, str(display_number), color='white', fontsize=10, ha='center', va='center', rotation=180, zorder=3)

    ax.set_xlim(0, FIELD_LENGTH)
    ax.set_ylim(0, FIELD_WIDTH)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Draw yard marker dots if correspondence points are provided
    if correspondence_points:
        draw_yard_marker_dots(ax, correspondence_points)

def draw_yard_marker_dots(ax, correspondence_points=None):
    """
    Draw white dots on the field to show yard marker positions
    
    Args:
        ax: Matplotlib axes object
        correspondence_points: List of correspondence points with field coordinates
    """
    if not correspondence_points:
        return
    
    # Field dimensions (in yards for plotting)
    for point in correspondence_points:
        field_coords = point.get('field_point', {})
        marker_info = point.get('yard_marker_info', {})
        
        # Convert feet to yards for plotting
        x_yards = field_coords.get('x', 0) / 3.0  # Convert feet to yards
        y_yards = field_coords.get('y', 0) / 3.0  # Convert feet to yards
        
        # Draw white dot
        ax.plot(x_yards, y_yards, 'wo', markersize=8, markeredgecolor='black', 
                markeredgewidth=1, zorder=10)
        
        # Add label
        label = marker_info.get('label', '')
        ax.text(x_yards, y_yards + 2, label, color='white', fontsize=8, 
                ha='center', va='bottom', zorder=11, 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

def update_field_with_correspondence_points(parent, correspondence_file, frame_number=0):
    """
    Update the virtual field to show yard marker dots from correspondence points
    
    Args:
        parent: Main window parent
        correspondence_file: Path to correspondence points JSON file
        frame_number: Frame number to display (for per-frame correspondence points)
    """
    import json
    
    try:
        if os.path.exists(correspondence_file):
            with open(correspondence_file, 'r') as f:
                correspondence_data = json.load(f)
            
            # Check if this is per-frame data or single correspondence points
            if 'frame_correspondences' in correspondence_data:
                # Per-frame correspondence points
                frame_correspondences = correspondence_data.get('frame_correspondences', {})
                correspondence_points = frame_correspondences.get(str(frame_number), [])
                
                if not correspondence_points:
                    # Try to find the closest frame with correspondence points
                    available_frames = [int(f) for f in frame_correspondences.keys() if frame_correspondences[f]]
                    if available_frames:
                        closest_frame = min(available_frames, key=lambda x: abs(x - frame_number))
                        correspondence_points = frame_correspondences.get(str(closest_frame), [])
                        print(f"Using correspondence points from frame {closest_frame} (requested: {frame_number})")
            else:
                # Single correspondence points (legacy format)
                correspondence_points = correspondence_data.get('correspondences', [])
            
            # Clear and redraw field with yard marker dots
            if hasattr(parent, 'field_axes'):
                parent.field_axes.clear()
                draw_field(parent.field_axes, correspondence_points)
                parent.field_canvas.draw()
                
                # Update title to show frame number if using per-frame data
                if 'frame_correspondences' in correspondence_data:
                    parent.field_axes.set_title(f"Virtual Field - Frame {frame_number} ({len(correspondence_points)} points)", 
                                              color='white', fontsize=12)
                
        else:
            print(f"Correspondence file not found: {correspondence_file}")
            
    except Exception as e:
        print(f"Error updating field with correspondence points: {e}")

def load_correspondence_video(parent):
    """Load the correspondence points video for playback"""
    try:
        # Try to find the correspondence video file
        cache_dir = "cache"
        video_files = []
        
        if os.path.exists(cache_dir):
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.endswith("_correspondence_video.mp4"):
                        video_files.append(os.path.join(root, file))
        
        if video_files:
            # Get the most recent video file
            latest_video = max(video_files, key=os.path.getmtime)
            
            # Load video
            if hasattr(parent, 'video_cap') and parent.video_cap:
                parent.video_cap.release()
            
            parent.video_cap = cv2.VideoCapture(latest_video)
            if parent.video_cap.isOpened():
                total_frames = int(parent.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                parent.frame_slider.setMaximum(total_frames - 1)
                parent.frame_slider.setValue(0)
                parent.frame_label.setText(f"0 / {total_frames}")
                
                # Load first frame
                show_video_frame(parent, 0)
                
                print(f"Loaded correspondence points video: {latest_video}")
                return True
            else:
                print("Failed to open video file")
                return False
        else:
            print("No correspondence points video found. Please run the processing pipeline first.")
            return False
            
    except Exception as e:
        print(f"Error loading correspondence points video: {e}")
        return False

def show_video_frame(parent, frame_number):
    """Display a specific video frame"""
    if not hasattr(parent, 'video_cap') or not parent.video_cap:
        return
    
    try:
        parent.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = parent.video_cap.read()
        
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Clear the field and show video frame
            parent.field_axes.clear()
            parent.field_axes.imshow(frame_rgb)
            parent.field_axes.axis('off')
            parent.field_canvas.draw()
            
            # Update frame counter
            total_frames = int(parent.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            parent.frame_label.setText(f"{frame_number} / {total_frames}")
            
    except Exception as e:
        print(f"Error displaying video frame: {e}")

def toggle_video_playback(parent):
    """Toggle video play/pause"""
    if not hasattr(parent, 'video_cap') or not parent.video_cap:
        # Try to load video first
        if load_correspondence_video(parent):
            parent.video_timer.start(33)  # ~30 FPS
            parent.play_button.setText("⏸️ Pause")
        return
    
    if parent.video_timer.isActive():
        parent.video_timer.stop()
        parent.play_button.setText("▶️ Play")
    else:
        parent.video_timer.start(33)  # ~30 FPS
        parent.play_button.setText("⏸️ Pause")

def next_video_frame(parent):
    """Advance to next video frame"""
    if not hasattr(parent, 'video_cap') or not parent.video_cap:
        return
    
    current_frame = int(parent.video_cap.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(parent.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if current_frame < total_frames - 1:
        show_video_frame(parent, current_frame + 1)
        parent.frame_slider.setValue(current_frame + 1)
    else:
        # End of video - pause
        parent.video_timer.stop()
        parent.play_button.setText("▶️ Play")

def seek_video_frame(parent, frame_number):
    """Seek to a specific video frame"""
    if not hasattr(parent, 'video_cap') or not parent.video_cap:
        return
    
    show_video_frame(parent, frame_number)

def toggle_scoreboard(parent, button):
    """Toggle scoreboard visibility and resize field accordingly"""
    if hasattr(parent, 'scoreboard_widget'):
        if button.isChecked():
            parent.scoreboard_widget.show()
            # Scoreboard visible - use normal size
            if hasattr(parent, 'field_figure'):
                parent.field_figure.set_size_inches(16, 10)
                parent.field_canvas.draw()
        else:
            parent.scoreboard_widget.hide()
            # Scoreboard hidden - make field larger
            if hasattr(parent, 'field_figure'):
                parent.field_figure.set_size_inches(20, 12)  # Much larger when scoreboard is hidden
                parent.field_canvas.draw()

def create_dock_title_bar(dock, parent):
    """Create a custom title bar for the dock widget with scoreboard toggle"""
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QPushButton, QLabel
    from PySide6.QtCore import Qt
    
    title_bar = QWidget()
    title_bar.setStyleSheet("""
        QWidget {
            background-color: #3c3c3c;
            border: none;
            padding: 2px;
        }
    """)
    
    layout = QHBoxLayout()
    layout.setContentsMargins(5, 2, 5, 2)
    layout.setSpacing(5)
    
    # Title label
    title_label = QLabel(dock.windowTitle())
    title_label.setStyleSheet("color: white; font-weight: bold;")
    layout.addWidget(title_label)
    
    layout.addStretch()
    
    # Scoreboard toggle button
    scoreboard_btn = QPushButton("📊")
    scoreboard_btn.setCheckable(True)
    scoreboard_btn.setChecked(False)  # Start with scoreboard hidden
    scoreboard_btn.setToolTip("Toggle Scoreboard")
    scoreboard_btn.setStyleSheet("""
        QPushButton {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 3px;
            padding: 4px 8px;
            font-size: 12px;
        }
        QPushButton:hover {
            background-color: #45a049;
        }
        QPushButton:checked {
            background-color: #2196F3;
        }
    """)
    scoreboard_btn.clicked.connect(lambda: toggle_scoreboard(parent, scoreboard_btn))
    layout.addWidget(scoreboard_btn)
    
    title_bar.setLayout(layout)
    return title_bar

def create_virtual_field_dock(parent):
    """Create a simplified virtual field dock with static field image and player dots"""
    dock = QDockWidget("Virtual Field", parent)
    dock.setAllowedAreas(Qt.AllDockWidgetAreas)
    dock.setFeatures(QDockWidget.DockWidgetMovable)
    
    # Set custom title bar with scoreboard toggle
    dock.setTitleBarWidget(create_dock_title_bar(dock, parent))
    
    # Main widget
    main_widget = QWidget()
    layout = QVBoxLayout()
    layout.setContentsMargins(5, 5, 5, 5)
    
    # Create scoreboard section
    scoreboard_widget = create_scoreboard(parent)
    scoreboard_widget.hide()  # Start with scoreboard hidden
    layout.addWidget(scoreboard_widget)
    
    # Field on the left, formation choice panel on the right, side by side so
    # the coach compares the recommended formation pictures against the field.
    field_row = QHBoxLayout()
    field_row.setSpacing(6)
    virtual_field = VirtualFieldWidget(parent)
    field_row.addWidget(virtual_field, stretch=1)

    try:
        from formationPanel import FormationPanel
        formation_panel = FormationPanel(parent)
        field_row.addWidget(formation_panel, stretch=0)
        parent.formation_panel = formation_panel
        # Confirm -> refresh the formation label/overlays (cache-only, deferred
        # import so this module doesn't import video at load time).
        def _on_formation_confirmed(_name, p=parent):
            try:
                from video import _load_formation_label_for_current_video
                _load_formation_label_for_current_video(p)
            except Exception as e:
                print(f"[FORMATION PANEL] refresh after confirm failed: {e}")
        formation_panel.formation_confirmed.connect(_on_formation_confirmed)
    except Exception as e:
        print(f"[FORMATION PANEL] not loaded: {e}")
        parent.formation_panel = None

    layout.addLayout(field_row)

    # Store reference for updates
    parent.virtual_field = virtual_field

    main_widget.setLayout(layout)
    dock.setWidget(main_widget)
    
    return dock

def update_virtual_field_with_video_frame(parent, frame_number):
    """Update the virtual field to show the current video frame's player positions"""
    if hasattr(parent, 'virtual_field'):
        parent.virtual_field.set_current_frame(frame_number)

def load_homography_data_for_virtual_field(parent, video_name, folder_name):
    """Load homography data for the virtual field"""
    if hasattr(parent, 'virtual_field'):
        return parent.virtual_field.load_homography_data(video_name, folder_name)
    return False


def load_formation_snapshot_for_virtual_field(parent, video_name, folder_name):
    """Load the pre-snap formation snapshot (offense/defense + LOS) for the field."""
    if hasattr(parent, 'virtual_field'):
        return parent.virtual_field.load_formation_snapshot(video_name, folder_name)
    return False

def create_scoreboard(parent):
    """Create a scoreboard widget with orange football scoreboard design"""
    from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QFrame
    from PySide6.QtCore import Qt
    
    scoreboard_widget = QFrame()
    scoreboard_widget.setStyleSheet("""
        QFrame {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                stop:0 #FF6B35, stop:1 #E55A2B);
            border: 3px solid #FF8C42;
            border-radius: 12px;
            padding: 15px;
        }
    """)
    
    layout = QHBoxLayout()
    layout.setSpacing(30)
    layout.setContentsMargins(20, 15, 20, 15)
    
    # Home team section
    home_layout = QVBoxLayout()
    home_layout.setAlignment(Qt.AlignCenter)
    home_team_label = QLabel("HOME")
    home_team_label.setStyleSheet("""
        color: white; 
        font-weight: bold; 
        font-size: 16px;
        background-color: rgba(0,0,0,0.3);
        padding: 5px 10px;
        border-radius: 5px;
    """)
    home_score_label = QLabel("0")
    home_score_label.setStyleSheet("""
        color: white; 
        font-size: 36px; 
        font-weight: bold;
        background-color: rgba(0,0,0,0.4);
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.3);
    """)
    home_layout.addWidget(home_team_label)
    home_layout.addWidget(home_score_label)
    layout.addLayout(home_layout)
    
    # Game info section
    game_layout = QVBoxLayout()
    game_layout.setAlignment(Qt.AlignCenter)
    quarter_label = QLabel("Q1")
    quarter_label.setStyleSheet("""
        color: white; 
        font-size: 20px; 
        font-weight: bold;
        background-color: rgba(0,0,0,0.3);
        padding: 8px 15px;
        border-radius: 6px;
    """)
    time_label = QLabel("15:00")
    time_label.setStyleSheet("""
        color: white; 
        font-size: 24px; 
        font-weight: bold;
        background-color: rgba(0,0,0,0.4);
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.3);
    """)
    game_layout.addWidget(quarter_label)
    game_layout.addWidget(time_label)
    layout.addLayout(game_layout)
    
    # Away team section
    away_layout = QVBoxLayout()
    away_layout.setAlignment(Qt.AlignCenter)
    away_team_label = QLabel("AWAY")
    away_team_label.setStyleSheet("""
        color: white; 
        font-weight: bold; 
        font-size: 16px;
        background-color: rgba(0,0,0,0.3);
        padding: 5px 10px;
        border-radius: 5px;
    """)
    away_score_label = QLabel("0")
    away_score_label.setStyleSheet("""
        color: white; 
        font-size: 36px; 
        font-weight: bold;
        background-color: rgba(0,0,0,0.4);
        padding: 10px 20px;
        border-radius: 8px;
        border: 2px solid rgba(255,255,255,0.3);
    """)
    away_layout.addWidget(away_team_label)
    away_layout.addWidget(away_score_label)
    layout.addLayout(away_layout)
    
    scoreboard_widget.setLayout(layout)
    
    # Store reference for toggling
    parent.scoreboard_widget = scoreboard_widget
    
    return scoreboard_widget