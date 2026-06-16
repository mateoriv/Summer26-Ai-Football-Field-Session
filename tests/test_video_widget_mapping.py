"""Offscreen Qt smoke test: widget click -> video pixel mapping for the QB
verification flow. Skips when no Qt platform can initialize (e.g. sandbox)."""

import pytest

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def qapp():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception:
            pytest.skip("no Qt platform available")
    return app


@pytest.fixture
def widget(qapp):
    from video import CustomVideoWidget
    w = CustomVideoWidget()
    w.frame_width, w.frame_height = 1280, 720
    w.resize(640, 480)  # scale 0.5 -> 640x360 drawn, letterbox bars top/bottom
    return w


def test_mapping_inverts_letterbox(widget):
    # Top-left of the drawn frame: y offset = (480-360)//2 = 60.
    assert widget.widget_to_video_coords(0, 60) == (0.0, 0.0)
    # Widget center maps to frame center.
    vx, vy = widget.widget_to_video_coords(320, 240)
    assert abs(vx - 640.0) < 2 and abs(vy - 360.0) < 2


def test_mapping_outside_frame_is_none(widget):
    assert widget.widget_to_video_coords(0, 0) is None       # top letterbox bar
    assert widget.widget_to_video_coords(0, 425) is None     # bottom bar
    assert widget.widget_to_video_coords(320, 419) is not None  # last valid row


def test_mapping_without_video_is_none(qapp):
    from video import CustomVideoWidget
    w = CustomVideoWidget()
    w.resize(640, 480)
    assert w.widget_to_video_coords(320, 240) is None  # no frame dims yet
