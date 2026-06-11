"""
Shared pytest fixtures + import-path setup.

The project is a collection of top-level scripts grouped into ``app/``,
``scripts/`` and ``formations/`` directories rather than an installable
package, and the modules import each other by bare name (``import ioutils``,
``from perFrameHomographyTransform import ...``). We replicate that runtime
import environment by putting all three directories on ``sys.path`` so the unit
tests can import the same way the app does.

Tests only touch the pure/deterministic helpers, so no Qt platform, GPU, video
files or trained models are required.
"""

import os
import sys

import numpy as np
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _sub in ("scripts", "formations", "app"):
    _path = os.path.join(_ROOT, _sub)
    if _path not in sys.path:
        sys.path.insert(0, _path)

# Headless Qt in case any imported helper ever pulls a Qt module transitively.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture
def project_root():
    return _ROOT


@pytest.fixture
def standard_formation_points():
    """A clean, axis-aligned 11-point offense in field yards (x=lateral, y=depth).

    Layout (right-strong, attacking +y downfield is not used here; geometry
    helpers infer their own axes via PCA):

      - 5 OL on the line at depth 0, laterally tight around x=0
      - QB centered, 3 yd behind the line
      - RB off-center, 6 yd behind the line
      - 1 TE just outside the right tackle, on the line
      - 3 WR split wide on the line

    Returned as a dict so individual indices are addressable in tests, plus the
    plain (11, 2) array under key ``points``.
    """
    rows = {
        "ol0": (-2.0, 0.0),
        "ol1": (-1.0, 0.0),
        "ol2": (0.0, 0.0),
        "ol3": (1.0, 0.0),
        "ol4": (2.0, 0.0),
        "qb": (0.0, -3.0),
        "rb": (3.0, -6.0),
        "te": (4.0, 0.0),
        "wr_l": (-14.0, 0.0),
        "wr_r": (14.0, 0.0),
        "wr_l2": (-18.0, 0.0),
    }
    names = list(rows.keys())
    pts = np.array([rows[n] for n in names], dtype=float)
    return {"names": names, "index": {n: i for i, n in enumerate(names)},
            "points": pts}
