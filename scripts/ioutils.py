#!/usr/bin/env python3
"""
Small shared helpers for the pipeline scripts.

These were previously copy-pasted across several modules (a plain JSON loader, a
JSON writer that ensures the parent directory exists, and the class-name
normaliser). Centralising them keeps behaviour identical in one place.

Depends only on the standard library so it stays cheap to import from both the
``scripts`` and ``formations`` packages.
"""

import json
import os


def load_json(path):
    """Load and return JSON data from a file. Raises if the file is missing."""
    with open(path, "r") as f:
        return json.load(f)


def save_json(data, path):
    """Write JSON data to a file, creating the parent directory if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def normalize_class(name):
    """Normalise a detection class name: trim, lowercase, spaces -> underscores."""
    return (name or "").strip().lower().replace(" ", "_")
