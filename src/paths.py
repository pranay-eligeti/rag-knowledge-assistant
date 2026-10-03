"""Locate public demo data in both editable installs and built wheels."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / "data"
if not DATA_DIR.is_dir():
    DATA_DIR = Path(__file__).resolve().parents[1] / "data"
