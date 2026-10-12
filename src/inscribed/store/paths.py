"""Locations of the gitignored output tree."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
OUTPUT_DIR = ROOT / "output"
CONFIGURATIONS_DIR = OUTPUT_DIR / "configurations"
RESULTS_DIR = ROOT / "results"
SEARCHES_DIR = RESULTS_DIR / "free"
CERTIFICATES_DIR = RESULTS_DIR / "certificates"
IMAGES_DIR = ROOT / "images"


def configuration_path(label: str) -> Path:
    return CONFIGURATIONS_DIR / f"{label}.json"


def case_path(dimension: int, n: int) -> Path:
    return RESULTS_DIR / f"d{dimension}-n{n:02d}.json"


def search_path(dimension: int, n: int) -> Path:
    return SEARCHES_DIR / f"d{dimension}-n{n:02d}.json"


def certificate_dir(dimension: int, n: int) -> Path:
    return CERTIFICATES_DIR / f"d{dimension}-n{n:02d}"
