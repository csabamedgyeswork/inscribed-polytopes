"""Simplicial 4-polytopes from Firsching's inscribed rational realisations (d = 4)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.spatial import QhullError

from ..combinatorics import CombinatoricsError
from ..geometry import oriented_hull
from .base import DATA_DIR, ComplexSource, TypeRecord


def _parse_rational_points(line: str) -> np.ndarray:
    """Parse one Firsching line: a list of points with rational coordinates."""
    rows = [
        [float(Fraction(token.strip())) for token in block.split(",")]
        for block in re.findall(r"\[([^\[\]]+)\]", line)
    ]
    return np.array(rows, dtype=float)


class FirschingSource(ComplexSource):
    """Simplicial 4-polytopes from Firsching's inscribed rational realisations.

    Each line of ``polytopalsimplicialspheres{n}_4.txt`` is one combinatorial
    type given by coordinates, so the type and a valid starting configuration
    arrive together. Types whose published realisation is not inscribed are
    skipped: those are the ones with no inscribed realisation at all.
    """

    d = 4

    def __init__(self, *, data_dir: Path = DATA_DIR, tolerance: float = 1e-9):
        self.data_dir = Path(data_dir)
        self.tolerance = tolerance

    def path(self, n: int) -> Path:
        return self.data_dir / f"polytopalsimplicialspheres{n}_4.txt"

    def _lines(self, n: int) -> list[str]:
        path = self.path(n)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} missing; run scripts/setup_data.py to fetch it"
            )
        return [line for line in path.read_text().splitlines() if line.strip()]

    def count(self, n: int) -> int:
        return sum(1 for _ in self.types(n))

    def types(self, n: int) -> Iterator[TypeRecord]:
        for index, line in enumerate(self._lines(n)):
            points = _parse_rational_points(line)
            if points.shape != (n, 4):
                continue
            radii = np.linalg.norm(points, axis=1)
            if np.max(np.abs(radii - 1.0)) > self.tolerance:
                continue
            try:
                cx = oriented_hull(points)
                cx.check()
            except (CombinatoricsError, QhullError):
                continue
            if cx.n != n:
                continue
            yield TypeRecord(
                complex=cx, label=f"firsching-{n}-{index}", points=points
            )
