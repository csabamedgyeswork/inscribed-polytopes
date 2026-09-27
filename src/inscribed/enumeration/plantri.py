"""Simplicial 2-spheres from the vendored plantri generator (d = 3)."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterator
from pathlib import Path

from ..combinatorics import OrientedComplex
from .base import PLANTRI, ComplexSource, TypeRecord


def _faces_from_embedding(adjacency: dict[int, list[int]]) -> tuple[tuple[int, ...], ...]:
    """Trace the faces of a planar embedding given by cyclic adjacency lists."""
    faces: list[tuple[int, ...]] = []
    seen: set[tuple[int, int]] = set()
    for start in adjacency:
        for first in adjacency[start]:
            if (start, first) in seen:
                continue
            face: list[int] = []
            a, b = start, first
            while (a, b) not in seen:
                seen.add((a, b))
                face.append(a)
                ring = adjacency[b]
                a, b = b, ring[(ring.index(a) - 1) % len(ring)]
            faces.append(tuple(face))
    return tuple(faces)


class PlantriSource(ComplexSource):
    """Simplicial 2-spheres from the vendored plantri generator."""

    d = 3

    def __init__(self, *, min_degree: int = 3, binary: Path = PLANTRI):
        self.min_degree = min_degree
        self.binary = Path(binary)

    def _run(self, n: int, *extra: str) -> subprocess.CompletedProcess:
        if not self.binary.exists():
            raise FileNotFoundError(
                f"plantri not built at {self.binary}; run scripts/setup_data.py"
            )
        args = [str(self.binary), f"-m{self.min_degree}", *extra, str(n)]
        return subprocess.run(args, capture_output=True, text=True, check=True)

    def count(self, n: int) -> int:
        output = self._run(n, "-u").stderr or self._run(n, "-u").stdout
        match = re.search(r"(\d+)\s+triangulations", output)
        if match is None:
            raise RuntimeError(f"cannot parse plantri output: {output!r}")
        return int(match.group(1))

    def types(self, n: int) -> Iterator[TypeRecord]:
        lines = self._run(n, "-a").stdout.splitlines()
        for index, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            adjacency = {
                v: [ord(c) - ord("a") for c in part]
                for v, part in enumerate(line.split(" ", 1)[1].split(","))
            }
            faces = _faces_from_embedding(adjacency)
            cx = OrientedComplex(d=3, n=n, facets=faces)
            cx.check()
            yield TypeRecord(complex=cx, label=f"plantri-{n}-{index}")
