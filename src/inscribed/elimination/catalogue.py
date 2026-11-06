"""The exact type catalogue of a case: oriented facets of every line of Firsching's list.

Each type is stored as one line of its shard, ``results/certificates/d<d>-n<n>/types/``, with
the facets as strings of vertex digits in the order and orientation used by the certificates.
The facets are proved in exact arithmetic from Firsching's rational vertices: every stored
4-set has all other vertices strictly on one side of its hyperplane, so it is a simplex facet,
and the stored facets form a closed coherently oriented sphere, so there are no others.
Each facet is oriented so that A_{F,j} > 0, which is the orientation with D > 0.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import re
from collections.abc import Callable, Iterable
from fractions import Fraction
from itertools import combinations
from math import lcm
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..certify import ExactCertificate
from ..combinatorics import OrientedComplex

DIGITS = "0123456789abcdefghijklmnopqrstuvwxyz"
SHARD = 10_000


def rational_points(line: str) -> list[tuple[Fraction, ...]]:
    """The exact vertices of one line of Firsching's file."""
    return [tuple(Fraction(t.strip()) for t in block.split(","))
            for block in re.findall(r"\[([^\[\]]+)\]", line)]


def _determinant(matrix: list[list[int]]) -> int:
    """Exact determinant of an integer matrix, by fraction-free Bareiss elimination."""
    a = [row[:] for row in matrix]
    size, sign, previous = len(a), 1, 1
    for k in range(size - 1):
        if a[k][k] == 0:
            swap = next((r for r in range(k + 1, size) if a[r][k] != 0), None)
            if swap is None:
                return 0
            a[k], a[swap] = a[swap], a[k]
            sign = -sign
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // previous
        previous = a[k][k]
    return sign * a[-1][-1]


class ExactHull:
    """The oriented boundary complex of the convex hull of rational points, proved exactly.

    The row (1, p_i), scaled by a positive integer, is integral; det of the rows of p_j and
    p_F is a positive multiple of A_{F,j}(p) = det(p_{F_1} - p_j, ..., p_{F_d} - p_j)."""

    def __init__(self, points: list[tuple[Fraction, ...]]):
        self.points, self.n, self.d = points, len(points), len(points[0])
        self.rows = []
        for p in points:
            scale = lcm(*(x.denominator for x in p))
            self.rows.append([scale] + [int(x * scale) for x in p])

    def side(self, facet: tuple[int, ...], j: int) -> int:
        """The sign of A_{F,j}(p)."""
        value = _determinant([self.rows[j]] + [self.rows[v] for v in facet])
        return (value > 0) - (value < 0)

    def oriented(self, facet: tuple[int, ...]) -> tuple[int, ...] | None:
        """``facet``, reordered if needed so that every A_{F,j} > 0; None unless every other
        point is strictly on one side, that is, unless it is a simplex facet."""
        signs = {self.side(facet, j) for j in range(self.n) if j not in facet}
        if signs == {1}:
            return facet
        if signs == {-1}:
            return facet[:-2] + (facet[-1], facet[-2])
        return None

    def _valid(self, facets: list[tuple[int, ...]]) -> bool:
        return OrientedComplex(d=self.d, n=self.n, facets=tuple(facets)).is_valid()

    def facets(self) -> tuple[tuple[int, ...], ...]:
        """qhull's facets in qhull's order, each proved and oriented exactly; if they do not
        form a closed coherent sphere, every d-subset is tested exactly instead."""
        try:
            proposal = ConvexHull(np.array(self.points, dtype=float), qhull_options="Qt")
            proposed = [self.oriented(tuple(int(i) for i in s)) for s in proposal.simplices]
            facets = [f for f in proposed if f is not None]
            if len(facets) == len(proposed) and self._valid(facets):
                return tuple(facets)
        except QhullError:
            pass
        found = map(self.oriented, combinations(range(self.n), self.d))
        facets = [f for f in found if f is not None]
        if not self._valid(facets):
            raise ValueError("the hull is not a simplicial polytope with these vertices")
        return tuple(facets)


def _encode(facets: Iterable[tuple[int, ...]]) -> str:
    return " ".join("".join(DIGITS[v] for v in f) for f in facets)


def _decode(line: str) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(DIGITS.index(c) for c in word) for word in line.split())


def _exact_line(line: str) -> str:
    return _encode(ExactHull(rational_points(line)).facets())


def canonical_form(facets: Iterable[Iterable[int]], n: int) -> tuple[int, ...]:
    """A form shared exactly by combinatorially equivalent facet systems: the least encoding
    over the leaves of colour refinement with individualisation (no pruning)."""
    sets = [frozenset(f) for f in facets]
    incident = [[f for f in sets if v in f] for v in range(n)]

    def refine(colours: list) -> list[int]:
        ranks = {c: i for i, c in enumerate(sorted(set(colours)))}
        current = [ranks[c] for c in colours]
        while True:
            sig = [(current[v], tuple(sorted(tuple(sorted(current[u] for u in f if u != v))
                                             for f in incident[v]))) for v in range(n)]
            ranks = {s: i for i, s in enumerate(sorted(set(sig)))}
            refined = [ranks[s] for s in sig]
            if len(ranks) == len(set(current)):
                return refined
            current = refined

    def search(colours: list[int]) -> tuple[int, ...]:
        cells: dict[int, list[int]] = {}
        for v, c in enumerate(colours):
            cells.setdefault(c, []).append(v)
        target = next((cells[c] for c in sorted(cells) if len(cells[c]) > 1), None)
        if target is None:
            return tuple(sorted(sum(1 << colours[v] for v in f) for f in sets))
        return min(search(refine([(c, 0 if u == v else 1) for u, c in enumerate(colours)]))
                   for v in target)

    return search(refine([0] * n))


def is_cycle_join(facets: Iterable[Iterable[int]], n: int, m: int, k: int) -> bool:
    """Whether the facets are those of the join of an m-cycle and a k-cycle, n = m + k: for
    some split of the vertices into an m-set and a k-set, the edges inside each part form a
    cycle, and the facets are the unions of an edge of one cycle with one of the other."""
    sets = {frozenset(f) for f in facets}
    if n != m + k or len(sets) != m * k:
        return False
    edges = {frozenset(p) for f in sets for p in combinations(sorted(f), 2)}
    for first in combinations(range(n), m):
        sides = [set(first), set(range(n)) - set(first)]
        cycles = [[e for e in edges if e <= s] for s in sides]
        if all(_is_cycle(c, s) for c, s in zip(cycles, sides)) and \
                sets == {a | b for a in cycles[0] for b in cycles[1]}:
            return True
    return False


def _is_cycle(edges: list[frozenset[int]], vertices: set[int]) -> bool:
    if len(edges) != len(vertices) or any(sum(v in e for e in edges) != 2 for v in vertices):
        return False
    seen, todo = set(), [min(vertices)]
    while todo:
        v = todo.pop()
        seen.add(v)
        todo += [w for e in edges if v in e for w in e - seen]
    return seen == vertices


class TypeCatalogue:
    """The committed exact facets of every type of a case, read shard by shard."""

    def __init__(self, directory: Path, prefix: str, d: int, n: int):
        self.directory, self.prefix, self.d, self.n = Path(directory), prefix, d, n
        self._lines: list[str] | None = None

    def label(self, k: int) -> str:
        return f"{self.prefix}-{k}"

    def index(self, label: str) -> int:
        prefix, _, k = label.rpartition("-")
        if prefix != self.prefix:
            raise KeyError(label)
        return int(k)

    def shard_path(self, lo: int) -> Path:
        return self.directory / f"{self.prefix}-{lo:06d}-{lo + SHARD - 1:06d}.txt.gz"

    def build(self, source_lines: list[str], workers: int = 1,
              log: Callable[[str], None] = print) -> str:
        """Prove and write the facets of every line; returns the catalogue hash."""
        self.directory.mkdir(parents=True, exist_ok=True)
        with Pool(workers) as pool:
            for lo in range(0, len(source_lines), SHARD):
                lines = pool.map(_exact_line, source_lines[lo:lo + SHARD], chunksize=50)
                buffer = io.BytesIO()
                with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as fh:
                    fh.write(("\n".join(lines) + "\n").encode())
                self.shard_path(lo).write_bytes(buffer.getvalue())
                log(f"{self.shard_path(lo).name}: {len(lines)} types")
        self._lines = None
        return self.sha256()

    def lines(self) -> list[str]:
        if self._lines is None:
            out, lo = [], 0
            while self.shard_path(lo).exists():
                out += gzip.decompress(self.shard_path(lo).read_bytes()).decode().splitlines()
                lo += SHARD
            self._lines = out
        return self._lines

    def __len__(self) -> int:
        return len(self.lines())

    def labels(self) -> list[str]:
        return [self.label(k) for k in range(len(self))]

    def __contains__(self, label: str) -> bool:
        try:
            return 0 <= self.index(label) < len(self)
        except (KeyError, ValueError):
            return False

    def facets(self, label: str) -> tuple[tuple[int, ...], ...]:
        return _decode(self.lines()[self.index(label)])

    def complex(self, label: str) -> OrientedComplex:
        return OrientedComplex(d=self.d, n=self.n, facets=self.facets(label))

    def fingerprint(self, label: str) -> str:
        """Identifies the stored facets of one type (and their order and orientation)."""
        return hashlib.sha256(self.lines()[self.index(label)].encode()).hexdigest()[:16]

    def sha256(self) -> str:
        return hashlib.sha256("\n".join(self.lines()).encode()).hexdigest()

    def resolver(self, cert: ExactCertificate) -> set[frozenset[int]] | None:
        """The facet sets of the type a certificate names, for the verifier's type check."""
        if cert.label not in self or (cert.d, cert.n) != (self.d, self.n):
            return None
        return {frozenset(f) for f in self.facets(cert.label)}

    def audit(self, winner: tuple[int, int], workers: int = 1) -> dict:
        """The combinatorial facts the proof needs: every type is a simplicial sphere with
        n vertices (re-checked here), the canonical forms are pairwise distinct, and exactly
        one type is the join of cycles of lengths ``winner``."""
        labels = self.labels()
        systems = [self.facets(label) for label in labels]
        spheres = all(self.complex(label).is_valid() for label in labels)
        with Pool(workers) as pool:
            forms = pool.starmap(canonical_form, [(f, self.n) for f in systems], chunksize=200)
        joins = [label for label, f in zip(labels, systems)
                 if is_cycle_join(f, self.n, *winner)]
        return {"types": len(labels), "spheres": spheres,
                "distinct_canonical_forms": len(set(forms)),
                "pairwise_distinct": len(set(forms)) == len(labels),
                "winner": joins[0] if len(joins) == 1 else None, "winner_candidates": joins,
                "catalogue_sha256": self.sha256()}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
