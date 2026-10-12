"""SO(d) gauge fixing.

D is invariant under the diagonal action of SO(d), so a configuration is
determined only up to an orthogonal map; see papers/sweep/sections/preliminaries.tex.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..algebra import generalized_cross
from ..combinatorics import OrientedComplex


def gauge_dimension(d: int) -> int:
    """Number of coordinates pinned by the echelon gauge: ``dim SO(d)``."""
    return d * (d - 1) // 2


def free_parameter_count(d: int, n: int) -> int:
    """Essential degrees of freedom of ``n`` points on ``S^{d-1}`` mod ``SO(d)``."""
    return n * (d - 1) - gauge_dimension(d)


@dataclass(frozen=True)
class GaugeFrame:
    """Echelon gauge: rotate so anchor ``a_k`` keeps only its last ``k`` coordinates."""

    d: int
    anchors: tuple[int, ...]
    tol: float = 1e-12

    def __post_init__(self) -> None:
        if len(self.anchors) != self.d - 1:
            raise ValueError(f"expected {self.d - 1} anchor vertices, got {len(self.anchors)}")

    @classmethod
    def default(cls, cx: OrientedComplex, *, prefer_low_degree: bool = True) -> GaugeFrame:
        """Anchors: an extreme-degree vertex and then its neighbours."""
        deg = cx.vertex_degrees()
        order = sorted(range(cx.n), key=lambda v: (deg[v], v), reverse=not prefer_low_degree)
        anchors = [order[0]]
        if cx.d > 2:
            nbrs = sorted(
                (w for e in cx.edges() if order[0] in e for w in e if w != order[0]),
                key=lambda w: (deg[w], w),
                reverse=not prefer_low_degree,
            )
            for w in nbrs:
                if len(anchors) >= cx.d - 1:
                    break
                anchors.append(w)
        return cls(d=cx.d, anchors=tuple(anchors[: cx.d - 1]))

    def pinned_coordinates(self) -> set[tuple[int, int]]:
        """``(vertex, coordinate)`` pairs forced to zero by the gauge."""
        pinned = set()
        for k, v in enumerate(self.anchors, start=1):
            for j in range(self.d - k):
                pinned.add((v, j))
        return pinned

    def transform(self, points: np.ndarray) -> np.ndarray:
        """Rotate ``points`` into the gauge frame by a determinant-+1 rotation."""
        points = np.asarray(points, dtype=float)
        n, d = points.shape
        basis: list[np.ndarray] = []
        for v in self.anchors:
            u = points[v].astype(float).copy()
            for b in basis:
                u -= (u @ b) * b
            norm = np.linalg.norm(u)
            # degenerate anchor: substitute any direction orthogonal to the basis so far
            u = _orthogonal_complement_vector(basis, d) if norm <= self.tol else u / norm
            basis.append(u)
        rows = np.zeros((d, d), dtype=float)
        for k, u in enumerate(basis, start=1):
            rows[d - k] = u
        rows[0] = _orthogonal_complement_vector(basis, d)
        if np.linalg.det(rows) < 0:
            rows[0] = -rows[0]
        return points @ rows.T

    def residual(self, points: np.ndarray) -> float:
        """How far ``points`` is from satisfying the gauge."""
        points = np.asarray(points, dtype=float)
        return max(
            (abs(float(points[v, j])) for v, j in self.pinned_coordinates()),
            default=0.0,
        )


def _orthogonal_complement_vector(basis: list[np.ndarray], d: int) -> np.ndarray:
    if len(basis) == d - 1:
        w = generalized_cross(np.array(basis))
        norm = np.linalg.norm(w)
        if norm > 0:
            return w / norm
    mat = np.array(basis) if basis else np.zeros((0, d))
    _, _, vh = np.linalg.svd(np.vstack([mat, np.zeros((1, d))]))
    return vh[-1]
