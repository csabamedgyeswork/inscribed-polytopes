"""Exact block-coordinate ascent.

Algorithm 1 of paper/sections/method.tex.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import QhullError

from ..combinatorics import OrientedComplex
from ..geometry import facet_unit_normals, hull_volume, oriented_hull
from ..optimality import OptimalityConditions
from ..volume import VolumeFunctional
from .base import SearchResult, Solver


class ExactVertexStep:
    """The closed-form coordinate move ``p_i = n_i / |n_i|``."""

    def __init__(self, form: VolumeFunctional):
        self.form = form

    def candidate(self, points: np.ndarray, v: int) -> np.ndarray | None:
        nv = self.form.vertex_normal(points, v)
        norm = float(np.linalg.norm(nv))
        if norm == 0.0:
            return None
        return nv / norm


class FixedTypeAscent(Solver):
    """Ascent on the signed volume of a prescribed combinatorial type.

    Fixed points are Property-Z configurations; folded ones can beat every
    polytope volume of the type, so pair this with a ``ConvexPosition``
    validator. See paper/sections/trap.tex.
    """

    def __init__(self, cx: OrientedComplex, *, tol: float = 1e-14, max_sweeps: int = 20000):
        self.cx = cx
        self.form = VolumeFunctional.build(cx)
        self.tol = tol
        self.max_sweeps = max_sweeps

    def solve(self, points: np.ndarray) -> SearchResult:
        pts = np.array(points, dtype=float, copy=True)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        step = ExactVertexStep(self.form)
        previous = self.form.value(pts)
        sweeps = 0
        for sweeps in range(1, self.max_sweeps + 1):
            for v in range(self.form.n):
                cand = step.candidate(pts, v)
                if cand is not None:
                    pts[v] = cand
            current = self.form.value(pts)
            if abs(current - previous) <= self.tol * max(1.0, abs(current)):
                previous = current
                break
            previous = current
        residual = OptimalityConditions(self.form).property_z_residual(pts)
        return SearchResult(
            points=pts,
            volume=float(previous),
            cx=self.cx,
            sweeps=sweeps,
            info={"property_z_residual": residual, "converged": residual < 1e-9},
        )


class HullAscent(Solver):
    """Ascent on the true convex hull volume; the combinatorial type emerges."""

    def __init__(self, *, tol: float = 1e-14, max_sweeps: int = 300):
        self.tol = tol
        self.max_sweeps = max_sweeps

    def solve(self, points: np.ndarray) -> SearchResult:
        pts = np.array(points, dtype=float, copy=True)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        n, _ = pts.shape
        current = hull_volume(pts)
        sweeps = 0
        for sweeps in range(1, self.max_sweeps + 1):
            start_value = current
            for v in range(n):
                current = self._improve_vertex(pts, v, current)
            if current - start_value <= self.tol * max(1.0, current):
                break
        try:
            cx = oriented_hull(pts)
        except QhullError:
            cx = None
        return SearchResult(points=pts, volume=float(current), cx=cx, sweeps=sweeps)

    def _improve_vertex(self, pts: np.ndarray, v: int, current: float) -> float:
        candidates = self._candidates(pts, v)
        best_value, best_point = current, None
        saved = pts[v].copy()
        for cand in candidates:
            pts[v] = cand
            value = hull_volume(pts)
            if value > best_value + 1e-15:
                best_value, best_point = value, cand.copy()
        pts[v] = saved if best_point is None else best_point
        return best_value if best_point is not None else current

    def _candidates(self, pts: np.ndarray, v: int) -> list[np.ndarray]:
        try:
            cx = oriented_hull(pts)
        except QhullError:
            return []
        if not any(v in f for f in cx.facets):
            # interior vertex: the best cap directions are the facet normals
            try:
                return list(facet_unit_normals(pts))
            except QhullError:
                return []
        step = ExactVertexStep(VolumeFunctional.build(cx))
        cand = step.candidate(pts, v)
        return [] if cand is None else [cand]
