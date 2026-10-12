"""Signed volume of an oriented simplicial polytope.

The functional D of papers/sweep/sections/preliminaries.tex.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from .algebra import generalized_cross
from .combinatorics import OrientedComplex


@dataclass(frozen=True)
class VolumeFunctional:
    """Precomputed data for evaluating the signed volume and its vertex normals.

    ``d! * vol = n_i . p_i + C_i`` for every vertex ``i``; ``n_i`` is the
    vertex normal and ``C_i`` the constant term.
    """

    cx: OrientedComplex
    # star[i] = tuple of (sign, other_vertices) with
    #   det(facet) == sign * det(p_i, p_others...)
    star: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]

    @classmethod
    def build(cls, cx: OrientedComplex) -> VolumeFunctional:
        star: list[tuple[tuple[int, tuple[int, ...]], ...]] = []
        for v in range(cx.n):
            entries = []
            for f in cx.facets:
                if v not in f:
                    continue
                k = f.index(v)
                # moving f[k] to the front is a cycle on k+1 elements
                entries.append(((-1) ** k, f[:k] + f[k + 1 :]))
            star.append(tuple(entries))
        return cls(cx=cx, star=tuple(star))

    @property
    def d(self) -> int:
        return self.cx.d

    @property
    def n(self) -> int:
        return self.cx.n

    def value(self, points: np.ndarray) -> float:
        """Signed volume; equals the polytope volume on the convex-position cell."""
        points = np.asarray(points, dtype=float)
        total = 0.0
        for f in self.cx.facets:
            total += np.linalg.det(points[list(f)])
        return total / factorial(self.d)

    def vertex_normal(self, points: np.ndarray, v: int) -> np.ndarray:
        """``n_v`` such that ``d! * vol == n_v . p_v + C_v``."""
        points = np.asarray(points, dtype=float)
        acc = np.zeros(self.d, dtype=float)
        for sign, others in self.star[v]:
            acc += sign * generalized_cross(points[list(others)])
        return acc

    def vertex_normals(self, points: np.ndarray) -> np.ndarray:
        return np.array([self.vertex_normal(points, v) for v in range(self.n)])

    def gradient(self, points: np.ndarray) -> np.ndarray:
        """Unconstrained gradient of the signed volume w.r.t. every vertex."""
        return self.vertex_normals(points) / factorial(self.d)

    def constant_term(self, points: np.ndarray, v: int) -> float:
        """``C_v / d!``: the part of the volume not involving ``p_v``."""
        points = np.asarray(points, dtype=float)
        total = 0.0
        for f in self.cx.facets:
            if v not in f:
                total += np.linalg.det(points[list(f)])
        return total / factorial(self.d)
