"""Convex-hull operations shared by validation and search."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from .algebra import generalized_cross
from .combinatorics import OrientedComplex


def hull_volume(points: np.ndarray) -> float:
    """Volume of the convex hull; 0 when the hull is degenerate."""
    try:
        return float(ConvexHull(np.asarray(points, dtype=float)).volume)
    except QhullError:
        return 0.0


def oriented_hull(points: np.ndarray) -> OrientedComplex:
    """Outward-oriented triangulated boundary of ``conv(points)``.

    Facets are oriented against the centroid of the hull vertices rather than
    against the origin: an inscribed polytope need not contain the centre of
    its sphere, and orienting against a point outside the hull yields an
    incoherent complex.
    """
    points = np.asarray(points, dtype=float)
    n, d = points.shape
    hull = ConvexHull(points, qhull_options="Qt")
    interior = points[hull.vertices].mean(axis=0)
    facets = []
    for simplex in hull.simplices:
        idx = [int(i) for i in simplex]
        if np.linalg.det(points[idx] - interior) < 0:
            idx[-1], idx[-2] = idx[-2], idx[-1]
        facets.append(tuple(idx))
    return OrientedComplex(d=d, n=n, facets=tuple(facets))


def facet_unit_normals(points: np.ndarray) -> np.ndarray:
    """Outward unit normals of the current hull facets."""
    points = np.asarray(points, dtype=float)
    hull = ConvexHull(points, qhull_options="Qt")
    eqs = hull.equations[:, :-1]
    norms = np.linalg.norm(eqs, axis=1, keepdims=True)
    return eqs / np.where(norms == 0, 1, norms)


def facet_normal(rows: np.ndarray) -> np.ndarray:
    """Unit normal of the hyperplane through ``d`` points, oriented away from the origin."""
    rows = np.asarray(rows, dtype=float)
    d = rows.shape[1]
    if rows.shape[0] != d:
        raise ValueError("expected d points spanning a facet")
    diffs = rows[1:] - rows[0]
    normal = np.cross(diffs[0], diffs[1]) if d == 3 else generalized_cross(diffs)
    norm = np.linalg.norm(normal)
    if norm == 0:
        return normal
    normal = normal / norm
    if normal @ rows[0] < 0:
        normal = -normal
    return normal
