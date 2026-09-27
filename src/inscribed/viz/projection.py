"""Schlegel diagrams: project a d-polytope through one facet into d-1 space."""

from __future__ import annotations

import numpy as np

from ..algebra import generalized_cross
from ..combinatorics import OrientedComplex


def _orthonormal_basis(normal: np.ndarray) -> np.ndarray:
    """Return a (d, d-1) matrix whose columns span the hyperplane ⟂ ``normal``."""
    d = normal.shape[0]
    a = np.column_stack([normal, np.eye(d)])
    q, _ = np.linalg.qr(a)
    return q[:, 1:]


def schlegel_projection(
    points: np.ndarray,
    cx: OrientedComplex,
    facet: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Project through ``facet`` (default: the first) into its hyperplane."""
    points = np.asarray(points, dtype=float)
    n, d = points.shape
    if facet is None:
        facet = cx.facets[0]
    facet = tuple(facet)

    rows = points[list(facet)]
    centroid = rows.mean(axis=0)
    diffs = rows[1:] - rows[0]
    normal = np.cross(diffs[0], diffs[1]) if d == 3 else generalized_cross(diffs)
    normal = normal / np.linalg.norm(normal)
    if normal @ (centroid - points.mean(axis=0)) < 0:
        normal = -normal

    basis = _orthonormal_basis(normal)

    offset = 1.0
    projected: np.ndarray = np.empty((0, d - 1))
    for _ in range(60):
        eye = centroid + offset * normal
        proj = []
        for p in points:
            direction = p - eye
            denom = float(direction @ normal)
            if abs(denom) < 1e-12:
                proj.append(np.zeros(d - 1))
                continue
            t = float((centroid - eye) @ normal) / denom
            proj.append(basis.T @ (eye + t * direction - centroid))
        projected = np.asarray(proj)
        if _others_inside(projected, facet):
            break
        offset *= 0.5
    return projected, basis


def _others_inside(projected: np.ndarray, facet: tuple[int, ...], tol: float = 1e-9) -> bool:
    """Whether every non-facet vertex lies in the facet's convex hull."""
    outer = projected[list(facet)]
    others = [i for i in range(projected.shape[0]) if i not in facet]
    if not others:
        return True
    return all(_point_in_hull(projected[i], outer, tol) for i in others)


def _point_in_hull(p: np.ndarray, hull_pts: np.ndarray, tol: float) -> bool:
    dim = hull_pts.shape[1]
    if dim == 1:
        lo, hi = hull_pts.min(axis=0), hull_pts.max(axis=0)
        return bool(np.all(p >= lo - tol) and np.all(p <= hi + tol))
    a = np.vstack([hull_pts.T, np.ones(hull_pts.shape[0])])
    b = np.concatenate([p, [1.0]])
    coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)
    return bool(np.all(coeffs >= -tol) and abs(coeffs.sum() - 1.0) <= 1e-6)
