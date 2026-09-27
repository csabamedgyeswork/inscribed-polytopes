"""Detect the orthogonal 2-plane split of a 4D free-sum configuration."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def orthogonal_plane_split(
    points: np.ndarray, *, tol: float = 1e-6
) -> tuple[np.ndarray, np.ndarray] | None:
    """Split 4D points into two orthogonal 2-plane polygons, or ``None``."""
    points = np.asarray(points, dtype=float)
    n, d = points.shape
    if d != 4:
        return None
    gram = points @ points.T

    others = [v for v in range(n) if v != 0]
    for size in range(1, n - 1):
        for combo in combinations(others, size - 1):
            a_idx = [0, *combo]
            b_idx = [v for v in range(n) if v not in a_idx]
            if not b_idx:
                continue
            pa, pb = points[a_idx], points[b_idx]
            if np.linalg.matrix_rank(pa, tol=1e-6) != 2:
                continue
            if np.linalg.matrix_rank(pb, tol=1e-6) != 2:
                continue
            if np.max(np.abs(gram[np.ix_(a_idx, b_idx)])) > 1e-4:
                continue
            return _to_2d(pa), _to_2d(pb)
    return None


def _to_2d(cluster: np.ndarray) -> np.ndarray:
    """Express a 2-plane cluster in its own orthonormal coordinates, ordered by angle."""
    centered = cluster - cluster.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    order = np.argsort(np.arctan2(coords[:, 1], coords[:, 0]))
    return coords[order]
