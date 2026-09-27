"""Linear-algebra primitives."""

from __future__ import annotations

import numpy as np


def generalized_cross(vectors: np.ndarray) -> np.ndarray:
    """Hodge dual of a wedge of d-1 vectors: x . n = det(x, v_0, ..., v_{d-2})."""
    vectors = np.asarray(vectors, dtype=float)
    k, d = vectors.shape
    if k != d - 1:
        raise ValueError(f"expected {d - 1} vectors in R^{d}, got {k}")
    if d == 3:
        return np.cross(vectors[0], vectors[1])
    out = np.empty(d, dtype=float)
    cols = np.arange(d)
    for j in range(d):
        out[j] = (-1) ** j * np.linalg.det(vectors[:, cols != j])
    return out
