"""Starting-configuration strategies."""

from __future__ import annotations

import numpy as np
from scipy.spatial import QhullError

from ..geometry import facet_unit_normals, hull_volume
from .base import Initializer


class RandomInitializer(Initializer):
    """Uniform random points on the sphere."""

    def __init__(self, d: int, n: int, *, seed: int = 0):
        self.d = d
        self.n = n
        self.rng = np.random.default_rng(seed)

    def initial(self) -> np.ndarray:
        pts = self.rng.normal(size=(self.n, self.d))
        return pts / np.linalg.norm(pts, axis=1, keepdims=True)


class ExplicitConfiguration(Initializer):
    """A fixed starting configuration."""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)

    def initial(self) -> np.ndarray:
        return self.points.copy()


class ChainedInitializer(Initializer):
    """Yield each leading initializer once, then delegate to the last one forever."""

    def __init__(self, *initializers: Initializer):
        if not initializers:
            raise ValueError("need at least one initializer")
        self.initializers = list(initializers)
        self._index = 0

    def initial(self) -> np.ndarray:
        if self._index < len(self.initializers) - 1:
            initializer = self.initializers[self._index]
            self._index += 1
            return initializer.initial()
        return self.initializers[-1].initial()


class GrowFromConfiguration(Initializer):
    """Append the sphere point that adds the most volume to a given hull."""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)

    def initial(self) -> np.ndarray:
        try:
            normals = facet_unit_normals(self.points)
        except QhullError:
            normals = np.empty((0, self.points.shape[1]))
        best_value, best_point = -np.inf, None
        for cand in normals:
            value = hull_volume(np.vstack([self.points, cand]))
            if value > best_value:
                best_value, best_point = value, cand
        if best_point is None:
            extra = np.zeros(self.points.shape[1])
            extra[0] = 1.0
            best_point = extra
        return np.vstack([self.points, best_point])
