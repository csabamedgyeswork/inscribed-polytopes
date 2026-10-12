"""First-order optimality conditions (Property Z).

See papers/sweep/sections/preliminaries.tex.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np

from .volume import VolumeFunctional


@dataclass(frozen=True)
class OptimalityConditions:
    """Berman--Hanes local optimality diagnostics for a volume functional."""

    form: VolumeFunctional

    def property_z_residual(self, points: np.ndarray) -> float:
        """Max over vertices of ``1 - p_v . n_v / |n_v|``; zero at Property-Z points."""
        points = np.asarray(points, dtype=float)
        worst = 0.0
        for v in range(self.form.n):
            nv = self.form.vertex_normal(points, v)
            norm = float(np.linalg.norm(nv))
            if norm == 0.0:
                return float("inf")
            worst = max(worst, 1.0 - float(points[v] @ nv) / norm)
        return worst

    def multipliers(self, points: np.ndarray) -> np.ndarray:
        """``lambda_v = |n_v| / d!``; at a Property-Z point ``sum_v lambda_v == d * vol``."""
        points = np.asarray(points, dtype=float)
        return np.array(
            [np.linalg.norm(self.form.vertex_normal(points, v)) for v in range(self.form.n)]
        ) / factorial(self.form.d)

    def euler_identity_error(self, points: np.ndarray) -> float:
        """``|sum_v lambda_v - d * vol|``, an independent check on candidates."""
        return float(
            abs(self.multipliers(points).sum() - self.form.d * self.form.value(points))
        )
