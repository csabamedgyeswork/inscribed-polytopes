"""Parametrised families of configurations with closed-form volumes."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np

from ..combinatorics import OrientedComplex, bipyramid, simplex


def regular_polygon(m: int, *, offset: float = 0.0) -> np.ndarray:
    """``m`` points on the unit circle of ``R^2``."""
    ang = offset + 2 * math.pi * np.arange(m) / m
    return np.stack([np.cos(ang), np.sin(ang)], axis=1)


def _normalized(points: np.ndarray) -> np.ndarray:
    return points / np.linalg.norm(points, axis=1, keepdims=True)


class ConfigurationFamily(ABC):
    """A parametric family of sphere-inscribed configurations."""

    name: str

    @abstractmethod
    def build(self, **params) -> np.ndarray:
        """Explicit coordinates for the configuration."""

    @abstractmethod
    def parameters_for(self, n: int, d: int) -> dict:
        """The parameters realising ``n`` vertices in ``R^d``."""

    def closed_form_volume(self, **params) -> float | None:
        """Exact volume of ``build(**params)``, when known in closed form."""
        return None

    def complex(self, **params) -> OrientedComplex | None:
        """The combinatorial type, when the family is simplicial."""
        return None


class RegularSimplex(ConfigurationFamily):
    name = "regular simplex"

    def build(self, d: int) -> np.ndarray:
        pts = np.eye(d + 1) - 1.0 / (d + 1)
        _, _, vh = np.linalg.svd(pts)
        return _normalized(pts @ vh[:d].T)

    def parameters_for(self, n: int, d: int) -> dict:
        if n != d + 1:
            raise ValueError(f"a regular simplex in R^{d} has {d + 1} vertices, not {n}")
        return {"d": d}

    def closed_form_volume(self, d: int) -> float:
        return math.sqrt(d + 1) * ((d + 1) / d) ** (d / 2) / math.factorial(d)

    def complex(self, d: int) -> OrientedComplex:
        return simplex(d)


class CrossPolytope(ConfigurationFamily):
    name = "cross polytope"

    def build(self, d: int) -> np.ndarray:
        rows = []
        for axis in range(d):
            for sign in (1.0, -1.0):
                vec = np.zeros(d)
                vec[axis] = sign
                rows.append(vec)
        return np.array(rows)

    def parameters_for(self, n: int, d: int) -> dict:
        if n != 2 * d:
            raise ValueError(f"a cross polytope in R^{d} has {2 * d} vertices, not {n}")
        return {"d": d}

    def closed_form_volume(self, d: int) -> float:
        return 2.0**d / math.factorial(d)


class Bipyramid(ConfigurationFamily):
    """Double ``m``-pyramid with the belt on the equator; its class maximum."""

    name = "bipyramid"

    def build(self, m: int) -> np.ndarray:
        belt = regular_polygon(m)
        rows = [[p[0], p[1], 0.0] for p in belt]
        rows += [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]
        return np.array(rows)

    def parameters_for(self, n: int, d: int) -> dict:
        if d != 3:
            raise ValueError("bipyramids are three-dimensional")
        return {"m": n - 2}

    def closed_form_volume(self, m: int) -> float:
        return (m / 3) * math.sin(2 * math.pi / m)

    def complex(self, m: int) -> OrientedComplex:
        return bipyramid(m)


class OrthogonalPolygonPair(ConfigurationFamily):
    """Regular ``m``-gon and ``k``-gon in orthogonal 2-planes of ``R^4``.

    ``vol_4(m, k) = (m k / 24) sin(2 pi / m) sin(2 pi / k)``; see
    paper/sections/results-4d.tex.
    """

    name = "orthogonal polygon pair"

    def build(self, m: int, k: int) -> np.ndarray:
        if m < 3 or k < 3:
            raise ValueError("both polygons must have at least 3 vertices")
        rows = []
        for ring, axes in ((regular_polygon(m), (0, 1)), (regular_polygon(k), (2, 3))):
            for point in ring:
                vec = np.zeros(4)
                vec[axes[0]], vec[axes[1]] = point
                rows.append(vec)
        return np.array(rows)

    def parameters_for(self, n: int, d: int) -> dict:
        if d != 4:
            raise ValueError("orthogonal polygon pairs are four-dimensional")
        best = max(
            ((m, n - m) for m in range(3, n - 2) if n - m >= 3),
            key=lambda mk: self.closed_form_volume(m=mk[0], k=mk[1]),
        )
        return {"m": best[0], "k": best[1]}

    def closed_form_volume(self, m: int, k: int) -> float:
        if m < 3 or k < 3:
            raise ValueError("both polygons must have at least 3 vertices")
        return (m * k / 24) * math.sin(2 * math.pi / m) * math.sin(2 * math.pi / k)


# The user's class 1, verbatim from the original problem statement, in their
# own 1-based labelling: the triaugmented triangular prism (Johnson solid J51).
TRIAUGMENTED_PRISM_FACETS = (
    (6, 2, 3), (6, 3, 4), (6, 5, 2), (6, 4, 5),
    (9, 4, 3), (9, 3, 8), (9, 8, 7), (9, 7, 4),
    (1, 5, 7), (1, 7, 8), (1, 8, 2), (1, 2, 5),
    (4, 7, 5), (8, 3, 2),
)


class TriaugmentedPrism(ConfigurationFamily):
    """The Hoehner--Ledford maximiser for 9 vertices on S^2, arXiv:2608.04392."""

    name = "triaugmented triangular prism"

    def build(self) -> np.ndarray:
        h = math.sqrt(2 * math.sqrt(3) - 3)
        r = math.sqrt(3) - 1

        def at(az_deg: float, rad: float, z: float):
            t = math.radians(az_deg)
            return [rad * math.cos(t), rad * math.sin(t), z]

        rows = {
            4: at(60, r, h), 3: at(60, r, -h),
            7: at(180, r, h), 8: at(180, r, -h),
            5: at(300, r, h), 2: at(300, r, -h),
            6: at(0, 1, 0), 9: at(120, 1, 0), 1: at(240, 1, 0),
        }
        return np.array([rows[i + 1] for i in range(9)])

    def parameters_for(self, n: int, d: int) -> dict:
        if (n, d) != (9, 3):
            raise ValueError("the triaugmented prism has 9 vertices in R^3")
        return {}

    def closed_form_volume(self) -> float:
        return 3 * math.sqrt(2 * math.sqrt(3) - 3)

    def complex(self) -> OrientedComplex:
        return OrientedComplex.from_labels(TRIAUGMENTED_PRISM_FACETS, one_indexed=True)
