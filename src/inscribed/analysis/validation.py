"""Certifying that a configuration is what it claims to be.

See paper/sections/trap.tex for why this is not optional.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..combinatorics import OrientedComplex
from ..geometry import facet_normal
from ..optimality import OptimalityConditions
from .gauge import GaugeFrame


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    message: str
    metrics: dict = field(default_factory=dict)


class Validator(ABC):
    """A check that a configuration satisfies some requirement."""

    @abstractmethod
    def check(self, points: np.ndarray) -> ValidationReport: ...


class ConvexPosition(Validator):
    """Points are in convex position and (if a complex is given) realise it."""

    def __init__(self, cx: OrientedComplex | None = None, *, tol: float = 1e-9):
        self.cx = cx
        self.tol = tol

    def check(self, points: np.ndarray) -> ValidationReport:
        points = np.asarray(points, dtype=float)
        n, d = points.shape
        if self.cx is not None and (n != self.cx.n or d != self.cx.d):
            raise ValueError("shape mismatch between points and complex")

        try:
            hull = ConvexHull(points)
        except QhullError as exc:
            return ValidationReport(False, f"degenerate hull: {exc}", {})

        n_hull = len(set(hull.vertices))
        hull_facets = {frozenset(s) for s in hull.simplices}

        margin = np.inf
        facet_margins: dict[tuple[int, ...], float] = {}
        facets = self.cx.facets if self.cx is not None else [tuple(s) for s in hull.simplices]
        for f in facets:
            rows = points[list(f)]
            normal = facet_normal(rows)
            offset = normal @ rows[0]
            f_margin = np.inf
            for w in range(n):
                if w not in f:
                    f_margin = min(f_margin, offset - float(normal @ points[w]))
            facet_margins[tuple(f)] = float(f_margin)
            margin = min(margin, f_margin)

        parts = []
        if n_hull != n:
            parts.append(f"only {n_hull}/{n} points are hull vertices")
        if margin <= self.tol:
            parts.append(f"support margin {margin:.3e} is not positive")
        type_matches = True
        if self.cx is not None:
            want = {frozenset(f) for f in self.cx.facets}
            type_matches = hull_facets == want
            if not type_matches:
                parts.append(
                    f"hull has {len(hull_facets - want)} unexpected and "
                    f"{len(want - hull_facets)} missing facets"
                )

        ok = not parts
        return ValidationReport(
            ok=ok,
            message="convex position, combinatorial type matches" if ok else "; ".join(parts),
            metrics={
                "hull_volume": float(hull.volume),
                "min_support_margin": float(margin),
                "n_hull_vertices": n_hull,
                "type_matches": type_matches,
                "facet_margins": facet_margins,
            },
        )


class PropertyZ(Validator):
    """The Berman--Hanes first-order condition holds to tolerance."""

    def __init__(self, conditions: OptimalityConditions, *, tol: float = 1e-9):
        self.conditions = conditions
        self.tol = tol

    def check(self, points: np.ndarray) -> ValidationReport:
        residual = self.conditions.property_z_residual(points)
        ok = residual < self.tol
        return ValidationReport(
            ok=ok,
            message="Property Z holds" if ok else f"Property Z residual {residual:.3e}",
            metrics={"property_z_residual": residual},
        )


class Gauge(Validator):
    """The configuration satisfies a gauge frame to tolerance."""

    def __init__(self, frame: GaugeFrame, *, tol: float = 1e-9):
        self.frame = frame
        self.tol = tol

    def check(self, points: np.ndarray) -> ValidationReport:
        residual = self.frame.residual(points)
        ok = residual < self.tol
        return ValidationReport(
            ok=ok,
            message="gauge satisfied" if ok else f"gauge residual {residual:.3e}",
            metrics={"gauge_residual": residual},
        )


class Composite(Validator):
    """All of several validators; reports the first failure."""

    def __init__(self, *validators: Validator):
        self.validators = validators

    def check(self, points: np.ndarray) -> ValidationReport:
        metrics: dict = {}
        for validator in self.validators:
            report = validator.check(points)
            metrics.update(report.metrics)
            if not report.ok:
                return ValidationReport(False, report.message, metrics)
        return ValidationReport(True, "all checks passed", metrics)
