"""Exhaustive maximisation over every combinatorial type of a source."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np

from ..analysis import ConvexPosition
from ..combinatorics import OrientedComplex
from ..enumeration import ComplexSource, TypeRecord
from .ascent import FixedTypeAscent
from .maximize import maximize_type


@dataclass
class TypeResult:
    """Best convex-position volume found within one combinatorial type."""

    label: str
    complex: OrientedComplex
    volume: float
    points: np.ndarray | None = None
    signed_best: float = -np.inf
    restarts: int = 0
    rejected: int = 0
    sweeps: int = 0
    refined: bool = False
    info: dict = field(default_factory=dict)

    @property
    def feasible(self) -> bool:
        """Whether any restart reached a valid inscribed polytope of this type."""
        return self.points is not None


class TypeSweep:
    """Run ``FixedTypeAscent`` inside every type, keeping convex-position optima.

    Two passes: a coarse screen over all types, then a warm-started precise
    re-solve of the leaders. Screening alone is not accurate enough to rank
    near-ties, and precise ascent on every type is dominated by the few types
    that actually have interior maxima.

    Ascent on a fixed type can converge to folded configurations whose signed
    volume exceeds every polytope of that type, so each candidate must pass
    ``ConvexPosition``; see papers/sweep/sections/trap.tex.
    """

    def __init__(self, source: ComplexSource, *, restarts: int = 20, seed: int = 0,
                 screen_tol: float = 1e-9, screen_sweeps: int = 400,
                 refine_tol: float = 1e-14, refine_sweeps: int = 20000,
                 refine_top: int = 10):
        self.source = source
        self.restarts = restarts
        self.seed = seed
        self.screen_tol = screen_tol
        self.screen_sweeps = screen_sweeps
        self.refine_tol = refine_tol
        self.refine_sweeps = refine_sweeps
        self.refine_top = refine_top

    def run(self, n: int, *, lower_bound: float = -np.inf,
            progress: Callable[[int, TypeResult], None] | None = None) -> list[TypeResult]:
        """Screen every type, refine the leaders; sorted by volume, best first."""
        results: list[TypeResult] = []
        for index, record in enumerate(self.source.types(n)):
            result = self.screen(record, index, lower_bound)
            results.append(result)
            if progress is not None:
                progress(index, result)
        results.sort(key=lambda r: (r.volume, r.signed_best), reverse=True)

        feasible = [r for r in results if r.feasible]
        refined = {id(r): self.refine(r) for r in feasible[: self.refine_top]}
        results = [refined.get(id(r), r) for r in results]
        results.sort(key=lambda r: (r.volume, r.signed_best), reverse=True)
        return results

    def screen(self, record: TypeRecord, index: int = 0,
               lower_bound: float = -np.inf) -> TypeResult:
        """Coarse best convex-position volume within one type, over several restarts."""
        outcome = maximize_type(
            record.complex,
            restarts=self.restarts,
            seed=self.seed + index,
            tol=self.screen_tol,
            max_sweeps=self.screen_sweeps,
            start=record.points,
            lower_bound=lower_bound,
        )
        return TypeResult(
            label=record.label, complex=record.complex, volume=outcome.volume,
            points=outcome.points, signed_best=outcome.signed_best,
            restarts=outcome.restarts, rejected=outcome.rejected,
            sweeps=outcome.sweeps,
        )

    def refine(self, result: TypeResult) -> TypeResult:
        """Re-solve to full precision, warm-started from the screened optimum."""
        if result.points is None:
            return result
        solver = FixedTypeAscent(
            result.complex, tol=self.refine_tol, max_sweeps=self.refine_sweeps
        )
        outcome = solver.solve(result.points)
        if not ConvexPosition(result.complex).check(outcome.points).ok:
            return result
        return replace(
            result,
            volume=max(result.volume, float(outcome.volume)),
            points=outcome.points.copy(),
            sweeps=result.sweeps + outcome.sweeps,
            refined=True,
        )
