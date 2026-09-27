"""Maximise the signed volume of one combinatorial type over several restarts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis import ConvexPosition
from ..combinatorics import OrientedComplex
from .ascent import FixedTypeAscent


@dataclass
class MaximizeOutcome:
    """Best convex-position result within one combinatorial type."""

    volume: float
    points: np.ndarray | None
    signed_best: float
    restarts: int
    rejected: int
    sweeps: int

    @property
    def feasible(self) -> bool:
        return self.points is not None


def maximize_type(
    cx: OrientedComplex,
    *,
    restarts: int = 20,
    seed: int = 0,
    tol: float = 1e-9,
    max_sweeps: int = 400,
    start: np.ndarray | None = None,
    lower_bound: float = -np.inf,
) -> MaximizeOutcome:
    """Best convex-position volume within ``cx``, over several random restarts."""
    rng = np.random.default_rng(seed)
    validator = ConvexPosition(cx)

    best, best_points = -np.inf, None
    signed_best, rejected, used, sweeps = -np.inf, 0, 0, 0
    for attempt in range(restarts):
        used = attempt + 1
        if attempt == 0 and start is not None:
            initial = np.asarray(start, dtype=float)
        else:
            initial = rng.normal(size=(cx.n, cx.d))
            initial /= np.linalg.norm(initial, axis=1, keepdims=True)

        outcome = FixedTypeAscent(cx, tol=tol, max_sweeps=max_sweeps).solve(initial)
        if outcome.volume < 0:
            flipped = FixedTypeAscent(
                cx.reversed_orientation(), tol=tol, max_sweeps=max_sweeps
            ).solve(initial)
            if flipped.volume > outcome.volume:
                outcome = flipped

        signed_best = max(signed_best, outcome.volume)
        sweeps = max(sweeps, outcome.sweeps)
        if validator.check(outcome.points).ok:
            if outcome.volume > best:
                best, best_points = outcome.volume, outcome.points.copy()
        else:
            rejected += 1

        if signed_best < lower_bound:
            break

    return MaximizeOutcome(
        volume=best,
        points=best_points,
        signed_best=signed_best,
        restarts=used,
        rejected=rejected,
        sweeps=sweeps,
    )
