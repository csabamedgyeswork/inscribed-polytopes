"""Multistart driver shared by all solvers."""

from __future__ import annotations

import numpy as np

from ..analysis import Validator
from .base import Initializer, SearchResult, Solver


class MultiStartSearch:
    """Run a solver from many starts, keep the best eligible result."""

    def __init__(
        self,
        solver: Solver,
        initializer: Initializer,
        validator: Validator | None = None,
    ):
        self.solver = solver
        self.initializer = initializer
        self.validator = validator

    def run(self, restarts: int) -> SearchResult:
        best: SearchResult | None = None
        values: list[float] = []
        rejected = 0
        for _ in range(restarts):
            result = self.solver.solve(self.initializer.initial())
            if self.validator is not None:
                try:
                    eligible = self.validator.check(result.points).ok
                except ValueError:
                    eligible = False
                if not eligible:
                    rejected += 1
                    continue
            values.append(result.volume)
            if best is None or result.volume > best.volume:
                best = result
        if best is None:
            raise RuntimeError(
                f"no eligible result in {restarts} restarts ({rejected} rejected)"
            )
        arr = np.array(values)
        best.info["restarts"] = restarts
        best.info["rejected"] = rejected
        best.info["distinct_values"] = int(
            len(np.unique(np.round(arr[np.isfinite(arr)], 9)))
        )
        best.info["fraction_at_best"] = float(np.mean(arr > best.volume - 1e-9))
        return best
