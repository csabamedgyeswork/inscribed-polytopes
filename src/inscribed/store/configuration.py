"""A maximised inscribed configuration as a first-class, serialisable record."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..combinatorics import OrientedComplex


@dataclass(frozen=True)
class Configuration:
    """An inscribed configuration with its combinatorial type and provenance."""

    points: np.ndarray
    complex: OrientedComplex
    volume: float
    label: str = ""
    source: str = "manual"
    metadata: dict = field(default_factory=dict)

    @property
    def d(self) -> int:
        return int(self.points.shape[1])

    @property
    def n(self) -> int:
        return int(self.points.shape[0])

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", np.asarray(self.points, dtype=float))
        if self.points.shape != (self.complex.n, self.complex.d):
            raise ValueError(
                f"points shape {self.points.shape} does not match complex "
                f"(n={self.complex.n}, d={self.complex.d})"
            )
