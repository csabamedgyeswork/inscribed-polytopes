"""Solver and initializer interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from ..combinatorics import OrientedComplex


@dataclass
class SearchResult:
    points: np.ndarray
    volume: float
    cx: OrientedComplex | None
    sweeps: int
    info: dict = field(default_factory=dict)


class Solver(ABC):
    """Improves a starting configuration and reports the result."""

    @abstractmethod
    def solve(self, points: np.ndarray) -> SearchResult: ...


class Initializer(ABC):
    """Produces starting configurations."""

    @abstractmethod
    def initial(self) -> np.ndarray: ...
