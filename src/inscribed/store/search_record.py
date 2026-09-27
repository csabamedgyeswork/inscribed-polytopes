"""A type-free search over one (dimension, n) as a single machine-readable record."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..combinatorics import OrientedComplex
from .case import Evidence
from .configuration import Configuration


@dataclass(frozen=True)
class SearchRecord:
    """The best configuration a type-free search reached at one (dimension, n)."""

    dimension: int
    n: int
    volume: float
    points: tuple[tuple[float, ...], ...]
    facets: tuple[tuple[int, ...], ...]
    degrees: tuple[int, ...]
    convex: bool
    symmetry: dict
    budget: dict
    provenance: dict = field(default_factory=dict)

    @property
    def evidence(self) -> Evidence:
        """Derived, never stored: a type-free search claims no completeness."""
        return Evidence.E0

    def configuration(self) -> Configuration:
        """Reconstruct the best-found polytope as a renderable Configuration."""
        cx = OrientedComplex(d=self.dimension, n=self.n, facets=self.facets)
        return Configuration(
            points=np.asarray(self.points, dtype=float),
            complex=cx,
            volume=float(self.volume),
            label=f"free-d{self.dimension}-n{self.n:02d}",
            source="free-search",
            metadata={"evidence": self.evidence.value},
        )


def build_search_record(dimension: int, n: int, *, points: np.ndarray,
                        complex: OrientedComplex, volume: float, convex: bool,
                        symmetry: dict, budget: dict,
                        provenance: dict | None = None) -> SearchRecord:
    """Assemble a SearchRecord from one completed multistart run."""
    points = np.asarray(points, dtype=float)
    if points.shape != (complex.n, complex.d):
        raise ValueError(
            f"points shape {points.shape} does not match complex "
            f"(n={complex.n}, d={complex.d})"
        )
    return SearchRecord(
        dimension=int(dimension),
        n=int(n),
        volume=float(volume),
        points=tuple(tuple(float(x) for x in row) for row in points),
        facets=tuple(tuple(int(v) for v in f) for f in complex.facets),
        degrees=tuple(sorted(complex.vertex_degrees().values())),
        convex=bool(convex),
        symmetry=dict(symmetry),
        budget=dict(budget),
        provenance=dict(provenance or {}),
    )
