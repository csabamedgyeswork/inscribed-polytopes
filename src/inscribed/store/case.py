"""A full sweep over one (dimension, n) as a single machine-readable record."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from ..combinatorics import OrientedComplex
from .configuration import Configuration


class Evidence(Enum):
    """Strength of a computed case; see the evidence ladder in papers/sweep/sections/method.tex."""

    E0 = "heuristic"
    E1 = "exhaustive-numerical"
    E2 = "exhibited"
    E3 = "relaxation-excluded"
    E4 = "certified"


@dataclass(frozen=True)
class TypeEntry:
    """One combinatorial type's outcome inside a case."""

    label: str
    degrees: tuple[int, ...]
    feasible: bool
    volume: float | None
    signed_max: float
    rejected: int
    restarts: int


@dataclass(frozen=True)
class CaseResult:
    """Every type at one (dimension, n), with the relaxation triage."""

    dimension: int
    n: int
    assumptions: tuple[str, ...]
    enumeration: dict
    budget: dict
    reference: dict | None
    winner: dict
    triage: dict
    types: tuple[TypeEntry, ...]
    provenance: dict = field(default_factory=dict)

    @property
    def evidence(self) -> Evidence:
        """Derived, never stored: E3 when every rival is bounded below the winner."""
        if not self.enumeration.get("complete", False):
            return Evidence.E1
        winner = self.winner["volume"]
        rivals = [t for t in self.types if t.label != self.winner["label"]]
        if rivals and all(t.signed_max < winner for t in rivals):
            return Evidence.E3
        return Evidence.E1

    def winner_configuration(self) -> Configuration:
        """Reconstruct the winning polytope as a renderable Configuration."""
        points = self.winner.get("points")
        if points is None:
            raise ValueError(f"case d{self.dimension}-n{self.n} stores no winner points")
        cx = OrientedComplex(
            d=self.dimension, n=self.n,
            facets=tuple(tuple(int(v) for v in f) for f in self.winner["facets"]),
        )
        return Configuration(
            points=np.asarray(points, dtype=float), complex=cx,
            volume=float(self.winner["volume"]), label=self.winner["label"],
            source=self.enumeration.get("source", "manual"),
            metadata={"evidence": self.evidence.value},
        )


def build_case(dimension: int, n: int, results, source_name: str,
               enumeration: dict, budget: dict, reference: dict | None,
               assumptions: tuple[str, ...] = (), provenance: dict | None = None,
               lower_bound: float = -math.inf) -> CaseResult:
    """Assemble a CaseResult from TypeSweep results; lower_bound must be -inf."""
    if math.isfinite(lower_bound):
        raise ValueError(
            "lower_bound must be -inf: pruning truncates signed_max and would "
            "mark types excludable on a truncated bound"
        )
    if not results:
        raise ValueError("no results to build a case from")
    winner = results[0]
    types = tuple(
        TypeEntry(
            label=r.label,
            degrees=tuple(sorted(r.complex.vertex_degrees().values())),
            feasible=r.feasible,
            volume=float(r.volume) if r.feasible else None,
            signed_max=float(r.signed_best),
            rejected=int(r.rejected),
            restarts=int(r.restarts),
        )
        for r in results
    )
    rivals = [t for t in types if t.label != winner.label]
    hard = [t for t in rivals if t.signed_max >= winner.volume]
    triage = {
        "target": float(winner.volume),
        "excluded": len(rivals) - len(hard),
        "hard": len(hard),
        "hard_labels": [t.label for t in hard],
    }
    return CaseResult(
        dimension=int(dimension),
        n=int(n),
        assumptions=tuple(assumptions),
        enumeration=dict(enumeration),
        budget=dict(budget),
        reference=reference,
        winner={
            "label": winner.label,
            "volume": float(winner.volume),
            "degrees": sorted(winner.complex.vertex_degrees().values()),
            "points": winner.points.tolist() if winner.points is not None else None,
            "facets": [list(f) for f in winner.complex.facets],
        },
        triage=triage,
        types=types,
        provenance=dict(provenance or {}),
    )
