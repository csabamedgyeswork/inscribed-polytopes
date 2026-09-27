"""Exact maximisation over an independent set of vertices.

D is linear in each vertex separately, so a set of pairwise non-adjacent
vertices may be updated simultaneously; see paper/sections/preliminaries.tex.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import factorial

import numpy as np

from ..combinatorics import OrientedComplex
from ..volume import VolumeFunctional


def _bron_kerbosch(
    clique: set[int],
    candidates: set[int],
    excluded: set[int],
    neighbours: list[set[int]],
    out: list[frozenset[int]],
) -> None:
    if not candidates and not excluded:
        out.append(frozenset(clique))
        return
    pivot = max(candidates | excluded, key=lambda v: len(neighbours[v]))
    for v in list(candidates - neighbours[pivot]):
        _bron_kerbosch(
            clique | {v},
            candidates & neighbours[v],
            excluded & neighbours[v],
            neighbours,
            out,
        )
        candidates = candidates - {v}
        excluded = excluded | {v}


def maximal_independent_sets(cx: OrientedComplex) -> list[frozenset[int]]:
    """Maximal sets of pairwise non-adjacent vertices."""
    adjacency: list[set[int]] = [set() for _ in range(cx.n)]
    for e in cx.edges():
        a, b = tuple(e)
        adjacency[a].add(b)
        adjacency[b].add(a)
    non_adjacent = [
        {w for w in range(cx.n) if w != v and w not in adjacency[v]}
        for v in range(cx.n)
    ]
    found: list[frozenset[int]] = []
    _bron_kerbosch(set(), set(range(cx.n)), set(), non_adjacent, found)
    return found


def max_independent_set(cx: OrientedComplex) -> frozenset[int]:
    return max(maximal_independent_sets(cx), key=len)


@dataclass(frozen=True)
class IndependentSetElimination:
    """Exact closed-form maximisation over an independent set of vertices.

    ``d! * vol_max = sum_{v in S} |n_v| + sum_{facets disjoint from S} det(...)``.
    """

    form: VolumeFunctional
    eliminated: frozenset[int]

    def __post_init__(self) -> None:
        for a, b in combinations(sorted(self.eliminated), 2):
            if frozenset((a, b)) in self.form.cx.edges():
                raise ValueError(f"vertices {a} and {b} are adjacent; not eliminable")

    @classmethod
    def maximal(cls, form: VolumeFunctional) -> IndependentSetElimination:
        return cls(form=form, eliminated=max_independent_set(form.cx))

    def value(self, points: np.ndarray) -> float:
        """Volume after exactly maximising over the eliminated vertices."""
        points = np.asarray(points, dtype=float)
        total = 0.0
        for v in self.eliminated:
            total += float(np.linalg.norm(self.form.vertex_normal(points, v)))
        for f in self.form.cx.facets:
            if self.eliminated.isdisjoint(f):
                total += np.linalg.det(points[list(f)])
        return total / factorial(self.form.d)

    def fill(self, points: np.ndarray) -> np.ndarray:
        """Place each eliminated vertex at its optimum ``n_v / |n_v|``."""
        out = np.array(points, dtype=float, copy=True)
        for v in self.eliminated:
            nv = self.form.vertex_normal(out, v)
            out[v] = nv / np.linalg.norm(nv)
        return out
