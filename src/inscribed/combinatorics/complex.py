"""Oriented simplicial spheres in arbitrary dimension.

See paper/sections/preliminaries.tex.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

Facet = tuple[int, ...]


class CombinatoricsError(ValueError):
    """Base class for invalid oriented complexes."""


class OrientationError(CombinatoricsError):
    """Facets are duplicated, degenerate, or incoherently oriented."""


class ClosureError(CombinatoricsError):
    """A ridge does not lie in exactly two facets."""


class TopologyError(CombinatoricsError):
    """The complex does not have the Euler characteristic of a sphere."""


def permutation_parity(perm: tuple[int, ...]) -> int:
    """+1 for an even permutation of ``sorted(perm)``, -1 for odd."""
    items = list(perm)
    seen = [False] * len(items)
    order = {v: i for i, v in enumerate(sorted(items))}
    parity = 1
    for start in range(len(items)):
        if seen[start]:
            continue
        length = 0
        node = start
        while not seen[node]:
            seen[node] = True
            node = order[items[node]]
            length += 1
        if length % 2 == 0:
            parity = -parity
    return parity


@dataclass(frozen=True)
class OrientedComplex:
    """An oriented simplicial ``(d-1)``-sphere on vertices ``0..n-1``."""

    d: int
    n: int
    facets: tuple[Facet, ...]

    @classmethod
    def from_labels(cls, facets, *, one_indexed: bool = False) -> OrientedComplex:
        shift = 1 if one_indexed else 0
        norm = tuple(tuple(int(v) - shift for v in f) for f in facets)
        d = len(norm[0])
        if any(len(f) != d for f in norm):
            raise OrientationError("all facets must have the same size")
        n = max(max(f) for f in norm) + 1
        return cls(d=d, n=n, facets=norm)

    def ridges(self):
        """Yield ``(sorted_ridge, induced_orientation, facet_index)`` triples."""
        for fi, f in enumerate(self.facets):
            for k in range(self.d):
                ridge = f[:k] + f[k + 1 :]
                sign = (-1) ** k * permutation_parity(ridge)
                yield tuple(sorted(ridge)), sign, fi

    def degrees(self) -> dict[int, int]:
        """Number of facets containing each vertex."""
        deg = {v: 0 for v in range(self.n)}
        for f in self.facets:
            for v in f:
                deg[v] += 1
        return deg

    def vertex_degrees(self) -> dict[int, int]:
        """Number of neighbours of each vertex (graph degree)."""
        nbrs: dict[int, set[int]] = {v: set() for v in range(self.n)}
        for f in self.facets:
            for a, b in combinations(f, 2):
                nbrs[a].add(b)
                nbrs[b].add(a)
        return {v: len(s) for v, s in nbrs.items()}

    def edges(self) -> set[frozenset[int]]:
        out: set[frozenset[int]] = set()
        for f in self.facets:
            for a, b in combinations(f, 2):
                out.add(frozenset((a, b)))
        return out

    def faces(self, k: int) -> set[frozenset[int]]:
        """All ``k``-element vertex subsets spanning a face."""
        out: set[frozenset[int]] = set()
        for f in self.facets:
            for c in combinations(f, k):
                out.add(frozenset(c))
        return out

    def euler_characteristic(self) -> int:
        chi = 0
        for k in range(1, self.d + 1):
            chi += (-1) ** (k - 1) * len(self.faces(k))
        return chi

    def star(self, v: int) -> tuple[Facet, ...]:
        return tuple(f for f in self.facets if v in f)

    def link_cycle(self, v: int) -> tuple[int, ...]:
        """For ``d == 3``: the neighbours of ``v`` in link order."""
        if self.d != 3:
            raise ValueError("link_cycle is only implemented for d == 3")
        succ = {}
        for f in self.star(v):
            k = f.index(v)
            a, b = f[(k + 1) % 3], f[(k + 2) % 3]
            succ[a] = b
        start = next(iter(succ))
        cycle = [start]
        while succ[cycle[-1]] != start:
            cycle.append(succ[cycle[-1]])
            if len(cycle) > self.n:
                raise CombinatoricsError(f"link of {v} is not a single cycle")
        return tuple(cycle)

    def check(self) -> None:
        """Raise unless closed, coherently oriented, with the Euler characteristic of
        ``S^(d-1)``."""
        if len(set(self.facets)) != len(self.facets):
            raise OrientationError("duplicate facets")
        for f in self.facets:
            if len(set(f)) != self.d:
                raise OrientationError(f"facet {f} has repeated vertices")

        incident: dict[tuple[int, ...], list[int]] = {}
        for ridge, sign, _ in self.ridges():
            incident.setdefault(ridge, []).append(sign)
        for ridge, signs in incident.items():
            if len(signs) != 2:
                raise ClosureError(f"ridge {ridge} lies in {len(signs)} facets, expected 2")
            if sum(signs) != 0:
                raise OrientationError(f"ridge {ridge} is incoherently oriented")

        if set(range(self.n)) != {v for f in self.facets for v in f}:
            raise TopologyError("some vertex label is unused")

        expected = 1 + (-1) ** (self.d - 1)
        chi = self.euler_characteristic()
        if chi != expected:
            raise TopologyError(
                f"Euler characteristic {chi}, expected {expected} for S^{self.d - 1}"
            )

    def is_valid(self) -> bool:
        try:
            self.check()
        except CombinatoricsError:
            return False
        return True

    def reversed_orientation(self) -> OrientedComplex:
        return OrientedComplex(
            d=self.d,
            n=self.n,
            facets=tuple(f[:-2] + (f[-1], f[-2]) for f in self.facets),
        )
