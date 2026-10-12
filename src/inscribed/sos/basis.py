"""Monomial bases and their orbits under a symmetry group."""

from __future__ import annotations

from itertools import combinations_with_replacement

from ..certify import Monomial, parity_key
from .symmetry import Symmetry


def monomials(size: int, degree: int) -> list[Monomial]:
    """Every monomial of degree at most ``degree`` in ``size`` variables."""
    return [m for t in range(degree + 1)
            for m in combinations_with_replacement(range(size), t)]


class ParityBlocks:
    """Monomials up to a degree in d*n variables, grouped by coordinate parity class."""

    def __init__(self, d: int, n: int, degree: int):
        self.d = d
        grouped: dict[int, list[Monomial]] = {}
        for m in monomials(d * n, degree):
            grouped.setdefault(parity_key(m, d), []).append(m)
        self.keys = sorted(grouped)
        self.blocks = [grouped[k] for k in self.keys]
        self.position = {m: (b, i) for b, block in enumerate(self.blocks)
                         for i, m in enumerate(block)}


class MomentOrbits:
    """Orbits of monomials under the group, with each monomial's sign relative to its orbit."""

    def __init__(self, monomials: list[Monomial], generators: list[Symmetry], d: int):
        self.index: dict[Monomial, tuple[int, int]] = {}
        zero: set[int] = set()
        count = 0
        for m in monomials:
            if m in self.index:
                continue
            orbit, count = count, count + 1
            self.index[m] = (orbit, 1)
            queue = [m]
            while queue:
                a = queue.pop()
                sign_a = self.index[a][1]
                for g in generators:
                    b, s = g.apply(a, d)
                    s *= sign_a
                    if b not in self.index:
                        self.index[b] = (orbit, s)
                        queue.append(b)
                    elif self.index[b][1] != s:
                        zero.add(orbit)
        live = [o for o in range(count) if o not in zero]
        self.column = {o: i for i, o in enumerate(live)}
        self.total = count

    @property
    def size(self) -> int:
        """Number of free moment variables."""
        return len(self.column)

    def lookup(self, monomial: Monomial) -> tuple[int, int] | None:
        """(column, sign) of a monomial's moment, or None when symmetry forces it to zero."""
        orbit, sign = self.index[monomial]
        column = self.column.get(orbit)
        return None if column is None else (column, sign)
