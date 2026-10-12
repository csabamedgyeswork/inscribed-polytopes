"""The facets of the cyclic polytope C_d(n) by Gale's evenness condition."""

from __future__ import annotations

from itertools import combinations


def cyclic_facets(n: int, d: int) -> set[frozenset[int]]:
    """The d-subsets of 0..n-1 satisfying Gale's evenness condition."""
    return {
        frozenset(s) for s in combinations(range(n), d)
        if all(sum(a < v < b for v in s) % 2 == 0
               for a, b in combinations([i for i in range(n) if i not in s], 2))
    }
