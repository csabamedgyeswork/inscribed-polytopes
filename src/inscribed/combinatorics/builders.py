"""Canonical small complexes used across tests and scripts."""

from __future__ import annotations

from .complex import OrientedComplex


def bipyramid(m: int) -> OrientedComplex:
    """Double ``m``-pyramid: an ``m``-cycle plus two apexes, ``m + 2`` vertices."""
    if m < 3:
        raise ValueError("need m >= 3")
    top, bot = m, m + 1
    facets = []
    for j in range(m):
        a, b = j, (j + 1) % m
        facets.append((top, a, b))
        facets.append((bot, b, a))
    return OrientedComplex(d=3, n=m + 2, facets=tuple(facets))


def simplex(d: int) -> OrientedComplex:
    """Boundary of the ``d``-simplex: ``d + 1`` vertices in ``R^d``."""
    verts = tuple(range(d + 1))
    facets = []
    for k in range(d + 1):
        f = verts[:k] + verts[k + 1 :]
        if k % 2 == 1:
            f = f[:-2] + (f[-1], f[-2])
        facets.append(f)
    return OrientedComplex(d=d, n=d + 1, facets=tuple(facets))
