"""Orient an abstract facet list into a coherent simplicial sphere."""

from __future__ import annotations

import json
from collections import deque

from .complex import (
    ClosureError,
    Facet,
    OrientedComplex,
    TopologyError,
    permutation_parity,
)


def _flip(facet: Facet) -> Facet:
    return facet[:-2] + (facet[-1], facet[-2])


def _ridge_signs(facet: Facet) -> dict[tuple[int, ...], int]:
    """Induced sign of each ridge of ``facet`` under its current order."""
    out: dict[tuple[int, ...], int] = {}
    for k in range(len(facet)):
        ridge = facet[:k] + facet[k + 1 :]
        out[tuple(sorted(ridge))] = (-1) ** k * permutation_parity(ridge)
    return out


def orient(facets, *, one_indexed: bool = False) -> OrientedComplex:
    """Return a coherently oriented complex on the given abstract facets."""
    shift = 1 if one_indexed else 0
    raw = [tuple(int(v) - shift for v in f) for f in facets]
    if not raw:
        raise ClosureError("no facets given")
    d = len(raw[0])
    if any(len(f) != d for f in raw):
        raise ClosureError("all facets must have the same size")
    for f in raw:
        if len(set(f)) != d:
            raise ClosureError(f"facet {f} has repeated vertices")

    ridge_to_facets: dict[tuple[int, ...], list[int]] = {}
    for i, f in enumerate(raw):
        for ridge in _ridge_signs(f):
            ridge_to_facets.setdefault(ridge, []).append(i)
    for ridge, owners in ridge_to_facets.items():
        if len(owners) != 2:
            raise ClosureError(
                f"ridge {ridge} lies in {len(owners)} facets, expected 2"
            )

    neighbours: dict[int, list[tuple[int, tuple[int, ...]]]] = {i: [] for i in range(len(raw))}
    for ridge, (a, b) in ridge_to_facets.items():
        neighbours[a].append((b, ridge))
        neighbours[b].append((a, ridge))

    oriented: list[Facet | None] = [None] * len(raw)
    oriented[0] = raw[0]
    signs = {0: _ridge_signs(raw[0])}
    queue: deque[int] = deque([0])
    while queue:
        i = queue.popleft()
        for j, ridge in neighbours[i]:
            if oriented[j] is None:
                cand = raw[j]
                if _ridge_signs(cand)[ridge] == signs[i][ridge]:
                    cand = _flip(cand)
                oriented[j] = cand
                signs[j] = _ridge_signs(cand)
                queue.append(j)
            elif signs[j][ridge] == signs[i][ridge]:
                raise TopologyError(
                    f"facets {i} and {j} induce the same sign on ridge {ridge}; "
                    "the complex is not orientable"
                )

    n = max(max(f) for f in oriented) + 1  # type: ignore[arg-type]
    cx = OrientedComplex(d=d, n=n, facets=tuple(oriented))  # type: ignore[arg-type]
    cx.check()
    return cx


def parse_facets(text: str) -> list[list[int]]:
    """Parse facets from a JSON list of lists or one facet per line."""
    text = text.strip()
    if not text:
        raise ValueError("empty facet input")
    if text.startswith("["):
        data = json.loads(text)
        return [[int(v) for v in f] for f in data]
    facets = []
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            facets.append([int(v) for v in line.replace(",", " ").split()])
    if not facets:
        raise ValueError("no facets found in input")
    return facets
