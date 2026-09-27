"""Symmetry group of a configuration from its Gram matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SymmetryReport:
    order: int
    proper: int
    improper: int
    orbits: list[list[int]]
    inner_products: list[tuple[float, int]]
    permutations: list[tuple[int, ...]]

    @property
    def orbit_sizes(self) -> list[int]:
        return sorted(len(o) for o in self.orbits)

    def summary(self) -> str:
        return (
            f"|G|={self.order} (proper {self.proper}, improper {self.improper}), "
            f"orbits {self.orbit_sizes}, "
            f"{len(self.inner_products)} distinct inner products"
        )


def _quantize(matrix: np.ndarray, tol: float) -> tuple[list[list[int]], list[float]]:
    values: list[float] = []
    colors: list[list[int]] = []
    for row in matrix:
        crow = []
        for x in row:
            for idx, v in enumerate(values):
                if abs(x - v) <= tol:
                    crow.append(idx)
                    break
            else:
                values.append(float(x))
                crow.append(len(values) - 1)
        colors.append(crow)
    return colors, values


def _gram_automorphisms(gram_colors, signature, n: int) -> list[tuple[int, ...]]:
    result: list[tuple[int, ...]] = []
    mapping = [-1] * n
    used = [False] * n

    def extend(k: int) -> None:
        if k == n:
            result.append(tuple(mapping))
            return
        for cand in range(n):
            if used[cand] or signature[cand] != signature[k]:
                continue
            if gram_colors[cand][cand] != gram_colors[k][k]:
                continue
            if any(gram_colors[k][j] != gram_colors[cand][mapping[j]] for j in range(k)):
                continue
            mapping[k] = cand
            used[cand] = True
            extend(k + 1)
            used[cand] = False
            mapping[k] = -1

    extend(0)
    return result


class SymmetryDetector:
    """Finds the orthogonal symmetries of a point configuration."""

    def __init__(self, *, tol: float = 1e-7):
        self.tol = tol

    def analyze(self, points: np.ndarray) -> SymmetryReport:
        points = np.asarray(points, dtype=float)
        gram = points @ points.T
        n = gram.shape[0]
        colors, _ = _quantize(gram, self.tol)
        signature = [tuple(sorted(colors[i])) for i in range(n)]
        perms = _gram_automorphisms(colors, signature, n)

        proper = improper = 0
        for perm in perms:
            q = np.linalg.pinv(points) @ points[list(perm)]
            if np.linalg.det(q) > 0:
                proper += 1
            else:
                improper += 1

        orbits = self._orbits(perms, n)
        inner_products = self._inner_product_histogram(gram)

        return SymmetryReport(
            order=len(perms),
            proper=proper,
            improper=improper,
            orbits=orbits,
            inner_products=inner_products,
            permutations=perms,
        )

    def _orbits(self, perms: list[tuple[int, ...]], n: int) -> list[list[int]]:
        parent = list(range(n))

        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for perm in perms:
            for i, j in enumerate(perm):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
        buckets: dict[int, list[int]] = {}
        for v in range(n):
            buckets.setdefault(find(v), []).append(v)
        return sorted(buckets.values(), key=lambda o: (len(o), o))

    def _inner_product_histogram(self, gram: np.ndarray) -> list[tuple[float, int]]:
        n = gram.shape[0]
        off = [gram[i, j] for i in range(n) for j in range(i + 1, n)]
        _, values = _quantize(np.array(off).reshape(1, -1), self.tol)
        counts = [(v, sum(1 for x in off if abs(x - v) <= self.tol)) for v in values]
        counts.sort()
        return counts
