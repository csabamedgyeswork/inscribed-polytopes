"""The computations printed in papers/maxvol-d4/sections/preliminaries.tex."""

from __future__ import annotations

import re
import unittest
from fractions import Fraction
from itertools import combinations, combinations_with_replacement, permutations
from math import ceil, factorial, floor, sqrt
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from inscribed.enumeration import FirschingSource

ROOT = Path(__file__).resolve().parents[3]
PRELIMINARIES = ROOT / "papers" / "maxvol-d4" / "sections" / "preliminaries.tex"


def preliminaries_text() -> str:
    """The preliminaries section with line breaks collapsed to single spaces."""
    return re.sub(r"\s+", " ", PRELIMINARIES.read_text())


def simplex_factor(k: int) -> float:
    """The factor (k+1)^((k+1)/2) k^(-k/2) of Horvath-Langi Theorem 3."""
    return (k + 1) ** ((k + 1) / 2) * k ** (-k / 2)


def gale_facets(n: int, d: int) -> list[tuple[int, ...]]:
    """The d-subsets of 0..n-1 satisfying Gale's evenness condition."""
    return [
        s for s in combinations(range(n), d)
        if all(sum(a < v < b for v in s) % 2 == 0
               for a, b in combinations([i for i in range(n) if i not in s], 2))
    ]


class TestCyclicFacets(unittest.TestCase):
    """'Labelling the vertices 1, ..., 7 along the curve, its 14 facets are 1234, 1237, ...'"""

    def setUp(self):
        text = preliminaries_text()
        match = re.search(r"its \$(\d+)\$ facets are~\\artref\{testprelim\} \\\[(.*?)\\\]", text)
        self.assertIsNotNone(match)
        self.count = int(match.group(1))
        listed = re.findall(r"\b[1-7]{4}\b", match.group(2))
        self.printed = [tuple(int(c) for c in s) for s in listed]

    def test_count(self):
        self.assertEqual(self.count, len(self.printed))

    def test_list_is_the_evenness_condition(self):
        facets = [tuple(v + 1 for v in s) for s in gale_facets(7, 4)]
        self.assertEqual(self.printed, facets)

    def test_list_is_the_hull_of_moment_curve_points(self):
        """Independent of Gale's theorem: the facets of the hull of 7 points on the curve."""
        points = np.array([[t ** k for k in range(1, 5)] for t in np.linspace(-1, 1, 7)])
        hull = {tuple(sorted(int(v) + 1 for v in s)) for s in ConvexHull(points).simplices}
        self.assertEqual(hull, set(self.printed))


class TestFirschingTypes(unittest.TestCase):
    """'Of the five combinatorial types of simplicial 4-polytopes with 7 vertices in
    Firsching's enumeration, C_4(7) is the first, labelled firsching-7-0 in our code;
    the other four are not cyclic.'"""

    def setUp(self):
        records = list(FirschingSource().types(7))
        self.labels = [r.label for r in records]
        self.types = [frozenset(frozenset(f) for f in r.complex.facets) for r in records]
        cyclic = frozenset(frozenset(f) for f in gale_facets(7, 4))
        self.relabelled = {
            frozenset(frozenset(p[v] for v in f) for f in cyclic)
            for p in permutations(range(7))
        }

    def test_printed(self):
        self.assertIn("Of the five combinatorial types of simplicial $4$-polytopes with $7$ "
                      "vertices", preliminaries_text())
        self.assertIn(r"$C_{4}(7)$ is the first, labelled \texttt{firsching-7-0}",
                      preliminaries_text())

    def test_count(self):
        self.assertEqual(len(self.types), 5)

    def test_only_the_first_is_cyclic(self):
        self.assertEqual(self.labels[0], "firsching-7-0")
        self.assertEqual([t in self.relabelled for t in self.types],
                         [True, False, False, False, False])


class TestNoncyclicMaximum(unittest.TestCase):
    """'For d = 4 the dimensions are floor(4/3) = 1, 1 and ceil(4/3) = 2, and the formula
    gives 1/24 * 2 * 2 * 3 sqrt(3)/2 = sqrt(3)/4.'"""

    def test_printed(self):
        self.assertIn(r"\frac{1}{24} \cdot 2 \cdot 2 \cdot \frac{3\sqrt{3}}{2} = "
                      r"\frac{\sqrt{3}}{4}", preliminaries_text())

    def test_dimensions(self):
        d = 4
        allowed = {floor(d / 3), ceil(d / 3)}
        self.assertEqual(allowed, {1, 2})
        found = [c for c in combinations_with_replacement(sorted(allowed), 3) if sum(c) == d]
        self.assertEqual(found, [(1, 1, 2)])

    def test_factors(self):
        self.assertAlmostEqual(simplex_factor(1), 2)
        self.assertAlmostEqual(simplex_factor(2), 3 * sqrt(3) / 2)

    def test_volume(self):
        volume = simplex_factor(1) ** 2 * simplex_factor(2) / factorial(4)
        self.assertAlmostEqual(volume, sqrt(3) / 4, places=15)
        self.assertEqual((Fraction(1, 24) * 2 * 2 * Fraction(3, 2)) ** 2 * 3, Fraction(3, 16))

    def test_attained_by_triangle_and_two_diameters(self):
        """'It is attained by the convex hull of a regular triangle and two diameters of S^3 in
        mutually orthogonal linear subspaces.'"""
        self.assertIn("convex hull of a regular triangle and two diameters", preliminaries_text())
        angles = 2 * np.pi * np.arange(3) / 3
        triangle = [[np.cos(a), np.sin(a), 0, 0] for a in angles]
        diameters = [[0, 0, 1, 0], [0, 0, -1, 0], [0, 0, 0, 1], [0, 0, 0, -1]]
        points = np.array(triangle + diameters)
        self.assertTrue(np.allclose(np.linalg.norm(points, axis=1), 1))
        self.assertAlmostEqual(ConvexHull(points).volume, sqrt(3) / 4, places=12)


if __name__ == "__main__":
    unittest.main()
