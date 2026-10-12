"""The numbers printed in papers/maxvol-d4/sections/seven.tex."""

from __future__ import annotations

import json
import re
import unittest
from fractions import Fraction
from math import factorial, sqrt
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from inscribed.certify import cyclic_facets
from inscribed.store import certificate_dir

from .configuration import hull_facets, regular_heptagon, stored_ascent

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "papers" / "maxvol-d4" / "sections" / "seven.tex"
LABEL = "cyclic-4-7"


def results_text() -> str:
    """The results section with line breaks collapsed to single spaces."""
    return re.sub(r"\s+", " ", RESULTS.read_text())


def summary() -> dict:
    """The stored summary of the cyclic type's certificate."""
    return json.loads((certificate_dir(4, 7) / f"{LABEL}.summary.json").read_text())


class TestBound(unittest.TestCase):
    """'... gives vol_4(P) <= beta/4! = 2/5.'"""

    def test_printed(self):
        self.assertIn(r"\beta/4! = 2/5", results_text())

    def test_bound_is_beta_over_factorial(self):
        data = summary()
        self.assertTrue(data["verification"]["ok"])
        self.assertEqual(Fraction(data["gamma"]) / factorial(4), Fraction(2, 5))
        self.assertEqual(Fraction(data["volume_bound"]), Fraction(2, 5))

    def test_below_the_noncyclic_maximum(self):
        """2/5 < sqrt(3)/4, compared exactly through squares."""
        self.assertLess(Fraction(2, 5) ** 2, Fraction(3, 16))


class TestNumericalMaximum(unittest.TestCase):
    """'... the largest volume of type C_4(7) found numerically is about 0.3891.'"""

    def test_printed_value(self):
        """The stored ascent volume is attained by an explicit inscribed polytope of type
        C_4(7)."""
        self.assertIn("about $0.3891$", results_text())
        points = regular_heptagon()
        self.assertTrue(np.allclose(np.linalg.norm(points, axis=1), 1))
        self.assertEqual(hull_facets(points), cyclic_facets(7, 4))
        volume = ConvexHull(points).volume
        self.assertAlmostEqual(volume, stored_ascent("convex_volume"), places=10)
        self.assertEqual(round(volume, 4), 0.3891)
        self.assertLess(volume, 2 / 5)
        self.assertLess(2 / 5, sqrt(3) / 4)


if __name__ == "__main__":
    unittest.main()
