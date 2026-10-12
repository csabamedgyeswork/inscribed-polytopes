"""The claims printed in papers/maxvol-d4/sections/method.tex."""

from __future__ import annotations

import gzip
import json
import re
import unittest
from fractions import Fraction
from itertools import combinations
from math import factorial, sqrt
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from inscribed.certify import SignedVolume, cyclic_facets, facet_sets
from inscribed.sos.basis import monomials
from inscribed.store import certificate_dir

from .configuration import (
    caratheodory,
    certificate_facets,
    hull_facets,
    regular_heptagon,
    signed_volume,
    stored_ascent,
)

ROOT = Path(__file__).resolve().parents[3]
METHOD = ROOT / "papers" / "maxvol-d4" / "sections" / "method.tex"
SCRIPT = ROOT / "scripts" / "sos_certificate.py"
LABEL = "cyclic-4-7"
D = 4


def method_text() -> str:
    """The method section with line breaks collapsed to single spaces."""
    return re.sub(r"\s+", " ", METHOD.read_text())


def certificate() -> dict:
    """The stored certificate of the cyclic type, as raw JSON."""
    path = certificate_dir(4, 7) / f"{LABEL}.certificate.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def parity(monomial) -> tuple[int, ...]:
    """The parity vector of a monomial, identified with its complement."""
    eps = tuple(sum(v % D == a for v in monomial) % 2 for a in range(D))
    return min(eps, tuple(1 - e for e in eps))


class TestDegree(unittest.TestCase):
    """'Let v(x) be the column vector of all monomials of degree at most 2.' and 'We use
    monomials of degree at most 2, the smallest degree for which v^T Q v reaches the degree 4
    of D.'"""

    def test_printed(self):
        self.assertIn("We use monomials of degree at most $2$", method_text())

    def test_stored_blocks(self):
        self.assertEqual(max(len(m) for b in certificate()["blocks"] for m in b), 2)

    def test_blocks_partition_all_monomials_of_degree_two(self):
        self.assertIn("the column vector of all monomials of degree at most $2$", method_text())
        stored = [tuple(m) for b in certificate()["blocks"] for m in b]
        self.assertEqual(len(stored), len(set(stored)))
        self.assertEqual(set(stored), set(monomials(D * 7, 2)))

    def test_signed_volume_has_degree_four(self):
        facets = [tuple(f) for f in certificate()["facets"]]
        degrees = {len(m) for m in SignedVolume.from_facets(facets, D).polynomial.terms}
        self.assertEqual(degrees, {D})

    def test_signed_volume_has_integer_coefficients(self):
        self.assertIn("with integer coefficients", method_text())
        facets = [tuple(f) for f in certificate()["facets"]]
        coefficients = SignedVolume.from_facets(facets, D).polynomial.terms.values()
        self.assertTrue(all(c.denominator == 1 for c in coefficients))


class TestSignedVolumeIsVolume(unittest.TestCase):
    """'If the orientation of C is coherent and p realises C, then d! vol_d(P) = |D(p)|',
    for the oriented facets stored in the certificate."""

    def test_on_inscribed_realisations(self):
        facets = certificate_facets()
        rng = np.random.default_rng(0)
        angle_sets = [2 * np.pi * np.arange(7) / 7]
        angle_sets += [np.sort(rng.uniform(0, 2 * np.pi, 7)) for _ in range(5)]
        for angles in angle_sets:
            with self.subTest(angles=angles):
                points = caratheodory(angles)
                self.assertTrue(np.allclose(np.linalg.norm(points, axis=1), 1))
                self.assertEqual(hull_facets(points), facet_sets(facets))
                volume = ConvexHull(points).volume
                self.assertAlmostEqual(abs(signed_volume(points, facets)) / (factorial(D) * volume),
                                       1, places=10)


class TestFacets(unittest.TestCase):
    """'The facet vertex sets must also be exactly those of the type: for C_4(7) the
    fourteen facets listed in the cyclic remark, which the verifier rebuilds from Gale's
    condition.'"""

    def test_count(self):
        self.assertIn("for $C_{4}(7)$ the fourteen facets", method_text())
        self.assertEqual(len(certificate()["facets"]), 14)

    def test_facets_are_gales_list(self):
        self.assertIn("rebuilds from Gale's", method_text())
        self.assertEqual(facet_sets(tuple(f) for f in certificate()["facets"]),
                         cyclic_facets(7, D))


class TestComputation(unittest.TestCase):
    """The solver, blocks, rounding and choice of beta in 'How the certificate was found'."""

    def setUp(self):
        self.text = method_text()
        self.cert = certificate()

    def test_denominator(self):
        self.assertIn("a rational with denominator $10^{9}$", self.text)
        self.assertIn("default=10**9", SCRIPT.read_text())
        for row in self.cert["multipliers"]:
            for x in row:
                self.assertEqual(10**9 % Fraction(x).denominator, 0)

    def test_solver(self):
        self.assertIn("Clarabel", self.text)
        summary = json.loads((certificate_dir(4, 7) / f"{LABEL}.summary.json").read_text())
        self.assertEqual(summary["solver"], "CLARABEL")

    def test_blocks_are_sign_classes(self):
        """Each block of Q is one class of monomials under sign changes of coordinate pairs."""
        self.assertIn("with one block per group", self.text)
        for block in self.cert["blocks"]:
            self.assertEqual(len({parity(m) for m in block}), 1)

    def test_two_sign_changes_leave_signed_volume_unchanged(self):
        """'Changing the signs of two coordinates of every point ... leaves D unchanged.'"""
        self.assertIn("leaves $D$ unchanged", self.text)
        facets = [tuple(f) for f in self.cert["facets"]]
        terms = SignedVolume.from_facets(facets, D).polynomial.terms
        for a, b in combinations(range(D), 2):
            flipped = {m: c * (-1) ** sum(v % D in (a, b) for v in m) for m, c in terms.items()}
            self.assertEqual(flipped, terms)

    def test_beta_above_numerical_maximum(self):
        """The maximum found is attained by an explicit configuration on (S^3)^7."""
        self.assertIn("at least the value of about $9.34$ found numerically", self.text)
        points = regular_heptagon()
        self.assertTrue(np.allclose(np.linalg.norm(points, axis=1), 1))
        found = abs(signed_volume(points, certificate_facets()))
        self.assertAlmostEqual(found, factorial(4) * stored_ascent("lower_bound_volume"),
                               places=9)
        self.assertEqual(round(found, 2), 9.34)
        self.assertGreater(Fraction(self.cert["gamma"]), found)

    def test_beta_below_the_threshold(self):
        self.assertIn(r"$\beta < 4! \cdot \sqrt{3}/4 \approx 10.39$", self.text)
        self.assertEqual(round(factorial(4) * sqrt(3) / 4, 2), 10.39)
        self.assertLess(Fraction(self.cert["gamma"]) ** 2, Fraction(factorial(4) ** 2 * 3, 16))


class TestBeta(unittest.TestCase):
    """'The stored certificate for C_4(7) has beta = 48/5.'"""

    def test_printed_value_is_stored(self):
        match = re.search(r"has \$\\beta = (\d+)/(\d+)\$", method_text())
        self.assertIsNotNone(match)
        printed = Fraction(int(match.group(1)), int(match.group(2)))
        self.assertEqual(Fraction(certificate()["gamma"]), printed)


if __name__ == "__main__":
    unittest.main()
