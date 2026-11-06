"""Tests for inscribed.certify.convexity: facet polynomials, positive definiteness, and that the
verifier rejects every kind of tampering with a convexity certificate."""

from __future__ import annotations

import unittest
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from inscribed.certify import ConvexityCertificate, ConvexityVerifier, facet_polynomial, facet_sets
from inscribed.certify.convexity import congruence_positive_definite, positive_definite
from inscribed.geometry import oriented_hull


def flat(p: np.ndarray) -> np.ndarray:
    return p.reshape(-1)


class TestFacetPolynomial(unittest.TestCase):
    def test_matches_the_determinant(self):
        p = np.random.default_rng(0).normal(size=(6, 4))
        poly = facet_polynomial((0, 1, 2, 3), 5, 4)
        self.assertAlmostEqual(poly.value(flat(p)), np.linalg.det(p[[0, 1, 2, 3]] - p[5]))

    def test_positive_on_inscribed_hulls_and_not_on_nonconvex(self):
        """Lemma 2.2 of the paper: on a realisation with D > 0 every A_{F,j} is positive."""
        rng = np.random.default_rng(1)
        for _ in range(50):
            p = rng.normal(size=(10, 4))
            p /= np.linalg.norm(p, axis=1, keepdims=True)
            cx = oriented_hull(p)
            values = [np.linalg.det(p[list(f)] - p[j]) for f in cx.facets
                      for j in range(10) if j not in f]
            self.assertGreater(min(values), 0)
        q = p.copy()
        q[cx.facets[0][0]] *= 0.05
        self.assertLess(min(np.linalg.det(q[list(f)] - q[j]) for f in cx.facets
                            for j in range(10) if j not in f), 0)


class TestPositiveDefinite(unittest.TestCase):
    def fractions(self, m):
        return [[Fraction(x).limit_denominator(10**6) for x in row] for row in m]

    def test_accepts_positive_definite(self):
        a = np.random.default_rng(2).normal(size=(12, 12))
        gram = self.fractions(a @ a.T + 0.1 * np.eye(12))
        gram = [[gram[min(i, j)][max(i, j)] for j in range(12)] for i in range(12)]
        self.assertTrue(congruence_positive_definite(gram))
        self.assertTrue(positive_definite(gram))

    def test_rejects_indefinite_singular_and_asymmetric(self):
        self.assertFalse(positive_definite(self.fractions([[1, 2], [2, 1]])))
        self.assertFalse(positive_definite(self.fractions([[1, 1], [1, 1]])))
        self.assertFalse(positive_definite(self.fractions([[2, 1], [0, 2]])))


@pytest.mark.slow
class TestVerifierRejectsTampering(unittest.TestCase):
    """One real convexity certificate, then each hypothesis of Proposition 2.4 broken."""

    @classmethod
    def setUpClass(cls):
        from inscribed.enumeration import FirschingSource
        from inscribed.sos import ConvexitySearch

        record = next(FirschingSource().types(8))
        cls.cx = record.complex
        search = ConvexitySearch(cls.cx, Fraction(1), record.label)
        cls.cc = search.round_exact(search.solve("convex"), "convex")
        cls.verifier = ConvexityVerifier(resolvers=(lambda c: facet_sets(cls.cx.facets),),
                                         threshold=lambda b: b <= 1)

    def check(self, cc):
        return self.verifier.check(cc)

    def test_genuine_passes(self):
        self.assertTrue(self.cc.multipliers)
        self.assertTrue(self.check(self.cc).ok)

    def test_round_trip(self):
        again = ConvexityCertificate.from_json(self.cc.to_json())
        self.assertEqual(again.digest(), self.cc.digest())

    def test_negative_lambda(self):
        (f, j, w), *rest = self.cc.multipliers
        report = self.check(replace(self.cc, multipliers=((f, j, -w), *rest)))
        self.assertFalse(report.multipliers_valid)

    def test_reversed_facet(self):
        (f, j, w), *rest = self.cc.multipliers
        flipped = f[:-2] + (f[-1], f[-2])
        self.assertFalse(self.check(replace(self.cc, multipliers=((flipped, j, w), *rest))).ok)

    def test_vertex_in_facet(self):
        (f, j, w), *rest = self.cc.multipliers
        self.assertFalse(self.check(replace(self.cc, multipliers=((f, f[0], w), *rest))).ok)

    def test_changed_coefficient(self):
        (f, j, w), *rest = self.cc.multipliers
        changed = (f, j, w + Fraction(1, 10**9))
        self.assertFalse(self.check(replace(self.cc, multipliers=(changed, *rest))).identity)

    def test_wrong_type(self):
        verifier = ConvexityVerifier(resolvers=(lambda c: {frozenset({0, 1, 2, 3})},),
                                     threshold=self.verifier.threshold)
        self.assertFalse(verifier.check(self.cc).type_matches)

    def test_threshold(self):
        verifier = ConvexityVerifier(resolvers=self.verifier.resolvers,
                                     threshold=lambda b: b < Fraction(1, 2))
        self.assertFalse(verifier.check(self.cc).below_threshold)
