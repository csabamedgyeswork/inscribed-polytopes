"""Tests for inscribed.certify.polynomial."""

from __future__ import annotations

import unittest
from fractions import Fraction

import numpy as np

from inscribed.certify import Polynomial, SignedVolume, parity_key
from inscribed.combinatorics import OrientedComplex
from inscribed.enumeration import FirschingSource


class TestParityKey(unittest.TestCase):
    def test_empty_is_zero(self):
        self.assertEqual(parity_key((), 4), 0)

    def test_single_variable(self):
        self.assertEqual(parity_key((3,), 4), min(1 << 3, (1 << 3) ^ 15))

    def test_complement_identified(self):
        self.assertEqual(parity_key((0, 1, 2, 3), 4), 0)
        self.assertEqual(parity_key((0, 1), 4), parity_key((2, 3), 4))

    def test_repeated_variable_cancels(self):
        self.assertEqual(parity_key((1, 1), 4), 0)


class TestPolynomial(unittest.TestCase):
    def test_add_and_scale(self):
        p = Polynomial({(0,): Fraction(1), (1,): Fraction(2)})
        q = Polynomial({(0,): Fraction(3), (2,): Fraction(-1)})
        r = p + q
        self.assertEqual(r.terms[(0,)], Fraction(4))
        self.assertEqual(r.terms[(1,)], Fraction(2))
        self.assertEqual(r.terms[(2,)], Fraction(-1))
        s = p.scale(Fraction(2))
        self.assertEqual(s.terms[(0,)], Fraction(2))

    def test_sub(self):
        p = Polynomial({(0,): Fraction(1)})
        q = Polynomial({(0,): Fraction(1)})
        self.assertTrue((p - q).is_zero())

    def test_zero_coefficients_dropped(self):
        p = Polynomial({(0,): Fraction(0), (1,): Fraction(1)})
        self.assertEqual(p.terms, {(1,): Fraction(1)})

    def test_substitute(self):
        p = Polynomial({(0,): Fraction(2), (1,): Fraction(1)})
        image = {(0,): (1, (1,)), (1,): (-1, (0,))}
        q = p.substitute(image)
        self.assertEqual(q.terms[(1,)], Fraction(2))
        self.assertEqual(q.terms[(0,)], Fraction(-1))

    def test_value(self):
        p = Polynomial({(0, 1): Fraction(3), (2,): Fraction(1)})
        self.assertAlmostEqual(p.value(np.array([2.0, 3.0, 4.0])), 3 * 2 * 3 + 4)


class TestSignedVolume(unittest.TestCase):
    def test_tetrahedron_coefficients(self):
        cx = OrientedComplex.from_labels([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
                                         one_indexed=True)
        sv = SignedVolume(cx)
        self.assertEqual(sv.d, 3)
        self.assertEqual(len(sv.polynomial.terms), 4 * 6)

    def test_from_facets_matches_constructor(self):
        cx = OrientedComplex.from_labels([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
                                         one_indexed=True)
        a = SignedVolume(cx).polynomial
        b = SignedVolume.from_facets(list(cx.facets), 3).polynomial
        self.assertEqual(a, b)

    def test_value_matches_determinant_sum(self):
        cx = OrientedComplex.from_labels([[1, 2, 3], [1, 2, 4], [1, 3, 4], [2, 3, 4]],
                                         one_indexed=True)
        sv = SignedVolume(cx)
        rng = np.random.default_rng(0)
        points = rng.normal(size=(4, 3))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        expected = sum(np.linalg.det(points[list(f)]) for f in cx.facets)
        self.assertAlmostEqual(sv.polynomial.value(points.reshape(-1)), expected)

    def test_value_matches_determinant_sum_for_the_cyclic_4_polytope(self):
        cx = next(r for r in FirschingSource().types(7) if r.label == "firsching-7-0").complex
        sv = SignedVolume(cx)
        rng = np.random.default_rng(1)
        for _ in range(5):
            points = rng.normal(size=(7, 4))
            expected = sum(np.linalg.det(points[list(f)]) for f in cx.facets)
            self.assertAlmostEqual(sv.polynomial.value(points.reshape(-1)), expected, places=10)


if __name__ == "__main__":
    unittest.main()
