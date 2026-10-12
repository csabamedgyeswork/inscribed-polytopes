"""Tests for inscribed.certify.verify."""

from __future__ import annotations

import unittest
from dataclasses import replace
from fractions import Fraction

from inscribed.certify import (
    CertificateVerifier,
    ExactCertificate,
    bareiss_positive_definite,
)
from inscribed.enumeration import PlantriSource
from inscribed.sos import (
    CertificateRounder,
    CertificateSearch,
    SolverSettings,
    enumerated_types,
)


def tetrahedron():
    """The unique simplicial type on 4 vertices."""
    return next(r for r in PlantriSource().types(4))


def certify(bound: str) -> ExactCertificate:
    """A verified certificate for the tetrahedron at a decimal bound."""
    record = tetrahedron()
    search = CertificateSearch.for_volume_bound(record.complex, Fraction(bound))
    solution = search.solve(SolverSettings())
    return CertificateRounder(10**9).round(search, solution, record.label).certificate


class TestBareiss(unittest.TestCase):
    def test_positive_definite(self):
        self.assertTrue(bareiss_positive_definite(
            [[Fraction(2), Fraction(1)], [Fraction(1), Fraction(2)]]))

    def test_indefinite(self):
        self.assertFalse(bareiss_positive_definite(
            [[Fraction(1), Fraction(2)], [Fraction(2), Fraction(1)]]))

    def test_semidefinite(self):
        self.assertFalse(bareiss_positive_definite(
            [[Fraction(1), Fraction(1)], [Fraction(1), Fraction(1)]]))

    def test_not_symmetric(self):
        self.assertFalse(bareiss_positive_definite(
            [[Fraction(1), Fraction(2)], [Fraction(1), Fraction(2)]]))

    def test_zero_matrix(self):
        self.assertFalse(bareiss_positive_definite([[Fraction(0)]]))


class TestVerifier(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cert = certify("0.52")
        cls.verifier = CertificateVerifier(resolvers=(enumerated_types,))

    def test_residual_zero(self):
        self.assertTrue(self.verifier.residual(self.cert).is_zero())

    def test_type_matches(self):
        self.assertTrue(self.verifier.type_matches(self.cert))

    def test_verify_ok(self):
        report = self.verifier.verify(self.cert)
        self.assertTrue(report.ok)
        self.assertEqual(report.volume_bound, Fraction(13, 25))

    def test_coherent(self):
        self.assertTrue(self.verifier.coherent(self.cert))

    def test_wrong_gamma_breaks_identity(self):
        tampered = replace(self.cert, gamma=self.cert.gamma + 1)
        self.assertFalse(self.verifier.residual(tampered).is_zero())

    def test_reordered_facets_accepted(self):
        facets = list(self.cert.facets)
        facets[0], facets[1] = facets[1], facets[0]
        self.assertTrue(self.verifier.verify(replace(self.cert, facets=facets)).ok)

    def test_flipped_facet_rejected_as_incoherent(self):
        facets = list(self.cert.facets)
        facets[0] = facets[0][:-2] + (facets[0][-1], facets[0][-2])
        report = self.verifier.verify(replace(self.cert, facets=facets))
        self.assertFalse(report.coherent)
        self.assertFalse(report.ok)

    def test_reversed_orientation_is_coherent(self):
        facets = [f[:-2] + (f[-1], f[-2]) for f in self.cert.facets]
        self.assertTrue(self.verifier.coherent(replace(self.cert, facets=facets)))

    def test_wrong_label_rejected(self):
        self.assertFalse(self.verifier.type_matches(replace(self.cert, label="plantri-4-99")))

    def test_indefinite_block_rejected(self):
        grams = [[row[:] for row in g] for g in self.cert.grams]
        grams[0][0][0] = -grams[0][0][0]
        report = self.verifier.verify(replace(self.cert, grams=grams))
        self.assertFalse(report.ok)

    def test_changed_gram_entry_breaks_identity(self):
        grams = [[row[:] for row in g] for g in self.cert.grams]
        grams[0][0][0] += Fraction(1, 10**9)
        tampered = replace(self.cert, grams=grams)
        self.assertFalse(self.verifier.residual(tampered).is_zero())


if __name__ == "__main__":
    unittest.main()
