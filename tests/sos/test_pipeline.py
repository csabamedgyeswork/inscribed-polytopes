"""End-to-end tests for the SOS pipeline on small types."""

from __future__ import annotations

import unittest
from fractions import Fraction

from inscribed.certify import CertificateVerifier
from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.sos import (
    CertificateRounder,
    CertificateSearch,
    SolverSettings,
    enumerated_types,
)

VERIFIER = CertificateVerifier(resolvers=(enumerated_types,))


def tetrahedron():
    """The unique simplicial type on 4 vertices."""
    return next(r for r in PlantriSource().types(4))


def simplex4():
    """The unique simplicial type on 5 vertices in d = 4."""
    return next(r for r in FirschingSource().types(5))


class TestPipeline(unittest.TestCase):
    def test_tetrahedron_certified(self):
        record = tetrahedron()
        search = CertificateSearch.for_volume_bound(record.complex, Fraction("0.52"))
        solution = search.solve(SolverSettings())
        self.assertGreater(solution.margin, 0)
        cert = CertificateRounder(10**9).round(search, solution, record.label).certificate
        report = VERIFIER.verify(cert)
        self.assertTrue(report.ok)
        self.assertEqual(report.volume_bound, Fraction(13, 25))

    def test_tetrahedron_refused_below_maximum(self):
        record = tetrahedron()
        search = CertificateSearch.for_volume_bound(record.complex, Fraction("0.51"))
        solution = search.solve(SolverSettings())
        self.assertLessEqual(solution.margin, 0)

    def test_simplex4_certified(self):
        record = simplex4()
        search = CertificateSearch.for_volume_bound(record.complex, Fraction("0.15"))
        solution = search.solve(SolverSettings())
        self.assertGreater(solution.margin, 0)
        cert = CertificateRounder(10**9).round(search, solution, record.label).certificate
        report = VERIFIER.verify(cert)
        self.assertTrue(report.ok)
        self.assertEqual(report.volume_bound, Fraction(3, 20))


if __name__ == "__main__":
    unittest.main()
