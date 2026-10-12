"""Tests for inscribed.sos.rounding."""

from __future__ import annotations

import unittest
from fractions import Fraction

from inscribed.certify import CertificateVerifier
from inscribed.enumeration import PlantriSource
from inscribed.sos import CertificateRounder, CertificateSearch, SolverSettings


class TestCertificateRounder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        record = next(r for r in PlantriSource().types(4))
        cls.search = CertificateSearch.for_volume_bound(record.complex, Fraction("0.52"))
        cls.solution = cls.search.solve(SolverSettings())
        cls.result = CertificateRounder(10**9).round(cls.search, cls.solution, record.label)

    def test_repaired_certificate_satisfies_the_identity(self):
        self.assertTrue(CertificateVerifier().residual(self.result.certificate).is_zero())

    def test_repairs_are_below_the_margin(self):
        self.assertLess(self.result.largest_repair, Fraction(self.solution.margin))

    def test_rounding_twice_gives_equal_independent_certificates(self):
        again = CertificateRounder(10**9).round(self.search, self.solution, "plantri-4-0")
        self.assertEqual(again.certificate, self.result.certificate)
        self.assertIsNot(again.certificate.grams[0], self.result.certificate.grams[0])


if __name__ == "__main__":
    unittest.main()
