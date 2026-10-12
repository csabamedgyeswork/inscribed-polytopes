"""Tests for inscribed.certify.certificate."""

from __future__ import annotations

import json
import unittest
from fractions import Fraction

from inscribed.certify import ExactCertificate


def sample() -> ExactCertificate:
    """A minimal certificate for the tetrahedron at bound 13/25."""
    return ExactCertificate(
        label="plantri-4-0", d=3, n=4,
        facets=[(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)],
        gamma=Fraction(78, 25),
        blocks=[[(), (0,)], [(1,)]],
        grams=[[[Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)]],
               [[Fraction(2)]]],
        multiplier_basis=[()],
        multipliers=[[Fraction(0)], [Fraction(0)], [Fraction(0)], [Fraction(0)]],
    )


class TestExactCertificate(unittest.TestCase):
    def test_volume_bound(self):
        self.assertEqual(sample().volume_bound, Fraction(13, 25))

    def test_json_roundtrip(self):
        cert = sample()
        self.assertEqual(ExactCertificate.from_json(cert.to_json()), cert)

    def test_json_serialisable(self):
        self.assertEqual(json.loads(json.dumps(sample().to_json())), sample().to_json())


if __name__ == "__main__":
    unittest.main()
