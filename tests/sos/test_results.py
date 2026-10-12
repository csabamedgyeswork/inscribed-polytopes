"""Regression tests for the committed certificates of the d = 4, n = 7 case."""

from __future__ import annotations

import gzip
import hashlib
import json
import unittest
from fractions import Fraction
from itertools import permutations
from math import factorial

import numpy as np
from scipy.spatial import ConvexHull

from inscribed.certify import (
    CertificateVerifier,
    ExactCertificate,
    cyclic_facets,
    cyclic_types,
    facet_sets,
)
from inscribed.enumeration import FirschingSource
from inscribed.sos import enumerated_types
from inscribed.store import certificate_dir

VERIFIER = CertificateVerifier(resolvers=(cyclic_types, enumerated_types))
DIRECTORY = certificate_dir(4, 7)
BOUNDS = {
    "cyclic-4-7": Fraction(2, 5),
    "firsching-7-1": Fraction(41, 100),
    "firsching-7-3": Fraction(37, 100),
    "firsching-7-4": Fraction(9, 25),
}


def load(label: str) -> tuple[ExactCertificate, dict, bytes]:
    """The stored certificate, its summary and its uncompressed payload."""
    payload = gzip.decompress((DIRECTORY / f"{label}.certificate.json.gz").read_bytes())
    summary = json.loads((DIRECTORY / f"{label}.summary.json").read_text())
    return ExactCertificate.from_json(json.loads(payload)), summary, payload


def isomorphic(a: set[frozenset[int]], b: set[frozenset[int]], n: int) -> bool:
    """Whether some relabelling of 0..n-1 carries facet set a onto b."""
    return len(a) == len(b) and any(
        {frozenset(p[v] for v in f) for f in a} == b for p in permutations(range(n)))


class TestCommittedCertificates(unittest.TestCase):
    def test_each_certificate_verifies_at_its_recorded_bound(self):
        for label, bound in BOUNDS.items():
            with self.subTest(label=label):
                cert, summary, _ = load(label)
                report = VERIFIER.verify(cert)
                self.assertTrue(report.ok)
                self.assertEqual(report.volume_bound, bound)
                self.assertEqual(Fraction(summary["volume_bound"]), bound)

    def test_summaries_hash_their_certificates(self):
        for label in BOUNDS:
            with self.subTest(label=label):
                _, summary, payload = load(label)
                self.assertEqual(summary["sha256_uncompressed"],
                                 hashlib.sha256(payload).hexdigest())

    def test_identity_holds_with_determinants_computed_independently(self):
        """gamma - sum det(p_F) - sum h_i (|p_i|^2 - 1) - sum v^T Q v vanishes on R^28."""
        cert, _, _ = load("cyclic-4-7")
        grams = [np.array([[float(x) for x in row] for row in g]) for g in cert.grams]
        multipliers = np.array([[float(x) for x in row] for row in cert.multipliers])
        rng = np.random.default_rng(0)
        for _ in range(5):
            x = rng.normal(size=cert.d * cert.n) * 0.7
            points = x.reshape(cert.n, cert.d)

            def monomial(m: tuple[int, ...]) -> float:
                return float(np.prod(x[list(m)])) if m else 1.0

            det_sum = sum(np.linalg.det(points[list(f)]) for f in cert.facets)
            squares = sum(v @ g @ v for g, block in zip(grams, cert.blocks)
                          for v in [np.array([monomial(m) for m in block])])
            constraint = sum(
                sum(h * monomial(m) for h, m in zip(multipliers[i], cert.multiplier_basis))
                * (points[i] @ points[i] - 1) for i in range(cert.n))
            residual = float(cert.gamma) - det_sum - constraint - squares
            self.assertLess(abs(residual), 1e-9 * max(1.0, abs(squares)))


class TestCyclicCase(unittest.TestCase):
    def test_the_certified_type_is_the_cyclic_polytope(self):
        """The certificate's facets are exactly Gale's list, in the same vertex order."""
        cert, _, _ = load("cyclic-4-7")
        self.assertEqual(facet_sets(cert.facets), cyclic_facets(7, 4))

    def test_no_other_type_is_cyclic(self):
        cyclic = cyclic_facets(7, 4)
        others = [r for r in FirschingSource().types(7) if r.label != "firsching-7-0"]
        self.assertEqual(len(others), 4)
        for record in others:
            with self.subTest(label=record.label):
                self.assertFalse(isomorphic(facet_sets(record.complex.facets), cyclic, 7))

    def test_cyclic_bound_lies_below_the_noncyclic_maximum(self):
        """2/5 < sqrt(3)/4, i.e. (2/5)^2 < 3/16."""
        self.assertLess(BOUNDS["cyclic-4-7"] ** 2, Fraction(3, 16))

    def test_signed_volume_is_d_factorial_times_volume_at_the_realisation(self):
        record = next(r for r in FirschingSource().types(7) if r.label == "firsching-7-0")
        points = np.asarray(record.points, dtype=float)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        det_sum = sum(np.linalg.det(points[list(f)]) for f in record.complex.facets)
        self.assertAlmostEqual(abs(det_sum), factorial(4) * ConvexHull(points).volume, places=10)


class TestRelaxationRecord(unittest.TestCase):
    def test_relaxation_value_is_in_volume_units(self):
        data = json.loads((DIRECTORY / "cyclic-4-7.relaxation.json").read_text())
        self.assertAlmostEqual(data["relaxation_value_volume"],
                               data["relaxation"]["value"] / factorial(4))

    def test_relaxation_value_matches_the_ascent_optimum(self):
        """The level-2 value agrees with the found maximum; it is evidence, not a bound."""
        data = json.loads((DIRECTORY / "cyclic-4-7.relaxation.json").read_text())
        self.assertAlmostEqual(data["relaxation_value_volume"], data["lower_bound_volume"],
                               delta=1e-5)


if __name__ == "__main__":
    unittest.main()
