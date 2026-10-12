"""Tests for inscribed.sos.store."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

from inscribed.certify import CertificateVerifier
from inscribed.enumeration import PlantriSource
from inscribed.sos import (
    CertificateRounder,
    CertificateSearch,
    CertificateStore,
    RelaxationResult,
    SolverSettings,
    enumerated_types,
)

VERIFIER = CertificateVerifier(resolvers=(enumerated_types,))


def tetrahedron():
    """The unique simplicial type on 4 vertices."""
    return next(r for r in PlantriSource().types(4))


class TestCertificateStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def test_save_and_load_certificate(self):
        record = tetrahedron()
        search = CertificateSearch.for_volume_bound(record.complex, Fraction("0.52"))
        solution = search.solve(SolverSettings())
        cert = CertificateRounder(10**9).round(search, solution, record.label).certificate
        report = VERIFIER.verify(cert)

        with patch("inscribed.sos.store.certificate_dir", return_value=self.tmp):
            store = CertificateStore(3, 4)
            cert_path, summary_path = store.save_certificate(
                cert, report, margin=solution.margin, largest_repair="0", solver="CLARABEL")
            loaded = store.load_certificate(record.label)

        self.assertEqual(loaded, cert)
        summary = json.loads(summary_path.read_text())
        self.assertTrue(summary["verification"]["ok"])
        self.assertEqual(summary["volume_bound"], "13/25")
        self.assertEqual(summary["sha256_uncompressed"],
                         hashlib.sha256(gzip.decompress(cert_path.read_bytes())).hexdigest())
        self.assertTrue(cert_path.name.endswith(".certificate.json.gz"))

    def test_unverified_certificate_refused(self):
        record = tetrahedron()
        search = CertificateSearch.for_volume_bound(record.complex, Fraction("0.52"))
        solution = search.solve(SolverSettings())
        cert = CertificateRounder(10**9).round(search, solution, record.label).certificate
        tampered = replace(cert, gamma=cert.gamma + 1)
        report = VERIFIER.verify(tampered)
        with patch("inscribed.sos.store.certificate_dir", return_value=self.tmp):
            with self.assertRaises(ValueError):
                CertificateStore(3, 4).save_certificate(
                    tampered, report, margin=solution.margin, largest_repair="0", solver="CLARABEL")
        self.assertEqual(list(self.tmp.iterdir()), [])

    def test_save_relaxation(self):
        result = RelaxationResult(status="optimal", value=3.08, seconds=0.1, variables=16,
                                  equalities=5, block_sizes=(31, 20))
        with patch("inscribed.sos.store.certificate_dir", return_value=self.tmp):
            store = CertificateStore(3, 4)
            path = store.save_relaxation("plantri-4-0", result, level=2, solver="CLARABEL",
                                         group_order=576, lower_bound=0.5132,
                                         convex_volume=0.5132)
        data = json.loads(path.read_text())
        self.assertEqual(path.name, "plantri-4-0.relaxation.json")
        self.assertEqual(data["label"], "plantri-4-0")
        self.assertAlmostEqual(data["relaxation_value_volume"], 3.08 / 6)
        self.assertEqual(data["relaxation"]["value"], 3.08)
        self.assertNotIn("upper_bound_volume", data)
        self.assertEqual(data["group_order"], 576)


if __name__ == "__main__":
    unittest.main()
