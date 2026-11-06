"""Tests for the exact type catalogue and the convexity campaign, on the 37 types of d = 4,
n = 8, where the winner is the cross-polytope, the join of two 4-cycles."""

from __future__ import annotations

import gzip
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import pytest

from inscribed.elimination import (
    ConvexCampaign,
    Ledger,
    LedgerEntry,
    TypeCatalogue,
    below_two_pentagons,
    canonical_form,
    is_cycle_join,
)
from inscribed.elimination.catalogue import ExactHull, rational_points
from inscribed.enumeration import FirschingSource
from inscribed.geometry import oriented_hull

LINES = FirschingSource()._lines(8)
BOUND = Fraction(13, 20)
CROSS = 23


def catalogue_of(directory: Path, lines: list[str]) -> TypeCatalogue:
    catalogue = TypeCatalogue(directory / "types", "firsching-8", 4, 8)
    catalogue.build(lines, workers=1, log=lambda _: None)
    return catalogue


class TestThreshold(unittest.TestCase):
    def test_two_pentagons(self):
        self.assertTrue(below_two_pentagons(Fraction(471, 500)))
        self.assertTrue(below_two_pentagons(Fraction(9421, 10000)))
        self.assertFalse(below_two_pentagons(Fraction(9422, 10000)))


class TestCatalogue(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.catalogue = catalogue_of(Path(cls.tmp.name), LINES)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_exact_facets_are_the_qhull_facets(self):
        for k, line in enumerate(LINES):
            points = rational_points(line)
            hull = oriented_hull([[float(x) for x in p] for p in points])
            self.assertEqual(self.catalogue.facets(self.catalogue.label(k)), tuple(hull.facets))

    def test_exact_orientation(self):
        for k, line in enumerate(LINES):
            hull = ExactHull(rational_points(line))
            for f in self.catalogue.facets(self.catalogue.label(k)):
                self.assertTrue(all(hull.side(f, j) > 0 for j in range(8) if j not in f))

    def test_rebuild_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as other:
            self.assertEqual(catalogue_of(Path(other), LINES).sha256(), self.catalogue.sha256())

    def test_audit(self):
        audit = self.catalogue.audit((4, 4), workers=1)
        self.assertEqual(audit["types"], 37)
        self.assertTrue(audit["spheres"] and audit["pairwise_distinct"])
        self.assertEqual(audit["winner"], self.catalogue.label(CROSS))
        self.assertEqual(audit["catalogue_sha256"], self.catalogue.sha256())

    def test_canonical_form_ignores_relabelling(self):
        facets = self.catalogue.facets(self.catalogue.label(5))
        perm = [3, 7, 1, 0, 6, 2, 5, 4]
        relabelled = [tuple(perm[v] for v in f) for f in facets]
        self.assertEqual(canonical_form(facets, 8), canonical_form(relabelled, 8))

    def test_audit_detects_a_duplicate(self):
        with tempfile.TemporaryDirectory() as other:
            twice = catalogue_of(Path(other), LINES[:3] + LINES[:1])
            self.assertFalse(twice.audit((4, 4), workers=1)["pairwise_distinct"])

    def test_cycle_join(self):
        winners = [k for k in range(37)
                   if is_cycle_join(self.catalogue.facets(self.catalogue.label(k)), 8, 4, 4)]
        self.assertEqual(winners, [CROSS])


class TestLedger(unittest.TestCase):
    def test_later_line_wins_and_complete_shards_are_sealed(self):
        with tempfile.TemporaryDirectory() as tmp:
            catalogue = catalogue_of(Path(tmp), LINES[:2])
            ledger = Ledger(Path(tmp) / "ledger", catalogue)
            a, b = catalogue.labels()

            def entry(label, status):
                return LedgerEntry(label, str(BOUND), status, "bare", 0.1, "00",
                                   catalogue.fingerprint(label), 1.0)

            ledger.append(entry(a, "failed"))
            ledger.append(entry(a, "proved"))
            self.assertEqual(ledger.entries()[a].status, "proved")
            self.assertEqual(ledger.seal(), [])
            ledger.append(entry(b, "proved"))
            self.assertEqual(len(ledger.seal()), 1)
            (sealed,) = ledger.directory.glob("*.jsonl.gz")
            self.assertEqual(gzip.decompress(sealed.read_bytes()).decode().count("\n"), 2)
            self.assertEqual(set(ledger.entries()), {a, b})
            ledger.append(entry(b, "proved"))
            self.assertFalse(sealed.exists())
            self.assertEqual(len(ledger.entries()), 2)


@pytest.mark.slow
class TestConvexCampaign(unittest.TestCase):
    """Run, resume and verify on four rivals and the cross-polytope."""

    def test_run_resume_verify_import(self):
        picks = [0, 1, 2, CROSS]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = catalogue_of(root, [LINES[k] for k in picks])
            campaign = ConvexCampaign(catalogue, Ledger(root / "ledger", catalogue), BOUND,
                                      winner=(4, 4), workers=2, log=lambda _: None)
            labels = catalogue.labels()
            counts = campaign.run(labels[:3])
            self.assertEqual(counts, {"proved": 3})
            self.assertEqual(campaign.pending(labels), labels[3:])
            self.assertEqual(campaign.run(labels), {"winner": 1})
            self.assertEqual(campaign.run(labels), {})

            out = root / "certs"
            result = campaign.verify(labels, write=out)
            self.assertEqual(sorted(result["match"]), sorted(labels[:3]))
            self.assertEqual(result["winner"], labels[3:])
            self.assertTrue(campaign.complete(catalogue.audit((4, 4), workers=1)))

            other = ConvexCampaign(catalogue, Ledger(root / "imported", catalogue), BOUND,
                                   winner=(4, 4), log=lambda _: None)
            counts = other.import_certificates(sorted(out.glob("*.cert.json")))
            self.assertEqual(counts, {"imported": 3, "rejected": 0, "skipped": 0})
            self.assertEqual({k: e.sha256 for k, e in other.ledger.entries().items()},
                             {k: e.sha256 for k, e in campaign.ledger.entries().items()
                              if e.status == "proved"})

    def test_a_changed_bound_is_not_done(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            catalogue = catalogue_of(root, LINES[:1])
            ledger = Ledger(root / "ledger", catalogue)
            (label,) = catalogue.labels()
            ledger.append(LedgerEntry(label, "2/3", "proved", "bare", 0.1, "00",
                                      catalogue.fingerprint(label), 1.0))
            campaign = ConvexCampaign(catalogue, ledger, BOUND, winner=(4, 4))
            self.assertEqual(campaign.pending([label]), [label])
