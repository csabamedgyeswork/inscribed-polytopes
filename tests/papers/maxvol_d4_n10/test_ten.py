"""The claims about Firsching's ten-vertex polytopes in papers/maxvol-d4-n10: the lemma on
his data and the proposition on the convexity certificates for every type other than
firsching-10-11.

The default run checks what the committed results state: the audit, the winner, the bound,
that the ledger has a proved entry for every rival, and the stored convexity certificate of
the cyclic polytope. The checks over all 162004 types are
marked slow (poetry run pytest -m slow, several minutes in INSCRIBED_WORKERS processes,
default 5): the exact facets are recomputed from Firsching's rational vertices, every type is
checked to be a coherently oriented sphere, and the canonical forms are recomputed. Rebuilding
certificates needs a semidefinite solve per type: the slow run rebuilds INSCRIBED_SAMPLE of
them (default 20), and all of them only if INSCRIBED_FULL_VERIFY is set, which takes hours.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
import re
import unittest
from fractions import Fraction
from itertools import combinations
from math import factorial
from multiprocessing import Pool
from pathlib import Path

import pytest

from inscribed.certify import ConvexityCertificate, ConvexityVerifier
from inscribed.elimination import ConvexCampaign, Ledger, TypeCatalogue
from inscribed.elimination.catalogue import (ExactHull, canonical_form, is_cycle_join,
                                             rational_points)
from inscribed.elimination.convex import below_two_pentagons
from inscribed.enumeration import FirschingSource
from inscribed.store import certificate_dir
from tests.papers.maxvol_d4_n9.test_nine import join_facets

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "maxvol-d4-n10"
RESULTS = certificate_dir(4, 10)
TYPES = 162004
WINNER = "firsching-10-11"
CYCLIC = "firsching-10-157156"
CYCLES = ([0, 2, 7, 5, 6], [1, 4, 3, 8, 9])
BOUND = Fraction(471, 500)
DATA_SHA256 = "9a770e1534d987931cef533e3728d79e41d235ab42ad0e8d5e01254607c7442a"
WORKERS = int(os.environ.get("INSCRIBED_WORKERS", "5"))
SAMPLE = int(os.environ.get("INSCRIBED_SAMPLE", "20"))


def text() -> str:
    """All sections of the paper with line breaks collapsed to single spaces."""
    sections = sorted((PAPER / "sections").glob("*.tex"))
    return re.sub(r"\s+", " ", " ".join(f.read_text() for f in sections))


def catalogue() -> TypeCatalogue:
    return TypeCatalogue(RESULTS / "types", "firsching-10", 4, 10)


def campaign() -> ConvexCampaign:
    c = catalogue()
    return ConvexCampaign(c, Ledger(RESULTS / "ledger", c), BOUND, workers=WORKERS,
                          log=lambda _: None)


def rivals() -> list[str]:
    return [f"firsching-10-{k}" for k in range(TYPES) if f"firsching-10-{k}" != WINNER]


def _exact_check(job: tuple[str, tuple[tuple[int, ...], ...]]) -> bool:
    """The stored facets are exactly the 4-sets with every other vertex strictly on one
    side, tested over all 210 of them, and each is stored with every A_{F,j} > 0."""
    line, stored = job
    hull = ExactHull(rational_points(line))
    found = {frozenset(f) for f in map(hull.oriented, combinations(range(10), 4))
             if f is not None}
    positive = all(hull.side(f, j) == 1 for f in stored for j in range(10) if j not in f)
    return found == {frozenset(f) for f in stored} and positive


def _form(facets: tuple[tuple[int, ...], ...]) -> tuple[int, ...]:
    return canonical_form(facets, 10)


class TestPrinted(unittest.TestCase):
    def test_lemma_and_proposition(self):
        paper = text()
        self.assertIn("$162004$ combinatorial types", paper)
        self.assertIn(r"\texttt{firsching-10-0} to \texttt{firsching-10-162003}", paper)
        self.assertIn(r"the type of \texttt{firsching-10-11} is that of the polytope", paper)
        self.assertIn(r"$(a_{0}, \ldots, a_{4}) = (0, 2, 7, 5, 6)$", paper)
        self.assertIn(r"$(b_{0}, \ldots, b_{4}) = (1, 4, 3, 8, 9)$", paper)
        self.assertIn("For each of these $162003$ values of $k$", paper)
        self.assertIn(r"$\beta = 2826/125$", paper)
        self.assertIn(r"\beta/4! = 471/500", paper)
        self.assertIn(r"$471/500 < V^{*}$ by \eqref{eq:vstar}", paper)
        self.assertNotIn("Hypothes", paper)
        self.assertEqual(factorial(4) * BOUND, Fraction(2826, 125))
        root5 = Fraction(2236, 1000)
        self.assertLess(root5 ** 2, 5)
        self.assertGreater(Fraction(25, 384) * (10 + 2 * root5), Fraction(9421, 10000))
        self.assertGreater(Fraction(9421, 10000), BOUND)


class TestCatalogue(unittest.TestCase):
    """The committed catalogue, audit and manifest agree, and the audit establishes the
    combinatorial part of the lemma on Firsching's data."""

    @classmethod
    def setUpClass(cls):
        cls.catalogue = catalogue()
        cls.audit = json.loads((RESULTS / "audit.json").read_text())
        cls.manifest = json.loads((RESULTS / "manifest.json").read_text())

    def test_one_line_per_type(self):
        self.assertEqual(len(self.catalogue), TYPES)

    def test_hashes_agree(self):
        sha = self.catalogue.sha256()
        self.assertEqual(self.audit["catalogue_sha256"], sha)
        self.assertEqual(self.manifest["catalogue_sha256"], sha)
        self.assertEqual(self.manifest["data_sha256"], DATA_SHA256)

    def test_audit(self):
        self.assertEqual(self.audit["types"], TYPES)
        self.assertTrue(self.audit["spheres"])
        self.assertTrue(self.audit["pairwise_distinct"])
        self.assertEqual(self.audit["distinct_canonical_forms"], TYPES)
        self.assertEqual(self.audit["winner"], WINNER)
        self.assertEqual(self.audit["winner_candidates"], [WINNER])

    def test_winner_is_the_join_of_the_printed_cycles(self):
        facets = {frozenset(f) for f in self.catalogue.facets(WINNER)}
        self.assertEqual(facets, join_facets(*CYCLES))

    def test_canonical_form_is_invariant(self):
        """The direction the proof needs: relabelled types keep their canonical form, so
        distinct forms mean distinct types."""
        rng = random.Random(0)
        for label in rng.sample(self.catalogue.labels(), 200):
            facets = self.catalogue.facets(label)
            relabel = list(range(10))
            rng.shuffle(relabel)
            moved = [tuple(relabel[v] for v in f) for f in facets]
            with self.subTest(label=label):
                self.assertEqual(canonical_form(moved, 10), canonical_form(facets, 10))


class TestCyclic(unittest.TestCase):
    """The remark on C4(10): the type is Gale's cyclic polytope, and at the two pentagons,
    labelled by the printed cycles, its signed volume is exactly 24 V*, above beta."""

    PENTAGONS = ((0, 6, 2, 4, 7), (1, 9, 3, 5, 8))

    def test_is_gales_cyclic_polytope(self):
        def even(s):
            out = [i for i in range(10) if i not in s]
            return all(sum(i < k < j for k in s) % 2 == 0
                       for i in out for j in out if i < j)
        gale = [s for s in combinations(range(10), 4) if even(s)]
        self.assertEqual(len(gale), 35)
        self.assertEqual(canonical_form(catalogue().facets(CYCLIC), 10),
                         canonical_form(gale, 10))

    def test_pentagons_exceed_beta(self):
        sp = pytest.importorskip("sympy")
        point = {}
        for k, (a, b) in enumerate(zip(*self.PENTAGONS)):
            c, s = sp.cos(2 * sp.pi * k / 5), sp.sin(2 * sp.pi * k / 5)
            point[a], point[b] = (c, s, 0, 0), (0, 0, c, s)
        D = sum(sp.Matrix([point[j] for j in f]).T.det() for f in catalogue().facets(CYCLIC))
        pair = sp.Rational(25, 16) * (10 + 2 * sp.sqrt(5))
        self.assertEqual(sp.simplify(sp.Abs(D) - pair), 0)
        self.assertGreater(pair, factorial(4) * BOUND)


class TestBound(unittest.TestCase):
    """'beta_k < 25/16 (10 + 2 sqrt 5)', decided exactly: with t = 16 beta / 25 - 10,
    if and only if t < 0 or t^2 < 20."""

    def test_below_the_pair(self):
        beta = factorial(4) * BOUND
        t = Fraction(16, 25) * beta - 10
        self.assertTrue(t < 0 or t * t < 20)
        self.assertTrue(below_two_pentagons(BOUND))
        self.assertFalse(below_two_pentagons(Fraction(943, 1000)))

    def test_manifest_bound(self):
        manifest = json.loads((RESULTS / "manifest.json").read_text())
        self.assertEqual(Fraction(manifest["bound"]), BOUND)


class TestLedger(unittest.TestCase):
    """Every rival has a proved ledger entry at the bound for its committed facets, and the
    winner has none. Fails until the campaign has covered all 162003 rivals."""

    def test_every_rival_is_proved(self):
        c = campaign()
        entries = c.ledger.entries()
        missing = [label for label in rivals() if not (
            c.done(entries.get(label), label) and entries[label].status == "proved")]
        self.assertEqual(len(missing), 0,
                         f"{len(missing)} of {TYPES - 1} rivals not proved, e.g. {missing[:5]}")

    def test_winner_is_not_certified(self):
        entry = campaign().ledger.entries().get(WINNER)
        self.assertTrue(entry is None or entry.status == "winner")


class TestCyclicCertificate(unittest.TestCase):
    """The one convexity certificate, for the cyclic polytope C_4(10), is stored and passes
    every exact check without a semidefinite solve."""

    def test_stored_certificate_verifies(self):
        cat = catalogue()
        raw = gzip.decompress((RESULTS / f"{CYCLIC}.convexity.json.gz").read_bytes())
        cc = ConvexityCertificate.from_json(json.loads(raw))
        self.assertEqual(cc.label, CYCLIC)
        self.assertEqual(cc.bound, BOUND)
        self.assertEqual(tuple(cc.certificate.facets), cat.facets(CYCLIC))
        self.assertTrue(cc.multipliers)
        self.assertEqual(cc.digest(), Ledger(RESULTS / "ledger", cat).entries()[CYCLIC].sha256)
        verifier = ConvexityVerifier(resolvers=(cat.resolver,), threshold=below_two_pentagons)
        report = verifier.check(cc)
        self.assertTrue(report.ok, report.failures())

    def test_only_the_cyclic_type_needs_multipliers(self):
        entries = Ledger(RESULTS / "ledger", catalogue()).entries()
        self.assertEqual([label for label, e in entries.items() if e.route == "convex"],
                         [CYCLIC])


@pytest.mark.slow
class TestFirschingData(unittest.TestCase):
    """The lemma on Firsching's data, from his rational vertices, over all 162004 types."""

    @classmethod
    def setUpClass(cls):
        path = FirschingSource().path(10)
        if not path.exists():
            raise unittest.SkipTest(f"{path} missing; run scripts/setup_data.py")
        if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
            raise AssertionError(f"{path}: SHA-256 differs from Firsching's published list")
        cls.lines = FirschingSource()._lines(10)
        cls.catalogue = catalogue()
        cls.systems = [cls.catalogue.facets(label) for label in cls.catalogue.labels()]

    def test_catalogue_is_the_exact_facets_of_the_data(self):
        self.assertEqual(len(self.lines), TYPES)
        with Pool(WORKERS) as pool:
            ok = pool.map(_exact_check, list(zip(self.lines, self.systems)), chunksize=200)
        bad = [k for k, good in enumerate(ok) if not good]
        self.assertEqual(bad, [])

    def test_simplicial_coherent_spheres(self):
        bad = [label for label in self.catalogue.labels()
               if not self.catalogue.complex(label).is_valid()]
        self.assertEqual(bad, [])

    def test_pairwise_not_equivalent(self):
        with Pool(WORKERS) as pool:
            forms = pool.map(_form, self.systems, chunksize=200)
        self.assertEqual(len(set(forms)), TYPES)

    def test_only_the_winner_is_a_join_of_two_pentagons(self):
        joins = [label for label, f in zip(self.catalogue.labels(), self.systems)
                 if is_cycle_join(f, 10, 5, 5)]
        self.assertEqual(joins, [WINNER])


@pytest.mark.slow
class TestCertificates(unittest.TestCase):
    """The proposition on the other types: certificates rebuilt from the catalogue and the
    recorded route pass
    every exact check of Section 3.3, including coherence and the facets of the type."""

    def test_sample_rebuilds_and_verifies(self):
        c = campaign()
        entries = c.ledger.entries()
        proved = sorted(label for label in rivals()
                        if c.done(entries.get(label), label))
        sample = random.Random(1).sample(proved, min(SAMPLE, len(proved)))
        result = c.verify(sample)
        self.assertEqual(result["failed"], [])
        self.assertEqual(result["missing"], [])
        self.assertEqual(len(result["match"]) + len(result["differ"]), len(sample))

    @unittest.skipUnless(os.environ.get("INSCRIBED_FULL_VERIFY"),
                         "set INSCRIBED_FULL_VERIFY=1 to rebuild every certificate (hours)")
    def test_every_rival_rebuilds_and_verifies(self):
        audit = json.loads((RESULTS / "audit.json").read_text())
        self.assertTrue(campaign().complete(audit))


if __name__ == "__main__":
    unittest.main()
