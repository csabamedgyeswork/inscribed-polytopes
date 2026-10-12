"""The claims about eight vertices in papers/maxvol-d4: the cross-polytope paragraph and
the remark on Firsching's polytopes in preliminaries.tex, and sections/eight.tex.

The exact re-verification of the 36 certificates takes several minutes and runs only when
INSCRIBED_FULL_VERIFY is set; the other checks always run.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import unittest
from fractions import Fraction
from itertools import combinations, permutations, product
from math import factorial
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

from inscribed.certify import CertificateVerifier, ExactCertificate, cyclic_types, facet_sets
from inscribed.enumeration import FirschingSource
from inscribed.sos import enumerated_types
from inscribed.store import RESULTS_DIR, certificate_dir

ROOT = Path(__file__).resolve().parents[3]
SECTIONS = ROOT / "papers" / "maxvol-d4" / "sections"
CROSS = "firsching-8-23"
BOUND = Fraction(13, 20)


def text(name: str) -> str:
    """One section with line breaks collapsed to single spaces."""
    return re.sub(r"\s+", " ", (SECTIONS / f"{name}.tex").read_text())


def rational_points(line: str) -> list[list[Fraction]]:
    """The exact vertices of one line of Firsching's file."""
    return [[Fraction(t.strip()) for t in block.split(",")]
            for block in re.findall(r"\[([^\[\]]+)\]", line)]


def determinant(rows: list[list[Fraction]]) -> Fraction:
    """Exact determinant by Gaussian elimination over the rationals."""
    m = [row[:] for row in rows]
    result = Fraction(1)
    for c in range(len(m)):
        pivot = next((r for r in range(c, len(m)) if m[r][c] != 0), None)
        if pivot is None:
            return Fraction(0)
        if pivot != c:
            m[c], m[pivot] = m[pivot], m[c]
            result = -result
        result *= m[c][c]
        for r in range(c + 1, len(m)):
            factor = m[r][c] / m[c][c]
            for k in range(c, len(m)):
                m[r][k] -= factor * m[c][k]
    return result


def exact_facets(points: list[list[Fraction]]) -> set[frozenset[int]]:
    """The 4-sets whose hyperplane has every other vertex strictly on one side."""
    n = len(points)
    found = set()
    for s in combinations(range(n), 4):
        base = points[s[0]]
        rows = [[x - y for x, y in zip(points[i], base)] for i in s[1:]]
        signs = {(determinant(rows + [[x - y for x, y in zip(points[q], base)]]) > 0)
                 - (determinant(rows + [[x - y for x, y in zip(points[q], base)]]) < 0)
                 for q in range(n) if q not in s}
        if signs in ({1}, {-1}):
            found.add(frozenset(s))
    return found


def canonical(facets: set[frozenset[int]], n: int) -> tuple[int, ...]:
    """A form shared exactly by combinatorially equivalent facet sets."""
    return min(tuple(sorted(sum(1 << p[v] for v in f) for f in facets))
               for p in permutations(range(n)))


def is_cross_polytope(facets: set[frozenset[int]], n: int) -> bool:
    """Whether the vertices pair up so that the facets are exactly the transversals."""
    edges = {frozenset(e) for f in facets for e in combinations(f, 2)}
    missing = [frozenset(e) for e in combinations(range(n), 2) if frozenset(e) not in edges]
    if len(missing) != n // 2 or len(set().union(*missing)) != n:
        return False
    pairs = [tuple(e) for e in missing]
    return facets == {frozenset(choice) for choice in product(*pairs)}


class EightTypes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = FirschingSource()
        cls.records = list(source.types(8))
        cls.points = [rational_points(line) for line in source._lines(8)]
        cls.facets = {r.label: facet_sets(r.complex.facets) for r in cls.records}


class TestFirschingPolytopes(EightTypes):
    """'Firsching gives a polytope of each type with rational vertices on S^3 ... these 37
    polytopes are simplicial and pairwise not combinatorially equivalent ... and
    firsching-8-23 is a cross-polytope.'"""

    def test_printed(self):
        prelim = text("preliminaries")
        self.assertIn(r"\texttt{firsching-8-0} to \texttt{firsching-8-36}", prelim)
        self.assertIn(r"\texttt{firsching-8-23} is a cross-polytope", prelim)
        self.assertIn("There are exactly $37$ combinatorial types", prelim)

    def test_labels(self):
        self.assertEqual([r.label for r in self.records],
                         [f"firsching-8-{k}" for k in range(37)])

    def test_vertices_lie_exactly_on_the_sphere(self):
        for label, points in zip(self.facets, self.points):
            with self.subTest(label=label):
                self.assertEqual(len(points), 8)
                self.assertTrue(all(sum(x * x for x in p) == 1 for p in points))

    def test_facets_are_exact_and_simplices(self):
        """The floating-point hull agrees with the exact supporting-hyperplane test, which
        also shows that every facet is a simplex."""
        for (label, facets), points in zip(self.facets.items(), self.points):
            with self.subTest(label=label):
                self.assertEqual(exact_facets(points), facets)

    def test_pairwise_not_equivalent(self):
        forms = {canonical(f, 8) for f in self.facets.values()}
        self.assertEqual(len(forms), 37)

    def test_only_firsching_8_23_is_a_cross_polytope(self):
        self.assertEqual([label for label, f in self.facets.items() if is_cross_polytope(f, 8)],
                         [CROSS])


class TestRegularCrossPolytope(unittest.TestCase):
    """'... it lies in P_d(2d) and has volume 2^d/d!, which is 2/3 for d = 4.'"""

    def test_printed(self):
        self.assertIn(r"volume $2^{d}/d!$, which is $2/3$ for $d = 4$", text("preliminaries"))

    def test_volume(self):
        self.assertEqual(Fraction(2 ** 4, factorial(4)), Fraction(2, 3))
        vertices = np.vstack([np.eye(4), -np.eye(4)])
        self.assertAlmostEqual(ConvexHull(vertices).volume, 2 / 3, places=12)

    def test_facets_are_the_transversals(self):
        vertices = np.vstack([np.eye(4), -np.eye(4)])
        hull = {frozenset(int(v) for v in s) for s in ConvexHull(vertices).simplices}
        self.assertTrue(is_cross_polytope(hull, 8))


def load(label: str) -> tuple[ExactCertificate, dict, bytes]:
    """The stored certificate, its summary and its uncompressed payload."""
    directory = certificate_dir(4, 8)
    payload = gzip.decompress((directory / f"{label}.certificate.json.gz").read_bytes())
    summary = json.loads((directory / f"{label}.summary.json").read_text())
    return ExactCertificate.from_json(json.loads(payload)), summary, payload


class TestCertificates(EightTypes):
    """'For each of them the stored certificate passes the checks ... with beta = 78/5 ...
    vol_4(P) <= beta/4! = 13/20.'"""

    def rivals(self) -> list[str]:
        return [label for label in self.facets if label != CROSS]

    def test_printed(self):
        eight = text("eight")
        self.assertIn(r"$\beta = 78/5$", eight)
        self.assertIn(r"\beta/4! = 13/20", eight)
        self.assertEqual(Fraction(78, 5) / factorial(4), BOUND)
        self.assertLess(BOUND, Fraction(2, 3))

    def test_every_rival_has_a_verified_summary_at_the_bound(self):
        self.assertEqual(len(self.rivals()), 36)
        for label in self.rivals():
            with self.subTest(label=label):
                _, summary, payload = load(label)
                self.assertTrue(summary["verification"]["ok"])
                self.assertEqual(Fraction(summary["gamma"]), Fraction(78, 5))
                self.assertEqual(Fraction(summary["volume_bound"]), BOUND)
                self.assertEqual(summary["sha256_uncompressed"],
                                 hashlib.sha256(payload).hexdigest())

    def test_each_certificate_has_the_facets_of_its_type(self):
        for label in self.rivals():
            with self.subTest(label=label):
                cert, _, _ = load(label)
                self.assertEqual(facet_sets(cert.facets), self.facets[label])

    @unittest.skipUnless(os.environ.get("INSCRIBED_FULL_VERIFY"),
                         "set INSCRIBED_FULL_VERIFY=1 for the exact re-verification")
    def test_each_certificate_verifies_exactly(self):
        verifier = CertificateVerifier(resolvers=(cyclic_types, enumerated_types))
        for label in self.rivals():
            with self.subTest(label=label):
                cert, _, _ = load(label)
                report = verifier.verify(cert)
                self.assertTrue(report.ok)
                self.assertEqual(report.volume_bound, BOUND)


class TestNumericalMaxima(unittest.TestCase):
    """'... among the 36 types, the largest value of D/4! found numerically is about 0.601,
    for firsching-8-22.'"""

    def test_printed(self):
        self.assertIn(r"about $0.601$, for \texttt{firsching-8-22}", text("eight"))

    def test_largest_rival(self):
        types = json.loads((RESULTS_DIR / "d4-n08.json").read_text())["types"]
        rivals = [t for t in types if t["label"] != CROSS]
        top = max(rivals, key=lambda t: t["signed_max"])
        self.assertEqual(top["label"], "firsching-8-22")
        self.assertEqual(round(top["signed_max"], 3), 0.601)
        self.assertLess(top["signed_max"], float(BOUND))


if __name__ == "__main__":
    unittest.main()
