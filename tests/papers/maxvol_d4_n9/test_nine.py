"""The claims about Firsching's nine-vertex polytopes in papers/maxvol-d4-n9: the lemma
on his data and the proposition on the other combinatorial types.

The checks that need the exact facets of all 1142 polytopes take several minutes and are
marked slow (run them with  poetry run pytest -m slow); the exact re-verification of the
1141 certificates additionally needs INSCRIBED_FULL_VERIFY to be set. Both run in
INSCRIBED_WORKERS processes (default 5).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import unittest
from fractions import Fraction
from functools import partial
from itertools import combinations, permutations, product
from math import factorial
from multiprocessing import Pool
from pathlib import Path

import pytest

from inscribed.certify import CertificateVerifier, ExactCertificate, cyclic_types, facet_sets
from inscribed.enumeration import FirschingSource
from inscribed.sos import enumerated_types
from inscribed.store import certificate_dir
from tests.papers.maxvol_d4.test_eight import exact_facets, rational_points

ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "papers" / "maxvol-d4-n9"
WINNER = "firsching-9-10"
NOT_INSCRIBED = ["firsching-9-323", "firsching-9-324"]
GAMMA = Fraction(468, 25)
BOUND = Fraction(39, 50)
WORKERS = int(os.environ.get("INSCRIBED_WORKERS", "5"))


def text() -> str:
    """All sections of the paper with line breaks collapsed to single spaces."""
    sections = sorted((PAPER / "sections").glob("*.tex"))
    return re.sub(r"\s+", " ", " ".join(f.read_text() for f in sections))


def refined_labels(facets: set[frozenset[int]], n: int) -> dict[int, object]:
    """Vertex labels preserved by every combinatorial equivalence: the number of facets and
    of neighbours, refined three times by the labels of the other vertices of each facet."""
    neighbours = {v: set().union(*(f for f in facets if v in f)) - {v} for v in range(n)}
    labels = {v: (sum(v in f for f in facets), len(neighbours[v])) for v in range(n)}
    for _ in range(3):
        labels = {v: (labels[v], tuple(sorted(tuple(sorted(labels[u] for u in f if u != v))
                                              for f in facets if v in f)))
                  for v in range(n)}
    return labels


def canonical(facets: set[frozenset[int]], n: int) -> tuple[int, ...]:
    """A form shared exactly by combinatorially equivalent facet sets: the least encoding
    over the relabellings that send each class of refined labels to a fixed block."""
    labels = refined_labels(facets, n)
    classes: dict[object, list[int]] = {}
    for v in range(n):
        classes.setdefault(labels[v], []).append(v)
    keys = sorted(classes)
    blocks, start = [], 0
    for key in keys:
        blocks.append((classes[key], range(start, start + len(classes[key]))))
        start += len(classes[key])
    best = None
    for choice in product(*(permutations(block) for _, block in blocks)):
        relabel = {v: q for (vertices, _), perm in zip(blocks, choice)
                   for v, q in zip(vertices, perm)}
        form = tuple(sorted(sum(1 << relabel[v] for v in f) for f in facets))
        if best is None or form < best:
            best = form
    return best


def join_facets(cycle_a: list[int], cycle_b: list[int]) -> set[frozenset[int]]:
    """The facets of the join of two cycles."""
    m, l = len(cycle_a), len(cycle_b)
    return {frozenset({cycle_a[i], cycle_a[(i + 1) % m], cycle_b[j], cycle_b[(j + 1) % l]})
            for i in range(m) for j in range(l)}


class NinePoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.points = {f"firsching-9-{k}": rational_points(line)
                      for k, line in enumerate(FirschingSource()._lines(9))}


class NineTypes(NinePoints):
    """The exact facets of all 1142 polytopes; several minutes of exact arithmetic."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        with Pool(WORKERS) as pool:
            facets = pool.map(exact_facets, list(cls.points.values()), chunksize=8)
        cls.facets = dict(zip(cls.points, facets))
        cls.loaded = {r.label: facet_sets(r.complex.facets)
                      for r in FirschingSource().types(9)}


class TestFirschingData(NinePoints):
    """'Firsching's data give a polytope with rational vertices for each line of a file with
    1142 lines ... The facets of firsching-9-10 are exactly those of J_{4,5} for the
    4-cycle (1, 3, 6, 4) and the 5-cycle (0, 2, 8, 7, 5).' The paper needs no claim about
    inscribability; the two realisations off the sphere are checked because the loader
    skips them and scripts/eliminate_non_inscribable.py certifies them."""

    def test_printed(self):
        paper = text()
        self.assertIn("for each line of a file with $1142$ lines", paper)
        self.assertIn(r"\texttt{firsching-9-0} to \texttt{firsching-9-1141}", paper)
        self.assertIn(r"$4$-cycle $(1, 3, 6, 4)$ and the $5$-cycle $(0, 2, 8, 7, 5)$", paper)
        self.assertNotIn("inscribable", paper)

    def test_nine_rational_vertices_each(self):
        self.assertEqual(len(self.points), 1142)
        for label, points in self.points.items():
            with self.subTest(label=label):
                self.assertEqual(len(points), 9)
                self.assertTrue(all(len(p) == 4 for p in points))

    def test_exactly_two_not_inscribed(self):
        off = [label for label, points in self.points.items()
               if not all(sum(x * x for x in p) == 1 for p in points)]
        self.assertEqual(off, NOT_INSCRIBED)

    def test_winner_is_the_join(self):
        self.assertEqual(exact_facets(self.points[WINNER]),
                         join_facets([1, 3, 6, 4], [0, 2, 8, 7, 5]))


@pytest.mark.slow
class TestFirschingTypes(NineTypes):
    """'... these 1142 polytopes are simplicial and pairwise not combinatorially
    equivalent.'"""

    def test_simplicial_with_nine_vertices(self):
        """The exact supporting-hyperplane test yields 4-sets only, every vertex lies on a
        facet, and every 3-subset of a facet lies in exactly two facets."""
        for label, facets in self.facets.items():
            with self.subTest(label=label):
                self.assertEqual(set().union(*facets), set(range(9)))
                ridges: dict[tuple[int, ...], int] = {}
                for f in facets:
                    for r in combinations(sorted(f), 3):
                        ridges[r] = ridges.get(r, 0) + 1
                self.assertEqual(set(ridges.values()), {2})

    def test_pairwise_not_equivalent(self):
        with Pool(WORKERS) as pool:
            forms = set(pool.map(partial(canonical, n=9), list(self.facets.values()),
                                 chunksize=8))
        self.assertEqual(len(forms), 1142)

    def test_loader_keeps_the_inscribed_ones_with_exact_facets(self):
        self.assertEqual(sorted(self.loaded), sorted(set(self.points) - set(NOT_INSCRIBED)))
        for label, facets in self.loaded.items():
            with self.subTest(label=label):
                self.assertEqual(facets, self.facets[label])


def stored(label: str) -> tuple[dict, bytes]:
    """The summary of the stored certificate and its uncompressed payload."""
    directory = certificate_dir(4, 9)
    payload = gzip.decompress((directory / f"{label}.certificate.json.gz").read_bytes())
    summary = json.loads((directory / f"{label}.summary.json").read_text())
    return summary, payload


def load(label: str) -> ExactCertificate:
    """The stored certificate."""
    return ExactCertificate.from_json(json.loads(stored(label)[1]))


_OFF_SPHERE: dict[str, set[frozenset[int]]] = {}


def _set_off_sphere(facets: dict[str, set[frozenset[int]]]) -> None:
    _OFF_SPHERE.update(facets)


def _off_sphere(cert: ExactCertificate) -> set[frozenset[int]] | None:
    """The exact facets of the two types the loader skips, which enumerated_types misses."""
    return _OFF_SPHERE.get(cert.label)


def _verify(label: str) -> tuple[str, bool, Fraction]:
    verifier = CertificateVerifier(resolvers=(cyclic_types, enumerated_types, _off_sphere))
    report = verifier.verify(load(label))
    return label, report.ok, report.volume_bound


def rivals() -> list[str]:
    return [f"firsching-9-{k}" for k in range(1142) if f"firsching-9-{k}" != WINNER]


class TestCertificates(unittest.TestCase):
    """'For each of these 1141 values of k, the stored certificate with beta = 468/25 passes
    the checks ... vol_4(P) <= beta/4! = 39/50.'"""

    def test_printed(self):
        paper = text()
        self.assertIn("For each of these $1141$ values of $k$", paper)
        self.assertIn(r"$\beta = 468/25$", paper)
        self.assertIn(r"\beta/4! = 39/50", paper)
        self.assertEqual(GAMMA / factorial(4), BOUND)

    def test_every_rival_has_a_verified_summary_at_the_bound(self):
        self.assertEqual(len(rivals()), 1141)
        for label in rivals():
            with self.subTest(label=label):
                summary, payload = stored(label)
                self.assertTrue(summary["verification"]["ok"])
                self.assertEqual(Fraction(summary["gamma"]), GAMMA)
                self.assertEqual(Fraction(summary["volume_bound"]), BOUND)
                self.assertEqual(summary["sha256_uncompressed"],
                                 hashlib.sha256(payload).hexdigest())


@pytest.mark.slow
class TestCertificateFacets(NineTypes):
    """'... including that its facets are those of firsching-9-k.'"""

    def test_each_certificate_has_the_facets_of_its_type(self):
        for label in rivals():
            with self.subTest(label=label):
                cert = load(label)
                self.assertEqual(facet_sets(cert.facets), self.facets[label])

    @unittest.skipUnless(os.environ.get("INSCRIBED_FULL_VERIFY"),
                         "set INSCRIBED_FULL_VERIFY=1 for the exact re-verification")
    def test_each_certificate_verifies_exactly(self):
        off_sphere = {label: self.facets[label] for label in NOT_INSCRIBED}
        with Pool(WORKERS, initializer=_set_off_sphere, initargs=(off_sphere,)) as pool:
            results = pool.imap_unordered(_verify, rivals(), chunksize=4)
            for label, ok, volume_bound in sorted(results):
                with self.subTest(label=label):
                    self.assertTrue(ok)
                    self.assertEqual(volume_bound, BOUND)


if __name__ == "__main__":
    unittest.main()
