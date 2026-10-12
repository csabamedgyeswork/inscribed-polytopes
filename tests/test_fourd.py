import unittest

from inscribed.analysis import ConvexPosition, SymmetryDetector
from inscribed.catalog import KNOWN, OrthogonalPolygonPair, RegularSimplex, Status
from inscribed.search import HullAscent, MultiStartSearch, RandomInitializer


def hull_search_4d(n, restarts, seed):
    return MultiStartSearch(HullAscent(), RandomInitializer(4, n, seed=seed)).run(restarts)


class TestSearchOnS3(unittest.TestCase):
    def test_recovers_horvath_langi_cases(self):
        family = OrthogonalPolygonPair()
        for n in (5, 6, 7):
            with self.subTest(n=n):
                res = hull_search_4d(n, 60, n)
                if n == 5:
                    expected = RegularSimplex().closed_form_volume(d=4)
                else:
                    expected = family.closed_form_volume(**family.parameters_for(n, 4))
                self.assertAlmostEqual(res.volume, expected, places=9)
                self.assertTrue(ConvexPosition(res.cx).check(res.points).ok)

    def test_n9_is_the_pentagon_square_pair(self):
        res = hull_search_4d(9, 80, 109)
        self.assertAlmostEqual(
            res.volume,
            OrthogonalPolygonPair().closed_form_volume(m=4, k=5),
            places=9,
        )
        report = SymmetryDetector().analyze(res.points)
        self.assertEqual(report.order, 80)
        self.assertEqual(report.orbit_sizes, [4, 5])

    def test_family_is_beaten_at_n_eleven(self):
        """Crossover: an explicit convex polytope beats the family at n = 11."""
        res = hull_search_4d(11, 120, 911)
        family = OrthogonalPolygonPair()
        best_family = family.closed_form_volume(**family.parameters_for(11, 4))
        self.assertGreater(res.volume, best_family + 1e-4)
        self.assertTrue(ConvexPosition(res.cx).check(res.points).ok)


class TestProvenCatalogEntries(unittest.TestCase):
    """Every catalog entry with a proof behind it is reproduced by search to 1e-9."""

    def test_proven_values_reproduced(self):
        cases = [(3, n, 60) for n in (4, 5, 6, 7, 8, 9)]
        cases += [(3, 12, 60)]
        cases += [(4, n, 60) for n in (5, 6, 7)]
        for d, n, restarts in cases:
            record = KNOWN[(d, n)]
            assert record.status in (Status.PROVEN, Status.CLASS_PROVEN)
            with self.subTest(d=d, n=n):
                res = MultiStartSearch(
                    HullAscent(), RandomInitializer(d, n, seed=1000 + n)
                ).run(restarts)
                self.assertAlmostEqual(res.volume, record.value, places=9)


if __name__ == "__main__":
    unittest.main()
