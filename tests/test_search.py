import unittest

import numpy as np

from inscribed.analysis import ConvexPosition
from inscribed.catalog import TriaugmentedPrism, known
from inscribed.geometry import hull_volume, oriented_hull
from inscribed.search import (
    ChainedInitializer,
    ExplicitConfiguration,
    FixedTypeAscent,
    GrowFromConfiguration,
    HullAscent,
    MultiStartSearch,
    RandomInitializer,
)
from inscribed.volume import VolumeFunctional


def hull_search(n, d, restarts, seed):
    return MultiStartSearch(HullAscent(), RandomInitializer(d, n, seed=seed)).run(restarts)


class TestHullSearch(unittest.TestCase):
    def test_recovers_known_optima_on_s2(self):
        for n, restarts in ((4, 40), (5, 30), (6, 30), (7, 40), (9, 60)):
            with self.subTest(n=n):
                res = hull_search(n, 3, restarts, n)
                self.assertAlmostEqual(res.volume, known(3, n).value, places=9)
                self.assertIsNotNone(res.cx)
                self.assertTrue(ConvexPosition(res.cx).check(res.points).ok)

    def test_signed_volume_of_emergent_type_matches_hull_volume(self):
        res = hull_search(9, 3, 60, 11)
        form = VolumeFunctional.build(res.cx)
        self.assertAlmostEqual(form.value(res.points), res.volume, places=10)

    def test_ascent_is_monotone(self):
        start = RandomInitializer(3, 8, seed=0).initial()
        before = hull_volume(start)
        res = HullAscent().solve(start)
        self.assertGreaterEqual(res.volume, before - 1e-12)

    def test_oriented_hull_facets_are_positively_oriented(self):
        pts = TriaugmentedPrism().build()
        cx = oriented_hull(pts)
        cx.check()
        for f in cx.facets:
            self.assertGreater(np.linalg.det(pts[list(f)]), 0)

    def test_grow_from_configuration_increases_volume(self):
        res = hull_search(6, 3, 20, 3)
        grown = GrowFromConfiguration(res.points).initial()
        self.assertEqual(grown.shape[0], 7)
        self.assertGreater(hull_volume(grown), res.volume)


class TestInterchangeability(unittest.TestCase):
    """The same multistart driver reaches the known optimum with either solver."""

    def test_fixed_type_and_hull_solvers_agree(self):
        target = known(3, 9).value
        via_hull = hull_search(9, 3, 60, 21)
        self.assertAlmostEqual(via_hull.volume, target, places=9)
        cx = via_hull.cx
        via_type = MultiStartSearch(
            FixedTypeAscent(cx), RandomInitializer(3, 9, seed=22), ConvexPosition(cx)
        ).run(60)
        self.assertAlmostEqual(via_type.volume, target, places=9)


class TestInitializers(unittest.TestCase):
    def test_random_initializer_is_unit_norm_and_seeded(self):
        init = RandomInitializer(4, 7, seed=5)
        a, b = init.initial(), init.initial()
        np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1.0)
        self.assertFalse(np.allclose(a, b))
        again = RandomInitializer(4, 7, seed=5).initial()
        np.testing.assert_allclose(a, again)

    def test_explicit_configuration_returns_a_copy(self):
        pts = np.eye(3)
        init = ExplicitConfiguration(pts)
        out = init.initial()
        out[0, 0] = 99.0
        self.assertEqual(pts[0, 0], 1.0)

    def test_chained_initializer_yields_each_once_then_delegates(self):
        fixed = np.zeros((2, 3))
        fixed[:, 0] = 1.0
        init = ChainedInitializer(ExplicitConfiguration(fixed), RandomInitializer(3, 2, seed=1))
        np.testing.assert_allclose(init.initial(), fixed)
        self.assertFalse(np.allclose(init.initial(), fixed))


if __name__ == "__main__":
    unittest.main()
