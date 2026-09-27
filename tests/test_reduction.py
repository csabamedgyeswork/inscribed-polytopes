import unittest

import numpy as np

from inscribed.analysis import (
    GaugeFrame,
    IndependentSetElimination,
    free_parameter_count,
    gauge_dimension,
    max_independent_set,
)
from inscribed.catalog import TriaugmentedPrism
from inscribed.combinatorics import simplex
from inscribed.volume import VolumeFunctional


class TestCounts(unittest.TestCase):
    def test_gauge_dimension_is_dim_so_d(self):
        self.assertEqual(gauge_dimension(3), 3)
        self.assertEqual(gauge_dimension(4), 6)
        self.assertEqual(gauge_dimension(5), 10)

    def test_free_parameters(self):
        # 9 points on S^2: 18 coordinates on the sphere, 3 killed by rotation
        self.assertEqual(free_parameter_count(3, 9), 15)
        # 9 points on S^3: 27 sphere coordinates, 6 killed by rotation
        self.assertEqual(free_parameter_count(4, 9), 21)

    def test_pinned_coordinate_count_matches_gauge_dimension(self):
        for d in range(3, 7):
            frame = GaugeFrame(d, tuple(range(d - 1)))
            self.assertEqual(len(frame.pinned_coordinates()), gauge_dimension(d))

    def test_wrong_anchor_count_rejected(self):
        with self.assertRaises(ValueError):
            GaugeFrame(4, (0, 1))


class TestGaugeFrame(unittest.TestCase):
    def test_pins_the_expected_coordinates(self):
        rng = np.random.default_rng(0)
        for d, n in ((3, 9), (4, 9), (5, 8)):
            pts = rng.normal(size=(n, d))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            frame = GaugeFrame(d, tuple(range(d - 1)))
            framed = frame.transform(pts)
            self.assertLess(frame.residual(framed), 1e-12)

    def test_anchor_becomes_the_pole(self):
        rng = np.random.default_rng(1)
        pts = rng.normal(size=(9, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        frame = GaugeFrame(3, (0, 1))
        framed = frame.transform(pts)
        np.testing.assert_allclose(framed[0], [0, 0, 1], atol=1e-12)
        self.assertAlmostEqual(framed[1, 0], 0.0, places=12)
        self.assertGreaterEqual(framed[1, 1], -1e-12)

    def test_is_a_rotation_so_volume_is_unchanged(self):
        rng = np.random.default_rng(2)
        cx = TriaugmentedPrism().complex()
        form = VolumeFunctional.build(cx)
        pts = rng.normal(size=(cx.n, cx.d))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        framed = GaugeFrame(3, (0, 1)).transform(pts)
        self.assertAlmostEqual(form.value(framed), form.value(pts), places=12)
        np.testing.assert_allclose(framed @ framed.T, pts @ pts.T, atol=1e-11)

    def test_idempotent(self):
        rng = np.random.default_rng(3)
        pts = rng.normal(size=(7, 4))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        frame = GaugeFrame(4, (0, 1, 2))
        once = frame.transform(pts)
        twice = frame.transform(once)
        np.testing.assert_allclose(once, twice, atol=1e-11)

    def test_handles_degenerate_anchors(self):
        """A repeated anchor must not crash; the slice is closed."""
        pts = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
        frame = GaugeFrame(3, (0, 1))
        framed = frame.transform(pts)
        self.assertLess(frame.residual(framed), 1e-12)

    def test_default_anchors_are_adjacent_and_correctly_sized(self):
        for cx in (TriaugmentedPrism().complex(), simplex(4), simplex(5)):
            frame = GaugeFrame.default(cx)
            self.assertEqual(len(frame.anchors), cx.d - 1)
            self.assertEqual(len(set(frame.anchors)), cx.d - 1)
            edges = cx.edges()
            for w in frame.anchors[1:]:
                self.assertIn(frozenset((frame.anchors[0], w)), edges)


class TestIndependentSetElimination(unittest.TestCase):
    def setUp(self):
        self.prism = TriaugmentedPrism()
        self.form = VolumeFunctional.build(self.prism.complex())

    def test_max_independent_set_is_the_three_apexes(self):
        self.assertEqual(set(max_independent_set(self.form.cx)), {0, 5, 8})

    def test_elimination_is_exact_at_the_optimum(self):
        elimination = IndependentSetElimination.maximal(self.form)
        pts = self.prism.build()
        self.assertAlmostEqual(elimination.value(pts), self.form.value(pts), places=13)

    def test_elimination_dominates_generic_configurations(self):
        rng = np.random.default_rng(4)
        elimination = IndependentSetElimination.maximal(self.form)
        for _ in range(20):
            pts = rng.normal(size=(9, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            self.assertGreaterEqual(elimination.value(pts) + 1e-12, self.form.value(pts))

    def test_filling_eliminated_vertices_attains_the_bound(self):
        rng = np.random.default_rng(5)
        elimination = IndependentSetElimination.maximal(self.form)
        pts = rng.normal(size=(9, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        filled = elimination.fill(pts)
        self.assertAlmostEqual(self.form.value(filled), elimination.value(pts), places=12)

    def test_adjacent_vertices_rejected(self):
        with self.assertRaises(ValueError):
            IndependentSetElimination(self.form, frozenset({0, 1}))


if __name__ == "__main__":
    unittest.main()
