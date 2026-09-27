import math
import unittest

import numpy as np

from inscribed.algebra import generalized_cross
from inscribed.catalog import Bipyramid, TriaugmentedPrism
from inscribed.combinatorics import simplex
from inscribed.volume import VolumeFunctional


class TestGeneralizedCross(unittest.TestCase):
    def test_defining_identity(self):
        """x . gcross(v_1..v_{d-1}) == det(x, v_1, ..., v_{d-1})."""
        rng = np.random.default_rng(0)
        for d in range(3, 7):
            for _ in range(20):
                vs = rng.normal(size=(d - 1, d))
                x = rng.normal(size=d)
                lhs = float(x @ generalized_cross(vs))
                rhs = float(np.linalg.det(np.vstack([x, vs])))
                self.assertAlmostEqual(lhs, rhs, places=9)

    def test_matches_numpy_cross_in_3d(self):
        rng = np.random.default_rng(1)
        a, b = rng.normal(size=3), rng.normal(size=3)
        np.testing.assert_allclose(generalized_cross(np.array([a, b])), np.cross(a, b))


class TestLinearity(unittest.TestCase):
    """d! * vol = n_i . p_i + C_i, for every vertex i."""

    def test_decomposition(self):
        rng = np.random.default_rng(2)
        prism = TriaugmentedPrism().complex()
        for cx in (prism, Bipyramid().complex(m=5), simplex(4), simplex(5)):
            form = VolumeFunctional.build(cx)
            pts = rng.normal(size=(cx.n, cx.d))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            total = form.value(pts)
            for v in range(cx.n):
                nv = form.vertex_normal(pts, v)
                rebuilt = float(nv @ pts[v]) / math.factorial(cx.d) + form.constant_term(pts, v)
                self.assertAlmostEqual(total, rebuilt, places=12)

    def test_gradient_matches_finite_differences(self):
        rng = np.random.default_rng(3)
        cx = TriaugmentedPrism().complex()
        form = VolumeFunctional.build(cx)
        pts = rng.normal(size=(cx.n, cx.d))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        grad = form.gradient(pts)
        eps = 1e-6
        for v in range(cx.n):
            for j in range(cx.d):
                plus = pts.copy()
                plus[v, j] += eps
                minus = pts.copy()
                minus[v, j] -= eps
                fd = (form.value(plus) - form.value(minus)) / (2 * eps)
                self.assertAlmostEqual(fd, grad[v, j], places=6)


class TestPublishedMaximizer(unittest.TestCase):
    """Hoehner--Ledford, arXiv:2608.04392, Theorem 1.1."""

    def setUp(self):
        self.prism = TriaugmentedPrism()
        self.form = VolumeFunctional.build(self.prism.complex())
        self.pts = self.prism.build()

    def test_points_on_unit_sphere(self):
        np.testing.assert_allclose(np.linalg.norm(self.pts, axis=1), 1.0, atol=1e-14)

    def test_volume_is_three_sqrt_two_sqrt_three_minus_three(self):
        self.assertAlmostEqual(
            self.form.value(self.pts), self.prism.closed_form_volume(), places=13
        )


if __name__ == "__main__":
    unittest.main()
