import unittest

import numpy as np

from inscribed.catalog import TriaugmentedPrism
from inscribed.optimality import OptimalityConditions
from inscribed.volume import VolumeFunctional


class TestPropertyZ(unittest.TestCase):
    def setUp(self):
        self.prism = TriaugmentedPrism()
        self.conditions = OptimalityConditions(
            VolumeFunctional.build(self.prism.complex())
        )
        self.pts = self.prism.build()

    def test_published_maximizer_satisfies_property_z(self):
        self.assertLess(self.conditions.property_z_residual(self.pts), 1e-12)

    def test_euler_relation_for_multipliers(self):
        """sum_v lambda_v == d * vol at any Property-Z configuration."""
        self.assertLess(self.conditions.euler_identity_error(self.pts), 1e-12)

    def test_generic_configuration_fails_property_z(self):
        rng = np.random.default_rng(0)
        pts = rng.normal(size=(9, 3))
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        self.assertGreater(self.conditions.property_z_residual(pts), 1e-3)

    def test_degenerate_normal_reported_as_infinite_residual(self):
        pts = self.pts.copy()
        pts[1] = pts[2]  # collapse an edge so some n_v vanishes or degenerates
        residual = self.conditions.property_z_residual(pts)
        self.assertTrue(residual > 1e-6 or residual == float("inf"))


if __name__ == "__main__":
    unittest.main()
