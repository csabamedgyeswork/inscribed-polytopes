"""Orthogonal 2-plane split detection for 4D free sums."""

import unittest

import numpy as np

from inscribed.analysis import orthogonal_plane_split
from inscribed.catalog import OrthogonalPolygonPair, RegularSimplex


class TestPlaneSplit(unittest.TestCase):
    def test_detects_free_sums(self):
        fam = OrthogonalPolygonPair()
        for n in range(6, 13):
            params = fam.parameters_for(n, 4)
            pts = fam.build(**params)
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            split = orthogonal_plane_split(pts)
            with self.subTest(n=n):
                self.assertIsNotNone(split)
                sizes = sorted(len(p) for p in split)
                self.assertEqual(sizes, sorted([params["m"], params["k"]]))

    def test_simplex_does_not_split(self):
        pts = RegularSimplex().build(d=4)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        self.assertIsNone(orthogonal_plane_split(pts))

    def test_wrong_dimension_returns_none(self):
        self.assertIsNone(orthogonal_plane_split(np.eye(5, 3)))


if __name__ == "__main__":
    unittest.main()
