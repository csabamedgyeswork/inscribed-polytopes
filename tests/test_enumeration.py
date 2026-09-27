"""Enumeration sources and the oriented_hull orientation convention."""

import unittest

import numpy as np

from inscribed.combinatorics import OrientedComplex
from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.geometry import oriented_hull


class TestFirschingSource(unittest.TestCase):
    """Counts must match Firsching's published enumeration exactly."""

    def test_type_counts(self):
        source = FirschingSource()
        # All simplicial 4-polytopes up to 8 vertices are inscribable, so the
        # source yields every type: 1, 2, 5, 37 for n = 5..8.
        for n, expected in ((5, 1), (6, 2), (7, 5), (8, 37)):
            with self.subTest(n=n):
                self.assertEqual(len(list(source.types(n))), expected)

    def test_records_carry_inscribed_points(self):
        record = next(iter(FirschingSource().types(8)))
        self.assertIsNotNone(record.points)
        radii = np.linalg.norm(record.points, axis=1)
        self.assertTrue(np.allclose(radii, 1.0, atol=1e-9))
        self.assertEqual(record.complex.d, 4)
        self.assertEqual(record.complex.n, 8)


class TestPlantriSource(unittest.TestCase):
    """Plantri yields isomorph-free simplicial 2-spheres."""

    def test_small_counts(self):
        source = PlantriSource()
        # Known counts of triangulations of S^2 on n vertices.
        for n, expected in ((4, 1), (5, 1), (6, 2), (7, 5)):
            with self.subTest(n=n):
                self.assertEqual(len(list(source.types(n))), expected)

    def test_types_are_valid_spheres(self):
        for record in PlantriSource().types(7):
            self.assertTrue(record.complex.is_valid())
            self.assertEqual(record.complex.d, 3)


class TestOrientedHullOrientation(unittest.TestCase):
    """oriented_hull orients against the hull centroid, not the origin.

    Regression test: an inscribed polytope need not contain the sphere centre,
    and orienting against a point outside the hull yields an incoherent
    complex. This is invisible in 3D, where maximisers always contain the
    centre, but it discarded valid 4D types before the fix.
    """

    def test_polytope_not_containing_origin(self):
        # Five points clustered near one pole of S^3 form a 4-simplex whose
        # interior does not contain the origin. Orienting against the origin
        # would produce an incoherent complex; the centroid does not.
        points = np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.6, 0.8, 0.0, 0.0],
                [0.6, 0.0, 0.8, 0.0],
                [0.6, 0.0, 0.0, 0.8],
                [0.6, -0.4, -0.4, -0.4],
            ]
        )
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        cx = oriented_hull(points)
        self.assertIsInstance(cx, OrientedComplex)
        self.assertTrue(cx.is_valid())


if __name__ == "__main__":
    unittest.main()
