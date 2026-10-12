"""Tests for inscribed.sos.symmetry."""

from __future__ import annotations

import unittest

from inscribed.certify import SignedVolume
from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.sos import Symmetry, SymmetryGroup


def tetrahedron():
    """The unique simplicial type on 4 vertices."""
    return next(r for r in PlantriSource().types(4))


def simplex4():
    """The unique simplicial type on 5 vertices in d = 4."""
    return next(r for r in FirschingSource().types(5))


def cyclic47():
    """The cyclic type C_4(7)."""
    return next(r for r in FirschingSource().types(7) if r.label == "firsching-7-0")


class TestSymmetry(unittest.TestCase):
    def test_identity(self):
        s = Symmetry.identity(3, 4)
        self.assertEqual(s.vertices, (0, 1, 2))
        self.assertEqual(s.coords, (0, 1, 2, 3))
        self.assertEqual(s.signs, (1, 1, 1, 1))

    def test_compose(self):
        flip = Symmetry((0, 1), (0, 1), (-1, 1))
        swap = Symmetry((1, 0), (0, 1), (1, 1))
        both = flip.compose(swap)
        self.assertEqual(both.vertices, (1, 0))
        self.assertEqual(both.signs, (-1, 1))

    def test_apply_sign_and_permutation(self):
        s = Symmetry((1, 0), (0, 1), (-1, 1))
        m, sign = s.apply((0,), 2)
        self.assertEqual(m, (2,))
        self.assertEqual(sign, -1)

    def test_acts_on_polynomial(self):
        p = SignedVolume(tetrahedron().complex).polynomial
        s = Symmetry.identity(4, 3)
        self.assertEqual(s.acts_on(p, 3), p)


class TestTypeAutomorphisms(unittest.TestCase):
    def test_tetrahedron_count(self):
        group = SymmetryGroup.for_signed_volume(tetrahedron().complex)
        self.assertEqual(len(group.automorphisms), 24)

    def test_cyclic_count(self):
        group = SymmetryGroup.for_signed_volume(cyclic47().complex)
        self.assertEqual(len(group.automorphisms), 14)


class TestSymmetryGroup(unittest.TestCase):
    def test_tetrahedron_order(self):
        self.assertEqual(SymmetryGroup.for_signed_volume(tetrahedron().complex).order(), 576)

    def test_cyclic_order(self):
        self.assertEqual(SymmetryGroup.for_signed_volume(cyclic47().complex).order(), 2688)

    def test_coordinate_only_order(self):
        group = SymmetryGroup(tetrahedron().complex, use_type_automorphisms=False)
        self.assertEqual(group.order(), 24)

    def test_generators_leave_signed_volume_invariant(self):
        SymmetryGroup.for_signed_volume(simplex4().complex)


if __name__ == "__main__":
    unittest.main()
