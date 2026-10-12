"""Tests for inscribed.sos.relaxation."""

from __future__ import annotations

import unittest

from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.sos import MomentRelaxation, SolverSettings, SymmetryGroup


def tetrahedron():
    """The unique simplicial type on 4 vertices."""
    return next(r for r in PlantriSource().types(4))


def simplex4():
    """The unique simplicial type on 5 vertices in d = 4."""
    return next(r for r in FirschingSource().types(5))


class TestMomentRelaxation(unittest.TestCase):
    def test_tetrahedron_tight(self):
        record = tetrahedron()
        group = SymmetryGroup.for_signed_volume(record.complex)
        result = MomentRelaxation(record.complex, group).solve(SolverSettings())
        self.assertAlmostEqual(result.value / 6, 8 * 3 ** 0.5 / 27, delta=1e-5)

    def test_simplex4_tight(self):
        record = simplex4()
        group = SymmetryGroup.for_signed_volume(record.complex)
        result = MomentRelaxation(record.complex, group).solve(SolverSettings())
        self.assertAlmostEqual(result.value / 24, 25 * 5 ** 0.5 / 384, delta=1e-5)

    def test_block_sizes(self):
        record = tetrahedron()
        group = SymmetryGroup.for_signed_volume(record.complex)
        relaxation = MomentRelaxation(record.complex, group)
        self.assertEqual(sorted(len(b) for b in relaxation.blocks()), [20, 31])

    def test_volume_value(self):
        record = tetrahedron()
        group = SymmetryGroup.for_signed_volume(record.complex)
        relaxation = MomentRelaxation(record.complex, group)
        result = relaxation.solve(SolverSettings())
        self.assertAlmostEqual(relaxation.volume_value(result), result.value / 6)


if __name__ == "__main__":
    unittest.main()
