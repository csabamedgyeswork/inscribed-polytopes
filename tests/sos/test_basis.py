"""Tests for inscribed.sos.basis."""

from __future__ import annotations

import unittest

from inscribed.sos import MomentOrbits, ParityBlocks, Symmetry
from inscribed.sos.basis import monomials


class TestMonomials(unittest.TestCase):
    def test_counts(self):
        self.assertEqual(len(monomials(3, 0)), 1)
        self.assertEqual(len(monomials(3, 1)), 4)
        self.assertEqual(len(monomials(3, 2)), 10)

    def test_sorted(self):
        for m in monomials(4, 2):
            self.assertEqual(tuple(sorted(m)), m)


class TestParityBlocks(unittest.TestCase):
    def test_tetrahedron_block_sizes(self):
        blocks = ParityBlocks(3, 4, 2)
        self.assertEqual(sorted(len(b) for b in blocks.blocks), [20, 20, 20, 31])

    def test_position_covers_all(self):
        blocks = ParityBlocks(4, 5, 2)
        total = sum(len(b) for b in blocks.blocks)
        self.assertEqual(len(blocks.position), total)


class TestMomentOrbits(unittest.TestCase):
    def test_identity_group(self):
        ms = monomials(2, 2)
        orbits = MomentOrbits(ms, [Symmetry.identity(1, 2)], 2)
        self.assertEqual(orbits.size, len(ms))

    def test_sign_flip_forces_zero(self):
        ms = monomials(2, 2)
        flip = Symmetry((0,), (0, 1), (-1, 1))
        orbits = MomentOrbits(ms, [flip], 2)
        self.assertIsNone(orbits.lookup((0,)))
        self.assertIsNotNone(orbits.lookup((0, 0)))

    def test_lookup_sign(self):
        ms = monomials(2, 2)
        flip = Symmetry((0,), (0, 1), (-1, 1))
        orbits = MomentOrbits(ms, [flip], 2)
        self.assertIsNone(orbits.lookup((0, 1)))


if __name__ == "__main__":
    unittest.main()
