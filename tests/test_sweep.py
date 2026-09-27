"""TypeSweep recovers the known maximiser over all types at a small case."""

import unittest

from inscribed.catalog import known
from inscribed.enumeration import PlantriSource
from inscribed.search import TypeSweep


class TestTypeSweep(unittest.TestCase):
    def test_octahedron_wins_at_d3_n6(self):
        # Two combinatorial types of simplicial 3-polytope on 6 vertices; the
        # octahedron is the proven maximiser with volume 4/3.
        sweep = TypeSweep(PlantriSource(), restarts=10, seed=11)
        results = sweep.run(6)
        feasible = [r for r in results if r.feasible]
        self.assertTrue(feasible)
        best = feasible[0]
        record = known(3, 6)
        self.assertIsNotNone(record)
        self.assertAlmostEqual(best.volume, record.value, places=9)
        self.assertAlmostEqual(best.volume, 4.0 / 3.0, places=9)


if __name__ == "__main__":
    unittest.main()
