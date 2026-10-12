import unittest

from inscribed.analysis import SymmetryDetector
from inscribed.catalog import CrossPolytope, TriaugmentedPrism, known
from inscribed.search import HullAscent, MultiStartSearch, RandomInitializer


class TestSymmetryDetection(unittest.TestCase):
    def test_triaugmented_prism_optimum_is_d3h(self):
        report = SymmetryDetector().analyze(TriaugmentedPrism().build())
        self.assertEqual(report.order, 12)
        self.assertEqual(report.proper, 6)
        self.assertEqual(report.improper, 6)
        self.assertEqual(report.orbit_sizes, [3, 6])

    def test_icosahedron_group_order(self):
        res = MultiStartSearch(HullAscent(), RandomInitializer(3, 12, seed=12)).run(40)
        self.assertAlmostEqual(res.volume, known(3, 12).value, places=9)
        self.assertEqual(SymmetryDetector().analyze(res.points).order, 120)

    def test_octahedron_group_order(self):
        res = MultiStartSearch(HullAscent(), RandomInitializer(3, 6, seed=6)).run(20)
        self.assertEqual(SymmetryDetector().analyze(res.points).order, 48)

    def test_cross_polytope_group_order(self):
        self.assertEqual(SymmetryDetector().analyze(CrossPolytope().build(d=4)).order, 384)


if __name__ == "__main__":
    unittest.main()
