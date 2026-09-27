import unittest

from inscribed.combinatorics import (
    MEDIAL_TYPE_COUNTS,
    OrientedComplex,
    bipyramid,
    is_medial,
    medial_degrees,
    medial_verdict,
    permutation_parity,
    simplex,
)
from inscribed.catalog import TriaugmentedPrism


def tetrahedron() -> OrientedComplex:
    return OrientedComplex(
        d=3, n=4, facets=((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3))
    )


class TestPermutationParity(unittest.TestCase):
    def test_known_parities(self):
        self.assertEqual(permutation_parity((0, 1, 2)), 1)
        self.assertEqual(permutation_parity((1, 0, 2)), -1)
        self.assertEqual(permutation_parity((1, 2, 0)), 1)
        self.assertEqual(permutation_parity((3, 1, 2, 0)), -1)


class TestMediality(unittest.TestCase):
    def test_admissible_degrees(self):
        self.assertEqual(medial_degrees(4), (3, 3))
        self.assertEqual(medial_degrees(9), (4, 5))
        self.assertEqual(medial_degrees(12), (5, 5))
        self.assertEqual(medial_degrees(14), (5, 6))

    def test_reference_maximisers_are_medial(self):
        self.assertTrue(is_medial(tetrahedron()))
        self.assertTrue(is_medial(bipyramid(3)))
        self.assertTrue(is_medial(bipyramid(4)))
        self.assertTrue(is_medial(bipyramid(5)))
        self.assertTrue(is_medial(TriaugmentedPrism().complex()))

    def test_heptagonal_bipyramid_is_not_medial(self):
        # Nine vertices, degrees 4^6 7^2: the apexes exceed the medial range.
        self.assertFalse(is_medial(bipyramid(7)))

    def test_mediality_is_three_dimensional_only(self):
        with self.assertRaises(ValueError):
            is_medial(simplex(4))

    def test_verdict_is_vacuous_where_no_medial_type_exists(self):
        self.assertEqual(MEDIAL_TYPE_COUNTS[11], 0)
        self.assertEqual(medial_verdict(11, [4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6]), "vacuous")

    def test_verdict_medial_and_not_medial(self):
        self.assertEqual(medial_verdict(12, [5] * 12), "medial")
        self.assertEqual(medial_verdict(12, [4] + [5] * 10 + [7]), "not medial")

    def test_verdict_beyond_tabulated_range_flags_unknown_existence(self):
        self.assertNotIn(40, MEDIAL_TYPE_COUNTS)
        self.assertEqual(medial_verdict(40, [3] * 40), "not medial (existence unknown)")


class TestValidation(unittest.TestCase):
    def test_reference_complexes_are_valid(self):
        prism = TriaugmentedPrism().complex()
        for cx in [tetrahedron(), prism] + [bipyramid(m) for m in range(3, 9)]:
            cx.check()

    def test_simplex_boundary_valid_in_every_dimension(self):
        for d in range(3, 7):
            simplex(d).check()

    def test_euler_characteristic(self):
        self.assertEqual(TriaugmentedPrism().complex().euler_characteristic(), 2)
        for d in range(3, 7):
            expected = 1 + (-1) ** (d - 1)
            self.assertEqual(simplex(d).euler_characteristic(), expected)

    def test_incoherent_orientation_rejected(self):
        cx = tetrahedron()
        bad = list(cx.facets)
        bad[0] = (bad[0][0], bad[0][2], bad[0][1])
        broken = OrientedComplex(d=3, n=4, facets=tuple(bad))
        self.assertFalse(broken.is_valid())

    def test_reversed_orientation_is_still_coherent(self):
        TriaugmentedPrism().complex().reversed_orientation().check()

    def test_open_surface_rejected(self):
        cx = tetrahedron()
        broken = OrientedComplex(d=3, n=4, facets=cx.facets[:-1])
        self.assertFalse(broken.is_valid())


class TestTriaugmentedPrism(unittest.TestCase):
    """The user's class 1 is combinatorially the triaugmented prism (J51)."""

    def setUp(self):
        self.cx = TriaugmentedPrism().complex()

    def test_f_vector(self):
        self.assertEqual(self.cx.n, 9)
        self.assertEqual(len(self.cx.edges()), 21)
        self.assertEqual(len(self.cx.facets), 14)

    def test_degree_sequence_is_medial(self):
        degs = sorted(self.cx.vertex_degrees().values())
        self.assertEqual(degs, [4, 4, 4, 5, 5, 5, 5, 5, 5])

    def test_degree_four_vertices_are_the_augmenting_ones(self):
        deg = self.cx.vertex_degrees()
        apexes = {v for v, k in deg.items() if k == 4}
        # user labels 1, 6, 9 -> zero-based 0, 5, 8
        self.assertEqual(apexes, {0, 5, 8})

    def test_apex_links_are_disjoint_four_cycles(self):
        for v in (0, 5, 8):
            self.assertEqual(len(self.cx.link_cycle(v)), 4)
        links = [set(self.cx.link_cycle(v)) for v in (0, 5, 8)]
        self.assertEqual(len(links[0] & links[1] & links[2]), 0)

    def test_prism_triangles_present(self):
        unoriented = {frozenset(f) for f in self.cx.facets}
        self.assertIn(frozenset({3, 6, 4}), unoriented)  # labels 4,7,5
        self.assertIn(frozenset({7, 2, 1}), unoriented)  # labels 8,3,2


if __name__ == "__main__":
    unittest.main()
