import math
import re
import unittest

from inscribed.catalog import (
    KNOWN,
    Bipyramid,
    CrossPolytope,
    OrthogonalPolygonPair,
    RegularSimplex,
    Status,
    TriaugmentedPrism,
    known,
)
from inscribed.geometry import hull_volume

_EXACT_NAMES = {"sqrt": math.sqrt, "sin": math.sin, "pi": math.pi}


def _evaluate_exact(exact: str) -> float:
    """Evaluate a catalog closed form, restoring the multiplications its notation omits."""
    expression = re.sub(r"(?<=[\d)])\s+(?=[\d(a-z])", "*", exact)
    return eval(expression, {"__builtins__": {}}, _EXACT_NAMES)


class TestFamilies(unittest.TestCase):
    """closed_form_volume matches the hull of build across a parameter grid."""

    def test_regular_simplex(self):
        family = RegularSimplex()
        for d in range(3, 7):
            self.assertAlmostEqual(
                hull_volume(family.build(d=d)), family.closed_form_volume(d=d), places=10
            )

    def test_cross_polytope(self):
        family = CrossPolytope()
        for d in range(3, 6):
            self.assertAlmostEqual(
                hull_volume(family.build(d=d)), family.closed_form_volume(d=d), places=10
            )

    def test_bipyramid(self):
        family = Bipyramid()
        for m in range(3, 9):
            self.assertAlmostEqual(
                hull_volume(family.build(m=m)), family.closed_form_volume(m=m), places=10
            )

    def test_orthogonal_polygon_pair(self):
        family = OrthogonalPolygonPair()
        for m in range(3, 9):
            for k in range(3, 9):
                with self.subTest(m=m, k=k):
                    self.assertAlmostEqual(
                        hull_volume(family.build(m=m, k=k)),
                        family.closed_form_volume(m=m, k=k),
                        places=10,
                    )

    def test_triaugmented_prism(self):
        family = TriaugmentedPrism()
        self.assertAlmostEqual(
            hull_volume(family.build()), family.closed_form_volume(), places=12
        )


class TestOrthogonalPolygonPairValues(unittest.TestCase):
    def setUp(self):
        self.family = OrthogonalPolygonPair()

    def test_known_exact_values(self):
        f = self.family.closed_form_volume
        self.assertAlmostEqual(f(m=3, k=3), 9 / 32, places=13)
        self.assertAlmostEqual(f(m=3, k=4), math.sqrt(3) / 4, places=13)
        self.assertAlmostEqual(f(m=4, k=4), 2 / 3, places=13)
        self.assertAlmostEqual(
            f(m=4, k=5), 5 * math.sqrt(10 + 2 * math.sqrt(5)) / 24, places=13
        )
        self.assertAlmostEqual(f(m=5, k=5), (125 + 25 * math.sqrt(5)) / 192, places=13)

    def test_four_four_split_is_the_cross_polytope(self):
        self.assertAlmostEqual(
            self.family.closed_form_volume(m=4, k=4),
            CrossPolytope().closed_form_volume(d=4),
            places=13,
        )

    def test_degenerate_polygons_rejected(self):
        with self.assertRaises(ValueError):
            self.family.closed_form_volume(m=2, k=5)
        with self.assertRaises(ValueError):
            self.family.build(m=2, k=5)

    def test_best_splits(self):
        for n, split in ((6, (3, 3)), (7, (3, 4)), (8, (4, 4)), (9, (4, 5)), (10, (5, 5))):
            params = self.family.parameters_for(n, 4)
            self.assertEqual((params["m"], params["k"]), split)

    def test_family_saturates_below_the_ball(self):
        """Large balanced splits tend to pi^2/6, well under the ball's pi^2/2."""
        big = self.family.closed_form_volume(m=400, k=400)
        self.assertLess(big, math.pi**2 / 6 + 1e-6)
        self.assertGreater(big, math.pi**2 / 6 - 1e-3)

    def test_wrong_dimension_rejected(self):
        with self.assertRaises(ValueError):
            self.family.parameters_for(9, 3)


class TestCatalog(unittest.TestCase):
    def test_registry_is_keyed_by_dimension_and_n(self):
        for (d, n), record in KNOWN.items():
            self.assertEqual(record.dimension, d)
            self.assertEqual(record.n, n)

    def test_proven_entries_carry_exact_forms_and_citations(self):
        for record in KNOWN.values():
            if record.status in (Status.PROVEN, Status.CLASS_PROVEN):
                self.assertIsNotNone(record.exact)
            self.assertTrue(record.citation.text)

    def test_exact_forms_agree_with_stored_values(self):
        for (d, n), record in KNOWN.items():
            if record.exact is None:
                continue
            with self.subTest(d=d, n=n):
                self.assertAlmostEqual(
                    _evaluate_exact(record.exact), record.value, places=12
                )

    def test_lookup(self):
        self.assertEqual(known(3, 9).status, Status.PROVEN)
        self.assertEqual(known(4, 9).status, Status.CONJECTURAL)
        self.assertIsNone(known(5, 7))


class TestOeisCitedValues(unittest.TestCase):
    """The n = 10 and n = 11 rows on S^2 are taken from OEIS, not from a paper.

    The digits, the n = 10 minimal polynomial and the degree of the n = 11 one
    are transcribed from the cited entries so the comparison stays offline.
    """

    A339262 = "2.21871113154539940324728275112841701381072537466334438175"
    A339263 = "2.35463449506861520323688059263892654160344864269342168599"

    def test_values_are_the_cited_decimal_expansions(self):
        for n, digits in ((10, self.A339262), (11, self.A339263)):
            with self.subTest(n=n):
                self.assertEqual(known(3, n).value, float(digits))

    def test_both_rows_are_conjectural_and_cite_oeis(self):
        """Both entries read "conjecturally maximum possible volume"."""
        for n in (10, 11):
            with self.subTest(n=n):
                self.assertEqual(known(3, n).status, Status.CONJECTURAL)
                self.assertIn("oeis.org", known(3, n).citation.url)

    def test_n10_is_a_root_of_the_cited_minimal_polynomial(self):
        """A339262 gives 65536 + 147456x - 59904x^2 - 139968x^3 + 59049x^4."""
        value = known(3, 10).value
        coefficients = (65536, 147456, -59904, -139968, 59049)
        residual = sum(c * value**i for i, c in enumerate(coefficients))
        self.assertLess(abs(residual), 1e-8)

    def test_the_n10_closed_form_is_the_cited_one_simplified(self):
        """Ours drops the (3 - sqrt2) factor, since (3 - sqrt2)^2 (10 + 6 sqrt2)
        = 38 + 6 sqrt2."""
        cited = (2 / 3) ** 4 * (
            3 + math.sqrt(2) + (3 - math.sqrt(2)) * math.sqrt(10 + 6 * math.sqrt(2))
        )
        self.assertAlmostEqual(_evaluate_exact(known(3, 10).exact), cited, places=15)

    def test_n11_carries_no_closed_form(self):
        """A339263 puts the volume at algebraic degree 100 with Galois group
        S_50, so no expression by radicals exists to print."""
        self.assertIsNone(known(3, 11).exact)


if __name__ == "__main__":
    unittest.main()
