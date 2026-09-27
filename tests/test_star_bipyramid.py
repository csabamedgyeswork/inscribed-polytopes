"""Claims made in paper/sections/trap.tex.

The spurious fixed points that block-coordinate ascent reaches on the double
m-pyramid are the configurations S_{m,k}: two polar apexes over a ring
advancing by 2 pi k / m. Every test here backs a sentence of that section.
"""

import json
import math
import unittest
import numpy as np

from inscribed.analysis import ConvexPosition
from inscribed.catalog import known
from inscribed.combinatorics.builders import bipyramid
from inscribed.enumeration import PlantriSource
from inscribed.optimality import OptimalityConditions
from inscribed.search import FixedTypeAscent
from inscribed.store.paths import case_path
from inscribed.volume import VolumeFunctional

HEPTAGONAL_BIPYRAMID = "plantri-9-24"


def star_bipyramid(m: int, k: int) -> np.ndarray:
    """``S_{m,k}``: equatorial ring of step ``2 pi k / m`` with apexes at the poles."""
    angles = 2 * math.pi * k * np.arange(m) / m
    points = np.zeros((m + 2, 3))
    points[:m, 0] = np.cos(angles)
    points[:m, 1] = np.sin(angles)
    points[m] = (0.0, 0.0, 1.0)
    points[m + 1] = (0.0, 0.0, -1.0)
    return points


def closed_form(m: int, k: int) -> float:
    """Right-hand side of (starbipyramid)."""
    return (m / 3) * math.sin(2 * math.pi * k / m)


def sphere_starts(m: int, count: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    starts = []
    for _ in range(count):
        p = rng.normal(size=(m + 2, 3))
        starts.append(p / np.linalg.norm(p, axis=1, keepdims=True))
    return starts


class TestTheType(unittest.TestCase):
    """"the heptagonal bipyramid, degree sequence 4^7 7^2 (label plantri-9-24)"."""

    def test_label_is_the_heptagonal_bipyramid(self):
        record = next(
            r for r in PlantriSource().types(9) if r.label == HEPTAGONAL_BIPYRAMID
        )
        degrees = sorted(
            sum(1 for f in record.complex.facets if v in f)
            for v in range(record.complex.n)
        )
        self.assertEqual(degrees, [4] * 7 + [7] * 2)
        self.assertEqual(len(record.complex.facets), 14)


class TestClosedForms(unittest.TestCase):
    """Equation (starbipyramid) and the Berman--Hanes value it generalises."""

    def test_signed_volume_matches_closed_form(self):
        for m in range(3, 13):
            for k in range(1, (m + 1) // 2):
                with self.subTest(m=m, k=k):
                    form = VolumeFunctional.build(bipyramid(m))
                    self.assertAlmostEqual(
                        form.value(star_bipyramid(m, k)), closed_form(m, k), places=12
                    )

    def test_berman_hanes_value_agrees_with_the_catalog(self):
        """(m/3) sin(2 pi / m) is the published maximum where the bipyramid wins."""
        for m in (3, 4, 5):  # n = 5, 6, 7: bipyramid is the known maximiser
            with self.subTest(m=m):
                record = known(3, m + 2)
                self.assertIsNotNone(record)
                self.assertAlmostEqual(closed_form(m, 1), record.value, places=12)

    def test_star_bipyramids_are_stationary(self):
        """Why ascent stops there; supports "every converged point ... meets"."""
        for m in range(3, 13):
            for k in range(1, (m + 1) // 2):
                with self.subTest(m=m, k=k):
                    conditions = OptimalityConditions(VolumeFunctional.build(bipyramid(m)))
                    self.assertLess(
                        conditions.property_z_residual(star_bipyramid(m, k)), 3e-16
                    )

    def test_only_the_convex_ring_realises_the_type(self):
        """"for k >= 2 ... S_{m,k} not in Omega_C"."""
        for m in range(5, 9):
            validator = ConvexPosition(bipyramid(m))
            self.assertTrue(validator.check(star_bipyramid(m, 1)).ok)
            for k in range(2, (m + 1) // 2):
                with self.subTest(m=m, k=k):
                    self.assertFalse(validator.check(star_bipyramid(m, k)).ok)


def ring_winding_number(ring: np.ndarray, point: tuple[float, float]) -> int:
    """Winding number of the closed planar polygon ``ring`` around ``point``."""
    angles = np.arctan2(ring[:, 1] - point[1], ring[:, 0] - point[0])
    turns = np.diff(np.concatenate([angles, angles[:1]]))
    turns = (turns + math.pi) % (2 * math.pi) - math.pi
    return round(turns.sum() / (2 * math.pi))


class TestRingGeometry(unittest.TestCase):
    """The description of the ring of S_{m,k} that derives (starbipyramid)."""

    CASES = [(m, k) for m in range(3, 13) for k in range(1, (m + 1) // 2)]

    def test_ring_labelling_joins_j_to_j_plus_one(self):
        """"Label the ring ... so that its facets join ring vertex j to j+1"."""
        for m in range(3, 13):
            with self.subTest(m=m):
                ring_edges = {
                    frozenset(v for v in f if v < m) for f in bipyramid(m).facets
                }
                self.assertEqual(ring_edges, {frozenset((j, (j + 1) % m)) for j in range(m)})

    def test_coprime_step_relabels_the_convex_ring(self):
        """"When gcd(m,k) = 1 these are the points of S_{m,1} relabelled"."""
        for m, k in self.CASES:
            if math.gcd(m, k) != 1:
                continue
            with self.subTest(m=m, k=k):
                star = {tuple(p) for p in np.round(star_bipyramid(m, k), 12)}
                convex = {tuple(p) for p in np.round(star_bipyramid(m, 1), 12)}
                self.assertEqual(star, convex)

    def test_common_divisor_makes_ring_vertices_coincide(self):
        """"When gcd(m,k) > 1 ... the ring {(m/g)/(k/g)} traversed g times"."""
        for m, k in self.CASES:
            g = math.gcd(m, k)
            if g == 1:
                continue
            with self.subTest(m=m, k=k):
                ring = star_bipyramid(m, k)[:m]
                distinct = {tuple(p) for p in np.round(ring, 12)}
                self.assertEqual(len(distinct), m // g)
                np.testing.assert_allclose(ring[: m // g], ring[m // g: 2 * (m // g)], atol=1e-12)
                self.assertEqual(ring_winding_number(ring[: m // g], (0.0, 0.0)), k // g)

    def test_signed_area_and_central_winding_number(self):
        """"signed area (m/2) sin(2 pi k/m), ... central region ... winding number k"."""
        for m, k in self.CASES:
            with self.subTest(m=m, k=k):
                ring = star_bipyramid(m, k)[:m, :2]
                nxt = np.roll(ring, -1, axis=0)
                area = 0.5 * float(np.sum(ring[:, 0] * nxt[:, 1] - nxt[:, 0] * ring[:, 1]))
                self.assertAlmostEqual(area, (m / 2) * math.sin(2 * math.pi * k / m), places=12)
                self.assertEqual(ring_winding_number(ring, (1e-6, 1e-7)), k)
                self.assertAlmostEqual(closed_form(m, k), 2 * area / 3, places=12)


def assert_is_star_bipyramid(case: unittest.TestCase, points: np.ndarray, m: int, k: int):
    """``points`` is ``S_{m,k}`` up to an orthogonal map: antipodal apexes, an
    equatorial ring, and consecutive ring vertices ``2 pi k / m`` apart."""
    top, bottom = points[m], points[m + 1]
    case.assertAlmostEqual(float(top @ bottom), -1.0, places=12)
    case.assertLess(np.abs(points[:m] @ top).max(), 1e-12)

    e1 = points[0] - (points[0] @ top) * top
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(top, e1)
    angles = np.array([math.atan2(p @ e2, p @ e1) for p in points[:m]])
    steps = np.diff(np.concatenate([angles, angles[:1]])) % (2 * math.pi)
    # The value converges to 1e-14, so coordinates carry about its square root.
    for step in steps:
        case.assertAlmostEqual(step, 2 * math.pi * k / m, delta=1e-6)


class TestConvergedFold(unittest.TestCase):
    """"The configuration reached above is S_{7,2}"."""

    def test_the_m7_fold_is_the_step_two_ring_with_antipodal_apexes(self):
        cx = bipyramid(7)
        validator = ConvexPosition(cx)
        best = None
        for start in sphere_starts(7, 150, seed=1):
            result = FixedTypeAscent(cx, tol=1e-14, max_sweeps=20000).solve(start)
            if validator.check(result.points).ok:
                continue
            if best is None or result.volume > best.volume:
                best = result
        self.assertIsNotNone(best)
        assert_is_star_bipyramid(self, best.points, 7, 2)


class TestDiscardTable(unittest.TestCase):
    """The table, and the two sentences drawn from it."""

    EXPECTED = {3: 0, 4: 0, 5: 18, 6: 81, 7: 132}

    def test_counts_values_and_surviving_runs(self):
        for m, expected in self.EXPECTED.items():
            cx = bipyramid(m)
            validator = ConvexPosition(cx)
            kept, discarded = [], []
            for start in sphere_starts(m, 150, seed=1):
                result = FixedTypeAscent(cx, tol=1e-14, max_sweeps=20000).solve(start)
                bucket = kept if validator.check(result.points).ok else discarded
                bucket.append(result)
            with self.subTest(m=m):
                self.assertEqual(len(discarded), expected)
                # "each discarded run here converged to S_{m,2}"
                for result in discarded:
                    self.assertAlmostEqual(result.volume, closed_form(m, 2), places=6)
                    assert_is_star_bipyramid(self, result.points, m, 2)
                # "all surviving runs reproduce the Berman--Hanes closed form"
                self.assertTrue(kept)
                for result in kept:
                    self.assertAlmostEqual(result.volume, closed_form(m, 1), delta=1e-14)

    def test_fold_need_not_exceed_the_convex_value(self):
        """"at m = 5 it carries less ... at m = 6 it carries the same value"."""
        self.assertLess(closed_form(5, 2), closed_form(5, 1))
        self.assertAlmostEqual(closed_form(6, 2), closed_form(6, 1), places=12)


class TestCommittedSweep(unittest.TestCase):
    """"our exhaustive n = 9 sweep discards 17 of its 20 restarts"."""

    def test_recorded_rejection_count(self):
        case = json.loads(case_path(3, 9).read_text())
        record = next(
            t for t in case["types"] if t["label"] == HEPTAGONAL_BIPYRAMID
        )
        self.assertEqual(record["rejected"], 17)
        self.assertEqual(record["restarts"], 20)
        self.assertAlmostEqual(record["volume"], closed_form(7, 1), places=12)


if __name__ == "__main__":
    unittest.main()
