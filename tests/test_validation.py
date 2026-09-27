import unittest

import numpy as np

from inscribed.analysis import (
    Composite,
    ConvexPosition,
    Gauge,
    GaugeFrame,
    PropertyZ,
    ValidationReport,
    Validator,
)
from inscribed.catalog import Bipyramid, TriaugmentedPrism, known
from inscribed.optimality import OptimalityConditions
from inscribed.search import FixedTypeAscent, MultiStartSearch, RandomInitializer
from inscribed.volume import VolumeFunctional


class TestConvexPosition(unittest.TestCase):
    def test_accepts_the_published_maximizer(self):
        prism = TriaugmentedPrism()
        report = ConvexPosition(prism.complex()).check(prism.build())
        self.assertTrue(report.ok)
        self.assertGreater(report.metrics["min_support_margin"], 0.1)
        self.assertAlmostEqual(
            report.metrics["hull_volume"], prism.closed_form_volume(), places=9
        )

    def test_rejects_folded_configurations(self):
        """Signed volume can exceed any polytope volume on folded surfaces."""
        cx = Bipyramid().complex(m=7)
        solver = FixedTypeAscent(cx)
        initializer = RandomInitializer(3, cx.n, seed=7)
        validator = ConvexPosition(cx)
        threshold = known(3, 9).value
        found_spurious = False
        for _ in range(60):
            res = solver.solve(initializer.initial())
            if res.volume > threshold + 1e-6:
                report = validator.check(res.points)
                self.assertFalse(report.ok)
                self.assertLess(report.metrics["hull_volume"], res.volume - 1e-6)
                found_spurious = True
        self.assertTrue(found_spurious, "expected to find folded fixed points")

    def test_without_complex_checks_convexity_only(self):
        prism = TriaugmentedPrism()
        report = ConvexPosition().check(prism.build())
        self.assertTrue(report.ok)


class TestOtherValidators(unittest.TestCase):
    def setUp(self):
        self.prism = TriaugmentedPrism()
        self.pts = self.prism.build()
        self.cx = self.prism.complex()

    def test_property_z_validator(self):
        conditions = OptimalityConditions(VolumeFunctional.build(self.cx))
        self.assertTrue(PropertyZ(conditions).check(self.pts).ok)
        shifted = self.pts.copy()
        shifted[0] = np.array([1.0, 0.0, 0.0])
        self.assertFalse(PropertyZ(conditions).check(shifted).ok)

    def test_gauge_validator(self):
        frame = GaugeFrame.default(self.cx)
        framed = frame.transform(self.pts)
        self.assertTrue(Gauge(frame).check(framed).ok)
        self.assertFalse(Gauge(frame).check(self.pts).ok)

    def test_composite_reports_first_failure(self):
        conditions = OptimalityConditions(VolumeFunctional.build(self.cx))
        frame = GaugeFrame.default(self.cx)
        both = Composite(ConvexPosition(self.cx), PropertyZ(conditions))
        self.assertTrue(both.check(self.pts).ok)
        bad = self.pts.copy()
        bad[0] = np.array([1.0, 0.0, 0.0])
        self.assertFalse(both.check(bad).ok)
        framed = frame.transform(self.pts)
        self.assertTrue(Composite(Gauge(frame), ConvexPosition(self.cx)).check(framed).ok)


class RejectAll(Validator):
    def check(self, points: np.ndarray) -> ValidationReport:
        return ValidationReport(False, "rejected by stub", {})


class TestValidatorInjection(unittest.TestCase):
    def test_rejecting_validator_stops_the_search(self):
        cx = Bipyramid().complex(m=5)
        search = MultiStartSearch(
            FixedTypeAscent(cx), RandomInitializer(3, cx.n, seed=0), RejectAll()
        )
        with self.assertRaises(RuntimeError):
            search.run(10)

    def test_search_without_validator_accepts_everything(self):
        cx = Bipyramid().complex(m=5)
        search = MultiStartSearch(FixedTypeAscent(cx), RandomInitializer(3, cx.n, seed=0))
        res = search.run(10)
        self.assertEqual(res.info["rejected"], 0)


if __name__ == "__main__":
    unittest.main()
