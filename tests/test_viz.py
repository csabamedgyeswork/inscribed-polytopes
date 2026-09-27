"""Schlegel projection math, the view table, and a render smoke test."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inscribed.catalog import RegularSimplex
from inscribed.combinatorics import bipyramid, simplex
from inscribed.store import RESULTS_DIR, load_case
from inscribed.viz import render_case, schlegel_projection
from inscribed.viz.views import VIEWS


class TestSchlegel(unittest.TestCase):
    def test_d3_projects_to_plane(self):
        cx = bipyramid(4)
        pts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                        [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
        proj, basis = schlegel_projection(pts, cx)
        self.assertEqual(proj.shape, (6, 2))
        self.assertEqual(basis.shape, (3, 2))

    def test_d4_projects_to_3space(self):
        cx = simplex(4)
        pts = RegularSimplex().build(d=4)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        proj, basis = schlegel_projection(pts, cx)
        self.assertEqual(proj.shape, (5, 3))
        self.assertEqual(basis.shape, (4, 3))

    def test_basis_is_orthonormal_and_perpendicular(self):
        cx = simplex(4)
        pts = RegularSimplex().build(d=4)
        pts /= np.linalg.norm(pts, axis=1, keepdims=True)
        _, basis = schlegel_projection(pts, cx)
        self.assertTrue(np.allclose(basis.T @ basis, np.eye(3), atol=1e-9))


class TestViews(unittest.TestCase):
    def test_views_are_three_orthographic(self):
        self.assertEqual([title for title, _, _ in VIEWS], ["front", "side", "top"])
        angles = [(elev, azim) for _, elev, azim in VIEWS]
        self.assertEqual(len(set(angles)), 3)


class TestRenderCase(unittest.TestCase):
    def test_render_case_writes_png(self):
        case = load_case(RESULTS_DIR / "d3-n04.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = render_case(case, Path(tmp))
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 20_000)

    def test_render_case_4d_writes_png(self):
        case = load_case(RESULTS_DIR / "d4-n05.json")
        with tempfile.TemporaryDirectory() as tmp:
            out = render_case(case, Path(tmp))
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 20_000)


if __name__ == "__main__":
    unittest.main()
