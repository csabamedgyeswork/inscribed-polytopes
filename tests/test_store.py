"""Configuration and CaseResult save/load round-trips."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from inscribed.combinatorics import bipyramid, simplex
from inscribed.store import (
    CaseResult,
    Configuration,
    Evidence,
    TypeEntry,
    build_search_record,
    load,
    load_case,
    load_many,
    load_search,
    save,
    save_case,
    save_many,
    save_search,
)
from inscribed.store.case import build_case


def _cfg():
    cx = bipyramid(4)
    pts = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0],
                    [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    return Configuration(points=pts, complex=cx, volume=4 / 3,
                         label="octa", source="manual",
                         metadata={"restarts": 10, "seed": 0})


class TestStore(unittest.TestCase):
    def test_round_trip_bit_for_bit(self):
        cfg = _cfg()
        with tempfile.TemporaryDirectory() as tmp:
            path = save(cfg, Path(tmp) / "octa.json")
            back = load(path)
        self.assertTrue(np.array_equal(back.points, cfg.points))
        self.assertEqual(back.volume, cfg.volume)
        self.assertEqual(back.label, cfg.label)
        self.assertEqual(back.source, cfg.source)
        self.assertEqual(back.metadata, cfg.metadata)
        self.assertEqual(back.complex.facets, cfg.complex.facets)

    def test_shape_mismatch_rejected(self):
        cx = bipyramid(4)
        with self.assertRaises(ValueError):
            Configuration(points=np.zeros((5, 3)), complex=cx, volume=0.0)

    def test_save_and_load_many(self):
        cfgs = [_cfg(), _cfg()]
        with tempfile.TemporaryDirectory() as tmp:
            save_many(cfgs, tmp)
            back = load_many(tmp)
        self.assertEqual(len(back), 2)


def _case(rival_signed: float, winner: float = 2.0) -> CaseResult:
    types = (
        TypeEntry("win", (4, 4, 4, 4), True, winner, winner, 0, 20),
        TypeEntry("rival", (5, 5, 5, 5), True, rival_signed, rival_signed, 0, 20),
    )
    triage = {
        "target": winner,
        "excluded": int(rival_signed < winner),
        "hard": int(rival_signed >= winner),
        "hard_labels": ["rival"] if rival_signed >= winner else [],
    }
    cx = simplex(3)
    return CaseResult(
        dimension=3, n=4, assumptions=(),
        enumeration={"source": "plantri", "total": 2, "complete": True},
        budget={"restarts": 20, "seed": 11},
        reference=None,
        winner={
            "label": "win", "volume": winner, "degrees": [3, 3, 3, 3],
            "points": np.eye(4, 3).tolist(),
            "facets": [list(f) for f in cx.facets],
        },
        triage=triage, types=types, provenance={},
    )


class TestCaseResult(unittest.TestCase):
    def test_round_trip(self):
        case = _case(rival_signed=1.5)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_case(case, Path(tmp) / "d3-n04.json")
            back = load_case(path)
        self.assertEqual(back.dimension, 3)
        self.assertEqual(back.n, 4)
        self.assertEqual(len(back.types), 2)
        self.assertEqual(back.types[1].signed_max, 1.5)
        self.assertEqual(back.winner["volume"], 2.0)
        self.assertEqual(back.triage["excluded"], 1)

    def test_winner_configuration_round_trip(self):
        case = _case(rival_signed=1.5)
        with tempfile.TemporaryDirectory() as tmp:
            back = load_case(save_case(case, Path(tmp) / "d3-n04.json"))
        cfg = back.winner_configuration()
        self.assertEqual(cfg.points.shape, (4, 3))
        self.assertEqual(cfg.label, "win")
        self.assertEqual(cfg.volume, 2.0)
        self.assertEqual(cfg.metadata["evidence"], "relaxation-excluded")
        self.assertEqual(len(cfg.complex.facets), len(simplex(3).facets))

    def test_evidence_e3_when_all_rivals_bounded(self):
        self.assertIs(_case(rival_signed=1.5).evidence, Evidence.E3)

    def test_evidence_e1_when_a_rival_is_hard(self):
        self.assertIs(_case(rival_signed=2.5).evidence, Evidence.E1)

    def test_evidence_e1_when_enumeration_incomplete(self):
        case = _case(rival_signed=1.5)
        incomplete = CaseResult(
            dimension=case.dimension, n=case.n, assumptions=case.assumptions,
            enumeration={"source": "plantri", "total": 2, "complete": False},
            budget=case.budget, reference=case.reference, winner=case.winner,
            triage=case.triage, types=case.types, provenance=case.provenance,
        )
        self.assertIs(incomplete.evidence, Evidence.E1)

    def test_finite_lower_bound_refused(self):
        class _R:
            label = "x"
            feasible = True
            volume = 1.0
            signed_best = 1.0
            rejected = 0
            restarts = 1

            class complex:
                @staticmethod
                def vertex_degrees():
                    return {0: 3}

        with self.assertRaises(ValueError):
            build_case(3, 4, [_R()], "plantri", {}, {}, None, lower_bound=0.5)


def _record():
    cx = simplex(3)
    pts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1],
                    [-0.577350269, -0.577350269, -0.577350269]], dtype=float)
    return build_search_record(
        3, 4, points=pts, complex=cx, volume=0.513200239280, convex=True,
        symmetry={"order": 24, "proper": 12, "improper": 12, "orbit_sizes": [4]},
        budget={"restarts": 80, "seed": 304, "fraction_at_best": 1.0},
        provenance={"numpy": "1.26.0"},
    )


class TestSearchRecord(unittest.TestCase):
    def test_round_trip(self):
        record = _record()
        with tempfile.TemporaryDirectory() as tmp:
            back = load_search(save_search(record, Path(tmp) / "d3-n04.json"))
        self.assertEqual(back, record)

    def test_evidence_is_always_heuristic(self):
        self.assertIs(_record().evidence, Evidence.E0)

    def test_configuration_reconstructs_the_polytope(self):
        cfg = _record().configuration()
        self.assertEqual(cfg.points.shape, (4, 3))
        self.assertEqual(cfg.volume, 0.513200239280)
        self.assertEqual(cfg.metadata["evidence"], "heuristic")
        self.assertEqual(len(cfg.complex.facets), len(simplex(3).facets))

    def test_points_must_match_the_complex(self):
        with self.assertRaises(ValueError):
            build_search_record(
                3, 4, points=np.zeros((5, 3)), complex=simplex(3), volume=1.0,
                convex=True, symmetry={}, budget={},
            )


if __name__ == "__main__":
    unittest.main()
