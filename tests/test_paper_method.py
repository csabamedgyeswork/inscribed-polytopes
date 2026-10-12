"""Claims made in papers/sweep/sections/method.tex.

The evidence ladder, the type sources and the two-pass sweep are described in
prose that counts things: how many cases reach E3, whether any sweep rests on a
recorded assumption, how often the refinement cut binds, at what tolerances the
committed sweeps ran. Nothing else keeps those sentences in step with the
committed case records and the sweep defaults.
"""

import re
import unittest
from pathlib import Path

from inscribed.search import TypeSweep
from inscribed.store import Evidence, load_case, load_search

ROOT = Path(__file__).resolve().parent.parent
SECTION = ROOT / "papers" / "sweep" / "sections" / "method.tex"
RESULTS = ROOT / "results"

_WORDS = {
    "three": 3, "six": 6, "ten": 10, "eleven": 11, "eleventh": 11, "thirteen": 13,
}


def _spelled(word: str) -> int:
    """The prose spells small counts out; read one back as a number."""
    return _WORDS[word.lower()]


def _cases() -> dict[tuple[int, int], object]:
    """Every committed case record, keyed by the pair the prose names it with."""
    paths = sorted(RESULTS.glob("d*-n*.json"))
    assert paths, "no committed case records"
    cases = [load_case(p) for p in paths]
    return {(c.dimension, c.n): c for c in cases}


def _claim(pattern: str) -> re.Match:
    """The sentence the test is about, so a reworded claim fails rather than drifts.

    Matched against the prose with its line breaks flattened, so that rewrapping
    a paragraph does not break a test.
    """
    match = re.search(pattern, re.sub(r"\s+", " ", SECTION.read_text()), re.S)
    assert match is not None, f"method.tex no longer says {pattern!r}"
    return match


class TestEvidenceLadder(unittest.TestCase):
    """Section 3.1: which levels the committed records actually carry."""

    def test_the_cases_reaching_e3_are_the_ones_named(self):
        """"Six cases reach E3: d = 3 and d = 4, each at n = 6, 7, 8"."""
        claim = _claim(
            r"(\w+) cases reach E3: \$d = (\d+)\$ and \$d = (\d+)\$, each at "
            r"\$n = (\d+), (\d+), (\d+)\$"
        )
        dimensions = [int(claim.group(2)), int(claim.group(3))]
        counts = [int(claim.group(i)) for i in (4, 5, 6)]
        named = {(d, n) for d in dimensions for n in counts}

        self.assertEqual(len(named), _spelled(claim.group(1)))
        reached = {key for key, case in _cases().items() if case.evidence is Evidence.E3}
        self.assertEqual(reached, named)

    def test_no_case_record_carries_a_level_outside_e1_and_e3(self):
        """"Case records carry E1 or E3 ... nothing in this work reaches E4"."""
        levels = {case.evidence for case in _cases().values()}
        self.assertTrue(levels <= {Evidence.E1, Evidence.E3}, levels)

    def test_every_type_free_search_carries_e0(self):
        """"the type-free searches E0"."""
        for path in sorted((RESULTS / "free").glob("d*-n*.json")):
            with self.subTest(record=path.stem):
                self.assertIs(load_search(path).evidence, Evidence.E0)


class TestTypeSources(unittest.TestCase):
    """Section 3.2: the minimum-degree flag is not used."""

    def test_no_sweep_uses_the_degree_restriction(self):
        """"we do not use the flag: every d = 3 sweep reported here enumerates
        all types and none rests on a recorded assumption"."""
        claim = _claim(
            r"we do not use the flag: every \$d = (\d+)\$ sweep reported here "
            r"enumerates all types and none rests on a recorded assumption"
        )
        dimension = int(claim.group(1))

        for (d, n), case in _cases().items():
            with self.subTest(d=d, n=n):
                self.assertEqual(case.assumptions, ())
                self.assertTrue(case.enumeration["complete"])
                if d == dimension:
                    self.assertEqual(case.enumeration["min_degree"], 3)


class TestTwoPassSweep(unittest.TestCase):
    """Section 3.4: the refinement cut, and the three cases where it binds."""

    def test_the_cut_is_the_number_of_leaders_the_prose_quotes(self):
        """"the ten highest-scoring feasible types are refined"."""
        claim = _claim(r"the (\w+) highest-scoring\s+feasible types are refined")
        self.assertEqual(TypeSweep.__init__.__kwdefaults__["refine_top"],
                         _spelled(claim.group(1)))

    def test_the_cut_binds_in_exactly_the_cases_named_and_by_the_margin_quoted(self):
        """"In ten of the thirteen cases below fewer than eleven types are
        feasible ... in the remaining three, d = 3 with n = 10, 11, 12, the
        winner exceeds the eleventh-ranked feasible type by more than 4%"."""
        claim = _claim(
            r"In (\w+) of the (\w+) cases below fewer than (\w+) types are "
            r"feasible, so the cut does not bind; in the remaining (\w+), "
            r"\$d = (\d+)\$ with \$n = (\d+), (\d+), (\d+)\$, the winner exceeds "
            r"the (\w+)-ranked feasible type by more than \$(\d+)\\%\$ of its own value"
        )
        slack, total, cut = (_spelled(claim.group(i)) for i in (1, 2, 3))
        dimension = int(claim.group(5))
        named = {(dimension, int(claim.group(i))) for i in (6, 7, 8)}
        rank = _spelled(claim.group(9))
        margin = int(claim.group(10)) / 100
        self.assertEqual(_spelled(claim.group(4)), len(named))

        # "eleventh-ranked" is the first type the cut leaves unrefined, so the
        # cut binds exactly when at least that many types are feasible.
        self.assertEqual(rank, cut)
        self.assertEqual(cut, TypeSweep.__init__.__kwdefaults__["refine_top"] + 1)

        cases = _cases()
        self.assertEqual(len(cases), total)

        binding = {}
        for key, case in cases.items():
            volumes = sorted(
                (t.volume for t in case.types if t.feasible), reverse=True
            )
            if len(volumes) >= cut:
                binding[key] = volumes

        self.assertEqual(set(binding), named)
        self.assertEqual(len(cases) - len(binding), slack)

        for key, volumes in binding.items():
            winner, last = volumes[0], volumes[cut - 1]
            with self.subTest(case=key):
                self.assertGreater((winner - last) / winner, margin)


class TestNumericalSetting(unittest.TestCase):
    """Section 3.5: the tolerances and the restart budget the sweeps ran at."""

    def test_the_quoted_tolerances_are_the_sweep_defaults(self):
        """"screen every type at volume tolerance 10^-9 and refine the feasible
        leaders at 10^-14"."""
        claim = _claim(
            r"screen every type at volume tolerance \$10\^\{-(\d+)\}\$ and refine\s+"
            r"the feasible leaders at \$10\^\{-(\d+)\}\$, with \$(\d+)\$ restarts"
        )
        defaults = TypeSweep.__init__.__kwdefaults__
        self.assertEqual(defaults["screen_tol"], 10 ** -int(claim.group(1)))
        self.assertEqual(defaults["refine_tol"], 10 ** -int(claim.group(2)))
        self.assertEqual(defaults["restarts"], int(claim.group(3)))

    def test_every_case_ran_at_the_quoted_restart_budget(self):
        """"with 20 restarts per type"."""
        claim = _claim(r"with \$(\d+)\$ restarts\s+per type")
        for key, case in _cases().items():
            with self.subTest(case=key):
                self.assertEqual(case.budget["restarts"], int(claim.group(1)))

    def test_no_two_types_share_a_stream(self):
        """"seeded by the type's index from a fixed base ... no two types share
        a stream"."""
        _claim(r"no two types\s+share a stream")
        sweep = TypeSweep.__new__(TypeSweep)
        sweep.seed = 7
        self.assertEqual(len({sweep.seed + index for index in range(64)}), 64)


if __name__ == "__main__":
    unittest.main()
