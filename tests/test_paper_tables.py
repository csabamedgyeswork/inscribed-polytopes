"""The LaTeX tables and quoted figures of the two results sections.

Every table in papers/sweep/sections/results-3d.tex and results-4d.tex is transcribed
by hand, so nothing else keeps the printed volumes, closed forms, statuses,
triage counts and margins in step with the committed sweeps.
"""

import math
import re
import subprocess
import unittest
from itertools import combinations, permutations
from pathlib import Path

import numpy as np

from inscribed.analysis import ConvexPosition, SymmetryDetector
from inscribed.catalog import OrthogonalPolygonPair, RegularSimplex, TriaugmentedPrism, known
from inscribed.combinatorics import OrientedComplex
from inscribed.enumeration import FirschingSource
from inscribed.enumeration.base import PLANTRI
from inscribed.geometry import oriented_hull
from inscribed.store import load_case, load_search
from inscribed.store.paths import case_path, search_path

ROOT = Path(__file__).resolve().parent.parent
SECTION_3D = ROOT / "papers" / "sweep" / "sections" / "results-3d.tex"
SECTION_4D = ROOT / "papers" / "sweep" / "sections" / "results-4d.tex"

AGREEMENT = 3e-15
"""The bound the 3D section claims: "Every row agrees to at most 3 x 10^-15"."""

_MATH_NAMES = {"sqrt": math.sqrt, "sin": math.sin, "pi": math.pi}


def _tabular_rows(label: str, *, path: Path, body_only: bool = True) -> list[list[str]]:
    """Cells of the rows of the tabular containing ``label``, header optional."""
    text = path.read_text()
    bodies = re.findall(r"\\begin\{tabular\}(.*?)\\end\{tabular\}", text, re.S)
    block = next(b for b in bodies if label in b)
    block = block.split(r"\toprule", 1)[-1].split(r"\bottomrule")[0]
    if body_only:
        block = block.split(r"\midrule", 1)[-1]
    else:
        block = block.replace(r"\midrule", "")
    return [
        [cell.strip() for cell in row.split("&")]
        for row in block.split(r"\\")
        if row.strip()
    ]


def _evaluate_latex(form: str) -> float:
    """Evaluate a closed form as printed, restoring the multiplications TeX omits."""
    expr = re.sub(r"\\(bigl|bigr|left|right)", "", form).replace("$", "")
    while r"\sqrt{" in expr:
        expr = re.sub(r"\\sqrt\{([^{}]*)\}", r"sqrt(\1)", expr)
    expr = expr.replace(r"\pi", "pi").replace(r"\sin", "sin")
    expr = re.sub(r"\s+", " ", expr).strip()
    expr = re.sub(r"(?<=[\d)])\s*(?=[a-z(])", "*", expr)
    return eval(expr, {"__builtins__": {}}, _MATH_NAMES)


def _power_of_ten(preceded_by: str, text: str) -> float:
    """The tolerance the prose quotes after ``preceded_by``, as a number."""
    match = re.search(
        preceded_by + r"\s+\$(\d+)\s*\\times\s*10\^\{-(\d+)\}\$", text
    )
    assert match is not None, f"no tolerance quoted after {preceded_by!r}"
    return int(match.group(1)) * 10 ** -int(match.group(2))


def _isomorphisms(a: OrientedComplex, b: OrientedComplex):
    """Vertex maps ``perm`` (a's vertex i to b's vertex perm[i]) carrying the
    facets of ``a`` onto those of ``b``."""
    if (a.n, a.d) != (b.n, b.d):
        return
    deg_a, deg_b = a.vertex_degrees(), b.vertex_degrees()
    faces_a = [frozenset(f) for f in a.facets]
    faces_b = {frozenset(f) for f in b.facets}
    for perm in permutations(range(a.n)):
        if any(deg_a[i] != deg_b[perm[i]] for i in range(a.n)):
            continue
        if {frozenset(perm[v] for v in f) for f in faces_a} == faces_b:
            yield perm


def _is_isomorphic(a: OrientedComplex, b: OrientedComplex) -> bool:
    """Whether two complexes are the same simplicial complex up to relabelling."""
    return next(_isomorphisms(a, b), None) is not None


def _congruence_error(a: np.ndarray, b: np.ndarray, maps) -> float:
    """Smallest max-coordinate distance between ``a`` and an orthogonal image of
    ``b``, over the given vertex correspondences."""
    best = math.inf
    for perm in maps:
        target = b[list(perm)]
        u, _, vt = np.linalg.svd(a.T @ target)
        best = min(best, float(np.abs(a @ (u @ vt) - target).max()))
    return best


def _plantri_count(*flags: str, n: int) -> int:
    """The number of graphs plantri reports generating, without writing them."""
    run = subprocess.run(
        [str(PLANTRI), *flags, "-u", str(n)], capture_output=True, text=True, check=True
    )
    match = re.search(r"(\d+) \w+ generated", run.stdout + run.stderr)
    assert match is not None, "plantri reported no count"
    return int(match.group(1))


def _cyclic_polytope(n: int, d: int) -> OrientedComplex:
    """The combinatorial type of C_d(n), as the hull of n points on the moment
    curve t -> (t, t^2, ..., t^d). Any n distinct parameters give the same type.
    """
    t = np.arange(1, n + 1, dtype=float)
    return oriented_hull(np.stack([t ** (k + 1) for k in range(d)], axis=1))


def _volume_rows():
    """(n, printed value, printed closed form, printed status) for each table row."""
    for cells in _tabular_rows("closed form", path=SECTION_3D):
        yield int(cells[0]), cells[1], cells[2], cells[3]


class TestVolumeTable(unittest.TestCase):
    """"Table 1 lists the maximum found for 4 <= n <= 12 against the published value"."""

    def test_table_covers_the_stated_range(self):
        self.assertEqual([n for n, _, _, _ in _volume_rows()], list(range(4, 13)))

    def test_found_values_are_the_committed_sweep_results(self):
        for n, printed, _, _ in _volume_rows():
            with self.subTest(n=n):
                case = load_case(case_path(3, n))
                self.assertEqual(printed, f"{case.winner['volume']:.12f}")

    def test_rows_agree_with_the_published_values_to_the_stated_bound(self):
        """The bound is on the computed maximum; the table prints it to 12 places."""
        self.assertIn(r"3 \times 10^{-15}", SECTION_3D.read_text())
        for n, _, _, _ in _volume_rows():
            with self.subTest(n=n):
                found = load_case(case_path(3, n)).winner["volume"]
                self.assertLessEqual(abs(found - known(3, n).value), AGREEMENT)

    def test_status_column_matches_the_catalog(self):
        for n, _, _, status in _volume_rows():
            with self.subTest(n=n):
                self.assertEqual(status, known(3, n).status.value)

    def test_closed_forms_evaluate_to_the_published_values(self):
        for n, _, form, _ in _volume_rows():
            with self.subTest(n=n):
                if known(3, n).exact is None:
                    self.assertEqual(form, "---")
                    continue
                self.assertNotEqual(form, "---")
                self.assertAlmostEqual(
                    _evaluate_latex(form), known(3, n).value, places=12
                )

    def test_every_winner_certifies_as_the_type_recorded_for_it(self):
        """"every optimum verifies as a convex-position configuration whose
        combinatorial type matches the one recorded for it"."""
        for n, _, _, _ in _volume_rows():
            with self.subTest(n=n):
                cfg = load_case(case_path(3, n)).winner_configuration()
                report = ConvexPosition(cfg.complex).check(cfg.points)
                self.assertTrue(report.ok, report.message)
                self.assertTrue(report.metrics["type_matches"])


class TestNineVertexParagraph(unittest.TestCase):
    """"plantri generates exactly those 50. Maximising within each gives the
    triaugmented triangular prism"."""

    def test_plantri_generates_the_number_of_types_claimed(self):
        claimed = re.search(r"\$(\d+)\$ simplicial ones", SECTION_3D.read_text())
        self.assertIsNotNone(claimed)
        self.assertEqual(int(claimed.group(1)), len(load_case(case_path(3, 9)).types))

    def test_the_winner_is_the_triaugmented_prism(self):
        winner = load_case(case_path(3, 9)).winner_configuration()
        prism = TriaugmentedPrism()
        self.assertAlmostEqual(winner.volume, prism.closed_form_volume(), places=12)
        self.assertTrue(_is_isomorphic(prism.complex(), winner.complex))

    def test_plantri_reproduces_the_hoehner_ledford_reduction(self):
        """"reduce the 2606 combinatorial types of nine-vertex polyhedra to the
        50 simplicial ones"."""
        flat = re.sub(r"\s+", " ", SECTION_3D.read_text())
        claim = re.search(
            r"reduce the \$(\d+)\$ combinatorial types of nine-vertex polyhedra to "
            r"the \$(\d+)\$ simplicial ones",
            flat,
        )
        self.assertIsNotNone(claim)
        self.assertEqual(_plantri_count("-p", n=9), int(claim.group(1)))
        self.assertEqual(_plantri_count(n=9), int(claim.group(2)))

    def test_the_figure_shows_the_swept_winner(self):
        """"the type is plantri-9-49, the triaugmented triangular prism"."""
        printed = re.search(r"the type is \\texttt\{([\w-]+)\}", SECTION_3D.read_text())
        self.assertIsNotNone(printed)
        self.assertEqual(printed.group(1), load_case(case_path(3, 9)).winner["label"])

    def test_the_reconstruction_has_the_group_described(self):
        """"a group of order 12, six proper and six improper elements with orbits
        of sizes 3 and 6"."""
        report = SymmetryDetector(tol=1e-7).analyze(TriaugmentedPrism().build())
        self.assertEqual((report.order, report.proper, report.improper), (12, 6, 6))
        self.assertEqual(report.orbit_sizes, [3, 6])

    def test_the_group_is_read_from_the_reconstruction_for_the_stated_reason(self):
        """"the swept optimum ... resolves only an order-2 subgroup" at 1e-7."""
        winner = load_case(case_path(3, 9)).winner_configuration()
        detector = SymmetryDetector(tol=1e-7)
        self.assertEqual(detector.analyze(TriaugmentedPrism().build()).order, 12)
        self.assertEqual(detector.analyze(winner.points).order, 2)

    def test_the_swept_optimum_is_congruent_to_the_reconstruction_at_the_stated_scale(self):
        """"the swept optimum is congruent to that configuration only to about
        5 x 10^-8", read as agreeing with the measured distance within a factor 2."""
        flat = re.sub(r"\s+", " ", SECTION_3D.read_text())
        printed = re.search(
            r"congruent to that configuration only to about "
            r"\$(?:(\d+)\s*\\times\s*)?10\^\{-(\d+)\}\$",
            flat,
        )
        self.assertIsNotNone(printed)
        scale = int(printed.group(1) or 1) * 10 ** -int(printed.group(2))

        winner = load_case(case_path(3, 9)).winner_configuration()
        prism = TriaugmentedPrism()
        error = _congruence_error(
            prism.build(),
            np.asarray(winner.points),
            _isomorphisms(prism.complex(), winner.complex),
        )
        self.assertLessEqual(scale / 2, error)
        self.assertLessEqual(error, 2 * scale)


class TestTwelveVertexParagraph(unittest.TestCase):
    """"The n = 12 sweep covers all 7595 types, including those with a degree-3
    vertex ... Its 31 feasible types include 9 with a degree-3 vertex, the
    largest of them at 2.370904"."""

    def _claim(self) -> re.Match:
        flat = re.sub(r"\s+", " ", SECTION_3D.read_text())
        match = re.search(
            r"The \$n = (\d+)\$ sweep covers all \$(\d+)\$ types, including those "
            r"with a degree-(\d+) vertex.*?Its \$(\d+)\$ feasible types include "
            r"\$(\d+)\$ with a degree-(\d+) vertex, the largest of them at "
            r"\$([\d.]+)\$, so restricting to minimum degree \$(\d+)\$",
            flat,
        )
        self.assertIsNotNone(match, "results-3d.tex no longer makes the n = 12 claim")
        return match

    def test_the_sweep_returns_the_icosahedron(self):
        """"and returns the icosahedron": twelve degree-5 vertices and the full
        icosahedral group, not only the icosahedron's volume."""
        case = load_case(case_path(3, 12))
        winner = case.winner_configuration()
        self.assertEqual(sorted(winner.complex.vertex_degrees().values()), [5] * 12)
        report = SymmetryDetector(tol=1e-6).analyze(winner.points)
        self.assertEqual((report.order, report.proper), (120, 60))

    def test_the_sweep_covers_every_type_including_the_trivalent_ones(self):
        claim = self._claim()
        case = load_case(case_path(3, int(claim.group(1))))
        self.assertEqual(len(case.types), int(claim.group(2)))
        self.assertTrue(case.enumeration["complete"])
        self.assertEqual(case.enumeration["min_degree"], int(claim.group(3)))

    def test_the_trivalent_feasible_types_are_counted_and_bounded_as_printed(self):
        claim = self._claim()
        case = load_case(case_path(3, int(claim.group(1))))
        feasible = [t for t in case.types if t.feasible]
        self.assertEqual(len(feasible), int(claim.group(4)))

        # Degrees are stored sorted, so the first entry is the minimum degree.
        discarded = [t for t in feasible if t.degrees[0] < int(claim.group(8))]
        self.assertEqual(len(discarded), int(claim.group(5)))
        self.assertTrue(all(t.degrees[0] == int(claim.group(6)) for t in discarded))

        printed = claim.group(7)
        places = len(printed.split(".")[1])
        self.assertEqual(f"{max(t.volume for t in discarded):.{places}f}", printed)


class TestTriageTable(unittest.TestCase):
    """The rivals / hard / level table of Section 5.2."""

    def _row(self, name: str) -> list[str]:
        rows = _tabular_rows("rivals", path=SECTION_3D, body_only=False)
        return next(cells for cells in rows if cells[0] == name)[1:]

    def test_columns_are_the_same_range(self):
        self.assertEqual([int(x) for x in self._row("$n$")], list(range(4, 13)))

    def test_rivals_hard_and_level_match_the_case_files(self):
        rivals, hard, level = (self._row(k) for k in ("rivals", "hard", "level"))
        for i, n in enumerate(range(4, 13)):
            with self.subTest(n=n):
                case = load_case(case_path(3, n))
                self.assertEqual(int(rivals[i]), len(case.types) - 1)
                self.assertEqual(int(hard[i]), case.triage["hard"])
                self.assertEqual(level[i], case.evidence.name)

    def test_the_heptagonal_bipyramid_is_one_of_the_three_hard_types_at_n9(self):
        """"the heptagonal bipyramid of Section 3 is one of the three hard types there"."""
        hard = load_case(case_path(3, 9)).triage["hard_labels"]
        self.assertEqual(len(hard), 3)
        self.assertIn("plantri-9-24", hard)


class TestFourDimensionalSweepTable(unittest.TestCase):
    """The types / winner / level table of Section 6.1."""

    def _row(self, name: str) -> list[str]:
        rows = _tabular_rows("winner", path=SECTION_4D, body_only=False)
        return next(cells for cells in rows if cells[0].strip() == name)[1:]

    def test_the_table_covers_the_exhaustively_enumerated_range(self):
        """Firsching's enumeration is complete for n <= 8, and starts at n = d+1."""
        self.assertEqual([int(x) for x in self._row("$n$")], list(range(5, 9)))

    def test_type_counts_winners_and_levels_match_the_case_files(self):
        types, winners, levels = (self._row(k) for k in ("types", "winner", "level"))
        for i, n in enumerate(range(5, 9)):
            with self.subTest(n=n):
                case = load_case(case_path(4, n))
                self.assertEqual(int(types[i]), len(case.types))
                self.assertAlmostEqual(
                    _evaluate_latex(winners[i]), case.winner["volume"], places=12
                )
                self.assertEqual(levels[i], case.evidence.name)

    def test_the_n5_winner_is_the_regular_simplex(self):
        """"which is why n = 5 falls outside the family and is the regular simplex
        instead": equal pairwise angles, not only the simplex's volume."""
        points = np.asarray(load_case(case_path(4, 5)).winner_configuration().points)
        gram = points @ points.T
        np.testing.assert_allclose(gram[~np.eye(5, dtype=bool)], -1 / 4, atol=1e-6)
        self.assertAlmostEqual(
            load_case(case_path(4, 5)).winner["volume"],
            RegularSimplex().closed_form_volume(d=4),
            places=12,
        )

    def test_the_n7_winner_is_printed_as_found_and_beats_the_cyclic_type(self):
        """"the winner over all five types is sqrt(3)/4 = 0.433012701892", and the
        Problem 2 reading needs the cyclic maximum to fall below it."""
        printed = re.search(r"\\sqrt\{3\}/4 = ([\d.]+)\$", SECTION_4D.read_text())
        self.assertIsNotNone(printed)
        case = load_case(case_path(4, 7))
        self.assertEqual(printed.group(1), f"{case.winner['volume']:.12f}")
        cyclic = [t for t in case.types if set(t.degrees) == {6}]
        self.assertEqual(len(cyclic), 1)
        self.assertLess(cyclic[0].volume, case.winner["volume"])

    def test_the_type_called_cyclic_at_n7_is_the_cyclic_polytope(self):
        """"The cyclic type C_4(7), degree sequence [6]^7". The degree sequence
        picks out one type of the five; naming it cyclic is a further claim, and
        the Problem 1 and Problem 2 readings that follow depend on it."""
        printed = re.search(
            r"cyclic type \$C_\{(\d+)\}\((\d+)\)\$, degree\s+sequence "
            r"\$\[(\d+)\]\^\{(\d+)\}\$",
            SECTION_4D.read_text(),
        )
        self.assertIsNotNone(printed)
        d, n, degree, count = (int(printed.group(i)) for i in (1, 2, 3, 4))
        self.assertEqual(count, n)

        entries = [t for t in load_case(case_path(d, n)).types if set(t.degrees) == {degree}]
        self.assertEqual(len(entries), 1)
        self.assertEqual(len(entries[0].degrees), count)

        types = {record.label: record.complex for record in FirschingSource().types(n)}
        self.assertTrue(
            _is_isomorphic(types[entries[0].label], _cyclic_polytope(n, d))
        )

    def test_the_cyclic_type_at_n7_agrees_with_the_published_closed_form(self):
        """"The cyclic type C_4(7), degree sequence [6]^7, maximises at ...,
        agreeing with the Horvath--Langi closed form ... to 6 x 10^-17"."""
        text = SECTION_4D.read_text()
        printed = re.search(r"maximises at \$([\d.]+)\$", text)
        self.assertIsNotNone(printed)

        cyclic = [t for t in load_case(case_path(4, 7)).types if set(t.degrees) == {6}]
        self.assertEqual(len(cyclic), 1)
        closed_form = (49 / 192) * (math.cos(math.pi / 7) + math.cos(2 * math.pi / 7))
        self.assertEqual(printed.group(1), f"{cyclic[0].volume:.12f}")
        self.assertLessEqual(
            abs(cyclic[0].volume - closed_form), _power_of_ten(r"to", text)
        )

    def test_the_two_margins_at_n8_are_both_shares_of_the_winner(self):
        """"ahead of ... by 20.0% of its own value" and "a 9.81% margin"."""
        case = load_case(case_path(4, 8))
        winner = case.winner["volume"]
        rivals = [t for t in case.types if t.label != case.winner["label"]]
        convex = max((t for t in rivals if t.feasible), key=lambda t: t.volume)
        signed = max(rivals, key=lambda t: t.signed_max)

        full = SECTION_4D.read_text()
        start = full.index("At $n = 8$")
        text = full[start:full.index(r"\subsection", start)]
        quoted = re.findall(r"\$([\d.]+)\$\s*\(\\texttt\{([\w-]+)\}\)", text)
        margins = re.findall(r"\$([\d.]+)\\%\$", text)
        self.assertEqual(len(quoted), 2)
        self.assertEqual(len(margins), 2)

        for (printed, label), rival, value, margin in zip(
            quoted, (convex, signed), (convex.volume, signed.signed_max), margins
        ):
            with self.subTest(label=label):
                self.assertEqual(label, rival.label)
                self.assertEqual(printed, f"{value:.6f}")
                self.assertAlmostEqual(
                    float(margin), 100 * (winner - value) / winner, places=1
                )


class TestPolygonPairFamily(unittest.TestCase):
    """Section 6.2: Table 2 and its agreement with the n <= 8 sweeps."""

    FAMILY = OrthogonalPolygonPair()

    def _best_split(self, n: int) -> tuple[float, tuple[int, int]]:
        """The family's largest volume over the splits of n into two polygons."""
        return max(
            (self.FAMILY.closed_form_volume(m=m, k=n - m), (m, n - m))
            for m in range(3, n - 2)
        )

    def _rows(self):
        """(n, split, exact volume, numeric, status) for each row of Table 2."""
        for cells in _tabular_rows("exact volume", path=SECTION_4D):
            split = tuple(int(x) for x in re.findall(r"\d+", cells[1]))
            yield int(cells[0]), split, cells[2], cells[3], cells[4]

    def test_table_rows_are_the_closed_form_of_their_split(self):
        """Equation (6.1) evaluated at the split, printed both ways."""
        for n, split, form, numeric, _ in self._rows():
            with self.subTest(n=n):
                self.assertEqual(sum(split), n)
                volume = self.FAMILY.closed_form_volume(m=split[0], k=split[1])
                self.assertAlmostEqual(_evaluate_latex(form), volume, places=12)
                self.assertEqual(numeric, f"{volume:.12f}")

    def test_table_rows_are_the_best_split_and_carry_the_catalog_status(self):
        rows = list(self._rows())
        self.assertEqual([n for n, *_ in rows], list(range(6, 9)))
        for n, split, _, _, status in rows:
            with self.subTest(n=n):
                best, argmax = self._best_split(n)
                self.assertEqual(sorted(split), sorted(argmax))
                self.assertAlmostEqual(
                    self.FAMILY.closed_form_volume(m=split[0], k=split[1]),
                    best,
                    places=15,
                )
                self.assertEqual(status, known(4, n).status.value)

    def test_table_rows_equal_the_sweep_winners(self):
        """"Each row ... equals the sweep winner above"."""
        for n, split, _, _, _ in self._rows():
            with self.subTest(n=n):
                winner = load_case(case_path(4, n)).winner["volume"]
                self.assertAlmostEqual(
                    winner, self.FAMILY.closed_form_volume(m=split[0], k=split[1]),
                    places=12,
                )

    def test_sweep_winners_are_regular_polygons_in_orthogonal_planes(self):
        """A matching volume does not fix the shape, so check the geometry: the
        winner splits into the row's two regular polygons, each spanning a
        2-plane, with every cross pair orthogonal."""
        tol = 1e-6
        for n, split, _, _, _ in self._rows():
            with self.subTest(n=n):
                points = np.asarray(load_case(case_path(4, n)).winner_configuration().points)
                gram = points @ points.T
                m = min(split)
                parts = next(
                    (list(s), [i for i in range(n) if i not in s])
                    for s in combinations(range(n), m)
                    if np.abs(gram[np.ix_(s, [i for i in range(n) if i not in s])]).max() < tol
                )
                self.assertEqual(sorted(map(len, parts)), sorted(split))
                for part in parts:
                    ring = points[part]
                    self.assertEqual(np.linalg.matrix_rank(ring, tol), 2)
                    side = math.sqrt(2 - 2 * math.cos(2 * math.pi / len(part)))
                    nearest = np.sort(
                        np.linalg.norm(ring[:, None] - ring[None], axis=2), axis=1
                    )[:, 1]
                    np.testing.assert_allclose(nearest, side, atol=tol)

    def test_free_search_matches_every_row_to_the_stated_bound(self):
        """"Free search matches every row to 10^-15"."""
        self.assertIn("$10^{-15}$", SECTION_4D.read_text())
        for n, split, _, _, _ in self._rows():
            with self.subTest(n=n):
                record = load_search(search_path(4, n))
                self.assertTrue(record.convex)
                self.assertLessEqual(
                    abs(
                        record.volume
                        - self.FAMILY.closed_form_volume(m=split[0], k=split[1])
                    ),
                    1e-15,
                )

    def _beyond_the_sweeps(self) -> re.Match:
        flat = re.sub(r"\s+", " ", SECTION_4D.read_text())
        claim = re.search(
            r"matches the best split at \$n = (\d+)\$ and \$n = (\d+)\$ to \$10\^\{-(\d+)\}\$, "
            r"but at \$n = (\d+), (\d+), (\d+)\$ .*?whose volumes \$([\d.]+)\$, \$([\d.]+)\$ "
            r"and \$([\d.]+)\$ exceed the best split by \$([\d.]+)\\%\$, \$([\d.]+)\\%\$ and "
            r"\$([\d.]+)\\%\$",
            flat,
        )
        self.assertIsNotNone(claim, "results-4d.tex no longer makes the n > 8 claim")
        return claim

    def test_free_search_ties_the_family_where_stated(self):
        """"matches the best split at n = 9 and n = 10 to 10^-15"."""
        claim = self._beyond_the_sweeps()
        bound = 10 ** -int(claim.group(3))
        for n in (int(claim.group(1)), int(claim.group(2))):
            with self.subTest(n=n):
                best, _ = self._best_split(n)
                self.assertLessEqual(abs(load_search(search_path(4, n)).volume - best), bound)

    def test_free_search_beats_the_family_by_the_printed_margins(self):
        """"at n = 11, 12, 13 it finds inscribed polytopes with all n points as
        vertices whose volumes ... exceed the best split by ..."."""
        claim = self._beyond_the_sweeps()
        for i in range(3):
            n = int(claim.group(4 + i))
            with self.subTest(n=n):
                record = load_search(search_path(4, n))
                points = np.asarray(record.points)
                np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1, atol=1e-12)
                self.assertTrue(ConvexPosition().check(points).ok)
                self.assertEqual(claim.group(7 + i), f"{record.volume:.6f}")
                best, _ = self._best_split(n)
                self.assertEqual(claim.group(10 + i), f"{100 * (record.volume - best) / best:.2f}")


if __name__ == "__main__":
    unittest.main()
