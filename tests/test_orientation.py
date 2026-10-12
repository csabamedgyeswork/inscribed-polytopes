"""orient() turns an abstract facet list into a coherent sphere."""

import unittest

from inscribed.combinatorics import (
    ClosureError,
    orient,
    parse_facets,
)
from inscribed.enumeration import PlantriSource


def _sorted_facets(cx):
    return sorted(tuple(sorted(f)) for f in cx.facets)


class TestOrient(unittest.TestCase):
    def test_sorted_octahedron(self):
        facets = [(0, 2, 4), (0, 2, 5), (0, 3, 4), (0, 3, 5),
                  (1, 2, 4), (1, 2, 5), (1, 3, 4), (1, 3, 5)]
        cx = orient(facets)
        self.assertTrue(cx.is_valid())
        self.assertEqual((cx.d, cx.n), (3, 6))

    def test_open_complex_rejected(self):
        with self.assertRaises(ClosureError):
            orient([(0, 1, 2)])

    def test_repeated_vertex_rejected(self):
        with self.assertRaises(ClosureError):
            orient([(0, 0, 1), (0, 1, 2)])

    def test_matches_plantri_up_to_global_sign(self):
        # The strongest check: re-orienting the sorted facets of every plantri
        # type at n = 8 must recover that type, up to the undecidable global sign.
        for record in PlantriSource().types(8):
            cx = record.complex
            recovered = orient(_sorted_facets(cx))
            same = _sorted_facets(recovered) == _sorted_facets(cx)
            flipped = _sorted_facets(recovered) == _sorted_facets(cx.reversed_orientation())
            self.assertTrue(same or flipped, f"mismatch on {record.label}")


class TestParseFacets(unittest.TestCase):
    def test_json(self):
        self.assertEqual(parse_facets("[[0,1,2],[0,1,3]]"), [[0, 1, 2], [0, 1, 3]])

    def test_lines(self):
        self.assertEqual(parse_facets("0 1 2\n0 1 3\n"), [[0, 1, 2], [0, 1, 3]])

    def test_comments_and_blanks_ignored(self):
        self.assertEqual(parse_facets("# comment\n\n0 1 2\n"), [[0, 1, 2]])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            parse_facets("   \n  ")


if __name__ == "__main__":
    unittest.main()
