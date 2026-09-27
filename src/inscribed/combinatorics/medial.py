"""Medial triangulations (Goldberg-Grace).

Exploratory: not used by the paper.
"""

from __future__ import annotations

from math import ceil, floor

from .complex import OrientedComplex

MEDIAL_TYPE_COUNTS = {
    4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 0,
    12: 1, 13: 0, 14: 1, 15: 1, 16: 2, 17: 3, 18: 6,
    19: 6, 20: 15, 21: 17, 22: 40,
}
"""Medial types per n, by exhaustive plantri enumeration; 0 means vacuous."""


def medial_degrees(n: int) -> tuple[int, int]:
    """The vertex degrees a medial triangulation on ``n`` vertices may use.

    The average degree is ``6 - 12/n`` by Euler; medial means every degree is
    one of the two integers bracketing it.
    """
    if n < 4:
        raise ValueError("need n >= 4")
    average = 6 - 12 / n
    return floor(average), ceil(average)


def is_medial(cx: OrientedComplex) -> bool:
    """Whether every vertex degree of ``cx`` is admissible for a medial triangulation."""
    if cx.d != 3:
        raise ValueError("mediality is only defined for d == 3")
    low, high = medial_degrees(cx.n)
    return all(low <= deg <= high for deg in cx.vertex_degrees().values())


def medial_verdict(n: int, degrees) -> str:
    """Classify a degree sequence against the conjecture at ``n``."""
    count = MEDIAL_TYPE_COUNTS.get(n)
    if count == 0:
        return "vacuous"
    low, high = medial_degrees(n)
    if all(low <= deg <= high for deg in degrees):
        return "medial"
    return "not medial" if count else "not medial (existence unknown)"
