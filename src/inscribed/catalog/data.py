"""Registry of known maxima, keyed by (dimension, n).

The values quoted in the paper's tables; see papers/sweep/sections/results-3d.tex and
papers/sweep/sections/results-4d.tex.
"""

from __future__ import annotations

import math

from .records import Citation, KnownValue, Status

_BERMAN_HANES = Citation(
    "Berman--Hanes, Volumes of polyhedra inscribed in the unit sphere in E^3, "
    "Math. Ann. 188 (1970), 78-84"
)
_HOEHNER_LEDFORD = Citation(
    "Hoehner--Ledford, The maximum volume polytope with nine vertices inscribed "
    "in the sphere (2026)",
    "https://arxiv.org/abs/2608.04392",
)
_HORVATH_LANGI = Citation(
    "Horvath--Langi, Maximum volume polytopes inscribed in the unit sphere",
    "https://arxiv.org/abs/1402.6496",
)
_HORVATH_LANGI_NONCYCLIC = Citation(
    "Horvath--Langi, Maximum volume polytopes inscribed in the unit sphere; for "
    "n = d + 3 with d even they solve only the noncyclic case",
    "https://arxiv.org/abs/1402.6496",
)
_HORVATH_LANGI_CROSS = Citation(
    "Horvath--Langi, Maximum volume polytopes inscribed in the unit sphere; the "
    "maximal inscribed cross-polytope is the regular one",
    "https://arxiv.org/abs/1402.6496",
)
_CLASSICAL = Citation("classical; see e.g. Fejes Toth, Regular Figures")
_OEIS_10 = Citation("conjectured value", "https://oeis.org/A339262")
_OEIS_11 = Citation("conjectured value", "https://oeis.org/A339263")

KNOWN: dict[tuple[int, int], KnownValue] = {}


def _add(dimension: int, n: int, value: float, name: str, status: Status,
         citation: Citation, exact: str | None = None) -> None:
    KNOWN[(dimension, n)] = KnownValue(
        dimension=dimension, n=n, value=value, name=name,
        status=status, citation=citation, exact=exact,
    )


_add(3, 4, 8 * math.sqrt(3) / 27, "regular tetrahedron", Status.PROVEN, _CLASSICAL,
     "8 sqrt(3) / 27")
_add(3, 5, math.sqrt(3) / 2, "triangular bipyramid", Status.PROVEN, _BERMAN_HANES,
     "sqrt(3) / 2")
_add(3, 6, 4 / 3, "regular octahedron", Status.PROVEN, _CLASSICAL, "4/3")
_add(3, 7, (5 / 3) * math.sin(2 * math.pi / 5), "pentagonal bipyramid", Status.PROVEN,
     _BERMAN_HANES, "(5/3) sin(2 pi / 5)")
_add(3, 8, math.sqrt((475 + 29 * math.sqrt(145)) / 250), "Grace's polyhedron",
     Status.PROVEN, _BERMAN_HANES, "sqrt((475 + 29 sqrt(145)) / 250)")
_add(3, 9, 3 * math.sqrt(2 * math.sqrt(3) - 3), "triaugmented triangular prism (J51)",
     Status.PROVEN, _HOEHNER_LEDFORD, "3 sqrt(2 sqrt(3) - 3)")
_add(3, 10, 2.21871113154539940324, "two square rings plus poles", Status.CONJECTURAL,
     _OEIS_10, "16 (3 + sqrt(2) + sqrt(38 + 6 sqrt(2))) / 81")
_add(3, 11, 2.35463449506861520323, "medial type with one degree-6 vertex",
     Status.CONJECTURAL, _OEIS_11, None)
_add(3, 12, (2 / 3) * math.sqrt(10 + 2 * math.sqrt(5)), "regular icosahedron",
     Status.PROVEN, _CLASSICAL, "(2/3) sqrt(10 + 2 sqrt(5))")

_add(4, 5, math.sqrt(5) * (5 / 4) ** 2 / 24, "regular 4-simplex", Status.PROVEN,
     _CLASSICAL, "25 sqrt(5) / 384")
_add(4, 6, 9 / 32, "two orthogonal triangles", Status.PROVEN, _HORVATH_LANGI, "9/32")
_add(4, 7, math.sqrt(3) / 4, "triangle + square in orthogonal planes",
     Status.CLASS_PROVEN, _HORVATH_LANGI_NONCYCLIC, "sqrt(3) / 4")
_add(4, 8, 2 / 3, "16-cell (cross polytope)", Status.CLASS_PROVEN, _HORVATH_LANGI_CROSS,
     "2/3")
_add(4, 9, 5 * math.sqrt(10 + 2 * math.sqrt(5)) / 24, "pentagon + square in orthogonal planes",
     Status.CONJECTURAL, Citation("this repository, papers/sweep/sections/results-4d.tex"),
     "5 sqrt(10 + 2 sqrt(5)) / 24")
_add(4, 10, (125 + 25 * math.sqrt(5)) / 192, "two pentagons in orthogonal planes",
     Status.CONJECTURAL, Citation("this repository, papers/sweep/sections/results-4d.tex"),
     "(125 + 25 sqrt(5)) / 192")


def known(dimension: int, n: int) -> KnownValue | None:
    return KNOWN.get((dimension, n))
