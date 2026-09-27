"""Combinatorial types of simplicial spheres: complexes, builders, mediality."""

from .builders import bipyramid, simplex
from .complex import (
    ClosureError,
    CombinatoricsError,
    Facet,
    OrientationError,
    OrientedComplex,
    TopologyError,
    permutation_parity,
)
from .medial import MEDIAL_TYPE_COUNTS, is_medial, medial_degrees, medial_verdict
from .orientation import orient, parse_facets

__all__ = [
    "MEDIAL_TYPE_COUNTS",
    "ClosureError",
    "CombinatoricsError",
    "Facet",
    "OrientationError",
    "OrientedComplex",
    "TopologyError",
    "bipyramid",
    "is_medial",
    "medial_degrees",
    "medial_verdict",
    "orient",
    "parse_facets",
    "permutation_parity",
    "simplex",
]
