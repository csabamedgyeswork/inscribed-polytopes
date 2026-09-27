"""Reference families and provenance-aware records of known maxima."""

from .data import KNOWN, known
from .families import (
    Bipyramid,
    ConfigurationFamily,
    CrossPolytope,
    OrthogonalPolygonPair,
    RegularSimplex,
    TriaugmentedPrism,
)
from .records import Citation, KnownValue, Status

__all__ = [
    "Bipyramid",
    "Citation",
    "ConfigurationFamily",
    "CrossPolytope",
    "KNOWN",
    "KnownValue",
    "OrthogonalPolygonPair",
    "RegularSimplex",
    "Status",
    "TriaugmentedPrism",
    "known",
]
