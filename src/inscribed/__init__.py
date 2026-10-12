"""Maximum-volume polytopes inscribed in the unit sphere, in any dimension.

See papers/ for the mathematics and README.md for the commands.
"""

from .analysis import (
    ConvexPosition,
    GaugeFrame,
    IndependentSetElimination,
    SymmetryDetector,
    SymmetryReport,
    free_parameter_count,
    orthogonal_plane_split,
)
from .combinatorics import (
    CombinatoricsError,
    OrientedComplex,
    bipyramid,
    is_medial,
    medial_degrees,
    simplex,
)
from .geometry import hull_volume
from .optimality import OptimalityConditions
from .volume import VolumeFunctional

__all__ = [
    "CombinatoricsError",
    "ConvexPosition",
    "GaugeFrame",
    "IndependentSetElimination",
    "OptimalityConditions",
    "OrientedComplex",
    "SymmetryDetector",
    "SymmetryReport",
    "VolumeFunctional",
    "bipyramid",
    "free_parameter_count",
    "hull_volume",
    "is_medial",
    "medial_degrees",
    "orthogonal_plane_split",
    "simplex",
]
