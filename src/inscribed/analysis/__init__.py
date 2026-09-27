"""Post-hoc analysis: validation, symmetry, gauge fixing, variable elimination."""

from .gauge import GaugeFrame, free_parameter_count, gauge_dimension
from .independent_sets import (
    IndependentSetElimination,
    max_independent_set,
    maximal_independent_sets,
)
from .planes import orthogonal_plane_split
from .symmetry import SymmetryDetector, SymmetryReport
from .validation import (
    Composite,
    ConvexPosition,
    Gauge,
    PropertyZ,
    ValidationReport,
    Validator,
)

__all__ = [
    "Composite",
    "ConvexPosition",
    "Gauge",
    "GaugeFrame",
    "IndependentSetElimination",
    "PropertyZ",
    "SymmetryDetector",
    "SymmetryReport",
    "ValidationReport",
    "Validator",
    "free_parameter_count",
    "gauge_dimension",
    "max_independent_set",
    "maximal_independent_sets",
    "orthogonal_plane_split",
]
