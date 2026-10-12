"""Sources of combinatorial types to sweep.

See papers/sweep/sections/method.tex.
"""

from .base import DATA_DIR, PLANTRI, ROOT, ComplexSource, TypeRecord
from .firsching import FirschingSource
from .plantri import PlantriSource

__all__ = [
    "DATA_DIR",
    "PLANTRI",
    "ROOT",
    "ComplexSource",
    "FirschingSource",
    "PlantriSource",
    "TypeRecord",
]
