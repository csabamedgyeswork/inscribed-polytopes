"""The ComplexSource abstraction shared by every enumeration backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..combinatorics import OrientedComplex

ROOT = Path(__file__).resolve().parent.parent.parent.parent
PLANTRI = ROOT / "vendor" / "plantri" / "plantri"
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class TypeRecord:
    """One combinatorial type, with a valid inscribed realisation when known."""

    complex: OrientedComplex
    label: str
    points: np.ndarray | None = None


class ComplexSource(ABC):
    """Yields the combinatorial types to maximise over, for one dimension."""

    d: int

    @abstractmethod
    def types(self, n: int) -> Iterator[TypeRecord]:
        """Yield every type with ``n`` vertices, without repetition up to isomorphism."""

    @abstractmethod
    def count(self, n: int) -> int:
        """Number of types ``types(n)`` will yield."""
