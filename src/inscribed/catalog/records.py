"""Provenance-aware records of known maxima."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Status(Enum):
    """Global theorem, theorem within one class only, or conjecture.

    The "status" column of the paper's tables.
    """

    PROVEN = "proven"
    CLASS_PROVEN = "class-proven"
    CONJECTURAL = "conjectural"


@dataclass(frozen=True)
class Citation:
    text: str
    url: str | None = None


@dataclass(frozen=True)
class KnownValue:
    dimension: int
    n: int
    value: float
    name: str
    status: Status
    citation: Citation
    exact: str | None = None
