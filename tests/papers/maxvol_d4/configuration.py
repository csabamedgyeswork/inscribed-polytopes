"""Explicit inscribed realisations of C_4(7), independent of any stored search output.

Points of the Caratheodory curve (cos t, sin t, cos 2t, sin 2t) / sqrt(2) lie on S^3,
and any seven of them, labelled in the order of t, span a cyclic polytope. The seven
equally spaced ones attain the largest volume of type C_4(7) found numerically.
"""

from __future__ import annotations

import gzip
import json

import numpy as np
from scipy.spatial import ConvexHull

from inscribed.store import certificate_dir

LABEL = "cyclic-4-7"


def caratheodory(angles) -> np.ndarray:
    """The points of the Caratheodory curve at the given angles, one per row."""
    t = np.asarray(angles, dtype=float)
    return np.stack([np.cos(t), np.sin(t), np.cos(2 * t), np.sin(2 * t)], axis=1) / np.sqrt(2)


def regular_heptagon() -> np.ndarray:
    """Seven equally spaced points of the Caratheodory curve."""
    return caratheodory(2 * np.pi * np.arange(7) / 7)


def certificate_facets() -> list[tuple[int, ...]]:
    """The oriented facets stored in the certificate of the cyclic type."""
    path = certificate_dir(4, 7) / f"{LABEL}.certificate.json.gz"
    return [tuple(f) for f in json.loads(gzip.decompress(path.read_bytes()))["facets"]]


def hull_facets(points: np.ndarray) -> set[frozenset[int]]:
    """The vertex sets of the facets of the convex hull of simplicial points."""
    return {frozenset(int(v) for v in s) for s in ConvexHull(points).simplices}


def signed_volume(points: np.ndarray, facets: list[tuple[int, ...]]) -> float:
    """D(p) = sum over the oriented facets of det(p_F)."""
    return float(sum(np.linalg.det(points[list(f)]) for f in facets))


def stored_ascent(key: str) -> float:
    """A value of the stored ascent run: ``lower_bound_volume`` is the largest D / 4! over
    all of (S^3)^7, ``convex_volume`` the volume of that point checked to realise C_4(7)."""
    data = json.loads((certificate_dir(4, 7) / f"{LABEL}.relaxation.json").read_text())
    return data[key]
