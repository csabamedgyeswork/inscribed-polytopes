"""Resolve enumerated type labels (plantri, Firsching) to their facet vertex sets."""

from __future__ import annotations

from ..certify import ExactCertificate, facet_sets
from ..enumeration import FirschingSource, PlantriSource


def enumerated_types(cert: ExactCertificate) -> set[frozenset[int]] | None:
    """The facet vertex sets of the enumerated type the certificate names, if any."""
    source = {3: PlantriSource, 4: FirschingSource}.get(cert.d)
    if source is None:
        return None
    for record in source().types(cert.n):
        if record.label == cert.label:
            return facet_sets(record.complex.facets)
    return None
