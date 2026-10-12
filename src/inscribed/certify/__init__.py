"""Exact verification of sum-of-squares certificates, with no solver dependency.

Everything here uses rational arithmetic only; importing this package never
imports cvxpy or the numerical search. The verifier in
:mod:`inscribed.certify.verify` is the part of the codebase a proof relies on.
"""

from .certificate import ExactCertificate
from .cyclic import cyclic_facets
from .polynomial import Monomial, Polynomial, SignedVolume, parity_key
from .verify import (
    CertificateVerifier,
    VerificationReport,
    bareiss_positive_definite,
    cyclic_types,
    facet_sets,
)

__all__ = [
    "CertificateVerifier",
    "ExactCertificate",
    "Monomial",
    "Polynomial",
    "SignedVolume",
    "VerificationReport",
    "bareiss_positive_definite",
    "cyclic_facets",
    "cyclic_types",
    "facet_sets",
    "parity_key",
]
