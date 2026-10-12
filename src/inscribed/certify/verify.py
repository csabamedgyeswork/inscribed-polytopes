"""Exact verification of a certificate, independent of the solver that produced it.

The identity gamma - D = sum v^T Q v + sum h_i (|p_i|^2 - 1) is expanded with
rational arithmetic and must vanish exactly; every Gram block must be symmetric and
positive definite by Sylvester's criterion, checked with fraction-free Bareiss
elimination on the scaled integer matrix. The facets must form a coherently oriented
sphere, so that D is d! times the volume, up to sign, on every convex realisation; and
their vertex sets must be those of the type the certificate names.

Every check uses exact rational arithmetic on the certificate data alone; no solver
output is trusted.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from fractions import Fraction
from math import lcm

from ..combinatorics import Facet, OrientedComplex
from .certificate import ExactCertificate
from .cyclic import cyclic_facets
from .polynomial import Polynomial, SignedVolume

TypeResolver = Callable[[ExactCertificate], "set[frozenset[int]] | None"]


def facet_sets(facets: Iterable[Facet]) -> set[frozenset[int]]:
    """The vertex sets of the facets, forgetting order and orientation."""
    return {frozenset(f) for f in facets}


def bareiss_positive_definite(gram: list[list[Fraction]]) -> bool:
    """Sylvester's criterion via fraction-free Bareiss elimination on the scaled integer matrix."""
    size = len(gram)
    if any(gram[i][j] != gram[j][i] for i in range(size) for j in range(i)):
        return False
    scale = lcm(*(x.denominator for row in gram for x in row))
    a = [[int(x * scale) for x in row] for row in gram]
    previous = 1
    for k in range(size):
        if a[k][k] <= 0:
            return False
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                a[i][j] = (a[i][j] * a[k][k] - a[i][k] * a[k][j]) // previous
        previous = a[k][k]
    return True


def cyclic_types(cert: ExactCertificate) -> set[frozenset[int]] | None:
    """Resolve a label of the form ``cyclic-{d}-{n}`` by Gale's evenness condition."""
    parts = cert.label.split("-")
    if len(parts) != 3 or parts[0] != "cyclic":
        return None
    d, n = int(parts[1]), int(parts[2])
    if d != cert.d or n != cert.n:
        return None
    return cyclic_facets(n, d)


@dataclass(frozen=True)
class VerificationReport:
    """Outcome of checking a certificate; ``ok`` is the conjunction of the checks."""

    identity: bool
    positive_definite: bool
    coherent: bool
    type_matches: bool
    volume_bound: Fraction

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return self.identity and self.positive_definite and self.coherent and self.type_matches


@dataclass(frozen=True)
class CertificateVerifier:
    """Exact check of a certificate: the polynomial identity and positive definiteness.

    ``resolvers`` answer the type check: each maps a certificate to the facet vertex
    sets of the type it names, or ``None`` when it does not know the label. The first
    non-``None`` answer is used; the check fails when none knows the label.
    """

    resolvers: tuple[TypeResolver, ...] = field(default=(cyclic_types,))

    def residual(self, cert: ExactCertificate) -> Polynomial:
        """gamma - D - sum h_i (|p_i|^2 - 1) - sum v^T Q v, expanded exactly."""
        signed_volume = SignedVolume.from_facets(cert.facets, cert.d).polynomial
        terms = dict((Polynomial({(): cert.gamma}) - signed_volume).terms)

        def accumulate(monomial, coeff):
            terms[monomial] = terms.get(monomial, Fraction(0)) + coeff

        for i, row in enumerate(cert.multipliers):
            for beta, coeff in zip(cert.multiplier_basis, row):
                if not coeff:
                    continue
                for k in range(cert.d):
                    var = cert.d * i + k
                    accumulate(tuple(sorted(beta + (var, var))), -coeff)
                accumulate(beta, coeff)
        for block, gram in zip(cert.blocks, cert.grams):
            for a, row in zip(block, gram):
                for b, coeff in zip(block, row):
                    if coeff:
                        accumulate(tuple(sorted(a + b)), -coeff)
        return Polynomial(terms)

    def coherent(self, cert: ExactCertificate) -> bool:
        """Whether the facets form a closed, coherently oriented complex with a sphere's Euler
        characteristic."""
        return OrientedComplex(d=cert.d, n=cert.n, facets=tuple(cert.facets)).is_valid()

    def type_matches(self, cert: ExactCertificate) -> bool:
        """Whether the facet vertex sets are exactly those of the type it names."""
        for resolve in self.resolvers:
            expected = resolve(cert)
            if expected is not None:
                return facet_sets(cert.facets) == expected
        return False

    def verify(self, cert: ExactCertificate) -> VerificationReport:
        """Run every check and return the report."""
        return VerificationReport(
            identity=self.residual(cert).is_zero(),
            positive_definite=all(bareiss_positive_definite(g) for g in cert.grams),
            coherent=self.coherent(cert),
            type_matches=self.type_matches(cert),
            volume_bound=cert.volume_bound,
        )
