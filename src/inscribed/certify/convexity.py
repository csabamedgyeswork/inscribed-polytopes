"""Convexity certificates: sum-of-squares identities with nonnegative facet-polynomial multiples.

A convexity certificate for a coherently oriented type C and a bound gamma = d! * bound is

    gamma - D = sum_B v_B^T Q_B v_B + sum lambda_{F,j} A_{F,j} + sum_i h_i (|p_i|^2 - 1),

with every Q_B positive definite and every lambda_{F,j} >= 0, where
A_{F,j} = det(p_{F_1} - p_j, ..., p_{F_d} - p_j) for a facet F and a vertex j not in F. Every
positively oriented realisation of C has every A_{F,j} > 0, so the identity bounds the volume
of every inscribed realisation by ``bound`` without bounding D on all of the spheres.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import permutations
from math import lcm

import numpy as np

from ..combinatorics.complex import permutation_parity
from .certificate import ExactCertificate
from .polynomial import Monomial, Polynomial
from .verify import CertificateVerifier, bareiss_positive_definite

Facet = tuple[int, ...]


def _bracket(points: Facet, coeff: int, d: int, out: dict[Monomial, Fraction]) -> None:
    """Add coeff * det(p_{points[0]}, ..., p_{points[d-1]}) to ``out``."""
    for s in permutations(range(d)):
        m = tuple(sorted(d * points[k] + s[k] for k in range(d)))
        out[m] = out.get(m, Fraction(0)) + coeff * permutation_parity(s)


def facet_polynomial(facet: Facet, j: int, d: int) -> Polynomial:
    """A_{F,j} = det(p_{F_1} - p_j, ..., p_{F_d} - p_j), expanded by multilinearity; terms
    with p_j in two columns vanish, so it is det(p_F) minus det(p_F with p_j in column k)."""
    terms: dict[Monomial, Fraction] = {}
    _bracket(facet, 1, d, terms)
    for k in range(d):
        _bracket(facet[:k] + (j,) + facet[k + 1:], -1, d, terms)
    return Polynomial(terms)


def canonical_json(payload: dict) -> bytes:
    """The bytes that are hashed: sorted keys, no whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class ConvexityCertificate:
    """An exact certificate together with the convexity multipliers (F, j, lambda)."""

    bound: Fraction
    certificate: ExactCertificate
    multipliers: tuple[tuple[Facet, int, Fraction], ...] = ()

    @property
    def label(self) -> str:
        return self.certificate.label

    def to_json(self) -> dict:
        return {"bound": str(self.bound), "certificate": self.certificate.to_json(),
                "convexity_multipliers": [{"facet": list(f), "vertex": j, "lambda": str(w)}
                                          for f, j, w in self.multipliers]}

    @classmethod
    def from_json(cls, data: dict) -> ConvexityCertificate:
        return cls(Fraction(data["bound"]), ExactCertificate.from_json(data["certificate"]),
                   tuple((tuple(m["facet"]), int(m["vertex"]), Fraction(m["lambda"]))
                         for m in data["convexity_multipliers"]))

    def digest(self) -> str:
        """SHA-256 of the canonical JSON; equal digests mean byte-identical certificates."""
        return hashlib.sha256(canonical_json(self.to_json())).hexdigest()


def congruence_positive_definite(gram: list[list[Fraction]], bits: int = 40) -> bool:
    """Sufficient exact test: with R an integer upper triangular matrix with nonzero diagonal
    (a rounded inverse Cholesky factor), if M = R^T A R is strictly diagonally dominant with
    positive diagonal then A is positive definite. The floating point only chooses R."""
    k = len(gram)
    scale = lcm(*(x.denominator for row in gram for x in row))
    a = np.array([[x.numerator * (scale // x.denominator) for x in row] for row in gram],
                 dtype=object)
    try:
        lower = np.linalg.cholesky(np.array([[float(x) for x in row] for row in gram]))
    except np.linalg.LinAlgError:
        return False
    r = np.linalg.inv(lower).T
    r = r / np.abs(r).max()
    rint = np.array([[int(round(v * 2**bits)) if j >= i else 0 for j, v in enumerate(row)]
                     for i, row in enumerate(r)], dtype=object)
    if any(rint[i, i] == 0 for i in range(k)):
        return False
    m = rint.T.dot(a).dot(rint)
    return all(m[i, i] > sum(abs(m[i, j]) for j in range(k) if j != i) for i in range(k))


def positive_definite(gram: list[list[Fraction]]) -> bool:
    """Exact: symmetric, and positive definite by the congruence test or else by Bareiss."""
    k = len(gram)
    if any(gram[i][j] != gram[j][i] for i in range(k) for j in range(i)):
        return False
    return congruence_positive_definite(gram) or bareiss_positive_definite(gram)


@dataclass(frozen=True)
class ConvexityReport:
    """Every check of one convexity certificate; ``ok`` is their conjunction."""

    type_matches: bool
    coherent: bool
    multipliers_valid: bool
    identity: bool
    positive_definite: bool
    below_threshold: bool

    @property
    def ok(self) -> bool:
        return all(vars(self).values())

    def failures(self) -> list[str]:
        return [name for name, passed in vars(self).items() if not passed]


@dataclass(frozen=True)
class ConvexityVerifier(CertificateVerifier):
    """Exact check of a convexity certificate. ``resolvers`` give the facet sets of the type a
    label names; ``threshold`` decides whether the certified volume bound is good enough."""

    threshold: Callable[[Fraction], bool] = field(kw_only=True)

    def multipliers_valid(self, cc: ConvexityCertificate) -> bool:
        stored = set(cc.certificate.facets)
        return all(w >= 0 and f in stored and 0 <= j < cc.certificate.n and j not in f
                   for f, j, w in cc.multipliers)

    def convexity_residual(self, cc: ConvexityCertificate) -> Polynomial:
        """gamma - D - sum v^T Q v - sum h_i (|p_i|^2 - 1) - sum lambda_{F,j} A_{F,j}."""
        d = cc.certificate.d
        terms: dict[Monomial, Fraction] = {}
        for f, j, w in cc.multipliers:
            for m, c in facet_polynomial(f, j, d).terms.items():
                terms[m] = terms.get(m, Fraction(0)) + w * c
        return self.residual(cc.certificate) - Polynomial(terms)

    def check(self, cc: ConvexityCertificate) -> ConvexityReport:
        cert = cc.certificate
        valid = self.multipliers_valid(cc)
        return ConvexityReport(
            type_matches=self.type_matches(cert),
            coherent=self.coherent(cert),
            multipliers_valid=valid,
            identity=valid and self.convexity_residual(cc).is_zero(),
            positive_definite=all(positive_definite(g) for g in cert.grams),
            below_threshold=cc.bound == cert.volume_bound and self.threshold(cert.volume_bound),
        )


__all__ = ["ConvexityCertificate", "ConvexityReport", "ConvexityVerifier", "canonical_json",
           "congruence_positive_definite", "facet_polynomial", "positive_definite"]
