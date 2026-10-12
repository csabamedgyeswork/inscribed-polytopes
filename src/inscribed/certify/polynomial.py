"""Sparse multivariate polynomials with exact rational coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from itertools import permutations

import numpy as np

from ..combinatorics import OrientedComplex
from ..combinatorics.complex import permutation_parity

Monomial = tuple[int, ...]


def parity_key(monomial: Monomial, d: int) -> int:
    """Coordinate parity bitmask of a monomial, identified with its complement."""
    mask = 0
    for var in monomial:
        mask ^= 1 << (var % d)
    return min(mask, mask ^ ((1 << d) - 1))


@dataclass(frozen=True)
class Polynomial:
    """A sparse map from sorted monomials (variable index tuples) to rational coefficients."""

    terms: dict[Monomial, Fraction]

    def __post_init__(self) -> None:
        object.__setattr__(self, "terms", {m: c for m, c in self.terms.items() if c})

    def __add__(self, other: Polynomial) -> Polynomial:
        """Sum of two polynomials."""
        terms = dict(self.terms)
        for m, c in other.terms.items():
            terms[m] = terms.get(m, Fraction(0)) + c
        return Polynomial(terms)

    def __sub__(self, other: Polynomial) -> Polynomial:
        """Difference of two polynomials."""
        return self + other.scale(Fraction(-1))

    def scale(self, factor: Fraction) -> Polynomial:
        """Multiply by a rational."""
        return Polynomial({m: factor * c for m, c in self.terms.items()})

    def is_zero(self) -> bool:
        """Whether the polynomial is identically zero."""
        return not self.terms

    def substitute(self, image: dict[Monomial, tuple[int, Monomial]]) -> Polynomial:
        """Replace each monomial by a scalar multiple of another monomial."""
        terms: dict[Monomial, Fraction] = {}
        for m, c in self.terms.items():
            factor, m2 = image[m]
            terms[m2] = terms.get(m2, Fraction(0)) + int(factor) * c
        return Polynomial(terms)

    def value(self, coordinates: np.ndarray) -> float:
        """Evaluate at a flattened coordinate vector."""
        flat = np.asarray(coordinates, dtype=float).reshape(-1)
        return float(sum(float(c) * np.prod(flat[list(m)]) for m, c in self.terms.items()))


class SignedVolume:
    """The polynomial D = sum over facets of det(p_F), in d*n variables."""

    def __init__(self, cx: OrientedComplex):
        self.d = cx.d
        terms: dict[Monomial, Fraction] = {}
        for f in cx.facets:
            for s in permutations(range(cx.d)):
                m = tuple(sorted(cx.d * f[k] + s[k] for k in range(cx.d)))
                terms[m] = terms.get(m, Fraction(0)) + permutation_parity(s)
        self.polynomial = Polynomial(terms)

    @classmethod
    def from_facets(cls, facets: list[tuple[int, ...]], d: int) -> SignedVolume:
        """The signed volume of a raw facet list, as stored in a certificate."""
        return cls(OrientedComplex(d=d, n=max(max(f) for f in facets) + 1,
                                   facets=tuple(facets)))
