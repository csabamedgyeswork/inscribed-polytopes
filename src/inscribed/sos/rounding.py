"""Round a floating-point certificate to rationals, then repair the identity exactly."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from ..certify import CertificateVerifier, ExactCertificate
from .search import CertificateSearch, FloatCertificate


@dataclass(frozen=True)
class RoundingResult:
    """The repaired exact certificate and the largest single repair made."""

    certificate: ExactCertificate
    largest_repair: Fraction


class CertificateRounder:
    """Fixed-denominator rounding with designated-entry repair."""

    def __init__(self, denominator: int):
        self.denominator = denominator

    def _rational(self, x: float) -> Fraction:
        """Nearest rational with the fixed denominator."""
        return Fraction(round(x * self.denominator), self.denominator)

    def round(self, search: CertificateSearch, solution: FloatCertificate,
              label: str) -> RoundingResult:
        """The repaired exact certificate and the largest single repair made."""
        grams = [[[self._rational(x) for x in row] for row in (g + g.T) / 2]
                 for g in solution.grams]
        multipliers = [[self._rational(x) for x in row] for row in solution.multipliers]
        rounded = self._certificate(search, label, grams, multipliers)
        largest = Fraction(0)
        for monomial, r in CertificateVerifier().residual(rounded).terms.items():
            half = len(monomial) // 2
            a, b = monomial[:half], monomial[half:]
            (block_a, i), (block_b, j) = search.basis.position[a], search.basis.position[b]
            if block_a != block_b:
                raise RuntimeError(f"residual monomial {monomial} fits no Gram block")
            share = r if i == j else r / 2
            grams[block_a][i][j] += share
            if i != j:
                grams[block_a][j][i] += share
            largest = max(largest, abs(share))
        repaired = self._certificate(search, label, grams, multipliers)
        return RoundingResult(certificate=repaired, largest_repair=largest)

    @staticmethod
    def _certificate(search: CertificateSearch, label: str, grams: list[list[list[Fraction]]],
                     multipliers: list[list[Fraction]]) -> ExactCertificate:
        """A certificate holding its own copies of the Gram blocks and multipliers."""
        return ExactCertificate(
            label=label, d=search.d, n=search.n, facets=list(search.cx.facets),
            gamma=search.gamma, blocks=search.basis.blocks,
            grams=[[row[:] for row in g] for g in grams],
            multiplier_basis=search.multiplier_basis,
            multipliers=[row[:] for row in multipliers],
        )
