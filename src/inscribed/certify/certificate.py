"""The exact data of a certified upper bound on the signed volume of one type."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial

from .polynomial import Monomial


@dataclass(frozen=True)
class ExactCertificate:
    """Rational data of the identity gamma - D = sum v^T Q v + sum h_i (|p_i|^2 - 1)."""

    label: str
    d: int
    n: int
    facets: list[tuple[int, ...]]
    gamma: Fraction
    blocks: list[list[Monomial]]
    grams: list[list[list[Fraction]]]
    multiplier_basis: list[Monomial]
    multipliers: list[list[Fraction]]

    @property
    def volume_bound(self) -> Fraction:
        """The certified bound on the volume, gamma / d!."""
        return self.gamma / factorial(self.d)

    def relabelled(self, permutation: tuple[int, ...], label: str) -> ExactCertificate:
        """The same certificate with the vertices renamed by ``permutation``.

        ``permutation[i]`` is the new name of vertex ``i``. The facet tuples are
        permuted with their order kept, so the orientation is unchanged, and the
        identity still holds because the variables are renamed throughout.
        """
        def rename(monomial: Monomial) -> Monomial:
            return tuple(sorted(self.d * permutation[v // self.d] + v % self.d
                                for v in monomial))

        basis = [rename(m) for m in self.multiplier_basis]
        order = sorted(range(len(basis)), key=lambda i: basis[i])
        inverse = [permutation.index(i) for i in range(self.n)]
        return ExactCertificate(
            label=label, d=self.d, n=self.n,
            facets=[tuple(permutation[v] for v in f) for f in self.facets],
            gamma=self.gamma,
            blocks=[[rename(m) for m in block] for block in self.blocks],
            grams=self.grams,
            multiplier_basis=[basis[i] for i in order],
            multipliers=[[self.multipliers[inverse[i]][j] for j in order] for i in range(self.n)],
        )

    def to_json(self) -> dict:
        """Serialise with every rational as an exact string."""
        return {
            "label": self.label, "d": self.d, "n": self.n,
            "facets": [list(f) for f in self.facets], "gamma": str(self.gamma),
            "blocks": [[list(m) for m in block] for block in self.blocks],
            "grams": [[[str(x) for x in row] for row in g] for g in self.grams],
            "multiplier_basis": [list(m) for m in self.multiplier_basis],
            "multipliers": [[str(x) for x in row] for row in self.multipliers],
        }

    @classmethod
    def from_json(cls, data: dict) -> ExactCertificate:
        """Inverse of ``to_json``."""
        return cls(
            label=data["label"], d=data["d"], n=data["n"],
            facets=[tuple(f) for f in data["facets"]], gamma=Fraction(data["gamma"]),
            blocks=[[tuple(m) for m in block] for block in data["blocks"]],
            grams=[[[Fraction(x) for x in row] for row in g] for g in data["grams"]],
            multiplier_basis=[tuple(m) for m in data["multiplier_basis"]],
            multipliers=[[Fraction(x) for x in row] for row in data["multipliers"]],
        )
