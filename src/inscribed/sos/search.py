"""SOS search for gamma - D on the product of spheres, maximising the Gram margin.

The search looks for a certificate of ``gamma - D = sum_B v_B^T Q_B v_B + sum_i
h_i (|p_i|^2 - 1)`` with every Gram matrix Q_B as safely positive definite as
possible, so that the floating-point solution can later be rounded to rationals.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import factorial

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from ..certify import Monomial, Polynomial, SignedVolume
from ..combinatorics import OrientedComplex
from .basis import ParityBlocks
from .relaxation import SolverSettings


@dataclass(frozen=True)
class FloatCertificate:
    """Floating-point Gram blocks and multipliers, with the smallest-eigenvalue margin."""

    grams: list[np.ndarray]
    multipliers: np.ndarray
    margin: float
    status: str


class CertificateSearch:
    """Maximise t with every Q_B - t I positive semidefinite and the identity matched."""

    def __init__(self, cx: OrientedComplex, gamma: Fraction, level: int = 2):
        self.cx, self.gamma, self.level = cx, gamma, level
        self.d, self.n = cx.d, cx.n
        self.basis = ParityBlocks(cx.d, cx.n, level)
        self.multiplier_basis = ParityBlocks(cx.d, cx.n, 2 * level - 2).blocks[0]
        self.target = Polynomial({(): gamma}) - SignedVolume(cx).polynomial
        self.rows: dict[Monomial, int] = {}

    @classmethod
    def for_volume_bound(cls, cx: OrientedComplex, bound: Fraction,
                         level: int = 2) -> CertificateSearch:
        """The search for volume at most ``bound``, that is D at most ``bound * d!``."""
        return cls(cx, bound * factorial(cx.d), level)

    def _row(self, monomial: Monomial) -> int:
        """Row index of a monomial in the coefficient-matching system."""
        return self.rows.setdefault(monomial, len(self.rows))

    def _gram_map(self, block: list[Monomial]) -> list[tuple[int, int]]:
        """(row, column of vec(Q)) pairs: Q[i, j] contributes to the monomial v_i v_j."""
        k = len(block)
        return [(self._row(tuple(sorted(a + b))), j * k + i)
                for i, a in enumerate(block) for j, b in enumerate(block)]

    def _multiplier_map(self) -> list[tuple[int, int, float]]:
        """(row, column, coefficient) of h_i[beta] * x^beta * (|p_i|^2 - 1)."""
        entries = []
        size = len(self.multiplier_basis)
        for i in range(self.n):
            for c, beta in enumerate(self.multiplier_basis):
                column = i * size + c
                for k in range(self.d):
                    var = self.d * i + k
                    entries.append((self._row(tuple(sorted(beta + (var, var)))), column, 1.0))
                entries.append((self._row(beta), column, -1.0))
        return entries

    def solve(self, settings: SolverSettings = SolverSettings()) -> FloatCertificate:
        """Solve the SDP and return the floating-point certificate."""
        gram_maps = [self._gram_map(block) for block in self.basis.blocks]
        mult_map = self._multiplier_map()
        for m in self.target.terms:
            self._row(m)
        rows = len(self.rows)

        grams = [cp.Variable((len(b), len(b)), symmetric=True) for b in self.basis.blocks]
        h = cp.Variable(self.n * len(self.multiplier_basis))
        t = cp.Variable()
        lhs = sp.csr_matrix(([e[2] for e in mult_map], ([e[0] for e in mult_map],
                             [e[1] for e in mult_map])), shape=(rows, h.size)) @ h
        for gram, gmap, block in zip(grams, gram_maps, self.basis.blocks):
            lift = sp.csr_matrix((np.ones(len(gmap)), ([r for r, _ in gmap], [c for _, c in gmap])),
                                 shape=(rows, len(block) ** 2))
            lhs = lhs + lift @ cp.vec(gram, order="F")
        rhs = np.zeros(rows)
        for m, c in self.target.terms.items():
            rhs[self.rows[m]] = float(c)

        constraints = [lhs == rhs]
        constraints += [g - t * np.eye(g.shape[0]) >> 0 for g in grams]
        problem = cp.Problem(cp.Maximize(t), constraints + [t <= 1])
        problem.solve(solver=settings.solver, verbose=settings.verbose)
        return FloatCertificate(
            grams=[np.asarray(g.value) for g in grams],
            multipliers=np.asarray(h.value).reshape(self.n, len(self.multiplier_basis)),
            margin=float(problem.value), status=str(problem.status),
        )
