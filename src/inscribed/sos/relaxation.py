"""The level-k moment relaxation of max D subject to |p_i|^2 = 1.

The relaxation is built only from the sphere equations and the exact symmetries of
the polynomial problem, so its exact optimum bounds max D from above for any type.
The value computed here is a floating-point SDP optimum and is not a bound: it can
fall below a feasible value by the solver tolerance. Only an exact certificate
(see :mod:`inscribed.sos.search` and :mod:`inscribed.certify.verify`) proves a bound.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import factorial

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from ..certify import Monomial, SignedVolume, parity_key
from ..combinatorics import OrientedComplex
from .basis import MomentOrbits, monomials
from .symmetry import SymmetryGroup


@dataclass(frozen=True)
class SolverSettings:
    """Choice of SDP solver."""

    solver: str = "CLARABEL"
    verbose: bool = False


@dataclass(frozen=True)
class RelaxationResult:
    """Outcome of one relaxation solve; ``value`` is the floating-point optimum, in units of D."""

    status: str
    value: float
    seconds: float
    variables: int
    equalities: int
    block_sizes: tuple[int, ...]


class MomentRelaxation:
    """Level-k moment relaxation of max D, reduced by the exact symmetry group."""

    def __init__(self, cx: OrientedComplex, group: SymmetryGroup, level: int = 2):
        self.cx, self.level = cx, level
        self.d, self.size = cx.d, cx.d * cx.n
        self.polynomial = SignedVolume(cx).polynomial
        group.check_invariance(self.polynomial)
        self.orbits = MomentOrbits(monomials(self.size, 2 * level), group.generators, self.d)
        self.basis = monomials(self.size, level)

    def _moment(self, *parts: Monomial) -> tuple[int, int] | None:
        """Column and sign of the moment of a product of monomials."""
        return self.orbits.lookup(tuple(sorted(sum(parts, ()))))

    def blocks(self) -> list[list[Monomial]]:
        """One basis block per parity class, keeping one class per coordinate-permutation orbit."""
        full = (1 << self.d) - 1
        classes: dict[int, list[Monomial]] = {}
        for m in self.basis:
            mask = parity_key(m, self.d)
            classes.setdefault(min(mask, mask ^ full), []).append(m)
        chosen: dict[int, list[Monomial]] = {}
        for key, members in sorted(classes.items()):
            weight = bin(key).count("1")
            chosen.setdefault(min(weight, self.d - weight), members)
        return list(chosen.values())

    def _block_matrix(self, block: list[Monomial], z: cp.Variable) -> cp.Expression:
        """Moment matrix of one basis block as an affine expression in z."""
        rows, cols, vals = [], [], []
        k = len(block)
        for i, a in enumerate(block):
            for j, b in enumerate(block):
                entry = self._moment(a, b)
                if entry is not None:
                    rows.append(i * k + j)
                    cols.append(entry[0])
                    vals.append(float(entry[1]))
        lift = sp.csr_matrix((vals, (rows, cols)), shape=(k * k, self.orbits.size))
        return cp.reshape(lift @ z, (k, k), order="C")

    def _localizing(self) -> sp.csr_matrix:
        """Rows of L(m * (|p_i|^2 - 1)) = 0 for every vertex i and low-degree monomial m."""
        unique: dict[tuple[tuple[int, int], ...], None] = {}
        for v in range(self.cx.n):
            squares = [(self.d * v + k,) * 2 for k in range(self.d)]
            for m in monomials(self.size, 2 * self.level - 2):
                row: dict[int, int] = {}
                for part, coeff in [(sq, 1) for sq in squares] + [((), -1)]:
                    entry = self._moment(m, part)
                    if entry is not None:
                        row[entry[0]] = row.get(entry[0], 0) + coeff * entry[1]
                items = tuple(sorted((c, x) for c, x in row.items() if x))
                if items:
                    lead = 1 if items[0][1] > 0 else -1
                    unique[tuple((c, lead * x) for c, x in items)] = None
        rows, cols, vals = [], [], []
        for r, items in enumerate(unique):
            for c, x in items:
                rows.append(r)
                cols.append(c)
                vals.append(float(x))
        return sp.csr_matrix((vals, (rows, cols)), shape=(len(unique), self.orbits.size))

    def _objective(self) -> np.ndarray:
        """Linear functional L(D) on the moment variables."""
        c = np.zeros(self.orbits.size)
        for m, coeff in self.polynomial.terms.items():
            entry = self.orbits.lookup(m)
            if entry is None:
                raise RuntimeError(f"monomial {m} of D is forced to zero by symmetry")
            c[entry[0]] += float(coeff) * entry[1]
        return c

    def solve(self, settings: SolverSettings = SolverSettings()) -> RelaxationResult:
        """Build and solve the SDP, returning its floating-point optimum."""
        z = cp.Variable(self.orbits.size)
        one = self.orbits.lookup(())
        assert one is not None
        localizing = self._localizing()
        blocks = self.blocks()
        constraints = [z[one[0]] == 1, localizing @ z == 0]
        constraints += [self._block_matrix(b, z) >> 0 for b in blocks]
        problem = cp.Problem(cp.Maximize(self._objective() @ z), constraints)
        start = time.time()
        problem.solve(solver=settings.solver, verbose=settings.verbose)
        return RelaxationResult(
            status=str(problem.status), value=float(problem.value), seconds=time.time() - start,
            variables=self.orbits.size, equalities=localizing.shape[0],
            block_sizes=tuple(len(b) for b in blocks),
        )

    def volume_value(self, result: RelaxationResult) -> float:
        """The relaxation value in units of volume, value / d!."""
        return result.value / factorial(self.d)
