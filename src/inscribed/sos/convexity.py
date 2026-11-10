"""Search for convexity certificates with SCS, and round them to exact ones.

The conic program is: maximise t subject to the identity

    gamma - D = sum_B v_B^T Q_B v_B + sum lambda_{F,j} A_{F,j} + sum_i h_i (|p_i|^2 - 1)

holding coefficientwise in floating point, Q_B - t I positive semidefinite, t <= 1 and every
lambda >= 0. The ``bare`` route has no lambda columns; the ``convex`` route has one per facet
F and vertex j not in F. SCS runs single-threaded from a cold start, so a solve is
deterministic for a fixed build. Nothing here is trusted: the exact verifier decides.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.sparse as sp
import scs

from ..certify import ConvexityCertificate, ExactCertificate, Polynomial
from ..certify.convexity import facet_polynomial
from ..certify.verify import CertificateVerifier
from ..combinatorics import OrientedComplex
from .search import CertificateSearch

ROUTES = ("bare", "convex")
_R2 = np.sqrt(2.0)


@dataclass(frozen=True)
class ConicSolution:
    margin: float
    iterations: int
    grams: list[np.ndarray]
    h: np.ndarray
    lam: np.ndarray


class ConvexitySearch:
    """One type and bound: build the conic program of a route, solve it, round it exactly."""

    def __init__(self, cx: OrientedComplex, bound: Fraction, label: str,
                 *, eps: float = 1e-5, max_iters: int = 50_000, denominator: int = 10**9):
        self.cx, self.bound, self.label = cx, bound, label
        self.search = CertificateSearch.for_volume_bound(cx, bound)
        self.eps, self.max_iters, self.denominator = eps, max_iters, denominator

    def convexity_terms(self, route: str) -> list[tuple[tuple[tuple[int, ...], int], Polynomial]]:
        if route not in ROUTES:
            raise ValueError(f"unknown route {route}")
        if route == "bare":
            return []
        return [((f, j), facet_polynomial(f, j, self.cx.d))
                for f in self.cx.facets for j in range(self.cx.n) if j not in f]

    def solve(self, route: str) -> ConicSolution:
        polys = self.convexity_terms(route)
        search, d, n = self.search, self.cx.d, self.cx.n
        blocks = [[tuple(m) for m in b] for b in search.basis.blocks]
        mbasis = [tuple(m) for m in search.multiplier_basis]
        rows: dict[tuple[int, ...], int] = {}

        def row(m: tuple[int, ...]) -> int:
            return rows.setdefault(m, len(rows))

        eq: list[tuple[int, int, float]] = []
        svec: list[tuple[int, int]] = []
        col = 0
        for block in blocks:
            k, start = len(block), col
            for j in range(k):
                for i in range(j, k):
                    eq.append((row(tuple(sorted(block[i] + block[j]))), col,
                               1.0 if i == j else _R2))
                    col += 1
            svec.append((start, k))
        h_col = col
        for i in range(n):
            for slot, beta in enumerate(mbasis):
                hc = col + i * len(mbasis) + slot
                for k in range(d):
                    v = d * i + k
                    eq.append((row(tuple(sorted(beta + (v, v)))), hc, 1.0))
                eq.append((row(beta), hc, -1.0))
        col += n * len(mbasis)
        lam_col = col
        for k, (_, poly) in enumerate(polys):
            for m, coeff in poly.terms.items():
                eq.append((row(m), col + k, float(coeff)))
        col += len(polys)
        t_col = col
        target = {tuple(m): float(c) for m, c in search.target.terms.items()}
        for m in target:
            row(m)
        nx, nz, nl = col + 1, len(rows), len(polys)

        zr, zc, zv = zip(*eq)
        a_eq = sp.csr_matrix((zv, (zr, zc)), shape=(nz, nx))
        a_lin = sp.csr_matrix(([1.0] + [-1.0] * nl,
                               ([0] + list(range(1, nl + 1)),
                                [t_col] + list(range(lam_col, lam_col + nl)))),
                              shape=(nl + 1, nx))
        psd: list[tuple[int, int, float]] = []
        r = 0
        for start, k in svec:
            idx = 0
            for j in range(k):
                for i in range(j, k):
                    psd.append((r + idx, start + idx, -1.0))
                    if i == j:
                        psd.append((r + idx, t_col, 1.0))
                    idx += 1
            r += k * (k + 1) // 2
        pr, pc, pv = zip(*psd)
        a_psd = sp.csr_matrix((pv, (pr, pc)), shape=(r, nx))
        a = sp.vstack([a_eq, a_lin, a_psd]).tocsc()
        b = np.zeros(a.shape[0])
        for m, value in target.items():
            b[rows[m]] = value
        b[nz] = 1.0
        cost = np.zeros(nx)
        cost[t_col] = -1.0
        cones = {"z": nz, "l": nl + 1, "s": [len(bl) for bl in blocks]}
        solver = scs.SCS({"A": a, "b": b, "c": cost}, cones, eps_abs=self.eps,
                         eps_rel=self.eps, max_iters=self.max_iters, verbose=False)
        sol = solver.solve(warm_start=False)
        x = sol["x"]
        grams = []
        for start, k in svec:
            q, idx = np.zeros((k, k)), start
            for j in range(k):
                for i in range(j, k):
                    q[i, j] = q[j, i] = x[idx] if i == j else x[idx] / _R2
                    idx += 1
            grams.append(q)
        return ConicSolution(float(x[t_col]), int(sol["info"]["iter"]), grams,
                             x[h_col:lam_col].reshape(n, len(mbasis)), x[lam_col:t_col])

    def _rational(self, x: float) -> Fraction:
        return Fraction(round(x * self.denominator), self.denominator)

    def _certificate(self, grams, h) -> ExactCertificate:
        s = self.search
        return ExactCertificate(label=self.label, d=s.d, n=s.n, facets=list(s.cx.facets),
                                gamma=s.gamma, blocks=s.basis.blocks,
                                grams=[[row[:] for row in g] for g in grams],
                                multiplier_basis=s.multiplier_basis,
                                multipliers=[row[:] for row in h])

    def round_exact(self, solution: ConicSolution, route: str) -> ConvexityCertificate:
        """Round to the fixed denominator, clip lambda at 0, and move the exact residual of the
        identity into the Gram entries of its monomial, so the identity holds exactly."""
        polys = self.convexity_terms(route)
        grams = [[[self._rational(x) for x in row] for row in (g + g.T) / 2]
                 for g in solution.grams]
        h = [[self._rational(x) for x in row] for row in solution.h]
        lam = [max(Fraction(0), self._rational(x)) for x in solution.lam]
        terms: dict = {}
        for weight, (_, poly) in zip(lam, polys):
            if weight:
                for m, c in poly.terms.items():
                    terms[m] = terms.get(m, Fraction(0)) + weight * c
        residual = (CertificateVerifier().residual(self._certificate(grams, h))
                    - Polynomial(terms))
        position = self.search.basis.position
        for monomial, r in residual.terms.items():
            half = len(monomial) // 2
            (block_a, i), (block_b, j) = position[monomial[:half]], position[monomial[half:]]
            if block_a != block_b:
                raise RuntimeError(f"{self.label}: residual monomial {monomial} fits no block")
            share = r if i == j else r / 2
            grams[block_a][i][j] += share
            if i != j:
                grams[block_a][j][i] += share
        return ConvexityCertificate(
            self.bound, self._certificate(grams, h),
            tuple((f, j, w) for w, ((f, j), _) in zip(lam, polys) if w))
