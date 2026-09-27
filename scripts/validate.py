"""Validate the engine against every known exact maximum on S^2.

Usage:  python scripts/validate.py
"""

from __future__ import annotations

import argparse
import math
import sys

from inscribed import (
    ConvexPosition,
    GaugeFrame,
    IndependentSetElimination,
    OptimalityConditions,
    VolumeFunctional,
    free_parameter_count,
    simplex,
)
from inscribed.catalog import Bipyramid, RegularSimplex, TriaugmentedPrism, known
from inscribed.search import FixedTypeAscent, MultiStartSearch, RandomInitializer


def class_maximum(cx, restarts: int, seed: int):
    search = MultiStartSearch(
        FixedTypeAscent(cx), RandomInitializer(cx.d, cx.n, seed=seed), ConvexPosition(cx)
    )
    return search.run(restarts)


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    print("Sanity: user's class-1 complex")
    prism = TriaugmentedPrism()
    tri = prism.complex()
    tri.check()
    form = VolumeFunctional.build(tri)
    conditions = OptimalityConditions(form)
    deg = tri.vertex_degrees()
    print(f"  valid oriented S^2, V={tri.n} E={len(tri.edges())} F={len(tri.facets)}")
    print(f"  degree sequence: {sorted(deg.values())}")
    print(f"  free parameters after gauge: {free_parameter_count(3, 9)}")
    elimination = IndependentSetElimination.maximal(form)
    ind = elimination.eliminated
    print(f"  max independent set (exactly eliminable): {sorted(v + 1 for v in ind)}")
    print(
        f"  parameters after elimination + gauge: "
        f"{free_parameter_count(3, tri.n - len(ind))}"
    )

    exact_pts = prism.build()
    v_exact = form.value(exact_pts)
    known9 = known(3, 9)
    print(f"  volume at published maximiser: {v_exact:.15f}")
    print(f"  3*sqrt(2*sqrt3-3):             {known9.value:.15f}")
    print(f"  Property Z residual there:     {conditions.property_z_residual(exact_pts):.3e}")
    print(f"  sum(lambda) - 3*vol:           {conditions.euler_identity_error(exact_pts):.3e}")
    print(f"  eliminated-form volume:        {elimination.value(exact_pts):.15f}")

    print("\nFixed-type ascent, restricted to convex-position fixed points")
    header = (
        f"{'type':<24} {'class max':>16} {'class theory':>16} {'err':>9} "
        f"{'rejected':>8} {'global?':>8}"
    )
    print(header)
    print("-" * len(header))

    failures = []
    cases = [
        (m + 2, f"double {m}-pyramid", Bipyramid().complex(m=m), Bipyramid().closed_form_volume(m=m))
        for m in range(3, 8)
    ]
    cases.append((9, "triaugmented prism", tri, prism.closed_form_volume()))

    for n, name, cx, theory in cases:
        cx.check()
        res = class_maximum(cx, restarts=150, seed=1)
        frame = GaugeFrame.default(cx)
        res.points = frame.transform(res.points)
        report = ConvexPosition(cx).check(res.points)
        err = abs(res.volume - theory)
        record = known(3, n)
        is_global = record is not None and abs(res.volume - record.value) < 1e-9
        print(
            f"{name:<24} {res.volume:>16.12f} {theory:>16.12f} {err:>9.2e} "
            f"{res.info['rejected']:>8} {('yes' if is_global else 'no'):>8}"
        )
        if frame.residual(res.points) > 1e-12:
            failures.append(f"{name}: gauge residual too large")
        euler = OptimalityConditions(VolumeFunctional.build(cx)).euler_identity_error(res.points)
        if euler > 1e-9:
            failures.append(f"{name}: Euler identity off by {euler:.2e}")
        if not report.ok:
            failures.append(f"{name}: not convex position ({report.message})")
        if err > 1e-9:
            failures.append(f"{name}: class maximum off by {err:.2e}")

    print("\nGeneral-dimension smoke test: boundary of the d-simplex")
    for d in (3, 4, 5):
        cx = simplex(d)
        cx.check()
        res = class_maximum(cx, restarts=30, seed=2)
        expected = RegularSimplex().closed_form_volume(d=d)
        print(
            f"  d={d}: found {res.volume:.12f}  regular simplex {expected:.12f}  "
            f"free params {free_parameter_count(d, d + 1)}"
        )
        if abs(res.volume - expected) > 1e-9:
            failures.append(f"d={d} simplex: {abs(res.volume - expected):.2e}")

    print()
    if failures:
        print("FAILURES:")
        for f in failures:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
