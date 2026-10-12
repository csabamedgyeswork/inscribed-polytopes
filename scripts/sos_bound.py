"""Estimate the maximal signed volume of one type: an ascent lower bound and a relaxation value.

    python scripts/sos_bound.py -d 4 -n 7 --label firsching-7-0 --target 0.4330127018922193
    python scripts/sos_bound.py -d 4 -n 7 --label firsching-7-2 --level 2

The relaxation value is the floating-point optimum of the level-k moment relaxation of
max D over the product of unit spheres, reduced by the exact symmetries of that
polynomial problem. It is not a bound; it only suggests which bound to certify with
scripts/sos_certificate.py. Results go to results/certificates/d<d>-n<n>/.
"""

from __future__ import annotations

import argparse
import sys
import time
from math import factorial

from inscribed.enumeration import FirschingSource, PlantriSource, TypeRecord
from inscribed.search import maximize_type
from inscribed.sos import CertificateStore, MomentRelaxation, SolverSettings, SymmetryGroup


def log(message: str) -> None:
    """Print a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def load_type(d: int, n: int, label: str) -> TypeRecord:
    """The enumerated type with the given label."""
    source = {3: PlantriSource, 4: FirschingSource}.get(d)
    if source is None:
        raise SystemExit(f"no enumeration source for d={d}")
    for record in source().types(n):
        if record.label == label:
            return record
    raise SystemExit(f"no type {label!r} at d={d}, n={n}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dimension", "-d", type=int, required=True)
    parser.add_argument("--vertices", "-n", type=int, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--level", type=int, default=2, help="relaxation level (degree 2k)")
    parser.add_argument("--solver", default="CLARABEL", choices=["CLARABEL", "SCS", "MOSEK"])
    parser.add_argument("--target", type=float, default=None,
                        help="volume to exclude, e.g. the best rival's value")
    parser.add_argument("--restarts", type=int, default=20,
                        help="ascent restarts for the lower bound")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-automorphisms", action="store_true",
                        help="use only the coordinate symmetries")
    parser.add_argument("--verbose", action="store_true", help="print solver progress")
    args = parser.parse_args()

    record = load_type(args.dimension, args.vertices, args.label)
    cx = record.complex
    scale = factorial(cx.d)
    log(f"type {record.label}: d={cx.d} n={cx.n}, {len(cx.facets)} facets")

    group = SymmetryGroup.for_signed_volume(cx, use_type_automorphisms=not args.no_automorphisms)
    preserving = sum(1 for _, o in group.automorphisms if o > 0)
    log(f"type automorphisms: {len(group.automorphisms)} ({preserving} orientation-preserving); "
        f"symmetry group order {group.order()}")

    log(f"ascent lower bound ({args.restarts} restarts) ...")
    outcome = maximize_type(cx, restarts=args.restarts, seed=args.seed,
                            tol=1e-13, max_sweeps=20000, start=record.points)
    convex = outcome.volume if outcome.feasible else None
    log(f"lower bound: max D/{scale} >= {outcome.signed_best:.12f}"
        + (f" (convex-position optimum {convex:.12f})" if convex is not None else ""))

    relaxation = MomentRelaxation(cx, group, level=args.level)
    log(f"{relaxation.orbits.total} orbits, {relaxation.orbits.size} not forced to zero")
    result = relaxation.solve(SolverSettings(args.solver, args.verbose))
    value = relaxation.volume_value(result)
    log(f"SDP: {result.variables} variables, {result.equalities} equalities, "
        f"PSD blocks {result.block_sizes}; status {result.status} after {result.seconds:.1f}s")
    log(f"floating-point relaxation value: max D/{scale} ~ {value:.12f} (not a proof)")
    log(f"difference from the lower bound {value - outcome.signed_best:.3e} "
        f"({(value - outcome.signed_best) / outcome.signed_best:.3%})")
    if args.target is not None:
        verdict = "numerically below" if value < args.target else "NOT below"
        log(f"target {args.target:.12f}: {verdict} at level {args.level} "
            f"(margin {(args.target - value) / args.target:+.3%}); certify with sos_certificate.py")

    path = CertificateStore(cx.d, cx.n).save_relaxation(
        record.label, result, level=args.level, solver=args.solver,
        group_order=group.order(), lower_bound=outcome.signed_best, convex_volume=convex)
    log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
