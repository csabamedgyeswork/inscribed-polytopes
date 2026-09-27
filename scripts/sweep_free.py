"""Type-free search for the maximum volume on S^(D-1), over a range of n.

Usage:  python scripts/sweep_free.py D [N_MIN] [N_MAX] [--medial] [--save]

HullAscent with carry-forward initialisation: the optimum at n seeds the
search at n + 1. With --medial (D = 3 only) each row is checked against the
Goldberg--Grace medial conjecture, which the paper does not use. With --save
each row is written to results/free/.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter

from inscribed import ConvexPosition, VolumeFunctional
from inscribed.analysis import SymmetryDetector
from inscribed.catalog import Status, known
from inscribed.combinatorics import medial_verdict
from inscribed.search import (
    ChainedInitializer,
    GrowFromConfiguration,
    HullAscent,
    MultiStartSearch,
    RandomInitializer,
)
from inscribed.store import build_search_record, capture, save_search, search_path


def search(d: int, n: int, restarts: int, carry):
    random = RandomInitializer(d, n, seed=100 * d + n)
    if carry is None:
        initializer, runs = random, restarts
    else:
        initializer = ChainedInitializer(GrowFromConfiguration(carry), random)
        runs = restarts + 1
    return MultiStartSearch(HullAscent(), initializer).run(runs)


def degree_summary(degrees: list[int]) -> str:
    counts = Counter(degrees)
    return " ".join(f"{deg}^{counts[deg]}" for deg in sorted(counts))


SYMMETRY_TOL = 1e-7
"""Gram tolerance for symmetry detection; these optima converge to about 1e-8."""


def persist(d: int, n: int, res, restarts: int, convex: bool):
    """Write one row to results/free/."""
    report = SymmetryDetector(tol=SYMMETRY_TOL).analyze(res.points)
    record = build_search_record(
        d, n,
        points=res.points,
        complex=res.cx,
        volume=res.volume,
        convex=convex,
        symmetry={
            "order": report.order,
            "proper": report.proper,
            "improper": report.improper,
            "orbit_sizes": report.orbit_sizes,
            "tol": SYMMETRY_TOL,
        },
        budget={
            "restarts": restarts,
            "seed": 100 * d + n,
            "fraction_at_best": float(res.info["fraction_at_best"]),
        },
        provenance=capture(),
    )
    return save_search(record, search_path(d, n))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("d", type=int, nargs="?", default=3, help="ambient dimension")
    parser.add_argument("n_min", type=int, nargs="?", default=None)
    parser.add_argument("n_max", type=int, nargs="?", default=None)
    parser.add_argument("--medial", action="store_true",
                        help="check each row against the medial conjecture (d = 3 only)")
    parser.add_argument("--save", action="store_true",
                        help="write each row to results/free/")
    args = parser.parse_args()
    d = args.d
    medial = args.medial
    n_min = args.n_min if args.n_min is not None else (4 if d == 3 else 5)
    n_max = args.n_max if args.n_max is not None else (12 if d == 3 else 10)
    if medial and d != 3:
        print("warning: --medial is only defined for d = 3; ignoring", file=sys.stderr)
        medial = False

    extra = "  verdict" if medial else ""
    header = (
        f"{'n':>3} {'best volume':>17} {'known exact':>17} {'err':>9} "
        f"{'degrees':<24} {'hit%':>5} {'cvx':>4} {'sec':>6}{extra}"
    )
    print(f"Type-free hull ascent on S^{d - 1}, n = {n_min}..{n_max}")
    print(header)
    print("-" * len(header))

    carry = None
    for n in range(n_min, n_max + 1):
        restarts = 80 if n < 10 else 160
        t0 = time.time()
        res = search(d, n, restarts, carry)
        elapsed = time.time() - t0
        carry = res.points

        record = known(d, n)
        proven = record is not None and record.status in (
            Status.PROVEN, Status.CLASS_PROVEN
        )
        known_val = record.value if proven else float("nan")
        err = abs(res.volume - record.value) if proven else float("nan")

        degs, convex = [], "-"
        if res.cx is not None:
            degs = sorted(res.cx.vertex_degrees().values())
            convex = "yes" if ConvexPosition(res.cx).check(res.points).ok else "NO"

        row = (
            f"{n:>3} {res.volume:>17.12f} {known_val:>17.12f} {err:>9.2e} "
            f"{degree_summary(degs):<24} {100 * res.info['fraction_at_best']:>5.1f} "
            f"{convex:>4} {elapsed:>6.1f}"
        )
        if medial:
            verdict = medial_verdict(n, degs)
            row += f"  {verdict.upper() if verdict.startswith('not medial') else verdict}"
        print(row)

        if res.cx is not None:
            signed = VolumeFunctional.build(res.cx).value(res.points)
            if abs(signed - res.volume) > 1e-9:
                print(f"     ! signed volume {signed:.12f} != hull volume")
            if args.save:
                print(f"     -> {persist(d, n, res, restarts, convex == 'yes').name}")
        elif args.save:
            print("     ! no hull complex; not saved", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
