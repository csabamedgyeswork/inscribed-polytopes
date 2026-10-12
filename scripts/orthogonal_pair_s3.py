"""Test the orthogonal-polygon-pair conjecture in R^4 against free search.

See papers/sweep/sections/results-4d.tex for the statement and its status.
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import numpy as np

from inscribed import ConvexPosition, SymmetryDetector, hull_volume
from inscribed.catalog import OrthogonalPolygonPair
from inscribed.search import (
    ChainedInitializer,
    GrowFromConfiguration,
    HullAscent,
    MultiStartSearch,
    RandomInitializer,
)

FAMILY_LIMIT = math.pi**2 / 6
BALL_VOLUME = math.pi**2 / 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("lo", type=int, nargs="?", default=6, help="smallest n")
    parser.add_argument("hi", type=int, nargs="?", default=13, help="largest n")
    parser.add_argument("--restarts", type=int, default=150)
    args = parser.parse_args()
    lo, hi, restarts = args.lo, args.hi, args.restarts

    family = OrthogonalPolygonPair()
    print("Closed form vs. explicit hull, for every split m + k = n")
    print(f"{'n':>3} {'best split':>11} {'formula':>15} {'hull of config':>15} {'diff':>9}")
    print("-" * 58)
    predictions = {}
    for n in range(6, hi + 1):
        params = family.parameters_for(n, 4)
        value = family.closed_form_volume(**params)
        pts = family.build(**params)
        pts = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        built = hull_volume(pts)
        predictions[n] = value
        split = f"({params['m']},{params['k']})"
        print(f"{n:>3} {split:>11} {value:>15.12f} {built:>15.12f} {abs(value - built):>9.1e}")

    print(f"\nfamily limit pi^2/6 = {FAMILY_LIMIT:.12f}")
    print(f"ball volume  pi^2/2 = {BALL_VOLUME:.12f}")

    print(f"\nFree search vs. prediction, n = {lo}..{hi}")
    header = (
        f"{'n':>3} {'search':>16} {'predicted':>16} {'search-pred':>12} "
        f"{'|G|':>5} {'orbits':<12} {'hit%':>5} {'sec':>6}"
    )
    print(header)
    print("-" * len(header))

    detector = SymmetryDetector()
    carry = None
    for n in range(lo, hi + 1):
        random = RandomInitializer(4, n, seed=900 + n)
        if carry is None:
            initializer, runs = random, restarts
        else:
            initializer = ChainedInitializer(GrowFromConfiguration(carry), random)
            runs = restarts + 1
        t0 = time.time()
        res = MultiStartSearch(HullAscent(), initializer).run(runs)
        elapsed = time.time() - t0
        carry = res.points
        predicted = predictions[n]
        sym = detector.analyze(res.points)
        flag = ""
        if res.cx is not None and not ConvexPosition(res.cx).check(res.points).ok:
            flag = " !nonconvex"
        print(
            f"{n:>3} {res.volume:>16.12f} {predicted:>16.12f} "
            f"{res.volume - predicted:>+12.2e} {sym.order:>5} "
            f"{str(sym.orbit_sizes):<12} {100 * res.info['fraction_at_best']:>5.1f} "
            f"{elapsed:>6.1f}{flag}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
