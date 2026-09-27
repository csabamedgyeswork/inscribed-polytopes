"""Reproduce the reference sweeps and check them against the recorded values.

This is the validation gate for the refactor: every case below must match its
expected volume to within ``TOL`` and report the expected number of feasible
combinatorial types. Exits nonzero on any drift.

Usage:  python scripts/reproduce.py
"""

from __future__ import annotations

import argparse
import sys
import time

from inscribed.catalog import known
from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.search import TypeSweep

TOL = 1e-9

# (d, n, restarts, expected volume, expected feasible types)
CASES = [
    (3, 8, 40, 1.815716104224, 5),
    (3, 9, 40, 2.043750115900, 7),
    (4, 7, 30, 0.433012701892, 4),
    (4, 8, 20, 0.666666666667, 10),
]


def source_for(d: int):
    return PlantriSource() if d == 3 else FirschingSource()


def main() -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    failures = []
    for d, n, restarts, expected, expected_feasible in CASES:
        t0 = time.time()
        sweep = TypeSweep(source_for(d), restarts=restarts, seed=11)
        results = sweep.run(n)
        elapsed = time.time() - t0

        feasible = [r for r in results if r.feasible]
        best = feasible[0].volume if feasible else float("nan")
        vol_ok = abs(best - expected) < TOL
        count_ok = len(feasible) == expected_feasible
        ok = vol_ok and count_ok

        status = "PASS" if ok else "FAIL"
        print(
            f"[{status}] d={d} n={n}  volume {best:.12f} (expect {expected:.12f})  "
            f"feasible {len(feasible)}/{len(results)} (expect {expected_feasible})  "
            f"{elapsed:.1f}s"
        )
        if not vol_ok:
            failures.append(f"d={d} n={n}: volume {best:.12f} != {expected:.12f}")
        if not count_ok:
            failures.append(
                f"d={d} n={n}: feasible {len(feasible)} != {expected_feasible}"
            )

    if failures:
        print("\nFAILURES:")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\nall reference sweeps reproduced")
    return 0


if __name__ == "__main__":
    sys.exit(main())
