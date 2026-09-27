"""Populate the committed results/ tree by sweeping the reference matrix.

Usage:  python scripts/build_results.py [--restarts R]
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from inscribed.catalog import known
from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.search import TypeSweep
from inscribed.store import build_case, capture, case_path, save_case

MATRIX = [
    (3, n, 3) for n in range(4, 12)
] + [
    (3, 12, 4),
] + [
    (4, n, 3) for n in range(5, 9)
]


def run_case(d: int, n: int, min_degree: int, restarts: int) -> None:
    source = PlantriSource(min_degree=min_degree) if d == 3 else FirschingSource()
    source_name = "plantri" if d == 3 else "firsching"
    sweep = TypeSweep(source, restarts=restarts, seed=11)
    started = time.time()
    results = sweep.run(n, lower_bound=-math.inf)
    elapsed = time.time() - started

    record = known(d, n)
    assumptions = ()
    complete = True
    if d == 3 and min_degree >= 4:
        complete = False
        assumptions = (
            f"min_degree>={min_degree} (no degree-3 vertex; proven 6<=n<=9, "
            "open for n>=10)",
        )
    case = build_case(
        d, n, results, source_name,
        enumeration={
            "source": source_name,
            "min_degree": min_degree,
            "total": len(results),
            "complete": complete,
        },
        budget={"restarts": restarts, "seed": 11, "seconds": round(elapsed, 1)},
        reference=(
            {"value": record.value, "status": record.status.value,
             "name": record.name, "exact": record.exact}
            if record is not None else None
        ),
        assumptions=assumptions,
        provenance=capture(),
        lower_bound=-math.inf,
    )
    path = save_case(case, case_path(d, n))
    t = case.triage
    print(f"d={d} n={n:>2}  {len(results)} types  winner {case.winner['volume']:.12f}  "
          f"triage {t['excluded']}/{t['excluded'] + t['hard']} excluded  "
          f"evidence {case.evidence.value}  ({elapsed:.0f}s) -> {path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--restarts", type=int, default=20)
    args = parser.parse_args()
    for d, n, min_degree in MATRIX:
        run_case(d, n, min_degree, args.restarts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
