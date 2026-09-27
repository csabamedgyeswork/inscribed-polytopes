"""Exhaustive maximisation over every combinatorial type with n vertices.

Usage:  python scripts/sweep_types.py D N [--restarts R] [--min-degree K] [--save]
"""

from __future__ import annotations

import argparse
import math
import sys
import time

from inscribed.catalog import known
from inscribed.combinatorics import is_medial
from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.search import TypeSweep
from inscribed.store import (
    Configuration,
    build_case,
    capture,
    case_path,
    configuration_path,
    save,
    save_case,
)


def source_for(d: int, min_degree: int):
    if d == 3:
        return PlantriSource(min_degree=min_degree)
    if d == 4:
        return FirschingSource()
    raise SystemExit(f"no enumeration source for d={d}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("d", type=int, nargs="?", default=3, help="ambient dimension")
    parser.add_argument("n", type=int, nargs="?", default=9, help="number of vertices")
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--min-degree", type=int, default=3)
    parser.add_argument("--save", action="store_true",
                        help="write the winner to output/configurations/")
    parser.add_argument("--record", action="store_true",
                        help="write the full case with triage to results/dD-nNN.json")
    args = parser.parse_args()
    d, n, restarts, min_degree = args.d, args.n, args.restarts, args.min_degree

    source = source_for(d, min_degree)
    sweep = TypeSweep(source, restarts=restarts, seed=11)

    started = time.time()
    state = {"feasible": 0}

    def report(index: int, result) -> None:
        state["feasible"] += int(result.feasible)
        print(
            f"\r  screening {index + 1} types, {state['feasible']} feasible, "
            f"{time.time() - started:.0f}s",
            end="",
            file=sys.stderr,
            flush=True,
        )

    results = sweep.run(n, lower_bound=-math.inf, progress=report)
    print(file=sys.stderr)
    elapsed = time.time() - started

    feasible = [r for r in results if r.feasible]
    print(f"d={d}  n={n}  {len(results)} types, {len(feasible)} with a valid "
          f"inscribed polytope  ({restarts} restarts each, {elapsed:.1f}s)")

    record = known(d, n)
    if record is not None:
        print(f"  known {record.status.value:<12} {record.value:.12f}  {record.name}")

    print(f"\n  {'rank':>4} {'volume':>17} {'signed':>17} {'degrees':<24} "
          f"{'rej':>4}  label")
    for rank, result in enumerate(feasible[:10], start=1):
        degrees = sorted(result.complex.vertex_degrees().values())
        print(f"  {rank:>4} {result.volume:>17.12f} {result.signed_best:>17.12f} "
              f"{str(degrees):<24} {result.rejected:>4}  {result.label}")

    if not feasible:
        print("  no type produced a convex-position optimum")
        return 1

    best = feasible[0]
    if record is not None:
        error = abs(best.volume - record.value)
        verdict = "MATCHES" if error < 1e-9 else "DIFFERS FROM"
        print(f"\n  winner {verdict} the known value, error {error:.2e}")
    if d == 3:
        print(f"  winner medial: {is_medial(best.complex)}")

    rivals = [r for r in results if r.label != best.label]
    hard = [r for r in rivals if r.signed_best >= best.volume]
    print(f"  triage: {len(rivals) - len(hard)} of {len(rivals)} rival types "
          f"bounded below the winner by the signed volume, {len(hard)} hard")

    if args.record:
        source_name = "plantri" if d == 3 else "firsching"
        assumptions = ()
        if d == 3 and min_degree >= 4:
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
                "complete": min_degree <= 3,
            },
            budget={"restarts": restarts, "seed": 11},
            reference=(
                {"value": record.value, "status": record.status.value,
                 "name": record.name, "exact": record.exact}
                if record is not None else None
            ),
            assumptions=assumptions,
            provenance=capture(),
            lower_bound=-math.inf,
        )
        print(f"  recorded {save_case(case, case_path(d, n))} "
              f"(evidence {case.evidence.value})")

    if args.save:
        source_name = "plantri" if d == 3 else "firsching"
        cfg = Configuration(
            points=best.points, complex=best.complex, volume=best.volume,
            label=f"d{d}-n{n}-best", source=source_name,
            metadata={"restarts": best.restarts, "rejected": best.rejected,
                      "sweeps": best.sweeps, "refined": best.refined, "seed": 11,
                      "solver": "TypeSweep"},
        )
        print(f"  saved {save(cfg, configuration_path(cfg.label))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
