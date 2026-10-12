"""Identify the structure of a computed optimum: symmetry, orbits, parameters.

Usage:  python scripts/analyze.py D N [--restarts R] [--save]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from inscribed import (
    ConvexPosition,
    IndependentSetElimination,
    OptimalityConditions,
    SymmetryDetector,
    VolumeFunctional,
    free_parameter_count,
)
from inscribed.catalog import known
from inscribed.search import HullAscent, MultiStartSearch, RandomInitializer
from inscribed.store import Configuration, configuration_path, save


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("d", type=int, nargs="?", default=3, help="ambient dimension")
    parser.add_argument("n", type=int, nargs="?", default=9, help="number of vertices")
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--save", action="store_true",
                        help="write the optimum to output/configurations/")
    args = parser.parse_args()
    d, n, restarts = args.d, args.n, args.restarts

    res = MultiStartSearch(HullAscent(), RandomInitializer(d, n, seed=17)).run(restarts)
    pts = res.points
    cx = res.cx
    assert cx is not None

    print(f"d={d}  n={n}  volume = {res.volume:.15f}")
    record = known(d, n) if d == 3 else None
    if record is not None:
        print(f"  known exact           {record.value:.15f}")
        print(f"  error                 {abs(res.volume - record.value):.2e}")
    print(f"  restarts hitting best {100 * res.info['fraction_at_best']:.1f}%")

    report = ConvexPosition(cx).check(pts)
    print(f"  convex position       {report.ok} ({report.message})")
    print(f"  support margin        {report.metrics['min_support_margin']:.4f}")

    form = VolumeFunctional.build(cx)
    conditions = OptimalityConditions(form)
    print(f"  Property Z residual   {conditions.property_z_residual(pts):.2e}")
    print(f"  sum(lambda) - d*vol   {abs(conditions.multipliers(pts).sum() - d * res.volume):.2e}")

    degs = sorted(cx.vertex_degrees().values())
    print(f"  vertex degrees        {degs}")
    print(f"  f-vector              f0={cx.n} f1={len(cx.edges())} f_{d - 1}={len(cx.facets)}")

    sym = SymmetryDetector().analyze(pts)
    print(f"  symmetry              {sym.summary()}")
    for orbit in sym.orbits:
        radii = {round(float(np.linalg.norm(pts[v])), 9) for v in orbit}
        star = {len([f for f in cx.facets if v in f]) for v in orbit}
        print(f"    orbit size {len(orbit):>2}: vertices {orbit}, "
              f"facets/vertex {sorted(star)}, |p|={radii}")
    print("  distinct inner products:")
    for value, count in sym.inner_products:
        print(f"    {value:+.12f}  x{count}")

    raw = free_parameter_count(d, n)
    print(f"  parameters: {raw} after gauge; {len(sym.orbits)} orbits under the detected group")

    elimination = IndependentSetElimination.maximal(form)
    ind = elimination.eliminated
    print(f"  max independent set (exactly eliminable): {sorted(ind)} -> "
          f"{free_parameter_count(d, n - len(ind))} parameters")

    if args.save:
        cfg = Configuration(
            points=pts, complex=cx, volume=res.volume,
            label=f"d{d}-n{n}-analyze", source="hull-ascent",
            metadata={"restarts": restarts, "seed": 17, "solver": "HullAscent"},
        )
        print(f"  saved {save(cfg, configuration_path(cfg.label))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
