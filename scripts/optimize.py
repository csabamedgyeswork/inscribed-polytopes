"""Maximise the volume of one combinatorial type.

The type comes either from a facet file or from one entry of an enumeration:

    python scripts/optimize.py mytype.json --restarts 40 --save
    python scripts/optimize.py --dimension 3 --vertices 9 --index 3 --save
    python scripts/optimize.py --dimension 4 --vertices 8 --label <label>

A facet file is a JSON list of lists, or one whitespace-separated facet per line.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from inscribed.catalog import known
from inscribed.combinatorics import orient, parse_facets
from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.search import maximize_type
from inscribed.store import Configuration, configuration_path, save


def source_for(d: int):
    if d == 3:
        return PlantriSource()
    if d == 4:
        return FirschingSource()
    raise SystemExit(f"no enumeration source for d={d}")


def resolve_complex(args) -> tuple:
    """Return (complex, label, source) from a file or an enumerated type."""
    if args.file is not None:
        text = Path(args.file).read_text()
        cx = orient(parse_facets(text))
        return cx, Path(args.file).stem, "manual"

    source = source_for(args.dimension)
    records = list(source.types(args.vertices))
    if args.index is not None:
        if not 0 <= args.index < len(records):
            raise SystemExit(f"--index {args.index} out of range 0..{len(records) - 1}")
        record = records[args.index]
    elif args.label is not None:
        matches = [r for r in records if r.label == args.label]
        if not matches:
            raise SystemExit(f"no type labelled {args.label!r} at d={args.dimension} n={args.vertices}")
        record = matches[0]
    else:
        raise SystemExit("give a facet file, or --index / --label with --dimension and --vertices")
    return record.complex, record.label, source.__class__.__name__.replace("Source", "").lower()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("file", nargs="?", help="facet file (JSON or one facet per line)")
    parser.add_argument("--dimension", "-d", type=int, default=3)
    parser.add_argument("--vertices", "-n", type=int, default=9)
    parser.add_argument("--index", type=int, default=None, help="index into the enumeration")
    parser.add_argument("--label", default=None, help="label of an enumerated type")
    parser.add_argument("--restarts", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", action="store_true", help="write the optimum to output/configurations/")
    args = parser.parse_args()

    cx, label, source = resolve_complex(args)
    print(f"type {label or '(unnamed)'}: d={cx.d} n={cx.n}, {len(cx.facets)} facets")

    outcome = maximize_type(cx, restarts=args.restarts, seed=args.seed,
                            tol=1e-14, max_sweeps=20000)
    if not outcome.feasible:
        print(f"  no convex-position optimum found ({outcome.rejected} restarts rejected)")
        return 1

    print(f"  volume {outcome.volume:.15f}  ({outcome.restarts} restarts, "
          f"{outcome.rejected} rejected)")
    record = known(cx.d, cx.n)
    if record is not None:
        error = abs(outcome.volume - record.value)
        verdict = "MATCHES" if error < 1e-9 else "DIFFERS FROM"
        print(f"  {verdict} the known value {record.value:.12f} ({record.name}), error {error:.2e}")

    if args.save:
        cfg = Configuration(
            points=outcome.points, complex=cx, volume=outcome.volume,
            label=label or f"d{cx.d}-n{cx.n}", source=source,
            metadata={"restarts": outcome.restarts, "rejected": outcome.rejected,
                      "sweeps": outcome.sweeps, "seed": args.seed,
                      "solver": "FixedTypeAscent"},
        )
        path = save(cfg, configuration_path(cfg.label))
        print(f"  saved {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
