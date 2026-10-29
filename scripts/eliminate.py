"""Eliminate every rival type of a case below a bound, with a per-type certificate cache.

    python scripts/eliminate.py -d 4 -n 9 --winner firsching-9-10 --bound 0.78 --status
    python scripts/eliminate.py -d 4 -n 9 --winner firsching-9-10 --bound 0.78 \
        --hardest-first --limit 3
    python scripts/eliminate.py -d 4 -n 9 --winner firsching-9-10 --bound 0.78 --shard 0/4

Every proved type is written to results/certificates/d<d>-n<n>/ immediately, so a
killed run resumes where it stopped. --shard i/k runs every k-th pending type, for
splitting the case across machines. Logs go to output/elimination/d<d>-n<n>.log.
"""

from __future__ import annotations

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

from inscribed.elimination import Campaign, Screening
from inscribed.enumeration import FirschingSource, PlantriSource
from inscribed.store import certificate_dir


def source_for(d: int):
    source = {3: PlantriSource, 4: FirschingSource}.get(d)
    if source is None:
        raise SystemExit(f"no enumeration source for d={d}")
    return source()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dimension", "-d", type=int, required=True)
    parser.add_argument("--vertices", "-n", type=int, required=True)
    parser.add_argument("--winner", required=True, help="label of the type to keep")
    parser.add_argument("--bound", required=True, help="volume bound, as an exact decimal")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", default="0/1", metavar="I/K",
                        help="run every K-th pending type, starting at I (zero-based)")
    parser.add_argument("--hardest-first", action="store_true",
                        help="start at the types closest to the bound (for pilots)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="redo types with a failed attempt at this bound")
    parser.add_argument("--restarts", type=int, default=50,
                        help="screening restarts per type, used only on first run")
    parser.add_argument("--status", action="store_true", help="report counts and exit")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    i, _, k = args.shard.partition("/")
    shard = (int(i), int(k))
    if not 0 <= shard[0] < shard[1]:
        parser.error("--shard needs 0 <= I < K")

    d, n, bound = args.dimension, args.vertices, Fraction(args.bound)
    directory = certificate_dir(d, n)
    source = source_for(d)

    log_path = Path("output/elimination") / f"d{d}-n{n:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a") as handle:
            handle.write(line + "\n")

    def screening_progress(index: int, record, row) -> None:
        if (index + 1) % 50 == 0:
            log(f"screening {index + 1} types ...")

    screening = Screening.load_or_run(source, n, directory, restarts=args.restarts,
                                      progress=screening_progress)
    campaign = Campaign(d, n, bound, args.winner, source=source, screening=screening,
                        shard=shard, hardest_first=args.hardest_first,
                        retry_failed=args.retry_failed)
    if args.status:
        status = campaign.status()
        log(f"d={d} n={n} bound={bound}: {status.proved} proved, {status.failed} failed, "
            f"{status.not_certifiable} not certifiable, {status.pending} pending of "
            f"{status.total}; failed attempts took {status.seconds:.0f}s")
        return 0

    from inscribed.elimination.campaign import run

    run(campaign, limit=args.limit, log=log)
    status = campaign.status()
    log(f"done: {status.proved} proved, {status.failed} failed, "
        f"{status.not_certifiable} not certifiable, {status.pending} pending")
    return 0 if status.pending == 0 and status.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
