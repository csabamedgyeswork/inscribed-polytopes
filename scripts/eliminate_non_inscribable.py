"""Certify the two non-inscribable d = 4, n = 9 types below the campaign bound.

    python scripts/eliminate_non_inscribable.py
    python scripts/eliminate_non_inscribable.py --type 323
    python scripts/eliminate_non_inscribable.py --bound 0.78 --status

FirschingSource skips firsching-9-323 and firsching-9-324, whose published
realisations are not inscribed, so scripts/eliminate.py never reaches them. A
certificate bounds the signed volume over all points on the sphere, convex or not, so
it exists for these types too; with both, the bound on the rivals no longer depends
on which types are inscribable. Certificates are stored next to the others in
results/certificates/d4-n09/, and the log is appended to output/elimination/d4-n09.log.
--type picks one of the two, so each can run in its own terminal.
"""

from __future__ import annotations

import argparse
import sys
import time
from fractions import Fraction
from pathlib import Path

from inscribed.certify import CertificateVerifier, cyclic_types, facet_sets
from inscribed.elimination import Campaign, Screening
from inscribed.enumeration import FirschingSource, TypeRecord
from inscribed.enumeration.firsching import _parse_rational_points
from inscribed.geometry import oriented_hull
from inscribed.sos import enumerated_types

D, N = 4, 9
WINNER = "firsching-9-10"
NON_INSCRIBABLE = (323, 324)


def non_inscribable_records(source: FirschingSource,
                            indices: tuple[int, ...]) -> dict[str, TypeRecord]:
    lines = source._lines(N)
    records = {}
    for index in indices:
        cx = oriented_hull(_parse_rational_points(lines[index]))
        cx.check()
        label = f"firsching-{N}-{index}"
        records[label] = TypeRecord(complex=cx, label=label)
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bound", default="0.78", help="volume bound, as an exact decimal")
    parser.add_argument("--type", type=int, choices=NON_INSCRIBABLE, default=None,
                        help="certify only firsching-9-<TYPE> (default: both)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="redo types with a failed attempt at this bound")
    parser.add_argument("--status", action="store_true", help="report and exit")
    args = parser.parse_args()
    bound = Fraction(args.bound)

    log_path = Path("output/elimination") / f"d{D}-n{N:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a") as handle:
            handle.write(line + "\n")

    source = FirschingSource()
    indices = NON_INSCRIBABLE if args.type is None else (args.type,)
    records = non_inscribable_records(source, indices)

    def resolve(cert):
        record = records.get(cert.label)
        return facet_sets(record.complex.facets) if record is not None else None

    campaign = Campaign(D, N, bound, WINNER, source=source, screening=Screening([]),
                        retry_failed=args.retry_failed)
    campaign.verifier = CertificateVerifier(
        resolvers=(cyclic_types, enumerated_types, resolve))

    todo = [record for label, record in records.items() if not campaign._done(label)]
    if args.status:
        for label in records:
            stored = campaign.store.stored_bound(label)
            state = ("proved" if stored is not None and stored <= bound
                     else "failed" if campaign.attempt_path(label).exists() else "pending")
            log(f"{label}: {state} at bound {bound}")
        return 0

    log(f"d={D} n={N} bound={bound}: {len(todo)} non-inscribable types to certify")
    failed = 0
    for index, record in enumerate(todo, start=1):
        started = time.time()
        log(f"[{index}/{len(todo)}] {record.label}: seeking volume <= {bound} "
            "(non-inscribable)")
        ok = campaign.run_one(record)
        failed += not ok
        log(f"[{index}/{len(todo)}] {record.label}: {'PROVED' if ok else 'failed'} "
            f"after {time.time() - started:.0f}s")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
