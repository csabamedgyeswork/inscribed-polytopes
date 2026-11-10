"""Eliminate every rival type of d = 4, n = 10 with convexity certificates, resumably.

    python scripts/eliminate_convex.py catalogue --workers 14     # exact facets of all types
    python scripts/eliminate_convex.py audit --workers 14         # distinct types, the winner
    python scripts/eliminate_convex.py import DIR [DIR ...]       # re-verify stored certificates
    python scripts/eliminate_convex.py run --workers 14           # every pending type
    python scripts/eliminate_convex.py run --range 0 1000 --routes convex
    python scripts/eliminate_convex.py verify --sample 200 --workers 14
    python scripts/eliminate_convex.py verify --labels firsching-10-7 --write /tmp/certs
    python scripts/eliminate_convex.py verify --complete --workers 14
    python scripts/eliminate_convex.py stats

Everything is written to results/certificates/d4-n10/: the exact type catalogue (types/),
audit.json, manifest.json and the ledger (ledger/). Certificates are not stored; ``verify``
rebuilds them from the catalogue, the bound and the recorded route, and checks them exactly.
The input is Firsching's list data/polytopalsimplicialspheres10_4.txt (scripts/setup_data.py),
needed only by ``catalogue``; its SHA-256 is pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from fractions import Fraction
from pathlib import Path

from inscribed.elimination import ConvexCampaign, Ledger, TypeCatalogue
from inscribed.elimination.catalogue import write_json
from inscribed.enumeration import FirschingSource
from inscribed.sos import ROUTES
from inscribed.store import certificate_dir

D, N = 4, 10
DATA_SHA256 = "9a770e1534d987931cef533e3728d79e41d235ab42ad0e8d5e01254607c7442a"
BOUND = Fraction(471, 500)


def data_lines() -> list[str]:
    path = FirschingSource().path(N)
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise SystemExit(f"{path}: SHA-256 differs from Firsching's published list")
    return FirschingSource()._lines(N)


def margins_from_summaries(directory: Path) -> dict[str, float]:
    """The margin of the proving attempt in the LABEL.json summaries of the pilot runs."""
    out = {}
    for path in directory.glob("*[0-9].json"):
        summary = json.loads(path.read_text())
        if summary.get("status", "").startswith("proved"):
            out[summary["label"]] = summary["attempts"][-1]["margin"]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["catalogue", "audit", "import", "run", "verify",
                                            "stats"])
    parser.add_argument("paths", nargs="*", type=Path, help="certificate folders for import")
    parser.add_argument("--bound", default=str(BOUND), help="volume bound, exact")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--range", nargs=2, type=int, metavar=("START", "STOP"))
    parser.add_argument("--labels", nargs="*", default=[])
    parser.add_argument("--sample", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--routes", nargs="+", choices=ROUTES, default=list(ROUTES))
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--write", type=Path, help="also save rebuilt certificates here")
    parser.add_argument("--results", type=Path, default=certificate_dir(D, N))
    args = parser.parse_args()

    root = args.results
    catalogue = TypeCatalogue(root / "types", f"firsching-{N}", D, N)
    if args.command == "catalogue":
        print("catalogue sha256", catalogue.build(data_lines(), args.workers))
        return 0
    if not len(catalogue):
        raise SystemExit(f"no catalogue in {catalogue.directory}; run the catalogue command")
    if args.command == "audit":
        audit = catalogue.audit((5, 5), args.workers)
        write_json(root / "audit.json", audit)
        print(json.dumps(audit, indent=1))
        return 0 if audit["pairwise_distinct"] and audit["winner"] else 1

    campaign = ConvexCampaign(catalogue, Ledger(root / "ledger", catalogue),
                              Fraction(args.bound), workers=args.workers)
    labels = list(args.labels)
    if args.range:
        labels += [catalogue.label(k) for k in range(*args.range)]
    if args.sample:
        labels += random.Random(args.seed).sample(catalogue.labels(), args.sample)
    labels = list(dict.fromkeys(labels))

    if args.command == "import":
        files = [p for d in args.paths for p in sorted(d.glob("*.cert.json*"))]
        margins = {k: v for d in args.paths for k, v in margins_from_summaries(d).items()}
        print(campaign.import_certificates(files, margins))
    elif args.command == "run":
        write_json(root / "manifest.json", campaign.manifest(DATA_SHA256))
        print(campaign.run(labels or catalogue.labels(), routes=tuple(args.routes),
                           retry_failed=args.retry_failed))
    elif args.command == "verify":
        if args.complete:
            return 0 if campaign.complete(json.loads((root / "audit.json").read_text())) else 1
        result = campaign.verify(labels, args.write)
        return 0 if not result["failed"] and not result["missing"] else 1
    print(json.dumps(campaign.stats(), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
