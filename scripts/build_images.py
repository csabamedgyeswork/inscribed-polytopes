"""Render every committed case winner in results/ to images/.

Usage:  python scripts/build_images.py [--case d3-n09]
"""

from __future__ import annotations

import argparse
import sys

from inscribed.store import IMAGES_DIR, RESULTS_DIR, load_case
from inscribed.viz import render_case


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", metavar="dD-nNN",
                        help="render a single case, e.g. d3-n09")
    args = parser.parse_args()

    if args.case:
        paths = [RESULTS_DIR / f"{args.case}.json"]
        if not paths[0].exists():
            print(f"no such case file: {paths[0]}", file=sys.stderr)
            return 1
    else:
        paths = sorted(RESULTS_DIR.glob("d*-n*.json"))

    for path in paths:
        case = load_case(path)
        out = render_case(case, IMAGES_DIR)
        print(f"d{case.dimension} n={case.n:>2}  {case.winner['label']:<24} -> {out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
