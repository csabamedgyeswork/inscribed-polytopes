"""Build vendored plantri and fetch external enumeration data into data/.

Usage:  python scripts/setup_data.py [--skip-download]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLANTRI_DIR = ROOT / "vendor" / "plantri"
DATA_DIR = ROOT / "data"

# Rational realisations of every simplicial 4-polytope with n vertices, most of
# them inscribed in S^3. Numbering follows Frank Lutz's enumeration.
# Index: https://firsching.ch/polytopes
FIRSCHING_BASE = "ftp://ftp.imp.fu-berlin.de/pub/moritz/inscribe"
FIRSCHING_FILES = {
    n: f"polytopalsimplicialspheres{n}_4.txt" for n in range(5, 11)
}


def build_plantri() -> Path:
    binary = PLANTRI_DIR / "plantri"
    source = PLANTRI_DIR / "plantri.c"
    if binary.exists() and binary.stat().st_mtime >= source.stat().st_mtime:
        print(f"plantri already built: {binary}")
        return binary
    print(f"compiling {source.name} ...")
    subprocess.run(
        ["cc", "-O3", "-o", str(binary), str(source)], check=True, cwd=PLANTRI_DIR
    )
    count = subprocess.run(
        [str(binary), "-u", "10"], capture_output=True, text=True, check=True
    )
    print(f"  {count.stderr.strip() or count.stdout.strip()}")
    return binary


def fetch(url: str, target: Path, *, timeout: int = 300) -> bool:
    """Download with curl; urllib stalls on this FTP server."""
    if target.exists() and target.stat().st_size > 0:
        print(f"  have {target.name} ({target.stat().st_size:,} bytes)")
        return True
    try:
        subprocess.run(
            ["curl", "-sS", "--max-time", str(timeout), "-o", str(target), url],
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"  FAILED {url}: {exc}")
        return False
    if not target.exists() or target.stat().st_size == 0:
        target.unlink(missing_ok=True)
        print(f"  FAILED {url}: empty response")
        return False
    print(f"  got {target.name} ({target.stat().st_size:,} bytes)")
    return True


def download_firsching() -> int:
    DATA_DIR.mkdir(exist_ok=True)
    print(f"fetching Firsching realisations into {DATA_DIR} ...")
    ok = 0
    for n, name in sorted(FIRSCHING_FILES.items()):
        if fetch(f"{FIRSCHING_BASE}/{name}", DATA_DIR / name):
            ok += 1
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-download", action="store_true",
                        help="build plantri only, do not fetch Firsching data")
    args = parser.parse_args()
    build_plantri()
    if args.skip_download:
        print("skipping download")
        return 0
    got = download_firsching()
    if got != len(FIRSCHING_FILES):
        print(
            f"\n{got}/{len(FIRSCHING_FILES)} files retrieved. "
            f"Check the file names listed at {FIRSCHING_BASE}."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
