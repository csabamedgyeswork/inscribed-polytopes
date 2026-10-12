"""Exact sum-of-squares certificate that the signed volume of one type stays below a bound.

    python scripts/sos_certificate.py -d 4 -n 7 --label cyclic-4-7 --bound 0.4
    python scripts/sos_certificate.py -d 4 -n 7 --all --bound 0.4
    python scripts/sos_certificate.py -d 4 -n 8 --all --skip firsching-8-23 --resume --bound 0.65
    python scripts/sos_certificate.py --verify \
        results/certificates/d4-n07/cyclic-4-7.certificate.json.gz

The certificate is an exact polynomial identity on R^(d n),

    gamma - D = sum_B v_B^T Q_B v_B + sum_i h_i (|p_i|^2 - 1),

with rational positive-definite Gram matrices Q_B and rational multipliers h_i. On the
product of unit spheres the last sum vanishes and the first is non-negative, so
D <= gamma there, hence every realisation of the type has volume <= gamma / d!. The
verifier rebuilds D from the facet list alone and checks the identity, the positivity,
the orientation and the type's identity in exact arithmetic, independently of the solver.
Only verified certificates are written.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from fractions import Fraction
from pathlib import Path

from inscribed.certify import (
    CertificateVerifier,
    ExactCertificate,
    VerificationReport,
    cyclic_types,
)
from inscribed.enumeration import FirschingSource, PlantriSource, TypeRecord
from inscribed.sos import (
    CertificateRounder,
    CertificateSearch,
    CertificateStore,
    SolverSettings,
    enumerated_types,
)

VERIFIER = CertificateVerifier(resolvers=(cyclic_types, enumerated_types))


def log(message: str) -> None:
    """Print a timestamped progress line."""
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def log_report(report: VerificationReport) -> None:
    """Print the outcome of each exact check."""
    checks = {"exact identity": report.identity,
              "Gram blocks positive definite": report.positive_definite,
              "facets coherently oriented": report.coherent,
              "type matches enumeration": report.type_matches}
    log("; ".join(f"{name}: {'OK' if passed else 'FAILED'}" for name, passed in checks.items()))


def source_for(d: int):
    """The enumeration source of one dimension."""
    source = {3: PlantriSource, 4: FirschingSource}.get(d)
    if source is None:
        raise SystemExit(f"no enumeration source for d={d}")
    return source()


def certify(record: TypeRecord, bound: Fraction, args: argparse.Namespace) -> int:
    """Search, round, verify and save a certificate for one type."""
    cx = record.complex
    log(f"type {record.label}: seeking volume <= {bound} = {float(bound):.6f}")

    search = CertificateSearch.for_volume_bound(cx, bound, level=args.level)
    solution = search.solve(SolverSettings(args.solver, args.verbose))
    log(f"status {solution.status}; smallest Gram eigenvalue margin t = {solution.margin:.3e}")
    if not solution.margin > 0:
        log("no strictly feasible certificate at this bound; raise --bound or --level")
        return 1

    rounded = CertificateRounder(args.denominator).round(search, solution, record.label)
    log(f"rounded to denominator {args.denominator}; "
        f"largest repair {float(rounded.largest_repair):.3e}")
    report = VERIFIER.verify(rounded.certificate)
    log_report(report)
    if not report.ok:
        log("verification failed; nothing written")
        return 1

    cert_path, summary_path = CertificateStore(cx.d, cx.n).save_certificate(
        rounded.certificate, report, margin=solution.margin,
        largest_repair=str(rounded.largest_repair), solver=args.solver)
    log(f"wrote {cert_path} and {summary_path}")
    log(f"PROVED: every configuration of {record.label} on the unit sphere has "
        f"signed volume <= {report.volume_bound} = {float(report.volume_bound):.12f}")
    return 0


def certify_all(records: list[TypeRecord], bound: Fraction, args: argparse.Namespace) -> int:
    """Certify every type not skipped, carrying on past failures, and report the outcome."""
    unknown = set(args.skip) - {r.label for r in records}
    if unknown:
        raise SystemExit(f"--skip names no type of this case: {', '.join(sorted(unknown))}")
    store = CertificateStore(args.dimension, args.vertices)
    proved, already, failed = [], [], []
    for index, record in enumerate(records, start=1):
        if record.label in args.skip:
            log(f"[{index}/{len(records)}] {record.label}: skipped")
            continue
        stored = store.stored_bound(record.label) if args.resume else None
        if stored is not None and stored <= bound:
            log(f"[{index}/{len(records)}] {record.label}: already certified at {stored}")
            already.append(record.label)
            continue
        log(f"[{index}/{len(records)}]")
        (failed if certify(record, bound, args) else proved).append(record.label)
    log(f"done: {len(proved)} certified, {len(already)} already certified, "
        f"{len(args.skip)} skipped, {len(failed)} failed")
    if failed:
        log(f"failed: {', '.join(failed)}")
    return 1 if failed else 0


def run_verify(path: Path) -> int:
    """Re-check a saved certificate from scratch."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as handle:
        cert = ExactCertificate.from_json(json.load(handle))
    log(f"certificate for {cert.label}: D <= {cert.gamma}, "
        f"{sum(len(b) for b in cert.blocks)} basis monomials in {len(cert.blocks)} blocks")
    report = VERIFIER.verify(cert)
    log_report(report)
    if report.ok:
        log(f"PROVED: signed volume <= {report.volume_bound} = {float(report.volume_bound):.12f}")
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", type=Path, default=None, help="re-check a saved certificate")
    parser.add_argument("--dimension", "-d", type=int)
    parser.add_argument("--vertices", "-n", type=int)
    parser.add_argument("--label")
    parser.add_argument("--all", action="store_true", help="certify every type of the case")
    parser.add_argument("--skip", action="append", default=[], metavar="LABEL",
                        help="with --all, leave out this type (repeatable), e.g. the winner")
    parser.add_argument("--resume", action="store_true",
                        help="with --all, leave out types already certified at this bound or lower")
    parser.add_argument("--bound", default="0.4",
                        help="volume bound to certify, as an exact decimal")
    parser.add_argument("--level", type=int, default=2)
    parser.add_argument("--solver", default="CLARABEL", choices=["CLARABEL", "SCS", "MOSEK"])
    parser.add_argument("--denominator", type=int, default=10**9)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.verify is not None:
        return run_verify(args.verify)
    if None in (args.dimension, args.vertices):
        parser.error("give --dimension and --vertices, or --verify")
    if not args.all and args.label is None:
        parser.error("give --label or --all")

    bound = Fraction(args.bound)
    records = list(source_for(args.dimension).types(args.vertices))
    if args.all:
        return certify_all(records, bound, args)
    matches = [r for r in records if r.label == args.label]
    if not matches:
        raise SystemExit(f"no type {args.label!r} at d={args.dimension}, n={args.vertices}")
    return certify(matches[0], bound, args)


if __name__ == "__main__":
    sys.exit(main())
