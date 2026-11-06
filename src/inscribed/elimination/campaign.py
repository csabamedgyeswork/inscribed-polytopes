"""Certify every rival type of a case below a common volume bound, resumably."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ..certify import CertificateVerifier, cyclic_types
from ..enumeration import TypeRecord
from ..sos import (
    CertificateRounder,
    CertificateSearch,
    CertificateStore,
    SolverSettings,
    enumerated_types,
)
from .screening import Screening


@dataclass(frozen=True)
class Attempt:
    """A failed certificate search, kept so it is not retried blindly."""

    label: str
    bound: str
    status: str
    margin: float
    seconds: float

    def save(self, directory: Path) -> Path:
        path = directory / f"{self.label}.attempt.json"
        path.write_text(json.dumps(vars(self), indent=1))
        return path


@dataclass
class Status:
    """Counts over the work list of a campaign."""

    proved: int = 0
    failed: int = 0
    not_certifiable: int = 0
    pending: int = 0
    total: int = 0
    seconds: float = 0.0


class Campaign:
    """One case: screen, then certify every rival below ``bound``.

    The cache is the certificate directory itself: a stored summary at a bound at
    most this campaign's counts as done, and an ``attempt.json`` at a bound at
    least this campaign's counts as failed. Types are easiest first by screened
    signed maximum, unless ``hardest_first``.
    """

    def __init__(self, d: int, n: int, bound: Fraction, winner: str,
                 *, source, screening: Screening,
                 directory: Path | None = None, shard: tuple[int, int] = (0, 1),
                 hardest_first: bool = False, retry_failed: bool = False):
        self.d, self.n, self.bound, self.winner = d, n, bound, winner
        self.store = CertificateStore(d, n)
        self.directory = directory if directory is not None else self.store.directory
        self.verifier = CertificateVerifier(resolvers=(cyclic_types, enumerated_types))
        self.records = [record for record in source.types(n) if record.label != winner]
        self.screening = screening
        self.shard = shard
        self.hardest_first = hardest_first
        self.retry_failed = retry_failed

    def work_list(self) -> list[TypeRecord]:
        """Types still to certify in this shard, ordered easiest or hardest first."""
        todo = [record for record in self.records if not self._done(record.label)]
        if self.hardest_first:
            todo.sort(key=lambda r: -self.screening.signed(r.label))
        else:
            todo.sort(key=lambda r: self.screening.signed(r.label))
        i, k = self.shard
        return todo[i::k]

    def _done(self, label: str) -> bool:
        stored = self.store.stored_bound(label)
        if stored is not None and stored <= self.bound:
            return True
        attempt = self.attempt_path(label)
        if not self.retry_failed and attempt.exists():
            previous = json.loads(attempt.read_text())
            if Fraction(previous["bound"]) >= self.bound:
                return True
        return False

    def attempt_path(self, label: str) -> Path:
        return self.directory / f"{label}.attempt.json"

    def run_one(self, record: TypeRecord, *, denominator: int = 10**9,
                solver: str = "CLARABEL", verbose: bool = False) -> bool:
        """Search, round, verify and store one certificate; False on any failure."""
        started = time.time()
        search = CertificateSearch.for_volume_bound(record.complex, self.bound)
        solution = search.solve(SolverSettings(solver, verbose))
        seconds = time.time() - started
        if not solution.margin > 0:
            Attempt(record.label, str(self.bound), solution.status,
                    solution.margin, seconds).save(self.directory)
            return False

        rounded = CertificateRounder(denominator).round(search, solution, record.label)
        report = self.verifier.verify(rounded.certificate)
        if not report.ok:
            Attempt(record.label, str(self.bound), "verification failed",
                    solution.margin, time.time() - started).save(self.directory)
            return False

        self.store.save_certificate(rounded.certificate, report, margin=solution.margin,
                                    largest_repair=str(rounded.largest_repair), solver=solver)
        self.attempt_path(record.label).unlink(missing_ok=True)
        return True

    def status(self) -> Status:
        """Counts over every rival: proved, failed, not certifiable, pending."""
        status = Status(total=len(self.records))
        for record in self.records:
            stored = self.store.stored_bound(record.label)
            if stored is not None and stored <= self.bound:
                status.proved += 1
                continue
            attempt = self.attempt_path(record.label)
            if attempt.exists():
                previous = json.loads(attempt.read_text())
                status.failed += 1
                status.seconds += previous.get("seconds", 0.0)
                continue
            if self.screening.signed(record.label) >= float(self.bound):
                status.not_certifiable += 1
                continue
            status.pending += 1
        return status


def run(campaign: Campaign, *, limit: int | None = None,
        log: Callable[[str], None] = print) -> Status:
    """Certify the work list, logging one line per type; returns the status."""
    work = campaign.work_list()
    if limit is not None:
        work = work[:limit]
    log(f"d={campaign.d} n={campaign.n} bound={campaign.bound}: "
        f"{len(work)} types in this shard, {len(campaign.records)} rivals in total")
    for index, record in enumerate(work, start=1):
        signed = campaign.screening.signed(record.label)
        if signed >= float(campaign.bound):
            log(f"[{index}/{len(work)}] {record.label}: signed max {signed:.6f} >= bound; "
                "not certifiable at this bound")
            continue
        started = time.time()
        log(f"[{index}/{len(work)}] {record.label}: seeking volume <= {campaign.bound} "
            f"(signed max {signed:.6f})")
        ok = campaign.run_one(record)
        log(f"[{index}/{len(work)}] {record.label}: {'PROVED' if ok else 'failed'} "
            f"after {time.time() - started:.0f}s")
    return campaign.status()
