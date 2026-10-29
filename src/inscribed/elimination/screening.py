"""Cached signed-maximum screen of every type of a case, from a JSON file when present."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

from ..enumeration import TypeRecord
from ..search.maximize import maximize_type


@dataclass(frozen=True)
class ScreenRow:
    """Coarse screen of one type: convex optimum and signed maximum."""

    label: str
    volume: float
    signed: float


def _screen(job: tuple[TypeRecord, int, int]) -> ScreenRow:
    """Screen one type; module-level so worker processes can run it."""
    record, restarts, seed = job
    outcome = maximize_type(record.complex, restarts=restarts, seed=seed, start=record.points)
    return ScreenRow(label=record.label, volume=outcome.volume, signed=outcome.signed_best)


class Screening:
    """Signed maximum of every type of one case, computed once and cached to disk."""

    def __init__(self, rows: list[ScreenRow]):
        self.rows = {row.label: row for row in rows}

    def __contains__(self, label: str) -> bool:
        return label in self.rows

    def signed(self, label: str) -> float:
        return self.rows[label].signed

    @classmethod
    def path(cls, directory: Path) -> Path:
        return directory / "screening.json"

    @classmethod
    def load_or_run(
        cls, source, n: int, directory: Path, *, restarts: int = 50, seed: int = 11,
        workers: int | None = None,
        progress: Callable[[int, TypeRecord, ScreenRow], None] | None = None,
    ) -> "Screening":
        """The screening of a case, from cache or by screening every type now.

        Each type's seed is ``seed`` plus its enumeration index, so the result does
        not depend on ``workers``.
        """
        path = cls.path(directory)
        if path.exists():
            rows = [ScreenRow(**row) for row in json.loads(path.read_text())]
            return cls(rows)
        records = list(source.types(n))
        jobs = [(record, restarts, seed + index) for index, record in enumerate(records)]
        rows = []
        with Pool(workers or os.cpu_count()) as pool:
            for index, row in enumerate(pool.imap(_screen, jobs)):
                rows.append(row)
                if progress is not None:
                    progress(index, records[index], row)
        directory.mkdir(parents=True, exist_ok=True)
        payload = [vars(row) for row in rows]
        path.write_text(json.dumps(payload, indent=1))
        return cls(rows)
