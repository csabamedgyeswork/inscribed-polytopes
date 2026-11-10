"""The convexity-certificate campaign: one exact certificate per rival type, recorded in a ledger.

Certificates are not stored. Each proved type is one ledger line with its bound, the route
that succeeded, the margin, the type fingerprint and the SHA-256 of the certificate; the
search is deterministic, so ``verify`` rebuilds the certificate from the committed catalogue,
checks it in exact arithmetic and compares the hash. A rebuilt certificate that passes every
check is a valid proof for its type even if its hash differs (another solver build): the
solver only proposes the rational data.
"""

from __future__ import annotations

import gzip
import io
import json
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from fractions import Fraction
from multiprocessing.pool import Pool
from pathlib import Path

from ..certify import ConvexityCertificate, ConvexityVerifier
from ..certify.convexity import canonical_json
from ..sos import ROUTES, ConvexitySearch
from ..store.provenance import capture
from .catalogue import SHARD, TypeCatalogue, is_cycle_join

DONE = ("proved", "winner")


def below_two_pentagons(bound: Fraction) -> bool:
    """Exactly: bound < 25 (10 + 2 sqrt 5) / 384 = 0.942196..., the volume of two orthogonal
    regular pentagons. With x = 384 bound / 25 - 10 this is x < 2 sqrt 5."""
    x = Fraction(384, 25) * bound - 10
    return x < 0 or x * x < 20


@dataclass(frozen=True)
class LedgerEntry:
    label: str
    bound: str
    status: str
    route: str | None
    margin: float | None
    sha256: str | None
    type_fp: str
    seconds: float
    attempts: tuple[tuple[str, float], ...] = ()
    error: str | None = None

    @classmethod
    def from_json(cls, data: dict) -> LedgerEntry:
        return cls(**{**data, "attempts": tuple(tuple(a) for a in data.get("attempts", ()))})


class Ledger:
    """Append-only JSON lines in shards of 10,000 labels; a later line for a label replaces an
    earlier one. A complete shard is rewritten sorted and gzipped. Only the parent writes. A
    line cut short by a killed run is ignored, so that type is simply run again."""

    def __init__(self, directory: Path, catalogue: TypeCatalogue):
        self.directory, self.catalogue = Path(directory), catalogue

    def _stem(self, label: str) -> Path:
        lo = self.catalogue.index(label) // SHARD * SHARD
        return self.directory / f"{self.catalogue.prefix}-{lo:06d}-{lo + SHARD - 1:06d}"

    def entries(self) -> dict[str, LedgerEntry]:
        out: dict[str, LedgerEntry] = {}
        for path in sorted(self.directory.glob("*.jsonl*")):
            raw = path.read_bytes()
            text = (gzip.decompress(raw) if path.suffix == ".gz" else raw).decode()
            for line in text.splitlines():
                try:
                    entry = LedgerEntry.from_json(json.loads(line))
                except (ValueError, TypeError):
                    continue
                out[entry.label] = entry
        return out

    def append(self, entry: LedgerEntry) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self._stem(entry.label)
        sealed = stem.with_suffix(".jsonl.gz")
        plain = stem.with_suffix(".jsonl")
        if sealed.exists():
            plain.write_bytes(gzip.decompress(sealed.read_bytes()))
            sealed.unlink()
        cut = False
        if plain.exists() and plain.stat().st_size:
            with plain.open("rb") as fh:
                fh.seek(-1, io.SEEK_END)
                cut = fh.read(1) != b"\n"
        with plain.open("a") as fh:
            fh.write(("\n" if cut else "") + json.dumps(asdict(entry), separators=(",", ":"))
                     + "\n")

    def seal(self) -> list[Path]:
        """Gzip every shard in which every label of the catalogue has a finished entry."""
        entries, sealed, total = self.entries(), [], len(self.catalogue)
        for plain in sorted(self.directory.glob("*.jsonl")):
            lo = int(plain.stem.rsplit("-", 2)[1])
            labels = [self.catalogue.label(k) for k in range(lo, min(lo + SHARD, total))]
            if not all(entries.get(label) and entries[label].status in DONE for label in labels):
                continue
            text = "".join(json.dumps(asdict(entries[label]), separators=(",", ":")) + "\n"
                           for label in labels)
            buffer = io.BytesIO()
            with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, filename="") as fh:
                fh.write(text.encode())
            plain.with_suffix(".jsonl.gz").write_bytes(buffer.getvalue())
            plain.unlink()
            sealed.append(plain)
        return sealed


@dataclass(frozen=True)
class Outcome:
    entry: LedgerEntry
    certificate: ConvexityCertificate | None


_CONTEXT: dict = {}


def _init(directory: str, prefix: str, d: int, n: int, threshold) -> None:
    catalogue = TypeCatalogue(Path(directory), prefix, d, n)
    _CONTEXT.update(catalogue=catalogue, verifier=ConvexityVerifier(
        resolvers=(catalogue.resolver,), threshold=threshold))


def certify(label: str, bound: Fraction, routes: tuple[str, ...] = ROUTES,
            winner: tuple[int, int] = (5, 5)) -> Outcome:
    """Search the routes in order; the first certificate that passes every exact check wins.
    An error in a route is recorded and the next route is tried, so one type cannot stop a
    campaign."""
    catalogue, verifier = _CONTEXT["catalogue"], _CONTEXT["verifier"]
    started = time.time()
    attempts: list[tuple[str, float]] = []
    errors: list[str] = []
    facets = catalogue.facets(label)

    def entry(status, route=None, margin=None, sha=None):
        return LedgerEntry(label, str(bound), status, route,
                           None if margin is None else round(margin, 6), sha,
                           catalogue.fingerprint(label), round(time.time() - started, 1),
                           tuple(attempts), "; ".join(errors) or None)

    if is_cycle_join(facets, catalogue.n, *winner):
        return Outcome(entry("winner"), None)
    try:
        search = ConvexitySearch(catalogue.complex(label), bound, label)
    except Exception as exc:
        errors.append(f"setup: {type(exc).__name__}: {exc}")
        return Outcome(entry("failed"), None)
    for route in routes:
        try:
            solution = search.solve(route)
            attempts.append((route, round(solution.margin, 6)))
            if not solution.margin > 0:
                continue
            cc = search.round_exact(solution, route)
            if verifier.check(cc).ok:
                return Outcome(entry("proved", route, solution.margin, cc.digest()), cc)
        except Exception as exc:
            errors.append(f"{route}: {type(exc).__name__}: {exc}")
    return Outcome(entry("failed"), None)


def _certify_job(job):
    label, bound, routes, winner = job
    return certify(label, bound, routes, winner)


def _check_file(job):
    path, bound = job
    raw = Path(path).read_bytes()
    cc = ConvexityCertificate.from_json(json.loads(gzip.decompress(raw) if path.endswith(".gz")
                                                   else raw))
    catalogue, verifier = _CONTEXT["catalogue"], _CONTEXT["verifier"]
    same = (cc.label in catalogue and cc.bound == bound
            and tuple(cc.certificate.facets) == catalogue.facets(cc.label))
    return cc.label, same and verifier.check(cc).ok, bool(cc.multipliers), cc.digest()


class ConvexCampaign:
    """Every type of a catalogue except the winner, the join of cycles of lengths ``winner``,
    at one bound, resumably. ``threshold`` must be a module-level function (it is pickled)."""

    def __init__(self, catalogue: TypeCatalogue, ledger: Ledger, bound: Fraction,
                 *, threshold: Callable[[Fraction], bool] = below_two_pentagons,
                 winner: tuple[int, int] = (5, 5), workers: int = 1,
                 log: Callable[[str], None] = print):
        if not threshold(bound):
            raise ValueError(f"bound {bound} does not exclude a rival")
        self.catalogue, self.ledger, self.bound = catalogue, ledger, bound
        self.threshold, self.winner, self.workers, self.log = threshold, winner, workers, log

    def _pool(self) -> Pool:
        c = self.catalogue
        return Pool(self.workers, initializer=_init,
                    initargs=(str(c.directory), c.prefix, c.d, c.n, self.threshold))

    def done(self, entry: LedgerEntry | None, label: str) -> bool:
        return (entry is not None and entry.status in DONE and Fraction(entry.bound) == self.bound
                and entry.type_fp == self.catalogue.fingerprint(label))

    def pending(self, labels: Iterable[str], retry_failed: bool = False) -> list[str]:
        entries = self.ledger.entries()
        return [label for label in labels if not self.done(entries.get(label), label)
                and (retry_failed or label not in entries or entries[label].status != "failed"
                     or Fraction(entries[label].bound) != self.bound)]

    def run(self, labels: list[str], *, routes: tuple[str, ...] = ROUTES,
            retry_failed: bool = False) -> dict[str, int]:
        todo = self.pending(labels, retry_failed)
        self.log(f"{len(labels)} labels, {len(labels) - len(todo)} already done")
        counts: dict[str, int] = {}
        with self._pool() as pool:
            jobs = [(label, self.bound, routes, self.winner) for label in todo]
            for k, outcome in enumerate(pool.imap_unordered(_certify_job, jobs), 1):
                e = outcome.entry
                self.ledger.append(e)
                counts[e.status] = counts.get(e.status, 0) + 1
                tried = ", ".join(f"{r} {m:+.4f}" for r, m in e.attempts)
                self.log(f"[{time.strftime('%H:%M:%S')}] {k}/{len(todo)} {e.label}: "
                         f"{e.status} ({tried}; {e.seconds:.0f}s)")
        self.ledger.seal()
        return counts

    def verify(self, labels: list[str], write: Path | None = None) -> dict[str, list[str]]:
        """Rebuild the certificate of every label with its recorded route and check it exactly.
        Returns the labels that passed with the recorded hash, passed with another hash,
        failed, or have no proved entry."""
        entries = self.ledger.entries()
        result: dict[str, list[str]] = {"match": [], "differ": [], "failed": [], "missing": [],
                                        "winner": []}
        jobs = []
        for label in labels:
            e = entries.get(label)
            if e is not None and e.status == "winner":
                result["winner"].append(label)
            elif e is None or not self.done(e, label):
                result["missing"].append(label)
            else:
                jobs.append((label, self.bound, (e.route,), self.winner))
        with self._pool() as pool:
            for outcome in pool.imap_unordered(_certify_job, jobs):
                label, cc = outcome.entry.label, outcome.certificate
                if cc is None:
                    result["failed"].append(label)
                    continue
                result["match" if cc.digest() == entries[label].sha256 else "differ"].append(label)
                if write is not None:
                    write.mkdir(parents=True, exist_ok=True)
                    (write / f"{label}.cert.json").write_bytes(canonical_json(cc.to_json()))
        self.log(" ".join(f"{k}: {len(v)}" for k, v in result.items()))
        return result

    def complete(self, audit: dict) -> bool:
        """The computations of the paper: the audit holds, and every label but the winner has
        a certificate that is rebuilt and passes every exact check."""
        labels = self.catalogue.labels()
        if not (audit.get("spheres") and audit.get("pairwise_distinct") and audit.get("winner")
                and audit.get("catalogue_sha256") == self.catalogue.sha256()):
            self.log("the audit does not establish the catalogue facts")
            return False
        result = self.verify([label for label in labels if label != audit["winner"]])
        ok = not result["failed"] and not result["missing"] and not result["winner"]
        self.log("COMPLETE: every type other than the winner is eliminated" if ok
                 else "INCOMPLETE")
        return ok

    def import_certificates(self, paths: list[Path], margins: dict[str, float] | None = None
                            ) -> dict[str, int]:
        """Record stored certificates that pass every exact check and whose facets are exactly
        the catalogue's, so that the campaign skips them and ``verify`` can rebuild them."""
        entries, counts = self.ledger.entries(), {"imported": 0, "rejected": 0, "skipped": 0}
        jobs = []
        for p in paths:
            label = p.name.split(".")[0]
            if self.done(entries.get(label), label):
                counts["skipped"] += 1
            else:
                jobs.append((str(p), self.bound))
        with self._pool() as pool:
            for label, ok, convex, sha in pool.imap_unordered(_check_file, jobs, chunksize=4):
                if not ok:
                    counts["rejected"] += 1
                    self.log(f"{label}: rejected")
                    continue
                route = "convex" if convex else "bare"
                margin = (margins or {}).get(label)
                self.ledger.append(LedgerEntry(label, str(self.bound), "proved", route, margin,
                                               sha, self.catalogue.fingerprint(label), 0.0,
                                               ((route, margin),) if margin is not None else ()))
                counts["imported"] += 1
        self.ledger.seal()
        return counts

    def stats(self) -> dict:
        entries = self.ledger.entries()
        counts: dict[str, int] = {}
        for e in entries.values():
            counts[e.status] = counts.get(e.status, 0) + 1
        size = sum(p.stat().st_size for p in self.ledger.directory.glob("*.jsonl*"))
        return {"types": len(self.catalogue), "entries": len(entries), "status": counts,
                "ledger_bytes": size}

    def manifest(self, data_sha256: str) -> dict:
        return {"bound": str(self.bound), "data_sha256": data_sha256,
                "catalogue_sha256": self.catalogue.sha256(), "types": len(self.catalogue),
                "routes": list(ROUTES), "denominator": 10**9,
                "versions": {**capture(), "scs": __import__("scs").__version__}}
