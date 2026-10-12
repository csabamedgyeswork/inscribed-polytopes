"""Read and write the results/certificates tree."""

from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import asdict
from fractions import Fraction
from math import factorial
from pathlib import Path

from ..certify import ExactCertificate, VerificationReport
from ..store import certificate_dir, provenance
from .relaxation import RelaxationResult


class CertificateStore:
    """One directory per (dimension, n): per type a certificate, its summary, and optionally
    the floating-point relaxation record."""

    def __init__(self, dimension: int, n: int):
        self.dimension = dimension
        self.directory = certificate_dir(dimension, n)

    def _path(self, label: str, suffix: str) -> Path:
        return self.directory / f"{label}.{suffix}"

    def save_relaxation(self, label: str, result: RelaxationResult, *, level: int, solver: str,
                        group_order: int, lower_bound: float | None,
                        convex_volume: float | None) -> Path:
        """Write the ascent lower bound and the floating-point relaxation value of one type."""
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(label, "relaxation.json")
        path.write_text(json.dumps({
            "label": label, "level": level, "solver": solver, "group_order": group_order,
            "lower_bound_volume": lower_bound, "convex_volume": convex_volume,
            "relaxation_value_volume": result.value / factorial(self.dimension),
            "relaxation": asdict(result),
            "provenance": provenance.capture(),
        }, indent=2))
        return path

    def save_certificate(self, cert: ExactCertificate, report: VerificationReport,
                         *, margin: float, largest_repair: str, solver: str) -> tuple[Path, Path]:
        """Write the gzipped certificate and its summary; only verified certificates are stored."""
        if not report.ok:
            raise ValueError(f"refusing to store unverified certificate for {cert.label}")
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(cert.to_json()).encode()
        cert_path = self._path(cert.label, "certificate.json.gz")
        cert_path.write_bytes(gzip.compress(payload, mtime=0))
        summary_path = self._path(cert.label, "summary.json")
        summary_path.write_text(json.dumps({
            "label": cert.label, "d": cert.d, "n": cert.n,
            "gamma": str(cert.gamma), "volume_bound": str(cert.volume_bound),
            "verification": {
                "identity": report.identity,
                "positive_definite": report.positive_definite,
                "coherent": report.coherent,
                "type_matches": report.type_matches,
                "ok": report.ok,
            },
            "margin": margin, "largest_repair": largest_repair, "solver": solver,
            "sha256_uncompressed": hashlib.sha256(payload).hexdigest(),
            "provenance": provenance.capture(),
        }, indent=2))
        return cert_path, summary_path

    def stored_bound(self, label: str) -> Fraction | None:
        """The volume bound of a stored verified certificate, or None if there is none."""
        path = self._path(label, "summary.json")
        if not path.exists():
            return None
        summary = json.loads(path.read_text())
        if not summary["verification"]["ok"]:
            return None
        return Fraction(summary["volume_bound"])

    def load_certificate(self, label: str) -> ExactCertificate:
        """Read a stored certificate."""
        path = self._path(label, "certificate.json.gz")
        return ExactCertificate.from_json(json.loads(gzip.decompress(path.read_bytes())))
