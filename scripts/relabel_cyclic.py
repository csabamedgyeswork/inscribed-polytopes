"""Relabel the firsching-7-0 certificate onto the vertex order of Gale's list.

    python scripts/relabel_cyclic.py

The stored certificate names the type firsching-7-0, whose vertex order differs
from the order along the moment curve used for C_4(7). This finds the
permutation between them, renames the certificate to cyclic-4-7, re-verifies
it, and replaces the stored files. The relaxation record is renamed with the
label field updated.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
from itertools import permutations

from inscribed.certify import CertificateVerifier, ExactCertificate, cyclic_facets, facet_sets
from inscribed.store import certificate_dir

OLD = "firsching-7-0"
NEW = "cyclic-4-7"


def find_permutation(cert: ExactCertificate) -> tuple[int, ...]:
    """The vertex renaming that carries the certificate's facets onto Gale's list."""
    target = cyclic_facets(cert.n, cert.d)
    source = facet_sets(cert.facets)
    for p in permutations(range(cert.n)):
        if {frozenset(p[v] for v in f) for f in source} == target:
            return tuple(p)
    raise SystemExit("no relabelling of firsching-7-0 onto C_4(7) exists")


def main() -> int:
    directory = certificate_dir(4, 7)
    payload = gzip.decompress((directory / f"{OLD}.certificate.json.gz").read_bytes())
    cert = ExactCertificate.from_json(json.loads(payload))
    permutation = find_permutation(cert)
    relabelled = cert.relabelled(permutation, NEW)

    report = CertificateVerifier().verify(relabelled)
    if not report.ok:
        raise SystemExit("the relabelled certificate does not verify; nothing written")

    summary = json.loads((directory / f"{OLD}.summary.json").read_text())
    summary.update({"label": NEW, "verification": {
        "identity": report.identity, "positive_definite": report.positive_definite,
        "coherent": report.coherent, "type_matches": report.type_matches, "ok": report.ok,
    }})

    new_payload = json.dumps(relabelled.to_json()).encode()
    summary["sha256_uncompressed"] = hashlib.sha256(new_payload).hexdigest()

    (directory / f"{NEW}.certificate.json.gz").write_bytes(gzip.compress(new_payload, mtime=0))
    (directory / f"{NEW}.summary.json").write_text(json.dumps(summary, indent=2))

    relaxation_path = directory / f"{OLD}.relaxation.json"
    if relaxation_path.exists():
        relaxation = json.loads(relaxation_path.read_text())
        relaxation["label"] = NEW
        (directory / f"{NEW}.relaxation.json").write_text(json.dumps(relaxation, indent=2))

    for suffix in ("certificate.json.gz", "summary.json", "relaxation.json"):
        path = directory / f"{OLD}.{suffix}"
        if path.exists():
            path.unlink()
    print(f"wrote {NEW}.* to {directory} and removed {OLD}.*")
    return 0


if __name__ == "__main__":
    sys.exit(main())
