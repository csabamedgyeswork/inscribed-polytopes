"""JSON serialisation for Configuration records; float repr round-trips exactly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..combinatorics import OrientedComplex
from .case import CaseResult, TypeEntry
from .configuration import Configuration
from .search_record import SearchRecord

SCHEMA_VERSION = 1


def _to_dict(cfg: Configuration) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "label": cfg.label,
        "source": cfg.source,
        "volume": cfg.volume,
        "d": cfg.d,
        "n": cfg.n,
        "points": cfg.points.tolist(),
        "facets": [list(f) for f in cfg.complex.facets],
        "metadata": cfg.metadata,
    }


def _from_dict(data: dict) -> Configuration:
    version = data.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version}")
    points = np.asarray(data["points"], dtype=float)
    facets = tuple(tuple(int(v) for v in f) for f in data["facets"])
    cx = OrientedComplex(d=int(data["d"]), n=int(data["n"]), facets=facets)
    return Configuration(
        points=points,
        complex=cx,
        volume=float(data["volume"]),
        label=data.get("label", ""),
        source=data.get("source", "manual"),
        metadata=data.get("metadata", {}),
    )


def save(cfg: Configuration, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_dict(cfg), indent=2))
    return path


def load(path: str | Path) -> Configuration:
    return _from_dict(json.loads(Path(path).read_text()))


def save_many(configs: list[Configuration], directory: str | Path) -> list[Path]:
    directory = Path(directory)
    paths = []
    for i, cfg in enumerate(configs):
        stem = cfg.label or f"config-{i}"
        candidate = directory / f"{stem}.json"
        if candidate in paths:
            candidate = directory / f"{stem}-{i}.json"
        paths.append(candidate)
        save(cfg, candidate)
    return paths


def load_many(directory: str | Path) -> list[Configuration]:
    directory = Path(directory)
    return [load(p) for p in sorted(directory.glob("*.json"))]


def _case_to_dict(case: CaseResult) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dimension": case.dimension,
        "n": case.n,
        "evidence": case.evidence.value,
        "assumptions": list(case.assumptions),
        "enumeration": case.enumeration,
        "budget": case.budget,
        "reference": case.reference,
        "winner": case.winner,
        "triage": case.triage,
        "types": [
            {
                "label": t.label,
                "degrees": list(t.degrees),
                "feasible": t.feasible,
                "volume": t.volume,
                "signed_max": t.signed_max,
                "rejected": t.rejected,
                "restarts": t.restarts,
            }
            for t in case.types
        ],
        "provenance": case.provenance,
    }


def _case_from_dict(data: dict) -> CaseResult:
    version = data.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version}")
    types = tuple(
        TypeEntry(
            label=t["label"],
            degrees=tuple(int(x) for x in t["degrees"]),
            feasible=bool(t["feasible"]),
            volume=None if t["volume"] is None else float(t["volume"]),
            signed_max=float(t["signed_max"]),
            rejected=int(t["rejected"]),
            restarts=int(t["restarts"]),
        )
        for t in data["types"]
    )
    return CaseResult(
        dimension=int(data["dimension"]),
        n=int(data["n"]),
        assumptions=tuple(data.get("assumptions", ())),
        enumeration=dict(data.get("enumeration", {})),
        budget=dict(data.get("budget", {})),
        reference=data.get("reference"),
        winner=dict(data["winner"]),
        triage=dict(data["triage"]),
        types=types,
        provenance=dict(data.get("provenance", {})),
    )


def save_case(case: CaseResult, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_case_to_dict(case), indent=2))
    return path


def load_case(path: str | Path) -> CaseResult:
    return _case_from_dict(json.loads(Path(path).read_text()))


def _search_to_dict(record: SearchRecord) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dimension": record.dimension,
        "n": record.n,
        "evidence": record.evidence.value,
        "volume": record.volume,
        "degrees": list(record.degrees),
        "convex": record.convex,
        "symmetry": record.symmetry,
        "budget": record.budget,
        "points": [list(row) for row in record.points],
        "facets": [list(f) for f in record.facets],
        "provenance": record.provenance,
    }


def _search_from_dict(data: dict) -> SearchRecord:
    version = data.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(f"unsupported schema_version {version}")
    return SearchRecord(
        dimension=int(data["dimension"]),
        n=int(data["n"]),
        volume=float(data["volume"]),
        points=tuple(tuple(float(x) for x in row) for row in data["points"]),
        facets=tuple(tuple(int(v) for v in f) for f in data["facets"]),
        degrees=tuple(int(x) for x in data["degrees"]),
        convex=bool(data["convex"]),
        symmetry=dict(data.get("symmetry", {})),
        budget=dict(data.get("budget", {})),
        provenance=dict(data.get("provenance", {})),
    )


def save_search(record: SearchRecord, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_search_to_dict(record), indent=2))
    return path


def load_search(path: str | Path) -> SearchRecord:
    return _search_from_dict(json.loads(Path(path).read_text()))
