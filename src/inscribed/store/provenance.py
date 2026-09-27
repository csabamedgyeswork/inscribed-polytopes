"""Capture the dependency versions a result was computed with."""

from __future__ import annotations


def _package_version(name: str) -> str | None:
    try:
        module = __import__(name)
    except ImportError:
        return None
    return getattr(module, "__version__", None)


def capture() -> dict:
    """Snapshot the dependency versions that affect the numerics."""
    return {
        "numpy": _package_version("numpy"),
        "scipy": _package_version("scipy"),
    }
