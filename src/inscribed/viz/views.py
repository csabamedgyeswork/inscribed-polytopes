"""The canonical orthographic views: front, side, top. No plotting dependency."""

from __future__ import annotations

# (title, elev, azim) for Axes3D.view_init
VIEWS: tuple[tuple[str, float, float], ...] = (
    ("front", 0.0, -90.0),
    ("side", 0.0, 0.0),
    ("top", 90.0, -90.0),
)
