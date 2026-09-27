"""Static PNG figures: three orthographic views of a case winner."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..catalog import known
from ..store.case import CaseResult
from .projection import schlegel_projection
from .views import VIEWS

DPI = 300
_LABEL_GAP = 0.20


def render_case(case: CaseResult, out_dir: Path) -> Path:
    """Render a case's winning polytope to ``out_dir/dD-nNN.png``."""
    cfg = case.winner_configuration()
    out = Path(out_dir) / f"d{case.dimension}-n{case.n:02d}.png"
    return render_configuration(cfg, out)


def render_configuration(cfg, path: Path) -> Path:
    """Render a Configuration as three orthographic views. Returns the path."""
    points, cx = cfg.points, cfg.complex
    if cfg.d == 4:
        points, _ = schlegel_projection(points, cx)
        # Centre and scale the bounding box to the frame; a Schlegel diagram
        # carries no metric meaning, so per-figure rescaling is free.
        points = points - points.mean(axis=0)
        points = points / np.abs(points).max()

    # Width is 3x the usable panel height (after the title band) so the square
    # box aspect leaves no horizontal slack between the three views.
    fig = plt.figure(figsize=(3 * 3.7 * 0.93, 3.7))
    for i, (_, elev, azim) in enumerate(VIEWS):
        ax = fig.add_subplot(1, len(VIEWS), i + 1, projection="3d")
        if cfg.d == 3:
            _draw_silhouette(ax, elev, azim)
            _draw_facets(ax, points, cx)
        else:
            _draw_edges(ax, points, cx)
        _label_vertices(ax, points, elev, azim)
        ax.view_init(elev=elev, azim=azim)
        ax.set_proj_type("ortho")
        # A Schlegel wireframe is anisotropic and never fills a cube-framed
        # panel the way a near-spherical solid does, so it gets more zoom.
        _frame(ax, zoom=1.5 if cfg.d == 3 else 1.68)
    fig.suptitle(_caption(cfg), fontsize=10, y=0.98)
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.93, wspace=-0.06)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return out


def _caption(cfg) -> str:
    label = (cfg.label or "configuration").capitalize()
    parts = [f"{label}   d={cfg.d} n={cfg.n}   volume={cfg.volume:.6f}"]
    record = known(cfg.d, cfg.n)
    if record is not None:
        parts.append(f"known={record.value:.6f} ({record.name})")
    return "   ".join(parts)


def _draw_facets(ax, points, cx) -> None:
    for f in cx.facets:
        tri = points[list(f)]
        ax.plot_trisurf(*tri.T, color="#9fb8c8", alpha=0.25,
                        edgecolor="k", linewidth=0.5)


def _draw_edges(ax, points, cx) -> None:
    for edge in cx.edges():
        ax.plot(*points[list(edge)].T, "k-", linewidth=0.6)


def _label_vertices(ax, points, elev: float, azim: float) -> None:
    """Place p_i beside its vertex.

    The nudge is applied in the screen plane, not in 3D: a vertex on the far
    side of the sphere projects near the image centre, so a radial 3D offset
    would still drop its label on top of the solid.
    """
    u, v = _screen_basis(elev, azim)
    for i, p in enumerate(points):
        screen = np.array([p @ u, p @ v])
        norm = np.linalg.norm(screen)
        away = screen / norm if norm > 1e-9 else np.array([1.0, 0.0])
        offset = _LABEL_GAP * (away[0] * u + away[1] * v)
        ax.text(*(p + offset), f"$p_{{{i}}}$", fontsize=8, ha="center", va="center")


def _screen_basis(elev: float, azim: float) -> tuple[np.ndarray, np.ndarray]:
    """Two orthonormal vectors spanning the plane facing the camera."""
    e, a = np.radians(elev), np.radians(azim)
    direction = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    helper = np.array([0.0, 0.0, 1.0])
    if abs(direction @ helper) > 0.9:
        helper = np.array([1.0, 0.0, 0.0])
    u = np.cross(direction, helper)
    u /= np.linalg.norm(u)
    return u, np.cross(direction, u)


def _draw_silhouette(ax, elev: float, azim: float) -> None:
    """The unit sphere's outline as seen from this view: a single faint circle."""
    u, v = _screen_basis(elev, azim)
    t = np.linspace(0, 2 * np.pi, 200)
    circle = np.outer(np.cos(t), u) + np.outer(np.sin(t), v)
    ax.plot(*circle.T, color="gray", linewidth=0.5, alpha=0.5)


def _frame(ax, zoom: float) -> None:
    # zoom counteracts the padding mplot3d reserves around the data.
    ax.set_box_aspect((1, 1, 1), zoom=zoom)
    ax.set_axis_off()
    limit = 1.0 + _LABEL_GAP + 0.08
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
