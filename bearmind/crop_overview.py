"""Crop overview of a series: each session's lit area, field and mouse box on its mean frame and activity map.

One PNG, `_bearmind/crop_overview.png`, redrawn by the runner after every placement of boxes: the author
checks the automatic crop by looking at it once per series.
"""
import math

import numpy as np

from bearmind import fileio

PER_ROW = 3  # sessions per row, two panels each
PANEL_IN = 2.6  # inches per panel side
DPI = 110
LIT = dict(color="#ffd400", linestyle=(0, (6, 3)), linewidth=1.3)  # yellow, long dashes
FIELD = dict(color="#2ecc40", linestyle=(0, (2, 2)), linewidth=1.3)  # green, short dashes
BOX = dict(color="#00d5ff", linestyle="-", linewidth=1.3)  # cyan, solid


def _projection(paths, sid, frame_shape) -> dict | None:
    """The session's saved mean frame and activity map; None when missing, unreadable or of another frame size
    (a projection left from an older detection)."""
    f = paths.crop_projection(sid)
    if not f.is_file():
        return None
    try:
        with np.load(f) as z:
            out = {k: z[k].astype(np.float32) for k in ("mean", "activity") if k in z.files}
    except (OSError, ValueError, KeyError):
        return None
    if "mean" not in out or list(out["mean"].shape) != list(frame_shape):
        return None
    if "activity" in out and out["activity"].shape != out["mean"].shape:
        del out["activity"]
    return out


def _items(paths, records) -> list:
    out = []
    for r in records:
        st = (getattr(r, "stages", None) or {}).get("crop")
        m = getattr(st, "metrics", None) or {}
        if m.get("frame_shape"):
            out.append((r.sid, m, _projection(paths, r.sid, m["frame_shape"])))
    return out


def _caption(sid, m) -> str:
    box = m.get("box")
    size = f"{box[2]}x{box[3]}" if isinstance(box, (list, tuple)) and len(box) == 4 else "none"
    return f"{sid}  box {size}  ({m.get('fov_method', 'brightness')})"


def _rect(ax, box, style):
    from matplotlib.patches import Rectangle

    if isinstance(box, (list, tuple)) and len(box) == 4:
        top, left, h, w = box
        # pixel centres sit on integer coordinates in imshow: a box edge lies half a pixel outside them
        ax.add_patch(Rectangle((left - 0.5, top - 0.5), w, h, fill=False, **style))


def _panel(ax, img, frame_shape, m, empty_text, cmap, vmin, vmax):
    h, w = frame_shape
    ax.set_facecolor("#202020")
    if img is None:
        ax.text(w / 2, h / 2, empty_text, color="#d0d0d0", ha="center", va="center", fontsize=7)
    else:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    _rect(ax, m.get("fov_lit"), LIT)
    _rect(ax, m.get("fov"), FIELD)
    _rect(ax, m.get("box"), BOX)
    ax.set_xticks([])
    ax.set_yticks([])


def write_overview(paths, records):
    """Draw every session with crop metrics; the path of the PNG, or None (nothing drawn, matplotlib not
    imported) when no session has a saved projection."""
    items = _items(paths, records)
    if not any(p is not None for _, _, p in items):
        return None
    # the object API with the Agg canvas: no pyplot state, safe next to the monitor's Qt event loop
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.lines import Line2D

    rows = math.ceil(len(items) / PER_ROW)
    cols = 2 * min(PER_ROW, len(items))
    fig = Figure(figsize=(cols * PANEL_IN, rows * (PANEL_IN + 0.35) + 0.5), dpi=DPI)
    FigureCanvasAgg(fig)
    for i, (sid, m, proj) in enumerate(items):
        r, k = divmod(i, PER_ROW)
        ax_mean = fig.add_subplot(rows, cols, r * cols + 2 * k + 1)
        ax_act = fig.add_subplot(rows, cols, r * cols + 2 * k + 2)
        mean = None if proj is None else proj["mean"]
        act = None if proj is None else proj.get("activity")
        lo, hi = (0, 1) if mean is None else np.percentile(mean, [1, 99.5])
        _panel(ax_mean, mean, m["frame_shape"], m, "no projection", "gray", lo, max(hi, lo + 1))
        top = 1.0 if act is None else float(max(act.max(), 1e-6))
        _panel(ax_act, act, m["frame_shape"], m, "no activity map" if proj is not None else "no projection",
               "magma", 0.0, top)
        ax_mean.set_title(_caption(sid, m), loc="left", fontsize=8)
    handles = [Line2D([], [], **LIT), Line2D([], [], **FIELD), Line2D([], [], **BOX)]
    fig.legend(handles, ["lit area", "field (crop.fov)", "mouse box"], loc="upper center", ncol=3,
               fontsize=8, frameon=False, title="mean frame | activity map; box height x width, px",
               title_fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.55 / fig.get_figheight(),
                        wspace=0.04, hspace=0.18)
    out = paths.crop_overview
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.png")
    try:
        fig.savefig(tmp, format="png", dpi=DPI)
        fileio.replace(tmp, out)  # retried while an image viewer holds the old file
    finally:
        tmp.unlink(missing_ok=True)
    return out
