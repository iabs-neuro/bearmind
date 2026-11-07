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
SEED_IN = dict(marker="o", linestyle="none", markersize=2.5, markerfacecolor="white", markeredgecolor="black",
               markeredgewidth=0.3)
SEED_OUT = dict(marker="o", linestyle="none", markersize=4.5, markerfacecolor="red", markeredgecolor="white",
                markeredgewidth=0.5)
UNREADABLE = "unreadable"


def _projection(paths, sid, frame_shape):
    """The session's saved mean frame, activity map and strong seeds; None when missing or of another frame
    size (left from an older detection); UNREADABLE when the file cannot be read (one damaged file must not
    cost the whole overview)."""
    f = paths.crop_projection(sid)
    if not f.is_file():
        return None
    try:
        with np.load(f) as z:
            out = {k: z[k].astype(np.float32) for k in ("mean", "activity") if k in z.files}
            if "seeds" in z.files:
                out["seeds"] = np.asarray(z["seeds"], dtype=np.int64).reshape(-1, 2)
    except Exception:
        return UNREADABLE
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


def _is_box(box) -> bool:
    return isinstance(box, (list, tuple)) and len(box) == 4


def _outside(seeds, box) -> np.ndarray:
    """Mask of the seeds outside the mouse box (none without a placed box: nothing to judge)."""
    if seeds is None or not len(seeds) or not _is_box(box):
        return np.zeros(0 if seeds is None else len(seeds), bool)
    top, left, h, w = box
    y, x = seeds[:, 0], seeds[:, 1]
    return (y < top) | (y >= top + h) | (x < left) | (x >= left + w)


def _caption(sid, m, n_out) -> str:
    box = m.get("box")
    size = f"{box[2]}x{box[3]}" if _is_box(box) else "none"
    why = str(m.get("fov_method", "brightness"))
    if m.get("activity_pos_neg") is not None:
        why += f", pos/neg {m['activity_pos_neg']:.3g}"
    text = f"{sid}  box {size}  ({why})"
    return text + (f"  {n_out} strong seeds outside the box" if n_out else "")


def _rect(ax, box, style):
    from matplotlib.patches import Rectangle

    if _is_box(box):
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
    fig = Figure(figsize=(cols * PANEL_IN, rows * (PANEL_IN + 0.35) + 0.7), dpi=DPI)
    FigureCanvasAgg(fig)
    for i, (sid, m, proj) in enumerate(items):
        r, k = divmod(i, PER_ROW)
        ax_mean = fig.add_subplot(rows, cols, r * cols + 2 * k + 1)
        ax_act = fig.add_subplot(rows, cols, r * cols + 2 * k + 2)
        empty = "projection unreadable" if proj is UNREADABLE else "no projection"
        proj = proj if isinstance(proj, dict) else None
        mean = None if proj is None else proj["mean"]
        act = None if proj is None else proj.get("activity")
        lo, hi = (0, 1) if mean is None else np.percentile(mean, [1, 99.5])
        _panel(ax_mean, mean, m["frame_shape"], m, empty, "gray", lo, max(hi, lo + 1))
        top = 1.0 if act is None else float(max(act.max(), 1e-6))
        _panel(ax_act, act, m["frame_shape"], m, "no activity map" if proj is not None else empty, "magma", 0.0, top)
        seeds = None if proj is None else proj.get("seeds")
        out = _outside(seeds, m.get("box"))
        if seeds is not None and len(seeds):
            ax_act.plot(seeds[~out, 1], seeds[~out, 0], **SEED_IN)
            ax_act.plot(seeds[out, 1], seeds[out, 0], **SEED_OUT)
        ax_mean.set_title(_caption(sid, m, int(out.sum())), loc="left", fontsize=8,
                          color="red" if out.any() else "black")
    handles = [Line2D([], [], **LIT), Line2D([], [], **FIELD), Line2D([], [], **BOX), Line2D([], [], **SEED_IN),
               Line2D([], [], **SEED_OUT)]
    labels = ["lit area", "field (crop.fov)", "mouse box", "strong seed (cn >= 0.95, pnr >= 15)",
              "strong seed outside the mouse box"]
    fig.legend(handles, labels, loc="upper center", ncol=5, fontsize=8, frameon=False,
               title="mean frame | activity map; box height x width, px; (fov_method, activity pos/neg)",
               title_fontsize=8)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.75 / fig.get_figheight(),
                        wspace=0.04, hspace=0.18)
    path = paths.crop_overview
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial.png")
    try:
        fig.savefig(tmp, format="png", dpi=DPI)
        fileio.replace(tmp, path)  # retried while an image viewer holds the old file
    finally:
        tmp.unlink(missing_ok=True)
    return path
