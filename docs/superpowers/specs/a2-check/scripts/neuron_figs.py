"""One figure per session of the neuron check: mean frame | activity map, old crop, new field and mouse box,
centres of the accepted components (outside the new mouse box in red).

usage: neuron_figs.py RESULT_DIR PROJECTION_ROOT FIG_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

LIT = dict(color="#ffd400", linestyle=(0, (6, 3)), linewidth=1.2)
FIELD = dict(color="#2ecc40", linestyle=(0, (2, 2)), linewidth=1.2)
BOX = dict(color="#00d5ff", linestyle="-", linewidth=1.4)
OLD = dict(color="#ff8c00", linestyle="-", linewidth=1.4)
IN = dict(marker="o", linestyle="none", markersize=2.2, markerfacecolor="#7CFC00", markeredgecolor="black",
          markeredgewidth=0.25)
OUT = dict(marker="o", linestyle="none", markersize=5, markerfacecolor="red", markeredgecolor="white",
           markeredgewidth=0.6)


def rect(ax, box, style):
    if box:
        top, left, h, w = box
        ax.add_patch(Rectangle((left - 0.5, top - 0.5), w, h, fill=False, **style))


def outside(pts, box):
    top, left, h, w = box
    y, x = pts[:, 0], pts[:, 1]
    return (y < top - 0.5) | (y > top + h - 0.5) | (x < left - 0.5) | (x > left + w - 0.5)


def main():
    res_dir, proj_root, fig_dir = map(Path, sys.argv[1:4])
    fig_dir.mkdir(parents=True, exist_ok=True)
    res = json.loads((res_dir / "neuron_check.json").read_text())
    arr = np.load(res_dir / "neuron_check_arrays.npz")
    vf = res_dir / "neuron_verify.json"
    ver = json.loads(vf.read_text()) if vf.is_file() else {}
    for sid, r in res.items():
        m = r["metrics"]
        with np.load(proj_root / "_bearmind" / "crop" / f"{sid}_projection.npz") as z:
            mean = z["mean"].astype(np.float32)
            act = z["activity"].astype(np.float32) if "activity" in z.files else None
        c = arr[f"{sid}_centres"]
        acc = arr[f"{sid}_accepted"].astype(int)
        pts = c[acc]
        out = outside(pts, r["new_box"])
        fig = Figure(figsize=(7.6, 4.3), dpi=100)
        FigureCanvasAgg(fig)
        for i, (img, cmap) in enumerate(((mean, "gray"), (act, "magma"))):
            ax = fig.add_subplot(1, 2, i + 1)
            h, w = mean.shape
            ax.set_facecolor("#202020")
            if img is not None:
                lo, hi = (np.percentile(img, [1, 99.5]) if cmap == "gray" else (0.0, max(float(img.max()), 1e-6)))
                ax.imshow(img, cmap=cmap, vmin=lo, vmax=max(hi, lo + 1e-6), interpolation="nearest")
            ax.set_xlim(-0.5, w - 0.5)
            ax.set_ylim(h - 0.5, -0.5)
            rect(ax, m.get("fov_lit"), LIT)
            rect(ax, m.get("fov"), FIELD)
            rect(ax, r["old_box"], OLD)
            rect(ax, r["new_box"], BOX)
            ax.plot(pts[~out, 1], pts[~out, 0], **IN)
            ax.plot(pts[out, 1], pts[out, 0], **OUT)
            ax.set_xticks([])
            ax.set_yticks([])
        nb, ob = r["new_box"], r["old_box"]
        n_lit = sum(o["in_lit_mask"] for o in ver[sid]["outside"]) if sid in ver else None
        where = "" if not out.any() or n_lit is None else f" ({int(out.sum()) - n_lit} of them outside the lit area)"
        title = (f"{sid}: {len(acc)} accepted, {int(out.sum())} outside the new box{where}\n"
                 f"old crop {ob[2]}x{ob[3]} -> new box {nb[2]}x{nb[3]} ({m['fov_method']})")
        fig.suptitle(title, fontsize=9, color="red" if out.any() else "black", y=0.99)
        fig.legend([Line2D([], [], **OLD), Line2D([], [], **BOX), Line2D([], [], **FIELD), Line2D([], [], **LIT),
                    Line2D([], [], **IN), Line2D([], [], **OUT)],
                   ["old crop (CNMF-E ran on it)", "new mouse box", "new field", "lit area",
                    "accepted neuron centre", "accepted, outside new box"],
                   loc="lower center", ncol=3, fontsize=7.5, frameon=False)
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.13, wspace=0.03)
        out_f = fig_dir / f"neurons_{sid}.png"
        fig.savefig(out_f, format="png", dpi=100)
        print(f"[OK] {out_f} {out_f.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
