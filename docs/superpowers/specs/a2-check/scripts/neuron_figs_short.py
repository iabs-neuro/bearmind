"""Figures of the series_short neuron check (same style as neuron_figs.py). usage: neuron_figs_short.py OUT_DIR FIG_DIR"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from neuron_figs import BOX, FIELD, IN, LIT, OLD, OUT, outside, rect  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

P = Path(r"H:\Projects\_bearmind_test\series_short\_bearmind\crop")
res_dir, fig_dir = Path(sys.argv[1]), Path(sys.argv[2])
res = json.loads((res_dir / "neuron_check_short.json").read_text())
arr = np.load(res_dir / "neuron_check_short_arrays.npz")
for sid, r in res.items():
    with np.load(P / f"{sid}_projection.npz") as z:
        mean = z["mean"].astype(np.float32)
        act = z["activity"].astype(np.float32) if "activity" in z.files else None
    pts = arr[f"{sid}_centres"][arr[f"{sid}_accepted"].astype(int)]
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
        rect(ax, r["fov_lit"], LIT)
        rect(ax, r["fov"], FIELD)
        rect(ax, r["old_box"], OLD)
        rect(ax, r["new_box"], BOX)
        ax.plot(pts[~out, 1], pts[~out, 0], **IN)
        ax.plot(pts[out, 1], pts[out, 0], **OUT)
        ax.set_xticks([])
        ax.set_yticks([])
    nb, ob = r["new_box"], r["old_box"]
    n_lit = sum(o["in_lit_mask"] for o in r["outside_box_accepted"])
    where = f" ({int(out.sum()) - n_lit} of them outside the lit area)" if out.any() else ""
    fig.suptitle(f"series_short {sid} (2000 frames): {len(pts)} accepted, {int(out.sum())} outside the new box{where}\n"
                 f"old crop {ob[2]}x{ob[3]} -> new box {nb[2]}x{nb[3]} ({r['fov_method']})",
                 fontsize=9, color="red" if out.any() else "black", y=0.99)
    fig.legend([Line2D([], [], **OLD), Line2D([], [], **BOX), Line2D([], [], **FIELD), Line2D([], [], **LIT),
                Line2D([], [], **IN), Line2D([], [], **OUT)],
               ["old crop (CNMF-E ran on it)", "new mouse box", "new field", "lit area",
                "accepted neuron centre", "accepted, outside new box"], loc="lower center", ncol=3, fontsize=7.5,
               frameon=False)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.13, wspace=0.03)
    f = fig_dir / f"neurons_series_short_{sid}.png"
    fig.savefig(f, format="png", dpi=100)
    print(f"[OK] {f} {f.stat().st_size // 1024} KB; lit median {r['lit_mean_median']}")
