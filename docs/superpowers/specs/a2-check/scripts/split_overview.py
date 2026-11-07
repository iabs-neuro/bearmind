"""Split a crop overview PNG into parts of whole rows (each with the legend strip on top), 256-colour palette,
full resolution, so each part stays under the docs size limit.

usage: split_overview.py SRC OUT_PREFIX N_ROWS ROWS_PER_PART
"""
import os
import sys

from PIL import Image

DPI, PANEL_IN, TOP_IN = 110, 2.6, 0.75  # crop_overview.py: row height PANEL_IN + 0.35 in, legend strip 0.75 in


def main():
    src, prefix, n_rows, per = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    im = Image.open(src).convert("RGB")
    top = round(TOP_IN * DPI)
    row_h = (im.height - top) / n_rows
    import numpy as np

    white = (np.asarray(im).min(axis=(1, 2)) >= 250)  # rows of pure background

    def cut(k):
        """Split line near row boundary k: the middle of the longest white run just above it (between the
        panels of row k - 1 and the caption of row k)."""
        if k == n_rows:
            return im.height
        b = round(top + k * row_h)
        best, run, start = (0, b), 0, None
        for y in range(b - 80, b + 10):
            if white[y]:
                start = y if run == 0 else start
                run += 1
                if run > best[0]:
                    best = (run, start + run // 2)
            else:
                run = 0
        return best[1]

    legend = im.crop((0, 0, im.width, cut(0)))
    for part, first in enumerate(range(0, n_rows, per), start=1):
        last = min(first + per, n_rows)
        body = im.crop((0, cut(first), im.width, cut(last)))
        out = Image.new("RGB", (im.width, legend.height + body.height), "white")
        out.paste(legend, (0, 0))
        out.paste(body, (0, legend.height))
        q = out.quantize(colors=256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
        path = f"{prefix}_{part}.png"
        q.save(path, optimize=True)
        print(f"[OK] {path}: rows {first + 1}-{last}, {out.width}x{out.height}, {os.path.getsize(path) // 1024} KB")


if __name__ == "__main__":
    main()
