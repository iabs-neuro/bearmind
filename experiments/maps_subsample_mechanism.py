"""Why the PNR map changes with frame subsets: per-pixel max and noise (sn) of the filtered movie.

Supports the report docs/superpowers/specs/2026-11-12-maps-subsample-validation.md (section on PNR). On a crop
around cells, the same steps as bm_qc.compute_init_maps (gSig 4 kernel, mean-subtracted, get_noise_fft 'mean')
for: all frames, every k-th frame, the block mean of k frames (PNR divided by sqrt(k), as CaImAn), and the first
600 s (long sessions only). Prints the median ratio variant / full of max, sn and PNR on cell pixels (top 5 % PNR
of the crop) and background pixels (bottom half).

Usage: python experiments/maps_subsample_mechanism.py <MC avi> <k> <fps> <y0> <x0> <crop side>
  e.g. ... LT1_J03_1D_MC.avi 5 19.76 170 240 120 ; ... LT1_J01_1D_MC.avi 8 29.78 151 304 48
The whole movie is decoded once, but only the crop (plus the filter margin) is kept in memory.
"""
import os
import sys
from pathlib import Path

# one BLAS thread: OpenBLAS otherwise commits a buffer per thread at import (about 4 GB on this machine)
for _k in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_k] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from caiman.source_extraction.cnmf.pre_processing import get_noise_fft  # noqa: E402

import bm_qc  # noqa: E402
from bearmind.video import _iter_avi  # noqa: E402

video, k, fps = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
y0, x0, n = int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
g = 4
psf = bm_qc._psf_kernel(g)
m = psf.shape[0] // 2 + 1  # filter margin, as the strips of compute_init_maps
fr = np.stack([f[y0 - m:y0 + n + m, x0 - m:x0 + n + m] for f in _iter_avi(Path(video))])
T = fr.shape[0]


def parts(blk, pnr_div=1.0):
    """max, sn, PNR and SD of the filtered crop (inner n x n pixels)."""
    blk = np.asarray(blk, np.float32)
    filt = np.stack([cv2.filter2D(b, -1, psf, borderType=1) for b in blk])
    filt -= filt.mean(0)
    tn = filt.shape[0]
    sn = get_noise_fft(filt.reshape(tn, -1).T, noise_method='mean')[0].reshape(filt.shape[1:])
    mx = filt.max(0)
    sd = filt.std(0)
    c = (slice(m, m + n), slice(m, m + n))
    return mx[c], sn[c], (mx / sn / pnr_div)[c], sd[c]


n10 = int(round(600 * fps))
sets = {"full": parts(fr), "dec": parts(fr[::k]), "tsubK": parts(bm_qc._tsub_mean(fr, k), np.sqrt(k))}
if T > n10 * 1.5:
    sets["first10"] = parts(fr[:n10])
mxf, snf, pnrf, sdf = sets["full"]
cell = pnrf >= np.percentile(pnrf, 95)
bg = pnrf <= np.percentile(pnrf, 50)
print(f"T {T}, k {k}, crop {n}x{n} at ({y0},{x0}); cell px {cell.sum()}, background px {bg.sum()}")
for name, (mx, sn, pnr, sd) in sets.items():
    def r(a, b, msk):
        return float(np.median(a[msk] / b[msk]))
    print(f"{name:8s} cell px: max x{r(mx, mxf, cell):.2f}  sn x{r(sn, snf, cell):.2f}  PNR x{r(pnr, pnrf, cell):.2f} | "
          f"background px: sn x{r(sn, snf, bg):.2f}  PNR x{r(pnr, pnrf, bg):.2f} | "
          f"sn/sd cell {float(np.median(sn[cell] / sd[cell])):.2f} bg {float(np.median(sn[bg] / sd[bg])):.2f}")
