r"""Merge sel.: the window's own time, measured inside the chain. CaImAn's Estimates.manual_merge is wrapped by a
timer, so the chain's time splits into manual_merge (the same call the window makes) and the rest (the window).
t7_speed.py compared the chain with a separate manual_merge call on a copy; that reference varies from call to
call. Inputs as t7_speed.py (series_short 4_Estimates, read only; 107 107 frames = tiled copy in memory).
Two merges per window: the first builds CNMFParams once (Task 2), the second does not.

usage: python t7_merge_split.py OUT_DIR
"""
import torch  # noqa: F401

import copy
import json
import math
import os
import pickle
import statistics
import sys
import time
from pathlib import Path

REPO = r"C:\Users\User\PycharmProjects\bearmind"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from caiman.source_extraction.cnmf.estimates import Estimates  # noqa: E402

import bm_examinator as be  # noqa: E402
from examinator_helpers import App, patch_doc_bytes  # noqa: E402

P4 = r"H:\Projects\_bearmind_test\series_short\4_Estimates\NOF_H01_1D_estimates.pickle"
OUT = Path(sys.argv[1])
KW = {"mode": "legacy", "end_frame": None, "enable_gpu_backend": True, "trace_points": 2000}
inside = []
_orig = Estimates.manual_merge


def timed_merge(self, *a, **k):
    t = time.perf_counter()
    try:
        return _orig(self, *a, **k)
    finally:
        inside.append(time.perf_counter() - t)


Estimates.manual_merge = timed_merge


def tiled(src, T):
    e = copy.copy(src)
    n0 = src.C.shape[1]
    reps = math.ceil(T / n0)
    for k, v in list(vars(src).items()):
        if sp.issparse(v) and v.ndim == 2 and v.shape[1] == n0:
            setattr(e, k, sp.hstack([v] * reps).tocsr()[:, :T])
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == n0:
            setattr(e, k, np.tile(v, (1, reps))[:, :T])
    if getattr(src, "R", None) is not None and src.R is src.YrA:
        e.R = e.YrA
    e.time = np.arange(T) / 30.0
    return e


with open(P4, "rb") as f:
    base4 = pickle.load(f)
R = {}
for T in (2000, 107107):
    e4 = tiled(base4, T)
    fname = str(OUT / f"t7_S{T}" / "NOF_H01_1D_estimates.pickle")
    e4.name = fname
    prep = be._prepare(fname, e4, 20, KW, None)
    rows_ = []
    for rep in range(3):
        app = App(be.make_app(fname, 20, KW, prepared=prep))
        for order, rows in enumerate(([3, 4], [10, 11])):  # second merge: rows of the list after the first
            app.select(rows)
            inside.clear()
            with app.events() as events:
                t0 = time.perf_counter()
                app.click("Merge sel.")
                dt = time.perf_counter() - t0
            mm = sum(inside)
            rows_.append({"rep": rep, "merge_in_window": order + 1, "rows": rows, "chain_s": dt,
                          "manual_merge_inside_s": mm, "window_s": dt - mm, "calls": len(inside),
                          "patch_bytes": patch_doc_bytes(events)})
        del app
    R[T] = rows_
    for k in (1, 2):
        sel = [r for r in rows_ if r["merge_in_window"] == k]
        print(f"[merge] T={T} merge #{k} in a window: chain median {statistics.median(r['chain_s'] for r in sel):.3f} s, "
              f"manual_merge {statistics.median(r['manual_merge_inside_s'] for r in sel):.3f} s, window "
              f"median {statistics.median(r['window_s'] for r in sel):.3f} s (max {max(r['window_s'] for r in sel):.3f}), "
              f"patch max {max(r['patch_bytes'] for r in sel)} B", flush=True)
    del prep, e4
(OUT / "t7_merge_split.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
print("[OK] written", OUT / "t7_merge_split.json")
