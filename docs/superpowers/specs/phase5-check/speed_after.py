r"""Step 4 of the phase 5 check: speed "after", measured the way the phase 5 page measured "before".

The real handlers of bm_examinator on a Bokeh document in memory, no server, no browser. Time is the Python time
of the whole button chain; size is the PATCH-DOC message built from the document events (what the server sends
to the browser), as tests/test_examinator_speed.py does. Inputs, read only:
  - series_short\5_Inspection\NOF_H01_1D\NOF_H01_1D_processed.pickle (Delete, selection, Save results, opening);
  - series_short\4_Estimates\NOF_H01_1D_estimates.pickle (Merge; the processed file's sparse S does not suit
    CaImAn's manual_merge, as on the page).
2000 frames = the real file (measured); 18 000 and 107 107 frames = the same file continued in time by tiling
every per-frame array (tiled copy), in memory only. The window options are the inspection server's
(legacy, full length, WebGL, 2000 trace points). Save results writes into OUT (scratch) as the server's
final_save does: the result pickle at once, the side files in the background thread (waited for before the next
measurement). Prints "[speed] ..." lines and writes t7_speed.json.

usage: python t7_speed.py OUT_DIR
"""
import torch  # noqa: F401  (torch before bm_examinator: shm.dll)

import copy
import gc
import json
import math
import os
import pickle
import shutil
import statistics
import sys
import time
from pathlib import Path

REPO = r"C:\Users\User\PycharmProjects\bearmind"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from bokeh.document import Document  # noqa: E402
from bokeh.document.events import ModelChangedEvent  # noqa: E402
from bokeh.protocol import Protocol  # noqa: E402
from caiman.source_extraction.cnmf.params import CNMFParams  # noqa: E402

import bm_examinator as be  # noqa: E402
from bearmind import wave  # noqa: E402
from examinator_helpers import App, changed_columns, message_bytes, patch_doc_bytes  # noqa: E402

P = r"H:\Projects\_bearmind_test\series_short\5_Inspection\NOF_H01_1D\NOF_H01_1D_processed.pickle"
P4 = r"H:\Projects\_bearmind_test\series_short\4_Estimates\NOF_H01_1D_estimates.pickle"
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
LENGTHS = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["2000", "18000", "107107"])]
KW = {"mode": "legacy", "end_frame": None, "enable_gpu_backend": True, "trace_points": 2000}
BULK = {"traces", "traces_recon", "times", "t", "xs", "ys"}
R = {"env": {k: os.environ.get(k) for k in ("MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                                             "OPENBLAS_CORETYPE", "CONDA_DEFAULT_ENV")},
     "cpus": os.cpu_count(), "inputs": {}, "by_length": {}}


def say(k, v):
    print(f"[speed] {k} {v}", flush=True)


def load(p):
    t = time.perf_counter()
    with open(p, "rb") as f:
        e = pickle.load(f)
    return e, time.perf_counter() - t


def tiled(src, T):
    """A copy of the estimates whose per-frame arrays (2-D, frames along axis 1) are continued to T frames."""
    e = copy.copy(src)
    n0 = src.C.shape[1]
    reps = math.ceil(T / n0)
    done = []
    for k, v in list(vars(src).items()):
        if sp.issparse(v) and v.ndim == 2 and v.shape[1] == n0:
            setattr(e, k, sp.hstack([v] * reps).tocsr()[:, :T])
            done.append(k)
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == n0:
            setattr(e, k, np.tile(v, (1, reps))[:, :T])
            done.append(k)
    if getattr(src, "R", None) is not None and src.R is src.YrA:  # CaImAn leaves R as the same object as YrA
        e.R = e.YrA
    e.time = np.arange(T) / 30.0  # as the page's tiled copies
    return e, done


def med(xs):
    return statistics.median(xs)


def open_app(prep, final_save=None):
    return App(be.make_app(prep.fname, 20, KW, prepared=prep, final_save=final_save))


def timed_click(app, label):
    with app.events() as events:
        t0 = time.perf_counter()
        app.click(label)
        dt = time.perf_counter() - t0
    return dt, patch_doc_bytes(events), events


base, t_load = load(P)
base4, t_load4 = load(P4)
R["inputs"] = {"processed": {"file": P, "bytes": os.path.getsize(P), "load_s": round(t_load, 3),
                             "components": int(base.A.shape[1]), "listed": len(base.idx_components),
                             "frames": int(base.C.shape[1]), "fov": list(np.shape(base.imax))},
               "estimates": {"file": P4, "bytes": os.path.getsize(P4), "load_s": round(t_load4, 3),
                             "components": int(base4.A.shape[1]), "listed": len(base4.idx_components),
                             "frames": int(base4.C.shape[1])}}
say("inputs", json.dumps(R["inputs"]))
proc = psutil.Process()

for T in LENGTHS:
    L = R["by_length"][T] = {"kind": "measured" if T == base.C.shape[1] else "tiled copy"}
    fname = str(OUT / f"t7_T{T}" / "NOF_H01_1D_estimates.pickle")  # a name only; nothing is read from it
    e, tiled_attrs = tiled(base, T)
    e.name = fname
    L["tiled_attrs"] = tiled_attrs
    gc.collect()

    # --- opening: preparation (contours computed, as with no cache), the wave share, the window on a document
    t = time.perf_counter()
    prep = be._prepare(fname, e, 20, KW, None)
    L["prepare_cold_s"] = time.perf_counter() - t
    t = time.perf_counter()
    idx = e.idx_components
    frac = wave.wave_frac(e.C, idx, k_mad=3.0, active_frac=0.3, sn=wave.component_noise(e, idx))
    L["wave_s"] = time.perf_counter() - t
    L["wave_frac"] = frac
    t = time.perf_counter()
    prep_cached = be._prepare(fname, e, 20, KW, prep.contours)
    L["prepare_with_contour_cache_s"] = time.perf_counter() - t
    del prep_cached
    apps = []
    for _ in range(3):
        doc = Document()
        t = time.perf_counter()
        be.make_app(fname, 20, KW, prepared=prep)(doc)
        apps.append(time.perf_counter() - t)
    L["window_on_document_s"] = apps  # what Next does on the IOLoop once the session is prepared
    L["first_document_bytes"] = message_bytes(Protocol().create("PULL-DOC-REPLY", "0", doc))
    del doc
    L["open_cold_s"] = L["prepare_cold_s"] + L["wave_s"] + med(apps)
    L["open_cached_s"] = L["prepare_with_contour_cache_s"] + med(apps)  # the share is cached with the contours
    say(f"T={T} open", {k: L[k] for k in ("prepare_cold_s", "wave_s", "prepare_with_contour_cache_s",
                                          "open_cold_s", "open_cached_s", "first_document_bytes")}
        | {"window_on_document_s": [round(x, 3) for x in apps], "wave_frac": frac})

    # --- Delete sel., the whole chain, three times on one window
    app = open_app(prep)
    dels = []
    for rows in ([3], [10, 11], [50]):
        app.select(rows)
        dt, size, events = timed_click(app, "Delete sel.")
        bulk = sorted({c for _, c in changed_columns(events)} & BULK)
        dels.append({"rows": rows, "s": dt, "patch_bytes": size, "bulk_columns_sent": bulk})
    L["delete"] = dels
    say(f"T={T} delete", [(d["rows"], round(d["s"], 4), d["patch_bytes"], d["bulk_columns_sent"]) for d in dels])

    # --- selection of another neuron (20 times)
    app = open_app(prep)
    ts = []
    with app.events() as events:
        for r in range(20):
            t0 = time.perf_counter()
            app.select([r])
            ts.append(time.perf_counter() - t0)
    kinds = sorted({type(ev.model).__name__ for ev in events if isinstance(ev, ModelChangedEvent)})
    L["select"] = {"median_ms": 1000 * med(ts), "max_ms": 1000 * max(ts), "data_columns_sent": len(changed_columns(events)),
                   "models_changed": kinds, "patch_bytes_20": patch_doc_bytes(events)}
    say(f"T={T} select", L["select"])

    # --- Save results until the window is free (final_save as the server's: result now, side files in background)
    out = OUT / f"t7_T{T}" / "5_Inspection" / "NOF_H01_1D"
    futures = []

    def final_save(est, out=out, futures=futures):
        t1 = time.perf_counter()
        be.write_result(est, out / "NOF_H01_1D_manual_estimates.pickle")
        futures.append((time.perf_counter() - t1,
                        be.write_side_files_in_background(est, out, "NOF_H01_1D_")))
        return "Saved."

    app = open_app(prep, final_save=final_save)
    app.select([3])
    app.click("Delete sel.")
    dt, size, _ = timed_click(app, "Save results")
    pickle_s, fut = futures[0]
    t = time.perf_counter()
    fut.result(timeout=900)
    side_after = time.perf_counter() - t
    L["save_results"] = {"until_free_s": dt, "pickle_s": pickle_s, "side_files_after_s": side_after,
                         "result_bytes": (out / "NOF_H01_1D_manual_estimates.pickle").stat().st_size,
                         "patch_bytes": size}
    say(f"T={T} save_results", L["save_results"])
    del app, prep, e
    shutil.rmtree(OUT / f"t7_T{T}", ignore_errors=True)
    gc.collect()

    # --- Merge sel. on the CNMF-E estimates (as the page): two rows, the whole chain vs CaImAn's manual_merge
    fname4 = str(OUT / f"t7_M{T}" / "NOF_H01_1D_estimates.pickle")
    e4, tiled4 = tiled(base4, T)
    e4.name = fname4
    prep4 = be._prepare(fname4, e4, 20, KW, None)
    merges = []
    for rows in ([3, 4], [10, 11]):  # the first Merge of a window builds CNMFParams once (about 0.2 s)
        app = open_app(prep4)
        comps = [app.storage.shown[r] for r in rows]
        ref = copy.copy(app.storage.estimates)
        params = CNMFParams(params_dict=ref.cnmf_dict)
        t0 = time.perf_counter()
        ref.manual_merge([comps], params=params)
        t_mm = time.perf_counter() - t0
        del ref
        app.select(rows)
        dt, size, events = timed_click(app, "Merge sel.")
        merges.append({"rows": rows, "components": [int(c) for c in comps], "s": dt, "manual_merge_s": t_mm,
                       "extra_s": dt - t_mm, "patch_bytes": size,
                       "bulk_columns_sent": sorted({c for _, c in changed_columns(events)} & BULK)})
        del app
    L["merge"] = merges
    L["merge_tiled_attrs"] = tiled4
    say(f"T={T} merge", [(m["rows"], round(m["s"], 3), round(m["manual_merge_s"], 3), m["patch_bytes"])
                         for m in merges])
    del prep4, e4
    gc.collect()
    L["rss_gb_after"] = proc.memory_info().rss / 1e9
    (OUT / "t7_speed.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")

R["peak_wset_gb"] = proc.memory_info().peak_wset / 1e9
(OUT / "t7_speed.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
print("[OK] written", OUT / "t7_speed.json")
