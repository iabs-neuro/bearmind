r"""Step 5 stand-in: open the longest ready CNMF-E estimates on the machine through prepare_session, read only.

No estimates longer than 90 000 frames exist on H:\Dataset\CalciumData or H:\Projects; the longest are the FOF
sessions of 64 300 frames. This opens one of them with the inspection server's window options: preparation time
(load included), the wave share and hint, the window on a document, the first document's size, peak memory of
this process, and the time axis (whole session; a zoom on the last 200 frames shows every frame).

usage: python t7_long.py OUT_DIR ESTIMATES_PICKLE N_FRAMES_HINT
"""
import torch  # noqa: F401  (torch before bm_examinator: shm.dll)

import json
import os
import sys
import time
from pathlib import Path

REPO = r"C:\Users\User\PycharmProjects\bearmind"
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tests"))

import numpy as np  # noqa: E402
import psutil  # noqa: E402
from bokeh.document import Document  # noqa: E402
from bokeh.protocol import Protocol  # noqa: E402

import bm_examinator as be  # noqa: E402
from bearmind import inspect_server, wave  # noqa: E402
from examinator_helpers import App, message_bytes  # noqa: E402

OUT, FILE, HINT = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
KW = {"mode": "legacy", "end_frame": None, "enable_gpu_backend": True, "trace_points": 2000}
proc = psutil.Process()
R = {"file": FILE, "bytes": os.path.getsize(FILE), "rss_gb_start": proc.memory_info().rss / 1e9}
st0 = os.stat(FILE)
fps = 20.0
time_s = np.arange(HINT + 10_000) / fps  # longer than the traces: LoadEstimates cuts it to them

t = time.perf_counter()
prep = be.prepare_session(FILE.replace("\\", "/"), time=time_s, default_fps=20, bkapp_kwargs=KW)
R["prepare_session_s"] = time.perf_counter() - t
est = prep.estimates
n = est.C.shape[1]
R["frames"] = int(n)
R["components"] = int(est.A.shape[1])
R["listed"] = int(len(est.idx_components))
R["time_len"] = int(len(est.time))
R["peak_gb_after_prepare"] = proc.memory_info().peak_wset / 1e9

t = time.perf_counter()
idx = est.idx_components
frac = wave.wave_frac(est.C, idx, k_mad=3.0, active_frac=0.3, sn=wave.component_noise(est, idx))
R["wave_s"] = time.perf_counter() - t
R["wave_frac"] = frac
R["wave_hint"] = inspect_server.wave_hint(frac, {"k_mad": 3.0, "active_frac": 0.3, "warn_frac": 0.025})

t = time.perf_counter()
app = App(be.make_app(prep.fname, 20, KW, prepared=prep))
R["window_on_document_s"] = time.perf_counter() - t
R["first_document_bytes"] = message_bytes(Protocol().create("PULL-DOC-REPLY", "0", app.doc))
ta = app.time_axis()
R["time_axis_points"] = int(len(ta))
R["time_axis_whole_session"] = bool(ta[0] == est.time[0] and ta[-1] == est.time[-1])
app.zoom(est.time[-200], est.time[-1])
tz = app.time_axis()
R["zoom_last_200_points"] = int(len(tz))
R["zoom_every_frame"] = bool(len(tz) >= 200 and np.array_equal(tz, est.time[-len(tz):]))
R["rows_listed"] = len(app.shown())
R["peak_gb"] = proc.memory_info().peak_wset / 1e9
R["file_unchanged"] = (os.stat(FILE).st_size, os.stat(FILE).st_mtime_ns) == (st0.st_size, st0.st_mtime_ns)
for k, v in R.items():
    print(f"[long] {k} {v}", flush=True)
(OUT / "t7_long.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
print("[OK] written", OUT / "t7_long.json")
