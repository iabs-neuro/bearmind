r"""Profile of the preparation with cached contours on FOF_F01_1D (read only): which calls take the 6.8 s.
usage: python t7_long_prof.py OUT FILE N"""
import torch  # noqa: F401
import cProfile, io, json, pstats, sys, time, os
from pathlib import Path
sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import numpy as np
import bm_examinator as be
OUT, FILE, HINT = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
KW = {"mode": "legacy", "end_frame": None, "enable_gpu_backend": True, "trace_points": 2000}
est = be.LoadEstimates(FILE.replace("\\", "/"), default_fps=20, time=np.arange(HINT + 10_000) / 20.0)
idx = est.idx_components
contours = be._component_contours(est, idx, 0.3)
pr = cProfile.Profile()
t = time.perf_counter(); pr.enable(); be._prepare(est.name, est, 20, KW, contours); pr.disable(); total = time.perf_counter() - t
st = pstats.Stats(pr)
rows = [(fn.split(os.sep)[-1].split("/")[-1], ln, name, round(ct, 3), round(tt, 3), nc)
        for (fn, ln, name), (cc, nc, tt, ct, callers) in st.stats.items() if ct > 0.1]
rows.sort(key=lambda r: -r[3])
print(f"[prof] total {total:.2f} s; MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}", flush=True)
for r in rows[:18]:
    print("[prof]", r, flush=True)
(OUT / "t7_long_prof.json").write_text(json.dumps({"total_s": total, "rows": rows[:30]}, indent=1), encoding="utf-8")
