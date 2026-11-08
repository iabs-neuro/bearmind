r"""Step 5 stand-in, second run: where prepare_session spends its time on FOF_F01_1D (read only): LoadEstimates,
then the preparation with contours computed, then with the contours of the first preparation (the server's
cache). The file may now be in the OS cache from the first run. usage: python t7_long_split.py OUT FILE N"""
import torch  # noqa: F401
import json, sys, time
from pathlib import Path
sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import numpy as np
import psutil
import bm_examinator as be
OUT, FILE, HINT = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
KW = {"mode": "legacy", "end_frame": None, "enable_gpu_backend": True, "trace_points": 2000}
R = {}
t = time.perf_counter(); est = be.LoadEstimates(FILE.replace("\\", "/"), default_fps=20, time=np.arange(HINT + 10_000) / 20.0)
R["load_s"] = time.perf_counter() - t
t = time.perf_counter(); prep = be._prepare(est.name, est, 20, KW, None); R["prepare_contours_computed_s"] = time.perf_counter() - t
t = time.perf_counter(); be._prepare(est.name, est, 20, KW, prep.contours); R["prepare_contours_cached_s"] = time.perf_counter() - t
R["peak_gb"] = psutil.Process().memory_info().peak_wset / 1e9
print("[long-split]", json.dumps(R), flush=True)
(OUT / "t7_long_split.json").write_text(json.dumps(R, indent=1), encoding="utf-8")
