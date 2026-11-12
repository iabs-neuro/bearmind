"""Re-run the thresholds suggestion (bm_qc.suggest_init_thresholds, gSig 4) on stored maps of the test root; read-only."""
import sys
import numpy as np
sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import bm_qc
from bearmind.paths import SeriesPaths
p = SeriesPaths(r"H:\Projects\_bearmind_test\lt1_J03_J06")
for sid in sys.argv[1:]:
    with np.load(p.threshold_maps(sid)) as z:
        maps = {"cn": z["cn_4"], "pnr": z["pnr_4"], "cn_neg": z["cn_mirror_4"], "pnr_neg": z["pnr_mirror_4"], "gSig": 4, "mean": z["mean"]}
    sug, grid = bm_qc.suggest_init_thresholds(maps, gSiz=17)
    for k, v in sug.items():
        print(sid, k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk in ("min_corr", "min_pnr", "n_seeds", "fdr", "note")})
    if grid is not None:
        f = grid["fdr"]; nr = grid["n_real"]
        print("   strictest corner (corr %.2f, pnr %.1f): n_real %d, fdr %.3f; min fdr on grid %.3f" % (
            grid["corr_grid"][-1], grid["pnr_grid"][-1], nr[-1, -1], f[-1, -1], f[nr > 0].min()))
