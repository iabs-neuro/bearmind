"""Around each gap (> 2 x median m): the time of the +-K intervals around it in units of the session's mean frame
period u = span / (N - 1), minus the number of intervals. About 0: the clock produced the frames seen (timing
jitter); about 1: one frame missing. Control: the same windows around random intervals with no gap. Read-only.
Sessions with a non-positive interval (a damaged row) are left out: their span is not a duration."""
import json, sys
import numpy as np
sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind.paths import SeriesPaths
from bearmind.registry import read_all
from bearmind.timestamps import session_timestamps
paths = SeriesPaths(r"H:\Projects\LT1\CalciumData")
recs, _ = read_all(paths)
K = 30
res = {"30": [], "20": []}
ctrl = {"30": [], "20": []}
rng = np.random.default_rng(0)
skipped = []
for r in recs:
    t = np.asarray(session_timestamps(paths, r).t_ms, float)
    d = np.diff(t)
    if np.any(d <= 0):
        skipped.append(r.sid)
        continue
    m = np.median(d)
    u = (t[-1] - t[0]) / (len(t) - 1)
    grp = "30" if r.fps > 25 else "20"
    gaps = np.where(d > 2 * m)[0]
    for i in gaps:
        a, b = i - K, i + K + 1
        if a < 0 or b > len(d):
            continue
        res[grp].append((r.sid, float(d[i] / m), float(d[a:b].sum() / u - (b - a))))
    for i in rng.integers(K, len(d) - K - 1, 40):
        if np.any(d[i - K:i + K + 1] > 2 * m):
            continue
        ctrl[grp].append(float(d[i - K:i + K + 1].sum() / u - (2 * K + 1)))
json.dump({"res": res, "ctrl": ctrl, "skipped": skipped}, open("dfr_gapwindow.json", "w"))
print("skipped (non-positive interval):", skipped)
for g in ("30", "20"):
    x = np.array([e[2] for e in res[g]]); c = np.array(ctrl[g])
    print(g, "gaps", len(x), "excess quantiles", np.round(np.percentile(x, [0, 10, 25, 50, 75, 90, 100]), 2).tolist() if len(x) else None)
    if len(x):
        print("    <0.5:", int((x < 0.5).sum()), " 0.5-1.5:", int(((x >= 0.5) & (x < 1.5)).sum()), " >=1.5:", int((x >= 1.5).sum()))
    print("   control", len(c), np.round(np.percentile(c, [0, 10, 25, 50, 75, 90, 100]), 2).tolist(), " control >=0.5:", int((c >= 0.5).sum()))
    print("   big gaps:", [(s, round(gm, 2), round(ex, 2)) for s, gm, ex in res[g] if gm > 2.5])
