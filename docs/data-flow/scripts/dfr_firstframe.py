"""Mean brightness of raw frames 0, 1, 2 and the MC shift of frame 0 for every LT1 session (2-3 frames decoded each)."""
import json, sys
import numpy as np
sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
from bearmind.paths import SeriesPaths
from bearmind.registry import read_all
from bearmind.video import VideoArray
paths = SeriesPaths(r"H:\Projects\LT1\CalciumData")
recs, _ = read_all(paths)
out = []
for r in recs:
    p = r.parts[0]
    v = VideoArray(paths.root / p.folder / p.chunks[0])
    fr = v[0:3]
    means = [float(f.mean()) for f in fr]
    s = np.load(paths.mc_shifts(r.sid))
    out.append({"sid": r.sid, "m0": means[0], "m1": means[1], "m2": means[2],
                "shift0": float(np.abs(s[0]).max()), "max_rest": float(np.abs(s[1:]).max()),
                "p99": float(np.percentile(np.abs(s).max(1), 99))})
json.dump(out, open("dfr_firstframe.json", "w"), indent=0)
m0 = np.array([o["m0"] for o in out]); m2 = np.array([o["m2"] for o in out])
ratio = m0 / m2
print("frame0/frame2 brightness ratio quantiles", np.round(np.percentile(ratio, [0, 10, 50, 90, 100]), 3).tolist())
print("black frame 0 (mean < 1):", [o["sid"] for o in out if o["m0"] < 1])
print("frame1/frame2 < 0.9:", [(o["sid"], round(o["m1"] / o["m2"], 2)) for o in out if o["m1"] / o["m2"] < 0.9])
print("sessions whose max |shift| comes from frame 0 only (shift0 > max over the rest):",
      sum(o["shift0"] > o["max_rest"] for o in out))
print("max over frames 1..: >10 px:", [(o["sid"], round(o["max_rest"], 1)) for o in out if o["max_rest"] > 10])
print("[OK]")
