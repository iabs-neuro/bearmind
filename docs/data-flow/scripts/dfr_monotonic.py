"""Non-positive timestamp intervals (time going back or standing still) in every LT1 timestamps file, raw rows."""
import sys, glob, os
import numpy as np
root = r"H:\Projects\LT1\CalciumData\1_Raw"
for f in sorted(glob.glob(os.path.join(root, "*", "*_Mini_TS.csv"))):
    with open(f, "rb") as fh:
        data = fh.read()
    lines = [ln for ln in data.decode("utf-8-sig").splitlines() if ln.strip()]
    v = np.array([int(x) for x in lines], dtype=np.int64)
    d = np.diff(v)
    bad = np.where(d <= 0)[0]
    lens = sorted({len(x.strip()) for x in lines})
    ends_nl = data.endswith(b"\n")
    if len(bad) or len(lens) > 1 or not ends_nl:
        print(os.path.basename(f), "rows", len(v), "nonpos", len(bad), "at", bad[:5].tolist(), "digit lengths", lens,
              "ends with newline", ends_nl, "last", lines[-1])
print("[OK] checked", len(glob.glob(os.path.join(root, "*", "*_Mini_TS.csv"))), "files")
