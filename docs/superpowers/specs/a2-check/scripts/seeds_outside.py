"""Strong seeds (saved in each projection npz) outside the session's mouse box and outside its field, per series.
Read-only. usage: seeds_outside.py ROOT [ROOT ...]"""
import json
import sys
from pathlib import Path

import numpy as np


def out(seeds, box):
    t, l, h, w = box
    return int(((seeds[:, 0] < t) | (seeds[:, 0] >= t + h) | (seeds[:, 1] < l) | (seeds[:, 1] >= l + w)).sum())


for root in map(Path, sys.argv[1:]):
    tot = [0, 0, 0]
    for f in sorted((root / "_bearmind" / "registry").glob("*.json")):
        r = json.loads(f.read_text())
        m = r["stages"]["crop"]["metrics"]
        with np.load(root / "_bearmind" / "crop" / f"{r['sid']}_projection.npz") as z:
            s = z["seeds"] if "seeds" in z.files else np.zeros((0, 2), int)
        ob, of = out(s, m["box"]), out(s, m["fov"])
        tot = [tot[0] + len(s), tot[1] + ob, tot[2] + of]
        if ob or of:
            print(f"  {r['sid']}: seeds {len(s)}, outside box {ob}, outside field {of}")
    print(f"{root.name}: strong seeds {tot[0]}, outside the mouse box {tot[1]}, outside the field {tot[2]}")
