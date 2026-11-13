"""Dry run of scan_series on LT1 with the merged code: every write is replaced by an in-memory capture.

Compares, per session, what a Rescan would store (parts, n_frames, fps, mode, excluded, scan anomalies) with the
registry now. Writes nothing under H:.
"""
import copy
import sys

sys.dont_write_bytecode = True
from collections import Counter
from pathlib import Path

import bearmind.scan as scan
from bearmind import registry

ROOT = Path(r"H:\Projects\LT1\CalciumData")
before, after = {}, {}


def fake_update(paths, sid, change):
    rec = registry.load_record(paths, sid)
    before[sid] = copy.deepcopy(rec)
    change(rec)
    after[sid] = rec
    return rec


def refuse(*a, **k):
    raise RuntimeError("write attempted in dry run")


scan.update_record = fake_update
scan.save_record = refuse
scan.replace_broken = refuse
scan._recreate = refuse
scan.fileio = type("F", (), {"write_atomic": staticmethod(lambda *a, **k: None)})
scan.config.update_yaml = refuse

recs = scan.scan_series(ROOT)
print("[OK] sessions", len(recs))
diff_core, flips, codes_b, codes_a = [], [], Counter(), Counter()
for sid in sorted(after):
    b, a = before[sid], after[sid]
    if (b.parts, b.n_frames, b.fps, b.mode) != (a.parts, a.n_frames, a.fps, a.mode):
        diff_core.append(sid)
    if b.excluded != a.excluded:
        flips.append((sid, b.excluded, a.excluded))
    for x in b.anomalies:
        codes_b[(x.code, x.level)] += 1
    for x in a.anomalies:
        codes_a[(x.code, x.level)] += 1
print("parts/n_frames/fps/mode changed:", diff_core or "none")
print("excluded flips:", flips or "none")
print("anomalies before:", dict(codes_b))
print("anomalies after :", dict(codes_a))
print("sessions with dropped_frames before/after:",
      sum(any(x.code == "dropped_frames" for x in before[s].anomalies) for s in before),
      sum(any(x.code == "dropped_frames" for x in after[s].anomalies) for s in after))

ex = next(x for s in after for x in after[s].anomalies if x.code == "dropped_frames")
print("data keys:", sorted(ex.data))
print(vars(ex))
import json as _j
rows = []
for s in sorted(after):
    for x in after[s].anomalies:
        if x.code == "dropped_frames":
            rows.append({"sid": s, "data": x.data})
Path("dropped_after.json").write_text(_j.dumps(rows, indent=1, default=str), encoding="utf-8")
