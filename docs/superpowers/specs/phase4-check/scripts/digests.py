"""Per-root digest of a snapshot.py output: sha256 over sorted 'path size mtime_ns sha256' lines (files) and
'path mtime_ns' lines (folders). usage: python digests.py SNAP.json [SNAP.json ...]"""
import hashlib
import json
import sys

for f in sys.argv[1:]:
    snap = json.load(open(f, encoding="utf-8"))
    for root, s in snap.items():
        lines = [f"{p} {v['size']} {v['mtime_ns']} {v['sha256']}" for p, v in sorted(s["files"].items())]
        lines += [f"{p}/ {m}" for p, m in sorted(s["dirs"].items())]
        d = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()[:16]
        hashed = all(v["sha256"] for v in s["files"].values())
        print(f"{f}  {root}  files {len(s['files'])}  folders {len(s['dirs'])}  "
              f"{'sha256' if hashed else 'size+mtime'}  digest {d}")
