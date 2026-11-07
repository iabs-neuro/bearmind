"""State of series_short: stage times and finished stamps from the registry, Preview/ listing, summary columns.
usage: short_state.py OUT.json   (read-only)"""
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(r"H:\Projects\_bearmind_test\series_short")
F = "%Y-%m-%dT%H:%M:%S.%f"
out = {"stages": {}, "preview_files": {}, "summary": None}
for f in sorted((ROOT / "_bearmind" / "registry").glob("*.json")):
    r = json.loads(f.read_text())
    st = {}
    for name, s in r["stages"].items():
        secs = None
        if s.get("started") and s.get("finished"):
            secs = round((datetime.strptime(s["finished"], F) - datetime.strptime(s["started"], F)).total_seconds(), 1)
        keep = {k: s["metrics"][k] for k in ("crop", "n_frames", "ssub_B", "need_gib", "map_workers", "n_components",
                                             "n_accepted", "patch_workers", "tbin", "ssub", "bytes", "box", "fov")
                if k in s.get("metrics", {})}
        st[name] = dict(state=s["state"], started=s.get("started"), finished=s.get("finished"), secs=secs,
                        outputs=sorted(s.get("outputs", {})), metrics=keep,
                        anomalies=[a["code"] for a in r.get("anomalies", []) if a.get("stage") == name] if False else None)
    out["stages"][r["sid"]] = st
    out["anomalies_" + r["sid"]] = [a.get("code") for a in r.get("anomalies", [])]
for p in sorted((ROOT / "Preview").iterdir()):
    s = p.stat()
    out["preview_files"][p.name] = [s.st_size, datetime.fromtimestamp(s.st_mtime).isoformat(timespec="seconds")]
df = pd.read_csv(ROOT / "_bearmind" / "summary.csv")
cols = [c for c in ("sid", "crop_box", "mc", "mc_preview", "cnmf", "cnmf_preview", "cnmf_n_components",
                    "export_n_neurons", "worst_check", "anomalies") if c in df.columns]
out["summary"] = df[cols].to_dict(orient="records")
Path(sys.argv[1]).write_text(json.dumps(out, indent=1, default=str))
for sid, st in out["stages"].items():
    print(sid, {k: (v["state"], v["secs"]) for k, v in st.items()})
    print("   mc", st["mc"]["metrics"], st["mc"]["outputs"])
    print("   cnmf", st["cnmf"]["metrics"], st["cnmf"]["outputs"])
    for side in ("mc_preview", "cnmf_preview"):
        if side in st:
            print("  ", side, st[side]["state"], st[side]["secs"], st[side]["metrics"])
    print("   anomalies", out["anomalies_" + sid])
for k, v in out["preview_files"].items():
    print("  ", k, v)
print(df[cols].to_string())
