"""Read-only snapshot of the LT1 batch for the actions page: per-session stage times from the registry.

Reads H:/Projects/LT1/CalciumData/_bearmind/registry/*.json and heartbeat.json; writes batch_snapshot.json
next to this script. Writes nothing under H:.
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(r"H:\Projects\LT1\CalciumData\_bearmind")
NEW_CODE = datetime(2026, 11, 13, 15, 57, 0)  # restart on the merged maps speedup (989c88b)


def ts(s):
    return datetime.fromisoformat(s) if s else None


def main():
    rows = []
    for p in sorted((ROOT / "registry").glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        row = {"sid": d["sid"], "mouse": d.get("mouse"), "n_frames": d["n_frames"], "fps": d["fps"],
               "minutes": d["n_frames"] / d["fps"] / 60.0, "excluded": d.get("excluded")}
        for st in ("crop", "mc", "thresholds", "cnmf"):
            s = d["stages"].get(st, {})
            a, b = ts(s.get("started")), ts(s.get("finished"))
            row[f"{st}_state"] = s.get("state")
            row[f"{st}_started"] = s.get("started")
            row[f"{st}_finished"] = s.get("finished")
            row[f"{st}_min"] = (b - a).total_seconds() / 60.0 if a and b else None
        box = (d["stages"]["mc"].get("params") or {}).get("box")
        row["box"] = box
        row["mpx_mframes"] = box[0] * box[1] * d["n_frames"] / 1e12 if box else None  # (pixels x frames) / 1e12
        tm = d["stages"]["thresholds"].get("metrics") or {}
        row["suggested"] = tm.get("suggested")
        row["map_workers"] = tm.get("map_workers")
        sess_cfg = ROOT / "config" / "sessions" / f"{d['sid']}.yaml"
        cn = (yaml.safe_load(sess_cfg.read_text(encoding="utf-8")) or {}).get("cnmf", {}) if sess_cfg.exists() else {}
        row["thresholds_chosen"] = all(cn.get(k) is not None for k in ("gSig", "min_corr", "min_pnr"))
        a = ts(row["thresholds_started"])
        row["maps_code"] = None if a is None else ("new" if a >= NEW_CODE else "old")
        rows.append(row)
    hb = json.loads((ROOT / "heartbeat.json").read_text(encoding="utf-8"))
    out = {"snapshot": datetime.now().isoformat(timespec="seconds"), "heartbeat": hb, "sessions": rows}
    dst = Path(__file__).with_name("batch_snapshot.json")
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("[OK]", dst, len(rows), "sessions")


if __name__ == "__main__":
    sys.exit(main())
