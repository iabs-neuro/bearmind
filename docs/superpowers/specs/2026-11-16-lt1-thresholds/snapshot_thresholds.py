"""Snapshot of the LT1 thresholds: the author's choices and the automatic suggestions, for later calibration.

Reads only (the series folder is not written):
  _bearmind/threshold_choices.csv      the window's log of every Accept (author's values + the suggestion shown)
  _bearmind/config/sessions/<SID>.yaml the values CNMF-E uses (cnmf.gSig / min_corr / min_pnr)
  _bearmind/registry/<SID>.json        n_frames, fps, the batch suggestion, stage states
  _bearmind/thresholds/<SID>_maps.npz  the maps; all three suggestion methods are recomputed from them

Writes next to this script:
  lt1_thresholds.csv          one row per session
  threshold_choices_log.csv   copy of the window's log (history of corrections included)
  seeds_lowered.csv           seeds on the maps at the author's thresholds and at lowered ones (sessions with a choice)
  grids/<SID>_g<g>.npz        the mirror_fdr grid (corr_grid, pnr_grid, n_real, n_null, fdr, marginal_fdr)

Run from the repo root with the bearmind env:  python docs/superpowers/specs/2026-11-16-lt1-thresholds/snapshot_thresholds.py
"""
import csv
import json
import shutil
import sys
from pathlib import Path

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

import bm_qc  # noqa: E402

SERIES = Path(r"H:\Projects\LT1\CalciumData\_bearmind")
OUT = Path(__file__).resolve().parent
CORR_DELTAS = (0.0, -0.02, -0.05, -0.1, -0.2, -0.4)
PNR_DELTAS = (0.0, -3.0, -6.0, -9.0, -12.0)


def maps_for(z, g):
    return {"cn": z[f"cn_{g}"], "pnr": z[f"pnr_{g}"], "cn_neg": z[f"cn_mirror_{g}"],
            "pnr_neg": z[f"pnr_mirror_{g}"], "mean": z["mean"], "max": z["max"], "gSig": g}


def last_choice_rows():
    rows = list(csv.DictReader(open(SERIES / "threshold_choices.csv", encoding="utf-8")))
    last = {}
    for r in rows:
        if r["when"] >= last.get(r["sid"], {}).get("when", ""):
            last[r["sid"]] = r
    n_acc = {}
    for r in rows:
        n_acc[r["sid"]] = n_acc.get(r["sid"], 0) + 1
    return last, n_acc


def fnum(x):
    return None if x in (None, "") else float(x)


def main():
    (OUT / "grids").mkdir(exist_ok=True)
    shutil.copyfile(SERIES / "threshold_choices.csv", OUT / "threshold_choices_log.csv")
    last, n_acc = last_choice_rows()
    rows, seeds = [], []
    for reg_path in sorted((SERIES / "registry").glob("*.json")):
        rec = json.loads(reg_path.read_text(encoding="utf-8"))
        sid = rec["sid"] if "sid" in rec else reg_path.stem
        st = rec.get("stages", {})
        th = st.get("thresholds", {})
        sug = (th.get("metrics") or {}).get("suggested") or {}
        cfg_path = SERIES / "config" / "sessions" / f"{sid}.yaml"
        cn = ((yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("cnmf") or {}) if cfg_path.exists() else {}
        chosen = all(cn.get(k) is not None for k in ("gSig", "min_corr", "min_pnr"))
        lc = last.get(sid, {})
        row = {
            "sid": sid, "mouse": rec.get("mouse"), "n_frames": rec.get("n_frames"), "fps": rec.get("fps"),
            "minutes": round(rec["n_frames"] / rec["fps"] / 60, 2) if rec.get("fps") and rec.get("n_frames") else None,
            "batch_sug_gSig": sug.get("gSig"), "batch_sug_min_corr": sug.get("min_corr"),
            "batch_sug_min_pnr": sug.get("min_pnr"), "batch_sug_method": sug.get("method"),
            "author_gSig": cn.get("gSig") if chosen else None,
            "author_min_corr": cn.get("min_corr") if chosen else None,
            "author_min_pnr": cn.get("min_pnr") if chosen else None,
            "author_source": lc.get("source"), "author_when": lc.get("when"), "n_accepts": n_acc.get(sid, 0),
            "author_equals_suggestion": (chosen and sug and cn.get("gSig") == sug.get("gSig")
                                         and abs(float(cn["min_corr"]) - float(sug["min_corr"])) < 1e-9
                                         and abs(float(cn["min_pnr"]) - float(sug["min_pnr"])) < 1e-9) or False,
            "cnmf_state": st.get("cnmf", {}).get("state"), "gate_state": st.get("gate", {}).get("state"),
        }
        maps_path = SERIES / "thresholds" / f"{sid}_maps.npz"
        if maps_path.exists():
            with np.load(maps_path) as z:
                g = int(cn["gSig"]) if chosen else int(sug.get("gSig", 4))
                m = maps_for(z, g)
                out, grid = bm_qc.suggest_init_thresholds(m, gSiz=4 * g + 1)
                for meth in ("mirror_fdr", "robust_bg", "pct_margin"):
                    r = out.get(meth) or {}
                    row[f"{meth}_min_corr"] = r.get("min_corr")
                    row[f"{meth}_min_pnr"] = r.get("min_pnr")
                    row[f"{meth}_n_seeds"] = r.get("n_seeds")
                    row[f"{meth}_fdr"] = r.get("fdr")
                if grid is not None:
                    np.savez_compressed(OUT / "grids" / f"{sid}_g{g}.npz", **{k: np.asarray(v) for k, v in grid.items()})
                if chosen:
                    ev = bm_qc.evaluate_thresholds(m, float(cn["min_corr"]), float(cn["min_pnr"]), 4 * g + 1)
                    row["author_n_seeds"], row["author_fdr"] = ev.get("n_seeds"), ev.get("fdr")
                    for dc in CORR_DELTAS:
                        for dp in PNR_DELTAS:
                            c, p = round(float(cn["min_corr"]) + dc, 3), round(float(cn["min_pnr"]) + dp, 2)
                            if c <= 0 or p <= 0:
                                continue
                            e = bm_qc.evaluate_thresholds(m, c, p, 4 * g + 1)
                            seeds.append({"sid": sid, "gSig": g, "d_corr": dc, "d_pnr": dp, "min_corr": c,
                                          "min_pnr": p, "n_seeds": e.get("n_seeds"),
                                          "n_null_seeds": e.get("n_null_seeds"), "fdr": e.get("fdr")})
        rows.append(row)
        print(f"[OK] {sid}", flush=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(OUT / "lt1_thresholds.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "seeds_lowered.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(seeds[0]))
        w.writeheader()
        w.writerows(seeds)
    print(f"[OK] {len(rows)} sessions, {sum(1 for r in rows if r['author_min_corr'] is not None)} with a choice")


if __name__ == "__main__":
    main()
