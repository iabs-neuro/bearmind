"""Builds auto-thresholds-explained.html (self-contained: inline JS/SVG/canvas, no external requests).

Run after `auto_vs_author.py widen`, `frames`, `traces` and `report` (same env, same LT1_CACHE):
  python docs/superpowers/specs/2026-11-16-lt1-thresholds/build_explainer.py
Every number in the page text comes from report_numbers.json (printed by auto_vs_author.py report) or from the
arrays embedded here (grids, map crops, seeds), which the page itself displays.
"""
import base64
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import auto_vs_author as A  # noqa: E402
import numpy as np  # noqa: E402

PANEL = {"LT1_J03_1D": (170, 330, 250, 410), "LT1_J06_1D": (150, 310, 110, 270)}  # crop rows r0:r1, cols c0:c1
TRACE_SIDS = ("LT1_J06_1D", "LT1_J08_1D", "LT1_J18_2D", "LT1_J03_1D", "LT1_J01_1D", "LT1_J13_1D")


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")


def panel_data(sid, box):
    import bm_qc
    from bearmind.seed_mask import valid_mask
    out = A.CACHE / f"panel_{sid}.json"
    if out.exists():
        return json.loads(out.read_text(encoding="utf-8"))
    m = A.load_maps(sid)
    valid = valid_mask(m, border_pix=5)
    halos = {"cur": bm_qc.halo_mask(m, valid, A.GSIZ), "capped": A.halo_capped(m, valid)}
    r0, r1, c0, c1 = box
    md = 2 * A.G
    seeds = {"real": [], "cur": [], "capped": []}
    for c in A.WIDE_CORR:
        for p in A.WIDE_PNR:
            s = bm_qc.count_seeds(m["cn"], m["pnr"], c, p, A.GSIZ, valid=valid, min_dist=md)
            lists = {"real": s}
            for k, h in halos.items():
                lists[k] = bm_qc.count_seeds(m["cn_neg"], m["pnr_neg"], c, p, A.GSIZ, valid=valid & ~h, min_dist=md)
            for k, arr in lists.items():
                inb = arr[(arr[:, 0] >= r0) & (arr[:, 0] < r1) & (arr[:, 1] >= c0) & (arr[:, 1] < c1)]
                seeds[k].append((inb - [r0, c0]).astype(np.uint8).ravel().tolist())
    crop = (slice(r0, r1), slice(c0, c1))
    q = lambda a, s: np.clip(np.round(np.nan_to_num(a[crop]) * s), 0, 65535).astype("<u2")  # noqa: E731
    d = {"sid": sid, "box": box, "h": r1 - r0, "w": c1 - c0,
         "cn": b64(q(m["cn"], 10000)), "pnr": b64(q(m["pnr"], 100)),
         "cn_neg": b64(q(m["cn_neg"], 10000)), "pnr_neg": b64(q(m["pnr_neg"], 100)),
         "mean": b64(np.clip(m["mean"][crop], 0, 255).astype(np.uint8)),
         "valid": b64(valid[crop].astype(np.uint8)),
         "halo_cur": b64(halos["cur"][crop].astype(np.uint8)), "halo_capped": b64(halos["capped"][crop].astype(np.uint8)),
         "seeds": seeds}
    out.write_text(json.dumps(d), encoding="utf-8")
    return d


def grid_data(sid):
    z = np.load(A.CACHE / "wide" / f"{sid}.npz")
    raw_cur = np.round(z["n_null_cur"] / float(z["area_cur"])).astype("<u2")
    raw_cap = np.round(z["n_null_capped"] / float(z["area_capped"])).astype("<u2")
    return {"n_real": b64(z["n_real"].astype("<u2")), "null_cur": b64(raw_cur), "null_capped": b64(raw_cap),
            "area_cur": float(z["area_cur"]), "area_capped": float(z["area_capped"]),
            "robust": [round(float(x), 3) for x in z["robust"]]}


def trace_data(sid):
    z = np.load(A.CACHE / f"{sid}_trace.npz")
    g, fps = z["g"].astype(np.float64), float(z["fps"])
    medg = np.median(g)
    black = g < 0.5 * medg
    gg = g.copy()
    gg[black] = medg
    d = gg / A.binned_baseline(gg, fps) - 1
    step = max(1, int(round(fps / 2)))  # 2 points per second: max and min of each half second keep the peaks
    n = len(d) // step
    dd = d[:n * step].reshape(n, step)
    return {"sid": sid, "fps": fps, "step_s": step / fps, "hi": np.round(dd.max(1), 4).tolist(),
            "lo": np.round(dd.min(1), 4).tolist(), "black": np.flatnonzero(black).tolist()}


def main():
    rep = json.loads((HERE / "report_numbers.json").read_text(encoding="utf-8"))
    data = {"num": rep["numbers"], "sessions": rep["sessions"],
            "corr": A.WIDE_CORR.tolist(), "pnr": A.WIDE_PNR.tolist(), "n_orig_corr": len(A.ORIG_CORR),
            "n_orig_pnr": len(A.ORIG_PNR), "grids": {}, "panel": {}, "traces": [], "frames": {}}
    for s in rep["sessions"]:
        if (A.CACHE / "wide" / f"{s['sid']}.npz").exists():
            data["grids"][s["sid"]] = grid_data(s["sid"])
    for sid, box in PANEL.items():
        d = dict(panel_data(sid, box))
        packed = {}
        for key, lists in d["seeds"].items():  # one byte array + offsets per list type (keeps the page small)
            off = np.cumsum([0] + [len(x) for x in lists]).astype("<u4")
            xy = np.array([v for x in lists for v in x], np.uint8)
            packed[key] = {"xy": b64(xy), "off": b64(off)}
        d["seeds"] = packed
        data["panel"][sid] = d
        print(f"[OK] panel {sid}")
    for sid in TRACE_SIDS:
        if (A.CACHE / f"{sid}_trace.npz").exists():
            data["traces"].append(trace_data(sid))
    for f in sorted((HERE / "frames_test").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        r.pop("seed_peak_frames", None)
        data["frames"][f.stem] = r
    tpl = (HERE / "explainer_template.html").read_text(encoding="utf-8")
    import re
    missing = []
    for key, path in re.findall(r'data-num="([^"]+)"(?: data-path="([^"]*)")?', tpl):
        v = data["num"].get(key)
        for p in (path.split(".") if path else []):
            v = None if v is None else (v[int(p)] if isinstance(v, list) else v.get(p))
        if v is None:
            missing.append(f"{key}.{path}")
    if "<!--PROSE:" in tpl:
        missing.append("PROSE placeholder left")
    print("[OK] every number of the page text is in report_numbers.json" if not missing
          else "[FAILED] missing: " + ", ".join(missing))
    html = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    (HERE / "auto-thresholds-explained.html").write_text(html, encoding="utf-8")
    print(f"[OK] auto-thresholds-explained.html, {len(html) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
