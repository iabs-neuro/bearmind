"""Verify the neuron check: orientation of A (F order, as CaImAn) against the activity map, and describe every
accepted component outside the new mouse box (inside the lit area or not, SNR, the author's autoinspection
decision found by centre). Read-only.

usage: neuron_verify.py RESULT_DIR PROJECTION_ROOT
"""
import ast
import csv
import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import numpy as np
import scipy.sparse as sp

from bearmind.crop import lit_fov

DS = Path(r"H:\Dataset\CalciumData")


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def decisions(ds, sid):
    f = glob.glob(str(DS / ds / "4_Estimates_inspections" / f"inspection_artifacts_{sid}" / "*_metrics_with_decisions.csv"))
    if not f:
        return None
    rows = []
    with open(f[0], newline="") as fh:
        for r in csv.DictReader(fh):
            c = r["center"].strip("[]").split()
            rows.append((float(c[0]), float(c[1]), r.get("decision"), r.get("ml_keep_probability"), r.get("delete"),
                         r.get("caiman_snr")))
    return rows


def main():
    res_dir, proj_root = Path(sys.argv[1]), Path(sys.argv[2])
    res = json.loads((res_dir / "neuron_check.json").read_text())
    arr = np.load(res_dir / "neuron_check_arrays.npz")
    report = {}
    for sid, r in res.items():
        with np.load(proj_root / "_bearmind" / "crop" / f"{sid}_projection.npz") as z:
            mean = z["mean"].astype(np.float32)
            act = z["activity"].astype(np.float32)
        lit = lit_fov(mean)  # same defaults as the crop config (min_fov_frac 0.05, fov_threshold_frac 0.2)
        assert list(lit[0]) == list(r["metrics"]["fov_lit"]), (lit[0], r["metrics"]["fov_lit"])
        mask = lit[1]
        with open(r["estimates"], "rb") as fh:
            e = pickle.load(fh)
        A = sp.csc_matrix(e.A)
        top, left, h, w = r["old_box"]
        acc = arr[f"{sid}_accepted"].astype(int)
        s = np.asarray(A[:, acc].sum(axis=1)).ravel()
        img_f = np.zeros_like(act)
        img_f[top:top + h, left:left + w] = s.reshape((h, w), order="F")
        line = {"corr_F_activity": round(corr(img_f, act), 3)}
        if h == w:  # a C-order reading is only shape-compatible for a square crop
            img_c = np.zeros_like(act)
            img_c[top:top + h, left:left + w] = s.reshape((h, w), order="C")
            line["corr_C_activity"] = round(corr(img_c, act), 3)
        c = arr[f"{sid}_centres"]
        snr = arr[f"{sid}_snr"]
        nb = r["new_box"]
        rows = decisions(r["ds"], sid)
        outs = []
        for k in acc:
            y, x = c[k]
            if nb[0] - 0.5 <= y <= nb[0] + nb[2] - 0.5 and nb[1] - 0.5 <= x <= nb[1] + nb[3] - 0.5:
                continue
            yi, xi = int(round(y)), int(round(x))
            in_lit = bool(mask[min(max(yi, 0), mask.shape[0] - 1), min(max(xi, 0), mask.shape[1] - 1)])
            dist = max(nb[0] - 0.5 - y, y - (nb[0] + nb[2] - 0.5), nb[1] - 0.5 - x, x - (nb[1] + nb[3] - 0.5))
            item = dict(k=int(k), y=round(float(y), 1), x=round(float(x), 1), snr=round(float(snr[k]), 2),
                        in_lit_mask=in_lit, px_outside=round(float(dist), 1),
                        mean_at=round(float(mean[min(yi, mean.shape[0]-1), min(xi, mean.shape[1]-1)]), 1))
            if rows:
                # the table's centre is in the estimates' (cropped) frame; try both axis orders
                ly, lx = y - top, x - left
                best = min(rows, key=lambda q: min((q[0] - ly) ** 2 + (q[1] - lx) ** 2, (q[1] - ly) ** 2 + (q[0] - lx) ** 2))
                dd = min((best[0] - ly) ** 2 + (best[1] - lx) ** 2, (best[1] - ly) ** 2 + (best[0] - lx) ** 2) ** 0.5
                item.update(match_px=round(dd, 1), decision=best[2], ml_p=best[3], delete=best[4])
            outs.append(item)
        line["n_outside"] = len(outs)
        line["n_outside_in_lit_mask"] = sum(o["in_lit_mask"] for o in outs)
        line["outside"] = outs
        line["lit_mean_median"] = round(float(np.median(mean[mask])), 1)
        line["unlit_mean_median"] = round(float(np.median(mean[~mask])), 1)
        report[sid] = line
        kept = [o for o in outs if o.get("decision") not in (None, "delete") and o.get("match_px", 99) < 3]
        print(f"{sid}: corr F {line['corr_F_activity']} C {line.get('corr_C_activity')}; outside {len(outs)}, "
              f"in lit mask {line['n_outside_in_lit_mask']}; matched and not deleted by autoinspection {len(kept)}; "
              f"mean lit/unlit {line['lit_mean_median']}/{line['unlit_mean_median']}")
        for o in outs:
            print("   ", o)
    (res_dir / "neuron_verify.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
