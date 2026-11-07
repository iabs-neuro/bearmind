"""Rows of the author's autoinspection tables (May 2026) kept by autoinspection (delete != 1) whose centre lies
outside the new mouse box: direct count from the table centres (y, x in the estimates' frame) moved by the old
crop offsets. Read-only.

usage: neuron_kept.py RESULT_DIR PROJECTION_ROOT
"""
import csv
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import numpy as np

from bearmind.crop import lit_fov

DS = Path(r"H:\Dataset\CalciumData")


def main():
    res_dir, proj_root = Path(sys.argv[1]), Path(sys.argv[2])
    res = json.loads((res_dir / "neuron_check.json").read_text())
    out = {}
    for sid, r in res.items():
        f = glob.glob(str(DS / r["ds"] / "4_Estimates_inspections" / f"inspection_artifacts_{sid}" /
                          "*_metrics_with_decisions.csv"))
        if not f:
            print(f"{sid}: no autoinspection table")
            continue
        with np.load(proj_root / "_bearmind" / "crop" / f"{sid}_projection.npz") as z:
            mean = z["mean"].astype(np.float32)
        mask = lit_fov(mean)[1]
        top0, left0 = r["old_box"][:2]
        t, l, h, w = r["new_box"]
        n_rows = n_kept = 0
        items = []
        with open(f[0], newline="") as fh:
            for row in csv.DictReader(fh):
                n_rows += 1
                if row["delete"] == "1":
                    continue
                n_kept += 1
                cy, cx = (float(v) for v in row["center"].strip("[]").split())
                y, x = cy + top0, cx + left0
                if t - 0.5 <= y <= t + h - 0.5 and l - 0.5 <= x <= l + w - 0.5:
                    continue
                yi, xi = min(int(round(y)), mask.shape[0] - 1), min(int(round(x)), mask.shape[1] - 1)
                items.append(dict(idx=row["component_idx"], y=round(y, 1), x=round(x, 1), decision=row["decision"],
                                  ml_p=row["ml_keep_probability"], snr=row["caiman_snr"],
                                  in_lit_mask=bool(mask[yi, xi]), mean_at=round(float(mean[yi, xi]), 1)))
        out[sid] = dict(table=f[0], n_rows=n_rows, n_kept=n_kept, kept_outside_box=items)
        print(f"{sid}: table rows {n_rows}, kept (delete != 1) {n_kept}, kept outside the new box {len(items)}, "
              f"of them in the lit mask {sum(i['in_lit_mask'] for i in items)}")
        for i in items:
            print("   ", i)
    (res_dir / "neuron_kept.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
