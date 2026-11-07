"""Neuron check on series_short's own records: the CNMF-E estimates of the 2000-frame clips made before A2 (old crop,
copied to the scratchpad before the re-run) against the new crop of the same series (registry after the re-run).
Read-only for the series.

usage: neuron_check_short.py BEFORE_DIR OUT_DIR
"""
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import numpy as np
import scipy.sparse as sp

from bearmind.crop import lit_fov
from bearmind.paths import SeriesPaths
from bearmind.registry import load_record

P = SeriesPaths(Path(r"H:\Projects\_bearmind_test\series_short"))
SIDS = ("NOF_H01_1D", "NOF_H03_3D")


def centres(A, d1):
    A = sp.csc_matrix(A)
    out = np.zeros((A.shape[1], 2))
    for k in range(A.shape[1]):
        s, e = A.indptr[k], A.indptr[k + 1]
        idx, v = A.indices[s:e], np.abs(A.data[s:e])
        out[k] = (np.sum((idx % d1) * v) / v.sum(), np.sum((idx // d1) * v) / v.sum())
    return out


def outside(pts, box):
    top, left, h, w = box
    return (pts[:, 0] < top - 0.5) | (pts[:, 0] > top + h - 0.5) | (pts[:, 1] < left - 0.5) | (pts[:, 1] > left + w - 0.5)


def main():
    before, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
    res = {}
    arrays = {}
    for sid in SIDS:
        old = json.loads((before / "registry" / f"{sid}.json").read_text())
        old_box = old["stages"]["mc"]["metrics"]["crop"]  # [top, left, h, w] MC (and so CNMF-E) ran with
        rec = load_record(P, sid)
        m = rec.stages["crop"].metrics
        with open(before / "4_Estimates" / f"{sid}_estimates.pickle", "rb") as fh:
            e = pickle.load(fh)
        A = sp.csc_matrix(e.A)
        assert A.shape[0] == old_box[2] * old_box[3], (A.shape, old_box)
        c = centres(A, old_box[2]) + np.array(old_box[:2])
        acc = np.arange(A.shape[1]) if e.idx_components is None else np.asarray(e.idx_components, int)
        snr = np.asarray(e.SNR_comp, float)
        with np.load(P.crop_projection(sid)) as z:
            mean = z["mean"].astype(np.float32)
        lit = lit_fov(mean)
        mask = lit[1]
        ob = outside(c, m["box"])
        of = outside(c, m["fov"])
        outs = []
        for k in acc:
            if ob[k]:
                y, x = c[k]
                yi, xi = min(int(round(y)), mask.shape[0] - 1), min(int(round(x)), mask.shape[1] - 1)
                outs.append(dict(k=int(k), y=round(float(y), 1), x=round(float(x), 1), snr=round(float(snr[k]), 2),
                                 in_lit_mask=bool(mask[yi, xi]), mean_at=round(float(mean[yi, xi]), 1)))
        res[sid] = dict(old_box=old_box, new_box=m["box"], fov=m["fov"], fov_lit=m["fov_lit"],
                        fov_method=m["fov_method"], n_components=int(A.shape[1]), n_accepted=int(len(acc)),
                        n_accepted_outside_box=int(ob[acc].sum()), n_accepted_outside_field=int(of[acc].sum()),
                        n_all_outside_box=int(ob.sum()), outside_box_accepted=outs,
                        lit_mean_median=round(float(np.median(mean[mask])), 1))
        arrays[f"{sid}_centres"], arrays[f"{sid}_accepted"], arrays[f"{sid}_snr"] = c, acc, snr
        print(f"{sid}: accepted {len(acc)} of {A.shape[1]}; outside new box {int(ob[acc].sum())}, outside field "
              f"{int(of[acc].sum())}; old box {old_box} new box {m['box']}")
        for o in outs:
            print("   ", o)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "neuron_check_short.json").write_text(json.dumps(res, indent=1))
    np.savez_compressed(out_dir / "neuron_check_short_arrays.npz", **arrays)


if __name__ == "__main__":
    main()
