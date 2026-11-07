"""Read-only check of the new crop against existing full-session CNMF-E estimates made with an older crop.

For each session: the crop stage's own detect() on the raw video (records built in memory; the raw folders
are only read; the projection npz goes to a scratch root), the mouse box from the checked sessions of that
mouse (fewer sessions -> a smaller or equal box, so the check is conservative), then the centres of the
accepted components (idx_components) of the old estimates moved to raw-frame coordinates with the old crop
offsets, counted outside the new mouse box and outside the new field.

usage: neuron_check.py OUT_DIR [SID ...]
"""
import glob
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:\Users\User\PycharmProjects\bearmind")
import numpy as np
import scipy.sparse as sp

from bearmind import config
from bearmind.crop import mouse_box_size, place_box
from bearmind.naming import mouse_id
from bearmind.paths import SeriesPaths
from bearmind.registry import RawPart, SessionRecord
from bearmind.stages.crop import detect
from bearmind.video import video_info

DS = Path(r"H:\Dataset\CalciumData")
SCRATCH_ROOT = Path(__file__).resolve().parent / "neuron_root"  # detect() writes its projection here only
SESSIONS = [
    ("RFC", "RFC_F01_1D"), ("RFC", "RFC_F01_3D"), ("RFC", "RFC_F20_1D"), ("RFC", "RFC_F20_3D"),
    ("NOF", "NOF_H01_1D"), ("NOF", "NOF_H03_3D"), ("NOF", "NOF_H03_4D"), ("CC", "CC_H03_1D"),
]


def raw_folder(ds, sid):
    for d, _, files in os.walk(DS / ds / "1_Raw"):
        if any(f.lower().endswith(".avi") for f in files) and (sid in Path(d).parts):
            return Path(d)
    raise FileNotFoundError(sid)


def build_record(sid, folder):
    def num(name):
        return int(re.findall(r"(\d+)\.avi$", name, re.I)[0])
    chunks = sorted((f for f in os.listdir(folder) if f.lower().endswith(".avi")), key=num)
    info = [video_info(folder / c) for c in chunks]
    counts = [i[0] for i in info]
    part = RawPart(folder=str(folder), chunks=chunks, n_frames=sum(counts), shape=(info[0][1], info[0][2]),
                   timestamps_file=None, ts_format=None, start_ms=None, metadata_file=None,
                   chunk_frames=counts)
    return SessionRecord(sid=sid, mouse=mouse_id(sid), mode="check", parts=[part], n_frames=sum(counts),
                         fps=info[0][3], fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})


def old_crop(ds, sid):
    f = glob.glob(str(DS / ds / "cropp" / f"{sid}_l=*_cropping.pickle"))
    assert len(f) == 1, f
    with open(f[0], "rb") as fh:
        return pickle.load(fh), f[0]


def estimates_file(ds, sid):
    f = sorted(glob.glob(str(DS / ds / "4_Estimates" / f"{sid}*estimates.pickle")))
    assert len(f) == 1, f
    return f[0]


def centres(A, dims):
    """(n, 2) y, x centres of mass of the columns of A, pixels in CaImAn's Fortran order."""
    A = sp.csc_matrix(A)
    d1 = dims[0]
    out = np.zeros((A.shape[1], 2))
    for k in range(A.shape[1]):
        s, e = A.indptr[k], A.indptr[k + 1]
        idx, v = A.indices[s:e], np.abs(A.data[s:e])
        y, x = idx % d1, idx // d1
        out[k] = (np.sum(y * v) / v.sum(), np.sum(x * v) / v.sum())
    return out


def outside(pts, box):
    top, left, h, w = box
    y, x = pts[:, 0], pts[:, 1]
    return (y < top - 0.5) | (y > top + h - 0.5) | (x < left - 0.5) | (x > left + w - 0.5)


def edge_dist(pts, box):
    """Distance (px) of each centre to the nearest box edge; negative outside."""
    top, left, h, w = box
    y, x = pts[:, 0], pts[:, 1]
    return np.minimum.reduce([y - (top - 0.5), (top + h - 0.5) - y, x - (left - 0.5), (left + w - 0.5) - x])


def main():
    out_dir = Path(sys.argv[1])
    want = set(sys.argv[2:])
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = SeriesPaths(SCRATCH_ROOT)
    cfg = config.resolve(paths)  # defaults + machine (no series config exists in the scratch root)
    results = {}
    for ds, sid in SESSIONS:
        if want and sid not in want:
            continue
        folder = raw_folder(ds, sid)
        rec = build_record(sid, folder)
        t0 = time.time()
        m = detect(paths, rec, cfg).metrics
        dt = time.time() - t0
        crop, crop_file = old_crop(ds, sid)
        H, W = rec.parts[0].shape
        dims = (H - crop["UP"] - crop["DOWN"], W - crop["LEFT"] - crop["RIGHT"])
        est_f = estimates_file(ds, sid)
        with open(est_f, "rb") as fh:
            e = pickle.load(fh)
        A = sp.csc_matrix(e.A)
        assert A.shape[0] == dims[0] * dims[1], (A.shape, dims)
        c = centres(A, dims) + np.array([crop["UP"], crop["LEFT"]])
        idx = np.arange(A.shape[1]) if e.idx_components is None else np.asarray(e.idx_components, int)
        snr = np.asarray(getattr(e, "SNR_comp", np.full(A.shape[1], np.nan)), float)
        results[sid] = dict(ds=ds, raw=str(folder), n_frames=rec.n_frames, frame=[H, W], detect_s=round(dt, 1),
                            metrics=m, old_crop=crop, old_crop_file=crop_file,
                            old_box=[crop["UP"], crop["LEFT"], dims[0], dims[1]], estimates=est_f,
                            n_components=int(A.shape[1]), accepted=idx.tolist(), centres=c.tolist(),
                            snr=snr.tolist(), T=int(e.C.shape[1]))
        print(f"[OK] {sid}: detect {dt:.0f} s, fov {m['fov']} ({m['fov_method']}, pos/neg {m['activity_pos_neg']}), "
              f"{len(idx)} accepted of {A.shape[1]}", flush=True)
    # mouse boxes from the checked sessions of each mouse
    by_mouse = {}
    for sid, r in results.items():
        by_mouse.setdefault(mouse_id(sid), []).append(sid)
    for mouse, sids in by_mouse.items():
        fovs = [tuple(results[s]["metrics"]["fov"]) for s in sids]
        frame = (min(results[s]["frame"][0] for s in sids), min(results[s]["frame"][1] for s in sids))
        size = mouse_box_size(fovs, frame, int(cfg["crop"]["margin_px"]))
        for s in sids:
            b = place_box(results[s]["metrics"]["fov"], size, frame)
            results[s]["new_box"] = [b.top, b.left, b.height, b.width]
            results[s]["mouse_sessions"] = sids
    for sid, r in results.items():
        c = np.asarray(r["centres"])
        acc = np.asarray(r["accepted"], int)
        out_box = outside(c, r["new_box"])
        out_field = outside(c, r["metrics"]["fov"])
        d = edge_dist(c, r["new_box"])
        r["n_accepted"] = int(len(acc))
        r["n_accepted_outside_box"] = int(out_box[acc].sum())
        r["n_accepted_outside_field"] = int(out_field[acc].sum())
        r["n_all_outside_box"] = int(out_box.sum())
        r["n_all_outside_field"] = int(out_field.sum())
        r["min_edge_dist_accepted"] = round(float(d[acc].min()), 1) if len(acc) else None
        r["n_accepted_within_10px_of_box_edge"] = int((d[acc] < 10).sum())
        r["outside_box_accepted"] = [dict(k=int(k), y=round(c[k, 0], 1), x=round(c[k, 1], 1),
                                          snr=round(r["snr"][k], 2)) for k in acc if out_box[k]]
        r["outside_field_accepted"] = [dict(k=int(k), y=round(c[k, 0], 1), x=round(c[k, 1], 1),
                                            snr=round(r["snr"][k], 2)) for k in acc if out_field[k]]
        r["outside_box_rejected"] = [dict(k=int(k), y=round(c[k, 0], 1), x=round(c[k, 1], 1),
                                          snr=round(r["snr"][k], 2))
                                     for k in range(len(c)) if out_box[k] and k not in set(acc.tolist())]
        print(f"{sid}: accepted {r['n_accepted']}, outside box {r['n_accepted_outside_box']}, outside field "
              f"{r['n_accepted_outside_field']}; all components outside box {r['n_all_outside_box']}; "
              f"min edge dist {r['min_edge_dist_accepted']} px; new box {r['new_box']} old box {r['old_box']}")
    slim = {s: {k: v for k, v in r.items() if k not in ("centres", "snr", "accepted")} for s, r in results.items()}
    (out_dir / "neuron_check.json").write_text(json.dumps(slim, indent=1, default=float))
    np.savez_compressed(out_dir / "neuron_check_arrays.npz",
                        **{f"{s}_centres": np.asarray(r["centres"]) for s, r in results.items()},
                        **{f"{s}_accepted": np.asarray(r["accepted"]) for s, r in results.items()},
                        **{f"{s}_snr": np.asarray(r["snr"]) for s, r in results.items()})
    print("[OK] written", out_dir / "neuron_check.json")


if __name__ == "__main__":
    main()
