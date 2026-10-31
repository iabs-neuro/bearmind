"""Threshold maps: CNMF-E correlation / PNR images per candidate gSig for the threshold GUI."""
import os

import numpy as np

import bm_qc
from bearmind.stages import StageResult
from bearmind.video import VideoArray

BLOCK = 1000  # frames per streamed block for mean/max projections


def _projections(movie):
    """Mean and max over time in one pass; the movie is never fully in memory."""
    T = movie.shape[0]
    total = np.zeros(movie.shape[1:], np.float64)
    mx = None
    for s in range(0, T, BLOCK):
        blk = np.asarray(movie[s:s + BLOCK])
        total += blk.sum(0, dtype=np.float64)
        m = blk.max(0).astype(np.float32)
        mx = m if mx is None else np.maximum(mx, m)
    return (total / T).astype(np.float32), mx


def _suggest(maps, g):
    sug, _ = bm_qc.suggest_init_thresholds(maps, gSiz=4 * g + 1)
    for method in ("mirror_fdr", "robust_bg"):
        r = sug.get(method)
        if r and r.get("min_corr") is not None and r.get("min_pnr") is not None:
            return {"gSig": int(g), "min_corr": float(r["min_corr"]),
                    "min_pnr": float(r["min_pnr"]), "method": method}
    raise RuntimeError("no threshold suggestion could be computed")


def run(paths, rec, cfg) -> StageResult:
    sid = rec.sid
    gsigs = [int(g) for g in cfg["thresholds"]["gsig_range"]]
    g_sug = 4 if 4 in gsigs else gsigs[len(gsigs) // 2]
    tsub = cfg["cnmf"]["tsub"]
    movie = VideoArray(paths.mc_video(sid))

    arrays = {"gsig": np.array(gsigs, dtype=int)}
    arrays["mean"], arrays["max"] = _projections(movie)
    suggested = None
    for g in gsigs:
        maps = bm_qc.compute_init_maps(movie, gSig=g, tsub=tsub, mirror=True, max_mem_gb=16,
                                       verbose=False)
        arrays[f"cn_{g}"] = np.asarray(maps["cn"], np.float32)
        arrays[f"pnr_{g}"] = np.asarray(maps["pnr"], np.float32)
        if "cn_neg" in maps:
            arrays[f"cn_mirror_{g}"] = np.asarray(maps["cn_neg"], np.float32)
            arrays[f"pnr_mirror_{g}"] = np.asarray(maps["pnr_neg"], np.float32)
        if g == g_sug:
            suggested = _suggest(maps, g)

    out = paths.threshold_maps(sid)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)
    return StageResult(outputs={"threshold_maps": paths.rel(out)},
                       metrics={"suggested": suggested, "gsig_range": gsigs, "n_frames": int(movie.shape[0])})
