"""Threshold maps: CNMF-E correlation / PNR images per candidate gSig for the threshold GUI."""
import gc
import os
import shutil
import time
import traceback

import numpy as np

import bm_qc
from bearmind.registry import Anomaly
from bearmind.stages import StageResult
from bearmind.stages.cnmf import mem_budget_gb
from bearmind.video import decode_to_memmap

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
    """Returns the suggestion dict; raises ValueError when no method gives finite values."""
    sug, _ = bm_qc.suggest_init_thresholds(maps, gSiz=4 * g + 1)
    for method in ("mirror_fdr", "robust_bg"):
        r = sug.get(method)
        if not r or r.get("min_corr") is None or r.get("min_pnr") is None:
            continue
        c, p = float(r["min_corr"]), float(r["min_pnr"])
        if np.isfinite(c) and np.isfinite(p):
            return {"gSig": int(g), "min_corr": c, "min_pnr": p, "method": method}
    raise ValueError("no suggestion method gave finite min_corr / min_pnr")


def run(paths, rec, cfg) -> StageResult:
    sid = rec.sid
    gsigs = [int(g) for g in cfg["thresholds"]["gsig_range"]]
    g_sug = 4 if 4 in gsigs else gsigs[len(gsigs) // 2]
    tsub = cfg["cnmf"]["tsub"]
    mem_gb = mem_budget_gb(cfg)
    tmp_dir = paths.temp_dir(cfg["machine"]["temp_dir"], sid)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    u8_path = tmp_dir / "mc_u8.dat"
    movie = None
    try:
        # one decode: compute_init_maps reads row strips, and each strip of a VideoArray re-decodes the file
        movie = decode_to_memmap(paths.mc_video(sid), u8_path)
        n_frames = int(movie.shape[0])
        arrays = {"gsig": np.array(gsigs, dtype=int)}
        arrays["mean"], arrays["max"] = _projections(movie)
        sug_maps = None
        for g in gsigs:
            maps = bm_qc.compute_init_maps(movie, gSig=g, tsub=tsub, mirror=True, max_mem_gb=mem_gb,
                                           verbose=False)
            arrays[f"cn_{g}"] = np.asarray(maps["cn"], np.float32)
            arrays[f"pnr_{g}"] = np.asarray(maps["pnr"], np.float32)
            if "cn_neg" in maps:
                arrays[f"cn_mirror_{g}"] = np.asarray(maps["cn_neg"], np.float32)
                arrays[f"pnr_mirror_{g}"] = np.asarray(maps["pnr_neg"], np.float32)
            if g == g_sug:
                sug_maps = maps
    except BaseException as e:
        traceback.clear_frames(e.__traceback__)  # traceback frames would keep the memmap (and its file) open
        raise
    finally:
        movie = None  # Windows cannot delete a mapped file
        gc.collect()
        try:
            u8_path.unlink(missing_ok=True)
        except OSError:
            pass
        for _ in range(5):
            shutil.rmtree(tmp_dir, ignore_errors=True)
            if not tmp_dir.exists():
                break
            time.sleep(0.5)

    out = paths.threshold_maps(sid)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.npz")
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, out)
    finally:
        tmp.unlink(missing_ok=True)

    # The suggestion only seeds the GUI: a failure must not lose the maps already written.
    suggested, anomalies = None, []
    if tmp_dir.exists():
        anomalies.append(Anomaly("temp_not_removed", "warning", str(tmp_dir)))
    try:
        suggested = _suggest(sug_maps, g_sug)
    except Exception as e:
        anomalies.append(Anomaly("no_threshold_suggestion", "warning", str(e) or type(e).__name__))
    return StageResult(outputs={"threshold_maps": paths.rel(out)}, anomalies=anomalies,
                       metrics={"suggested": suggested, "gsig_range": gsigs, "n_frames": n_frames})
