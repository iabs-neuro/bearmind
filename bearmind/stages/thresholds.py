"""Threshold maps: CNMF-E correlation / PNR images per candidate gSig for the threshold GUI."""
import gc
import math
import os
import shutil
import time
import traceback

import numpy as np

import bm_qc
from bearmind import config, pool
from bearmind.registry import Anomaly
from bearmind.stages import StageResult, require_temp_space
from bearmind.stages.cnmf import mem_budget_gb
from bearmind.video import decode_to_memmap, video_info

BLOCK = 1000  # frames per streamed block for mean/max projections
# Temp disk per pixel-frame of the MC movie: decode_to_memmap's uint8 copy; compute_init_maps works on row
# strips in RAM (max_mem_gb).
TEMP_BYTES_PER_PX_FRAME = 1


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


def map_peak_gb(T: int, h: int, w: int, g: int, tsub: int, mem_gb: float) -> float:
    """Peak RAM of one compute_init_maps call (the strip model lives in bm_qc, next to the strips)."""
    return bm_qc.init_maps_peak_bytes(mem_gb, math.ceil(T / tsub), h, w, g, tsub) / 1e9


def map_workers(T: int, h: int, w: int, gsigs: list, tsub: int, cfg) -> int:
    """Processes for the gSig maps: one per gSig, within the CPUs and the machine RAM budget.

    n workers may run any n of the gSig jobs at once, so the n largest per-gSig peaks must fit together.
    """
    cap = min(len(gsigs), int(cfg["machine"]["n_cpus"]) - 1)
    if cap <= 1:
        return 1
    mem_gb = mem_budget_gb(cfg)
    peaks = sorted((map_peak_gb(T, h, w, g, tsub, mem_gb) for g in gsigs), reverse=True)
    # strip RAM only: each worker's baseline (imports) and the returned maps fit in the headroom between
    # cnmf_ram_budget_gb and physical RAM
    budget = float(cfg["machine"]["cnmf_ram_budget_gb"])
    n, held = 0, 0.0
    for p in peaks[:cap]:
        if held + p > budget:
            break
        n, held = n + 1, held + p
    return max(1, n)


def _maps_job(args):
    """Pool worker: the maps of one gSig from the shared uint8 memmap (opened read-only here)."""
    u8_path, shape, g, tsub, mem_gb = args
    movie = np.memmap(u8_path, dtype=np.uint8, mode="r", shape=tuple(shape))
    try:
        return g, bm_qc.compute_init_maps(movie, gSig=g, tsub=tsub, mirror=True, max_mem_gb=mem_gb,
                                          verbose=False)
    finally:
        del movie


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


def _collect(results, arrays, g_sug):
    """Store each (g, maps) as it arrives; returns the maps of g_sug."""
    sug_maps = None
    for g, maps in results:
        arrays[f"cn_{g}"] = np.asarray(maps["cn"], np.float32)
        arrays[f"pnr_{g}"] = np.asarray(maps["pnr"], np.float32)
        if "cn_neg" in maps:
            arrays[f"cn_mirror_{g}"] = np.asarray(maps["cn_neg"], np.float32)
            arrays[f"pnr_mirror_{g}"] = np.asarray(maps["pnr_neg"], np.float32)
        if g == g_sug:
            sug_maps = maps
    return sug_maps


def _parallel_maps(u8_path, shape, gsigs, tsub, mem_gb, n_workers):
    """Maps of all gSig in a watched CaImAn pool, in the order of gsigs."""
    saved = os.environ.get("OPENBLAS_NUM_THREADS")
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # workers inherit it: one BLAS thread each, as in cnmf.run
    dview = None
    try:
        dview = pool.start_pool(n_workers)
        jobs = [(str(u8_path), tuple(shape), g, tsub, mem_gb) for g in gsigs]
        return dview.map(_maps_job, jobs)
    finally:
        if dview is not None:
            pool.stop_pool(dview)
        if saved is None:
            os.environ.pop("OPENBLAS_NUM_THREADS", None)
        else:
            os.environ["OPENBLAS_NUM_THREADS"] = saved


def run(paths, rec, cfg) -> StageResult:
    sid = rec.sid
    gsigs = [int(g) for g in cfg["thresholds"]["gsig_range"]]
    g_sug = config.fallback_gsig(gsigs)
    tsub = cfg["cnmf"]["tsub"]
    mem_gb = mem_budget_gb(cfg)
    tmp_dir = paths.temp_dir(cfg["machine"]["temp_dir"], sid)
    T, h, w, _ = video_info(paths.mc_video(sid))
    require_temp_space(tmp_dir, TEMP_BYTES_PER_PX_FRAME * T * h * w, "thresholds")
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
        n_workers = map_workers(n_frames, h, w, gsigs, tsub, cfg)
        if n_workers == 1:
            results = ((g, bm_qc.compute_init_maps(movie, gSig=g, tsub=tsub, mirror=True, max_mem_gb=mem_gb,
                                                   verbose=False)) for g in gsigs)
            sug_maps = _collect(results, arrays, g_sug)
        else:
            sug_maps = _collect(_parallel_maps(u8_path, movie.shape, gsigs, tsub, mem_gb, n_workers),
                                arrays, g_sug)
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
                       metrics={"suggested": suggested, "gsig_range": gsigs, "n_frames": n_frames,
                                "map_workers": n_workers})
