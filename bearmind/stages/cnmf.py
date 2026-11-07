"""CNMF-E stage: FFV1 MC video -> CaImAn C-order memmap in the temp folder -> CNMF-E -> estimates + residual maps.

The QC video is made by stages.previews from the stored estimates, not here.

bm_qc (CaImAn, cv2, matplotlib: about 7 s) is imported inside the functions that use it: the runner and
`status` import this module for cnmf_params on every stage-parameter check.
"""
from __future__ import annotations

import errno
import gc
import json
import math
import os
import pickle
import re
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from bearmind import pool
from bearmind.registry import Anomaly
from bearmind.stages import StageResult, WaitingForInput, require_temp_space
from bearmind.video import decode_to_memmap, video_info

# Frames per tile of the memmap writer's transpose: (TILE, h, cols) uint8 tiles stay cache-resident,
# measured 2-5x faster than one strided copy per column.
TILE = 64
# save_memmap in CaImAn 1.13.2 adds exactly np.float32(add_to_movie), default 0.0001, in one addition.
ADD_TO_MOVIE = np.float32(1e-4)
IMAX_PNR_CLIP = 70  # display image for manual inspection: PNR clipped as in the notebook DoCNMF
# Temp disk per pixel-frame at the peak: the uint8 copy (1, decode_to_memmap) and the float32 C-order memmap
# (4, write_memmap) exist together until _imax; after that the memmap (4) and cnmf_qc's float16 residual at
# its tsub 2 (2 bytes per 2 frames: 1) do. 567 x 520 x 108 000 frames (an hour at 30 fps): 159 GB.
TEMP_BYTES_PER_PX_FRAME = 5
_ENV = ("CAIMAN_TEMP", "OPENBLAS_NUM_THREADS")


def mem_budget_gb(cfg) -> float:
    """Working-memory budget of the streamed steps (memmap strips, seed maps): 10% of RAM, within [2, 32] GB."""
    return float(np.clip(0.1 * float(cfg["machine"]["ram_gb"]), 2, 32))


def cnmf_params(cfg, fps) -> dict:
    """CaImAn params dict: notebook base dict + session thresholds; fr is the session FPS, never divided by tsub."""
    base = cfg["cnmf"]
    if any(base.get(k) is None for k in ("gSig", "min_corr", "min_pnr")):
        raise WaitingForInput("thresholds")
    p = dict(base)
    g = int(base["gSig"])
    p.update(gSig=(g, g), gSiz=(4 * g + 1, 4 * g + 1), min_corr=float(base["min_corr"]),
             min_pnr=float(base["min_pnr"]), fr=float(fps))
    return p


def check(paths, rec, cfg) -> None:
    """Raises WaitingForInput while the thresholds are not set (the runner calls it before the stage)."""
    cnmf_params(cfg, rec.fps)


def patch_workers(T, tsub, cfg) -> int:
    """Pool size for the CNMF-E patch stage: workers whose measured peak fits the RAM budget."""
    table = cfg["memory"]["patch_gb_per_mframe"]
    per_mframe = table.get(tsub, table.get(str(tsub)))
    if per_mframe is None:
        raise ValueError(f"memory.patch_gb_per_mframe has no entry for tsub={tsub}")
    n = math.floor(cfg["machine"]["cnmf_ram_budget_gb"] / (per_mframe * T / 1e6))
    return int(max(1, min(n, int(cfg["machine"]["n_cpus"]) - 1)))


def memmap_name(base: str, dims, T: int) -> str:
    """File name cm.save_memmap(order='C') gives in 1.13.2 (paths.generate_fname_tot + '_frames_<T>.mmap')."""
    d1, d2 = dims
    return re.sub(r"(_)+", "_", f"{base}_d1_{d1}_d2_{d2}_d3_1_order_C") + f"_frames_{T}.mmap"


def write_memmap(movie, folder, mem_gb: float, base: str = "Yr") -> Path:
    """C-order (pixels x T) float32 memmap, byte-identical to cm.save_memmap([tif], order='C').

    `movie` is a (T, h, w) uint8 array, normally the decoded memmap. CaImAn numbers pixels in F order
    (p = i + h*j), so a strip of frame columns is a contiguous run of rows: the file is written sequentially,
    one float32 strip of at most `mem_gb` at a time, and the movie is never held whole.
    """
    T, h, w = movie.shape
    cols = max(1, int(round(mem_gb * 1e9)) // (4 * T * h))
    path = Path(folder) / memmap_name(base, (h, w), T)
    with open(path, "wb") as f:
        for j0 in range(0, w, cols):
            c = min(cols, w - j0)
            strip = np.empty((c, h, T), np.float32)  # rows i + h*(j0+j), j < c
            for t0 in range(0, T, TILE):
                # float32(uint8) + float32(1e-4): the same single addition as save_memmap
                np.add(movie[t0:t0 + TILE, :, j0:j0 + c].transpose(2, 1, 0), ADD_TO_MOVIE,
                       out=strip[:, :, t0:t0 + TILE])
            strip.tofile(f)
            del strip
    return path


def _imax(paths, sid, gSig, tsub, dims, movie, mem_gb) -> np.ndarray:
    """uint8 PNR image for the inspection GUI, from the threshold maps when they have this gSig.

    Otherwise the PNR is computed on `movie` (the decoded uint8 memmap) in row strips.
    """
    import bm_qc

    pnr = None
    maps = paths.threshold_maps(sid)
    if maps.exists():
        with np.load(maps) as d:
            key = f"pnr_{gSig}"
            if key in d.files and d[key].shape == tuple(dims):
                pnr = np.asarray(d[key], np.float32)
    if pnr is None:
        # correlation_pnr(images[::5]) would hold ~T/5 float32 frames several times over
        pnr = bm_qc.compute_init_maps(movie, gSig=gSig, tsub=tsub, mirror=False, max_mem_gb=mem_gb,
                                      verbose=False)["pnr"]
    pnr = np.clip(np.nan_to_num(np.asarray(pnr, np.float32), nan=0.0), 0, IMAX_PNR_CLIP)
    mx = float(pnr.max())
    if mx <= 0:
        return np.zeros(pnr.shape, np.uint8)
    return (pnr * 255 / mx).astype(np.uint8)


def _move(src, dst) -> None:
    """os.replace; across drives (temp folder on C:, series elsewhere) copy to <dst>.partial, then replace."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(src, dst)
        return
    except OSError as e:
        if e.errno != errno.EXDEV:
            raise
    part = dst.with_name(dst.name + ".partial")
    try:
        shutil.copyfile(src, part)
        os.replace(part, dst)
    finally:
        part.unlink(missing_ok=True)
    src.unlink()


def _unlink(path) -> None:
    """Delete a temp file; a failure (file still mapped) is left to the rmtree retries in run's finally."""
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def run(paths, rec, cfg) -> StageResult:
    params = cnmf_params(cfg, rec.fps)  # WaitingForInput before any heavy work
    sid = rec.sid
    mc_path = paths.mc_video(sid)
    T, h, w, _ = video_info(mc_path)
    tmp = paths.temp_dir(cfg["machine"]["temp_dir"], sid)
    require_temp_space(tmp, TEMP_BYTES_PER_PX_FRAME * T * h * w, "CNMF-E")

    import caiman as cm
    from caiman.source_extraction.cnmf.cnmf import CNMF
    from caiman.source_extraction.cnmf.params import CNMFParams

    import bm_qc

    n_workers = patch_workers(T, params["tsub"], cfg)
    mem_gb = mem_budget_gb(cfg)
    g = params["gSig"][0]

    tmp.mkdir(parents=True, exist_ok=True)  # CaImAn ignores a CAIMAN_TEMP folder that does not exist
    saved = {k: os.environ.get(k) for k in _ENV}
    os.environ["CAIMAN_TEMP"] = str(tmp)
    # pool workers inherit the env when spawned: one OpenBLAS thread each keeps idle commit at 0.67 GB, not 4.65
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    dview = cnm = est = Yr = images = u8 = None
    mmap = None
    u8_path = tmp / "mc_u8.dat"
    result = None
    try:
        # one decode of the FFV1; the strip writer and the imax fallback read this uint8 copy
        u8 = decode_to_memmap(mc_path, u8_path)
        mmap = write_memmap(u8, tmp, mem_gb)
        imax = _imax(paths, sid, g, params["tsub"], u8.shape[1:], u8, mem_gb)
        u8 = None  # last reference: unmaps the file, so it can go before the fit (T*h*w bytes of temp disk)
        _unlink(u8_path)

        Yr, dims, T = cm.load_memmap(str(mmap))
        images = Yr.T.reshape((T,) + dims, order="F")

        dview = pool.start_pool(n_workers)
        opts = CNMFParams(params_dict=params)
        cnm = CNMF(n_workers, dview=dview, params=opts)
        cnm.fit(images)
        est = cnm.estimates
        est.evaluate_components(images, opts, dview=dview)
        pool.stop_pool(dview)  # QC does not use the pool; free the workers early
        dview = None

        est.imax = imax
        est.cnmf_dict = params
        est.mc_video = paths.rel(mc_path)  # relative: the series folder may move
        est.patch_workers = n_workers  # the pool size can change the result, so it is recorded

        qc_base = str(tmp / sid)
        bm_qc.cnmf_qc(est, movie=images, out_base=qc_base, fps=rec.fps, tmp_dir=str(tmp), make_video=False,
                      verbose=False)
        _move(qc_base + "_qc_residual_maps.png", paths.residual_png(sid))
        with open(qc_base + "_qc.json", encoding="utf-8") as f:
            missed = json.load(f).get("residual_excess_outside")

        # written last: a failed QC must not leave a valid-looking estimates file behind
        out = paths.estimates(sid)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".partial")
        try:
            with open(part, "wb") as f:
                pickle.dump(est, f)
            os.replace(part, out)
        finally:
            part.unlink(missing_ok=True)

        n_comp = int(est.A.shape[1])
        idx = getattr(est, "idx_components", None)
        acc = np.arange(n_comp) if idx is None else np.asarray(idx, dtype=int)
        snr = getattr(est, "SNR_comp", None)
        median_snr = float(np.median(np.asarray(snr)[acc])) if snr is not None and len(acc) else None
        metrics = {
            "n_components": n_comp,
            "n_accepted": int(len(acc)),
            "missed_cells": missed,
            "median_snr": median_snr,
            "gSig": int(g),
            "min_corr": params["min_corr"],
            "min_pnr": params["min_pnr"],
            "patch_workers": n_workers,
        }
        outputs = {
            "estimates": paths.rel(out),
            "residual_png": paths.rel(paths.residual_png(sid)),
        }
        result = StageResult(outputs=outputs, metrics=metrics)
    except BaseException as e:
        # frames of fit / cnmf_qc in the traceback still reference the memmap and would block its deletion
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        if dview is not None:
            pool.stop_pool(dview)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Windows cannot delete a mapped file: drop every view of the memmaps first
        cnm = est = Yr = images = u8 = None
        gc.collect()
        for f in (mmap, u8_path):
            if f is not None:
                _unlink(f)
        for _ in range(5):
            shutil.rmtree(tmp, ignore_errors=True)
            if not tmp.exists():
                break
            time.sleep(0.5)
    if tmp.exists():
        result.anomalies.append(Anomaly("temp_not_removed", "warning", str(tmp)))
    return result
