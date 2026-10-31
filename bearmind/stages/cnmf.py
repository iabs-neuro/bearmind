"""CNMF-E stage: FFV1 MC video -> CaImAn C-order memmap in the temp folder -> CNMF-E -> estimates + QC previews."""
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

import bm_qc
from bearmind.registry import Anomaly
from bearmind.stages import StageResult, WaitingForInput
from bearmind.video import VideoArray

# uint8 columns decoded per strip of the memmap writer. Every strip re-decodes the whole FFV1 file,
# so the budget sets the number of decode passes: ceil(w / floor(STRIP_GB / (T*h))).
STRIP_GB = 16
# save_memmap in CaImAn 1.13.2 adds exactly np.float32(add_to_movie), default 0.0001, in one addition.
ADD_TO_MOVIE = np.float32(1e-4)
IMAX_PNR_CLIP = 70  # display image for manual inspection: PNR clipped as in the notebook DoCNMF
_ENV = ("CAIMAN_TEMP", "OPENBLAS_NUM_THREADS")


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


def write_memmap(video, folder, base: str = "Yr", strip_gb: float | None = None) -> Path:
    """C-order (pixels x T) float32 memmap, byte-identical to cm.save_memmap([tif], order='C').

    CaImAn numbers pixels in F order (p = i + h*j), so a strip of frame columns is a contiguous run of rows:
    the file is written sequentially, one strip at a time, and the movie is never held whole.
    """
    T, h, w = video.shape
    strip_gb = STRIP_GB if strip_gb is None else strip_gb
    cols = max(1, int(round(strip_gb * 1e9)) // (T * h))
    path = Path(folder) / memmap_name(base, (h, w), T)
    with open(path, "wb") as f:
        for j0 in range(0, w, cols):
            blk = video[:, :, j0:j0 + cols]  # (T, h, cols) uint8, one decode pass
            for j in range(blk.shape[2]):
                col = np.ascontiguousarray(blk[:, :, j].T)  # (h, T): rows i + h*(j0+j)
                (col.astype(np.float32) + ADD_TO_MOVIE).tofile(f)
            del blk
    return path


def _imax(paths, sid, gSig, tsub, dims) -> np.ndarray:
    """uint8 PNR image for the inspection GUI, from the threshold maps when they have this gSig."""
    pnr = None
    maps = paths.threshold_maps(sid)
    if maps.exists():
        with np.load(maps) as d:
            key = f"pnr_{gSig}"
            if key in d.files and d[key].shape == tuple(dims):
                pnr = np.asarray(d[key], np.float32)
    if pnr is None:
        # streamed in row strips; correlation_pnr(images[::5]) would hold ~T/5 float32 frames several times over
        pnr = bm_qc.compute_init_maps(VideoArray(paths.mc_video(sid)), gSig=gSig, tsub=tsub, mirror=False,
                                      max_mem_gb=16, verbose=False)["pnr"]
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


def _stop(dview) -> None:
    import caiman as cm
    try:
        cm.stop_server(dview=dview)
    except Exception:
        pass
    try:
        dview.terminate()
    except Exception:
        pass


def run(paths, rec, cfg) -> StageResult:
    params = cnmf_params(cfg, rec.fps)  # WaitingForInput before any heavy work

    import caiman as cm
    from caiman.cluster import setup_cluster
    from caiman.source_extraction.cnmf.cnmf import CNMF
    from caiman.source_extraction.cnmf.params import CNMFParams

    sid = rec.sid
    mc_path = paths.mc_video(sid)
    video = VideoArray(mc_path)
    T = int(video.shape[0])
    n_workers = patch_workers(T, params["tsub"], cfg)
    g = params["gSig"][0]

    tmp = Path(cfg["machine"]["temp_dir"]) / sid
    tmp.mkdir(parents=True, exist_ok=True)  # CaImAn ignores a CAIMAN_TEMP folder that does not exist
    saved = {k: os.environ.get(k) for k in _ENV}
    os.environ["CAIMAN_TEMP"] = str(tmp)
    # pool workers inherit the env when spawned: one OpenBLAS thread each keeps idle commit at 0.67 GB, not 4.65
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    dview = cnm = est = Yr = images = None
    mmap = None
    result = None
    try:
        mmap = write_memmap(video, tmp)
        Yr, dims, T = cm.load_memmap(str(mmap))
        images = Yr.T.reshape((T,) + dims, order="F")

        _, dview, _ = setup_cluster(backend="multiprocessing", n_processes=n_workers, single_thread=False)
        opts = CNMFParams(params_dict=params)
        cnm = CNMF(n_workers, dview=dview, params=opts)
        cnm.fit(images)
        est = cnm.estimates
        est.evaluate_components(images, opts, dview=dview)
        _stop(dview)  # QC and imax do not use the pool; free the workers early
        dview = None

        est.imax = _imax(paths, sid, g, params["tsub"], dims)
        est.cnmf_dict = params
        est.mc_video = paths.rel(mc_path)  # relative: the series folder may move
        est.patch_workers = n_workers  # the pool size can change the result, so it is recorded

        out = paths.estimates(sid)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".partial")
        try:
            with open(part, "wb") as f:
                pickle.dump(est, f)
            os.replace(part, out)
        finally:
            part.unlink(missing_ok=True)

        qc_base = str(tmp / sid)
        bm_qc.cnmf_qc(est, movie=images, out_base=qc_base, fps=rec.fps, tmp_dir=str(tmp), verbose=False)
        _move(qc_base + "_qc.mp4", paths.preview(sid, "cnmf_qc"))
        _move(qc_base + "_qc_residual_maps.png", paths.residual_png(sid))
        with open(qc_base + "_qc.json", encoding="utf-8") as f:
            missed = json.load(f).get("residual_excess_outside")

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
            "preview_cnmf_qc": paths.rel(paths.preview(sid, "cnmf_qc")),
            "residual_png": paths.rel(paths.residual_png(sid)),
        }
        result = StageResult(outputs=outputs, metrics=metrics)
    except BaseException as e:
        # frames of fit / cnmf_qc in the traceback still reference the memmap and would block its deletion
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        if dview is not None:
            _stop(dview)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        # Windows cannot delete a mapped file: drop every view of the memmap first
        cnm = est = Yr = images = None
        gc.collect()
        if mmap is not None:
            try:
                mmap.unlink(missing_ok=True)
            except OSError:
                pass
        for _ in range(5):
            shutil.rmtree(tmp, ignore_errors=True)
            if not tmp.exists():
                break
            time.sleep(0.5)
    if tmp.exists():
        result.anomalies.append(Anomaly("temp_not_removed", "warning", str(tmp)))
    return result
