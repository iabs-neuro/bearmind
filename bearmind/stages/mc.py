"""Motion correction stage: shifts from CaImAn on a cropped temp BigTIFF, applied by streaming to FFV1."""
from __future__ import annotations

import gc
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np
import tifffile

from bearmind.registry import Anomaly
from bearmind.stages import StageResult
from bearmind.video import CropBox, SessionReader, VideoArray, write_ffv1, write_preview_mp4

CHUNK = 1000
N_CORR_FRAMES = 200


def _crop(cfg) -> CropBox | None:
    c = cfg.get("crop") or {}
    pos, box = c.get("box_pos"), c.get("box")
    if pos is None or box is None:
        return None
    return CropBox(int(pos[0]), int(pos[1]), int(box[0]), int(box[1]))


def _write_tif(reader: SessionReader, path: Path) -> None:
    with tifffile.TiffWriter(str(path), bigtiff=True) as tw:
        for chunk in reader.iter_chunks(CHUNK):
            for fr in chunk:  # per frame: chunk-wise writes would give a 4-D (n_chunks, chunk, h, w) series
                tw.write(fr, contiguous=True)


def _corrected_chunks(tif: Path, shifts: np.ndarray, border_nan: str):
    """Apply per-frame rigid shifts chunk by chunk; same call and sign as MotionCorrect.apply_shifts_movie."""
    from caiman.motion_correction import apply_shift_iteration

    mm = tifffile.memmap(str(tif), mode="r")
    block = out = None
    try:
        for a in range(0, mm.shape[0], CHUNK):
            block = np.asarray(mm[a:a + CHUNK], dtype=np.float32)
            out = np.empty(block.shape, np.uint8)
            for i, (img, sh) in enumerate(zip(block, shifts[a:a + CHUNK])):
                x = apply_shift_iteration(img, tuple(sh), border_nan=border_nan)
                out[i] = np.clip(np.rint(x), 0, 255).astype(np.uint8)
            yield out
    finally:
        del block, out
        mm_base = getattr(mm, "_mmap", None)
        del mm
        if mm_base is not None:
            mm_base.close()


def _preview_chunks(video: VideoArray):
    for a in range(0, video.shape[0], CHUNK):
        yield video[a:a + CHUNK]


def _mean_template_corr(video: VideoArray, template: np.ndarray) -> float:
    idx = np.unique(np.linspace(0, video.shape[0] - 1, min(N_CORR_FRAMES, video.shape[0])).astype(int))
    t = np.asarray(template, dtype=np.float64).ravel()
    cs = [np.corrcoef(video[int(i)].astype(np.float64).ravel(), t)[0, 1] for i in idx]
    return float(np.nanmean(cs))


def run(paths, rec, cfg) -> StageResult:
    import caiman as cm
    import caiman.motion_correction as cmc
    from caiman.cluster import setup_cluster

    import bm_qc

    mcfg = dict(cfg["mc"])
    if mcfg.get("pw_rigid"):
        raise ValueError("pw_rigid motion correction is not supported by this stage")
    sid, fps = rec.sid, rec.fps
    crop = _crop(cfg)
    n_in = int(rec.n_frames)
    tmp = Path(cfg["machine"]["temp_dir"]) / sid
    tmp.mkdir(parents=True, exist_ok=True)  # CaImAn ignores a missing CAIMAN_TEMP folder
    saved = {k: os.environ.get(k) for k in ("CAIMAN_TEMP", "OPENBLAS_NUM_THREADS")}
    os.environ["CAIMAN_TEMP"] = str(tmp)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"  # workers inherit env at start; 4 threads each cost 7x RAM
    dview = None
    gen = None
    result = None
    pv = cfg["preview"]
    prev_kw = dict(crf=pv["crf"], speed=pv["speed"], tbin=pv["tbin"])
    try:
        tif = tmp / f"{sid}_crop.tif"
        _write_tif(SessionReader(paths.root, rec, crop=crop), tif)

        _, dview, _ = setup_cluster(backend="multiprocessing",
                                    n_processes=max(1, int(cfg["machine"]["n_cpus"]) - 1),
                                    single_thread=False)
        mc = cmc.MotionCorrect([str(tif)], dview=dview, **{
            **mcfg, "splits_rig": math.ceil(n_in / 500), "shifts_opencv": True})
        mc.motion_correct(save_movie=False)
        shifts = np.asarray(mc.shifts_rig, dtype=np.float32)
        template = np.asarray(mc.total_template_rig)

        paths.mc_video(sid).parent.mkdir(parents=True, exist_ok=True)
        np.save(paths.mc_shifts(sid), shifts)
        gen = _corrected_chunks(tif, shifts, mcfg["border_nan"])
        n = write_ffv1(gen, paths.mc_video(sid), fps)
        gen.close()
        mc_vid = VideoArray(paths.mc_video(sid))

        paths.preview(sid, "raw").parent.mkdir(parents=True, exist_ok=True)
        write_preview_mp4(SessionReader(paths.root, rec, crop=None).iter_chunks(CHUNK),
                          paths.preview(sid, "raw"), fps, **prev_kw)
        write_preview_mp4(_preview_chunks(mc_vid), paths.preview(sid, "mc"), fps, **prev_kw)
        # raw dF/F is of the cropped movie before correction (the temp tif), not of the uncropped raw
        bm_qc.make_dff_video(str(tif), out_path=str(paths.preview(sid, "raw_dff")), fps=fps, verbose=False)
        bm_qc.make_dff_video(mc_vid, out_path=str(paths.preview(sid, "mc_dff")), fps=fps, verbose=False)

        metrics = {
            "max_shift_px": float(np.max(np.abs(shifts))),
            "mean_template_corr": _mean_template_corr(mc_vid, template),
            "n_frames": int(n),
            "crop": None if crop is None else [crop.top, crop.left, crop.height, crop.width],
        }
        outputs = {
            "mc_video": paths.rel(paths.mc_video(sid)),
            "mc_shifts": paths.rel(paths.mc_shifts(sid)),
            "preview_raw": paths.rel(paths.preview(sid, "raw")),
            "preview_raw_dff": paths.rel(paths.preview(sid, "raw_dff")),
            "preview_mc": paths.rel(paths.preview(sid, "mc")),
            "preview_mc_dff": paths.rel(paths.preview(sid, "mc_dff")),
        }
        result = StageResult(outputs=outputs, metrics=metrics)
    finally:
        if gen is not None:
            gen.close()  # releases the tif memmap even after a writer error (Windows locks open files)
        if dview is not None:
            try:
                cm.stop_server(dview=dview)
            except Exception:
                pass
            try:
                dview.terminate()
            except Exception:
                pass
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        mc = mm = None
        gc.collect()
        for _ in range(5):
            shutil.rmtree(tmp, ignore_errors=True)
            if not tmp.exists():
                break
            time.sleep(0.5)
    if tmp.exists():
        result.anomalies.append(Anomaly("temp_not_removed", "warning", str(tmp)))
    return result
