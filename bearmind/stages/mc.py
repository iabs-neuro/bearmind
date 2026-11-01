"""Motion correction stage: shifts from CaImAn on a cropped temp BigTIFF, applied by streaming to FFV1."""
from __future__ import annotations

import gc
import math
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import tifffile

from bearmind import fileio, pool
from bearmind.registry import Anomaly
from bearmind.stages import StageResult, WaitingForInput, require_temp_space
from bearmind.video import CropBox, SessionReader, VideoArray, write_ffv1, write_preview_mp4

CHUNK = 1000
N_CORR_FRAMES = 200
# Temp disk per pixel-frame: _write_tif stores the cropped movie as a uint8 BigTIFF (plus a few hundred
# bytes of tags per frame); CaImAn 1.13.2 motion_correct(save_movie=False) on a file writes nothing to
# CAIMAN_TEMP (motion_correction_piecewise makes a memmap only when it saves the movie).
TEMP_BYTES_PER_PX_FRAME = 1


def _crop(cfg) -> CropBox | None:
    """Crop box of the session; None (explicit box_pos: null) means no FOV was found: uncropped."""
    c = cfg.get("crop") or {}
    if "box_pos" not in c:  # written by crop.finalize_mouse once every session of the mouse is detected
        raise WaitingForInput("crop_finalize")
    pos, box = c.get("box_pos"), c.get("box")
    if pos is None:  # finalize found no field of view (anomaly fov_not_found)
        return None
    if box is None:  # the mouse box was removed: finalize places one again, MC never runs uncropped
        raise WaitingForInput("crop_finalize", "the mouse crop box is missing")
    return CropBox(int(pos[0]), int(pos[1]), int(box[0]), int(box[1]))


def check(paths, rec, cfg) -> None:
    """Raises WaitingForInput while the crop box is not placed (the runner calls it before the stage)."""
    _crop(cfg)


def _save_shifts(path: Path, shifts: np.ndarray) -> None:
    """np.save to <name>.partial.npy, then fileio.replace: a killed stage never leaves a valid-looking file."""
    path = Path(path)
    part = path.with_name(path.stem + ".partial.npy")
    try:
        np.save(part, shifts)
        fileio.replace(part, path)
    finally:
        part.unlink(missing_ok=True)


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


def _mean_template_corr(video: VideoArray, template: np.ndarray, gSig_filt=None) -> float:
    """Mean correlation of sampled corrected frames with CaImAn's template.

    With gSig_filt CaImAn builds the template from high-pass filtered frames, so the frames are filtered
    with the same function first; raw 1p frames are dominated by a smooth background and correlate poorly
    with a filtered template however well they are aligned.
    """
    from caiman.motion_correction import high_pass_filter_space

    idx = np.unique(np.linspace(0, video.shape[0] - 1, min(N_CORR_FRAMES, video.shape[0])).astype(int))
    t = np.asarray(template, dtype=np.float64).ravel()
    cs = []
    for i in idx:
        fr = np.asarray(video[int(i)], dtype=np.float32)
        if gSig_filt is not None:
            fr = high_pass_filter_space(fr, tuple(gSig_filt))
        cs.append(np.corrcoef(fr.astype(np.float64).ravel(), t)[0, 1])
    return float(np.nanmean(cs))


def run(paths, rec, cfg) -> StageResult:
    crop = _crop(cfg)  # WaitingForInput before any heavy work
    h, w = (crop.height, crop.width) if crop is not None else rec.parts[0].shape
    tmp = paths.temp_dir(cfg["machine"]["temp_dir"], rec.sid)
    require_temp_space(tmp, TEMP_BYTES_PER_PX_FRAME * int(rec.n_frames) * int(h) * int(w), "MC")

    import caiman.motion_correction as cmc

    import bm_qc

    mcfg = dict(cfg["mc"])
    if mcfg.get("pw_rigid"):
        raise ValueError("pw_rigid motion correction is not supported by this stage")
    sid, fps = rec.sid, rec.fps
    n_in = int(rec.n_frames)
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

        dview = pool.start_pool(max(1, int(cfg["machine"]["n_cpus"]) - 1))
        mc = cmc.MotionCorrect([str(tif)], dview=dview, **{
            **mcfg, "splits_rig": math.ceil(n_in / 500), "shifts_opencv": True})
        mc.motion_correct(save_movie=False)
        shifts = np.asarray(mc.shifts_rig, dtype=np.float32)
        template = np.asarray(mc.total_template_rig)
        pool.stop_pool(dview)  # only motion_correct uses the pool; free the workers before the writing
        dview = None

        paths.mc_video(sid).parent.mkdir(parents=True, exist_ok=True)
        _save_shifts(paths.mc_shifts(sid), shifts)
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
            "mean_template_corr": _mean_template_corr(mc_vid, template, mcfg.get("gSig_filt")),
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
    except BaseException as e:
        # traceback frames (e.g. of make_dff_video) still reference the tif memmap and would block its deletion
        traceback.clear_frames(e.__traceback__)
        raise
    finally:
        if gen is not None:
            gen.close()  # releases the tif memmap even after a writer error (Windows locks open files)
        if dview is not None:
            pool.stop_pool(dview)
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
