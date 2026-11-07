"""Preview stages: the MC videos and the CNMF-E QC video, made from the stored MC video and estimates.

They run as side stages: when the checkbox is set, right after MC / CNMF-E, before the next main stage (that time is
the price of the checkbox); no main stage uses their result, so a disabled or failed preview holds nothing. bm_qc
(CaImAn, cv2, matplotlib: about 7 s) is imported inside the functions: the runner imports preview_params on every
stage-parameter check.
"""
from __future__ import annotations

import gc
import logging
import pickle
import shutil
import time
import traceback
from pathlib import Path

from bearmind import fileio
from bearmind.registry import Anomaly
from bearmind.stages import StageResult, require_temp_space
from bearmind.video import (CropBox, SessionReader, chunks_to_memmap, decode_to_memmap, video_info,
                            write_preview_mp4)

log = logging.getLogger(__name__)

CHUNK = 1000
# One uint8 copy at a time: the cropped raw movie, then the MC movie.
TEMP_BYTES_PER_PX_FRAME = 1
_CONFIG_HINT = "_bearmind/config/series.yaml (the default is in bearmind/defaults.yaml)"


def preview_params(cfg) -> dict:
    """The preview settings that matter: {"enabled": False} when off, as the settings of a disabled preview
    must not change a stage hash; else the validated integers tbin, ssub, crf, speed."""
    pv = cfg.get("preview") or {}
    if not pv.get("enabled"):
        return {"enabled": False}
    out = {"enabled": True}
    for key in ("tbin", "ssub", "crf", "speed"):
        v = pv.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or (key in ("tbin", "ssub") and v < 1):
            need = "an integer >= 1" if key in ("tbin", "ssub") else "an integer"
            raise ValueError(f"preview.{key} must be {need}, got {v!r}; fix it in {_CONFIG_HINT}")
        out[key] = int(v)
    return out


def _mc_crop(rec) -> CropBox | None:
    """The box MC really ran with (its recorded metric), not the config: the config may have changed since."""
    st = (rec.stages or {}).get("mc")
    metrics = getattr(st, "metrics", None) or {}
    if "crop" not in metrics:
        raise ValueError(f"{rec.sid}: the MC record has no crop metric; run MC again")
    c = metrics["crop"]
    return None if c is None else CropBox(*(int(x) for x in c))


def _rmtree(tmp: Path) -> None:
    gc.collect()  # Windows cannot delete a mapped file: the views must be gone first
    for _ in range(5):
        shutil.rmtree(tmp, ignore_errors=True)
        if not tmp.exists():
            break
        time.sleep(0.5)


def _finish(result: StageResult, tmp: Path) -> StageResult:
    _rmtree(tmp)
    if tmp.exists():
        result.anomalies.append(Anomaly("temp_not_removed", "warning", str(tmp)))
    return result


def _bytes(paths, outputs) -> int:
    return int(sum((paths.root / rel).stat().st_size for rel in outputs.values()))


def _remove_written(files) -> None:
    """The preview files a failed run wrote (or began to write), so it leaves no half-new set; files it had not
    reached yet stay as they were."""
    for f in files:
        try:
            f.unlink(missing_ok=True)
        except OSError as e:  # open in a video player: the stage error matters more than this file
            log.warning("preview file not removed after the failure: %s: %s", f, e)


def mc_previews(paths, rec, cfg) -> StageResult:
    """raw, raw dF/F, MC and MC dF/F videos of a session; the dF/F pair is of the cropped movie."""
    prm = preview_params(cfg)
    if not prm["enabled"]:
        raise ValueError("mc_previews called with preview.enabled false")
    tbin, ssub = prm["tbin"], prm["ssub"]
    sid, fps = rec.sid, rec.fps
    crop = _mc_crop(rec)
    T, h, w, _ = video_info(paths.mc_video(sid))
    tmp = paths.temp_dir(cfg["machine"]["temp_dir"], sid)
    require_temp_space(tmp, TEMP_BYTES_PER_PX_FRAME * T * h * w, "MC previews")

    import bm_qc

    kw = dict(crf=prm["crf"], speed=prm["speed"], tbin=tbin, ssub=ssub)
    names = {"raw": "preview_raw", "raw_dff": "preview_raw_dff", "mc": "preview_mc", "mc_dff": "preview_mc_dff"}
    arr = None
    result = None
    written = []  # Preview/ files this run writes, removed if it fails

    def out(kind):
        f = paths.preview(sid, kind)
        written.append(f)
        return f

    tmp.mkdir(parents=True, exist_ok=True)
    try:
        paths.preview(sid, "raw").parent.mkdir(parents=True, exist_ok=True)
        write_preview_mp4(SessionReader(paths.root, rec, crop=None).iter_chunks(CHUNK), out("raw"), fps, **kw)

        def dff(kind):
            made = tmp / f"{sid}_{kind}.mp4"
            bm_qc.make_dff_video(arr, out_path=str(made), fps=fps, tbin=tbin, ssub=ssub, verbose=False)
            fileio.move(made, out(kind))

        arr = chunks_to_memmap(SessionReader(paths.root, rec, crop=crop).iter_chunks(CHUNK),
                               tmp / "raw_crop.dat", (T, h, w))
        dff("raw_dff")
        arr = None
        gc.collect()  # Windows cannot delete a mapped file: views left in reference cycles must go first
        (tmp / "raw_crop.dat").unlink(missing_ok=True)

        arr = decode_to_memmap(paths.mc_video(sid), tmp / "mc_u8.dat")
        write_preview_mp4((arr[a:a + CHUNK] for a in range(0, T, CHUNK)), out("mc"), fps, **kw)
        dff("mc_dff")
        arr = None

        outputs = {names[k]: paths.rel(paths.preview(sid, k)) for k in names}
        result = StageResult(outputs=outputs,
                             metrics={"tbin": tbin, "ssub": ssub, "bytes": _bytes(paths, outputs)})
    except BaseException as e:
        # traceback frames (e.g. of make_dff_video) still reference the memmaps and would block their deletion
        traceback.clear_frames(e.__traceback__)
        arr = None
        _rmtree(tmp)
        _remove_written(written)
        raise
    return _finish(result, tmp)


def cnmf_preview(paths, rec, cfg) -> StageResult:
    """The CNMF-E QC video (raw | raw - background | A*C | residual) from the stored estimates and MC video."""
    prm = preview_params(cfg)
    if not prm["enabled"]:
        raise ValueError("cnmf_preview called with preview.enabled false")
    tbin, ssub = prm["tbin"], prm["ssub"]
    sid = rec.sid
    T, h, w, _ = video_info(paths.mc_video(sid))
    tmp = paths.temp_dir(cfg["machine"]["temp_dir"], sid)
    require_temp_space(tmp, TEMP_BYTES_PER_PX_FRAME * T * h * w, "CNMF-E preview")

    import bm_qc

    u8 = est = None
    result = None
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        with open(paths.estimates(sid), "rb") as f:
            est = pickle.load(f)
        # the movie here is uint8; CNMF-E fitted uint8 + 1e-4, which cannot be seen on the raw panel
        u8 = decode_to_memmap(paths.mc_video(sid), tmp / "mc_u8.dat")
        base = str(tmp / sid)
        bm_qc.cnmf_qc(est, movie=u8, out_base=base, fps=rec.fps, tmp_dir=str(tmp), make_video=True,
                      residual_maps=False, video_tbin=tbin, video_ssub=ssub, verbose=False)
        # its only Preview/ file, moved in last (one replace, or a .partial copy across drives): a failure
        # leaves no half-written video there
        fileio.move(base + "_qc.mp4", paths.preview(sid, "cnmf_qc"))
        outputs = {"preview_cnmf_qc": paths.rel(paths.preview(sid, "cnmf_qc"))}
        result = StageResult(outputs=outputs,
                             metrics={"tbin": tbin, "ssub": ssub, "bytes": _bytes(paths, outputs)})
    except BaseException as e:
        traceback.clear_frames(e.__traceback__)
        u8 = est = None
        _rmtree(tmp)
        raise
    u8 = est = None
    return _finish(result, tmp)
