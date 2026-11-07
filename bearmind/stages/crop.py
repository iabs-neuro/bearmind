"""Crop stage: per-session FOV detection, then one box size per mouse."""
import numpy as np

from bearmind import config, fileio
from bearmind.crop import (activity_fov, activity_map, fitted_size, lit_fov, mouse_box_size, mouse_frame_shape,
                           place_box)
from bearmind.registry import Anomaly
from bearmind.stages import StageResult
from bearmind.video import CropBox, SessionReader

WINDOWS = 3  # start, middle and end: cells that are active only in part of the session still count
# The field needs where transients are, not the exact cell size, so the seed maps use one gSig for every series.
ACTIVITY_GSIG = 4
FOV_METHODS = ("activity", "brightness")


def detect(paths, rec, cfg) -> StageResult:
    """The lit area of the mean frame (outer limit) and, with fov_method activity, the field with calcium
    activity inside it. metrics['fov'] is the field finalize_mouse centres the mouse box on: the activity
    field, or the lit area when the method is brightness or the activity field is not trusted."""
    c = cfg["crop"]
    method = c["fov_method"]
    if method not in FOV_METHODS:
        raise ValueError(f"crop.fov_method is {method!r}; expected one of {', '.join(FOV_METHODS)}")
    length = max(1, int(c["sample_frames"]) // WINDOWS)
    frames = SessionReader(paths.root, rec, crop=None).windows(WINDOWS, length)
    mean = frames.mean(axis=0, dtype=np.float64).astype(np.float32)  # no float copy of all frames
    lit = lit_fov(mean, c["min_fov_frac"], c["fov_threshold_frac"])
    activity = field = None
    if lit is not None and method == "activity":
        import bm_qc  # CaImAn: imported in the stage only, the runner imports this module for finalize_mouse

        from bearmind.stages.cnmf import mem_budget_gb

        maps = bm_qc.compute_init_maps(frames, gSig=ACTIVITY_GSIG, tsub=1, mirror=True,
                                       max_mem_gb=mem_budget_gb(cfg), verbose=False)
        activity = activity_map(maps, lit[1])
        field = activity_fov(activity, lit[0], float(c["activity_mass_cut"]), int(c["activity_margin_px"]))
    lit_box = None if lit is None else list(lit[0])
    out = _write_projection(paths, rec.sid, mean, activity)
    return StageResult(outputs={"projection": paths.rel(out)}, metrics={
        "fov": list(field) if field is not None else lit_box,
        "fov_lit": lit_box,
        "fov_activity": None if field is None else list(field),
        "fov_method": "activity" if field is not None else "brightness",
        "frame_shape": [int(frames.shape[1]), int(frames.shape[2])],
    })


def _write_projection(paths, sid, mean, activity):
    """The mean frame and the activity map (when computed) for the crop overview, as float16."""
    out = paths.crop_projection(sid)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial.npz")
    arrays = {"mean": mean.astype(np.float16)}
    if activity is not None:
        arrays["activity"] = activity.astype(np.float16)
    try:
        np.savez_compressed(tmp, **arrays)
        fileio.replace(tmp, out)  # retried while the overview or a viewer holds the old file
    finally:
        tmp.unlink(missing_ok=True)
    return out


def _mouse_box(paths, mouse, records, metrics, found, frame_shape, cfg) -> tuple[int, int]:
    """The mouse box size, written to the mouse config with crop.box_source (auto | manual).

    Computed from the fields of view when the box is missing, or when it is automatic (box_source auto, also
    when absent) and every session was detected again since its position was placed: a changed detection
    parameter re-runs crop for all of them, a session added later does not. A box other than the one the
    positions were placed with was changed by a person: it becomes manual and is kept from then on.
    """
    c = config.resolve(paths, mouse=mouse).get("crop") or {}  # reread: an earlier batch may have written it
    size, source = c.get("box"), c.get("box_source", "auto")
    placed = [m["box"] for m in (metrics[r.sid] for r in records) if m.get("box")]
    redetected = not any("box" in metrics[r.sid] for r in records)
    if size is not None and any(list(b[2:]) != fitted_size(size, frame_shape) for b in placed):
        source = "manual"
    elif size is None or (source == "auto" and redetected):
        size, source = mouse_box_size(found, frame_shape, cfg["crop"]["margin_px"]), "auto"
    size = [int(size[0]), int(size[1])]
    if (c.get("box"), c.get("box_source")) != (size, source):
        config.update_yaml(paths.mouse_config(mouse), {"crop": {"box": size, "box_source": source}})
    return size[0], size[1]


def finalize_mouse(paths, mouse, records, cfg) -> dict:
    """Write crop.box (mouse) and crop.box_pos (sessions); return the box per sid.

    Every session's box is placed again on each call: centred on its field of view, inside the frame
    (positions computed for another box size could pass the frame edge). Records without crop metrics
    (crop stage failed or older record) are skipped: no box_pos, no say in the box size, not in the result.
    """
    metrics = {r.sid: getattr(r.stages.get("crop"), "metrics", None) or {} for r in records}
    records = [r for r in records if metrics[r.sid].get("frame_shape")]
    fovs = {r.sid: metrics[r.sid].get("fov") for r in records}
    found = [tuple(f) for f in fovs.values() if f]
    if found:
        frame_shape = mouse_frame_shape(metrics[r.sid] for r in records)
        size = _mouse_box(paths, mouse, records, metrics, found, frame_shape, cfg)

    out = {}
    for r in records:
        fov = fovs[r.sid]
        if fov:
            box = place_box(fov, size, frame_shape)
            out[r.sid] = box
            config.update_yaml(paths.session_config(r.sid), {"crop": {"box_pos": [box.top, box.left]}})
        else:
            out[r.sid] = None
            config.update_yaml(paths.session_config(r.sid), {"crop": {"box_pos": None}})
            if not any(a.code == "fov_not_found" for a in r.anomalies):  # finalize runs again for late sessions
                r.anomalies.append(Anomaly(
                    "fov_not_found", "warning",
                    "no field of view found; session is processed uncropped, "
                    "its frame size differs from the other sessions of the mouse"))
    return out
