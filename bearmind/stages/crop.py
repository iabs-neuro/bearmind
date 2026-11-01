"""Crop stage: per-session FOV detection, then one box size per mouse."""
from bearmind import config
from bearmind.crop import detect_fov, fitted_size, mouse_box_size, mouse_frame_shape, place_box
from bearmind.registry import Anomaly
from bearmind.stages import StageResult
from bearmind.video import CropBox, SessionReader


def detect(paths, rec, cfg) -> StageResult:
    c = cfg["crop"]
    frames = SessionReader(paths.root, rec, crop=None).sample(c["n_sample_frames"])
    fov = detect_fov(frames, c["min_fov_frac"], c["fov_threshold_frac"])
    return StageResult(metrics={
        "fov": list(fov) if fov is not None else None,
        "frame_shape": [int(frames.shape[1]), int(frames.shape[2])],
    })


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
