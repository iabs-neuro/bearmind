"""Crop stage: per-session FOV detection, then one box size per mouse."""
from bearmind import config
from bearmind.crop import detect_fov, mouse_box_size, place_box
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


def finalize_mouse(paths, mouse, records, cfg) -> dict:
    """Write crop.box (mouse) and crop.box_pos (sessions); return the box per sid."""
    fovs = {r.sid: r.stages["crop"].metrics.get("fov") for r in records}
    found = [tuple(f) for f in fovs.values() if f]
    if not found:
        return {r.sid: None for r in records}

    frame_shape = (
        min(r.stages["crop"].metrics["frame_shape"][0] for r in records),
        min(r.stages["crop"].metrics["frame_shape"][1] for r in records),
    )
    # Reread: the mouse config may have been written by an earlier batch.
    size = (config.resolve(paths, mouse=mouse).get("crop") or {}).get("box")
    if size is None:
        size = mouse_box_size(found, frame_shape, cfg["crop"]["margin_px"])
        config.update_yaml(paths.mouse_config(mouse), {"crop": {"box": list(size)}})
    size = (int(size[0]), int(size[1]))

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
            r.anomalies.append(Anomaly(
                "fov_not_found", "warning",
                "no field of view found; session is processed uncropped, "
                "its frame size differs from the other sessions of the mouse"))
    return out
