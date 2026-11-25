"""Prepare review images and candidate wave and edge events without changing component decisions."""
from collections import defaultdict
import html
import json
from pathlib import Path
import pickle

import numpy as np
from scipy import ndimage, sparse
from scipy.spatial import cKDTree

from bearmind import bad_frames, crop, fileio, registry, wave
from bearmind.paths import SeriesPaths
from bearmind.stages import gate
from bearmind.timestamps import session_time_s


def accepted(est):
    selected = getattr(est, "idx_components", None)
    return np.arange(est.A.shape[1]) if selected is None else np.asarray(selected, dtype=int)


def _spatial(est, ids):
    spatial = sparse.csc_matrix(est.A[:, ids])
    total = np.asarray(spatial.sum(axis=0)).ravel()
    height, width = est.imax.shape
    y, x = np.indices((height, width))
    centres = np.column_stack((spatial.T @ y.ravel(order="F"), spatial.T @ x.ravel(order="F")))
    centres = np.divide(centres, total[:, None], out=np.full(centres.shape, np.nan), where=total[:, None] > 0)
    return spatial, centres


def _support(paths, rec, shape):
    with np.load(paths.crop_projection(rec.sid), allow_pickle=False) as projection:
        mean = projection["mean"]
        activity = projection["activity"] if "activity" in projection.files else None
    found = crop.lit_fov(mean)
    if found is None:
        return None, None
    lit = found[1]
    activity_mask = None
    stage = rec.stages.get("crop")
    mass = stage.metrics.get("activity_pos_neg") if stage is not None else None
    if activity is not None and mass is not None and mass >= crop.MIN_POS_NEG:
        labels, count = ndimage.label(lit & (activity > 0))
        if count:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            largest = labels == sizes.argmax()
            if largest.sum() >= .25 * lit.sum():
                activity_mask = ndimage.binary_fill_holes(largest)
    mc = rec.stages.get("mc")
    box = mc.metrics.get("crop") if mc is not None else None
    if box is not None:
        top, left, height, width = map(int, box)
        sl = np.s_[top:top + height, left:left + width]
        lit = lit[sl]
        if activity_mask is not None:
            activity_mask = activity_mask[sl]
    return (lit, activity_mask) if lit.shape == shape else (None, None)


def edge_candidates(spatial, ids, lit, activity=None, outside_fraction=.35):
    """Flag components with substantial spatial energy outside the supported field."""
    if lit is None:
        return None
    energy = spatial.astype(np.float64).power(2)
    total = np.asarray(energy.sum(axis=0)).ravel()
    outside_lit = np.asarray(energy[~lit.ravel(order="F")].sum(axis=0)).ravel()
    outside_activity = (np.asarray(energy[~activity.ravel(order="F")].sum(axis=0)).ravel()
                        if activity is not None else None)
    flags = []
    for j in np.flatnonzero(total > 0):
        outside = outside_lit[j] / total[j]
        active = float(outside_activity[j] / total[j]) if outside_activity is not None else None
        if outside >= outside_fraction or (active is not None and active >= outside_fraction):
            flags.append({"component": int(ids[j]), "outside_lit": round(float(outside), 4),
                          "outside_activity": None if active is None else round(active, 4)})
    return flags


def wave_candidates(est, ids, centres, time_s, fps, radius_px=24., k_mad=10., min_nearby=5,
                    min_event_s=.2, severe_share=.025):
    """Find sustained, locally synchronous calcium transients and report severe-session candidates."""
    valid = np.isfinite(centres).all(axis=1)
    ids, centres = ids[valid], centres[valid]
    n, frames = len(ids), est.C.shape[1]
    if n < min_nearby or not frames:
        return {"events": [], "frame_fraction": 0.0, "severe": False}
    if len(time_s) != frames:
        raise ValueError(f"The estimates contain {frames} frames and the time axis contains {len(time_s)}.")
    neighborhoods = cKDTree(centres).query_ball_point(centres, radius_px)
    counts = [len(group) for group in neighborhoods]
    tree = sparse.csr_matrix((np.ones(sum(counts), dtype=np.int32),
                              np.concatenate(neighborhoods), np.r_[0, np.cumsum(counts)]), shape=(n, n))
    active = np.empty((n, frames), dtype=bool)
    noise = wave.component_noise(est, ids)
    # A local baseline prevents slow changes in calcium signal across the recording from dominating.
    window = max(1, round(30 * fps))
    for first in range(0, frames, window):
        block = est.C[:, first:first + window]
        for start in range(0, n, 16):
            rows = ids[start:start + 16]
            active[start:start + len(rows), first:first + block.shape[1]] = wave.active_mask(
                block, rows, noise[rows], k_mad)
    enough = np.empty(frames, dtype=bool)
    for start in range(0, frames, 1000):
        block = active[:, start:start + 1000]
        local = block & (tree @ block.astype(np.int32) >= min_nearby)
        enough[start:start + block.shape[1]] = local.sum(axis=0) >= min_nearby
    events = []
    spans = []
    for a, b in bad_frames.ranges_from_indices(np.flatnonzero(enough)):
        cuts = np.flatnonzero(np.diff(time_s[a:b + 1]) > 3 / fps) + a
        spans.extend(zip([a, *(cuts + 1)], [*cuts, b]))
    for a, b in spans:
        if b + 1 < frames and time_s[b + 1] - time_s[b] > 3 / fps:
            duration = float(time_s[b] - time_s[a]) + 1 / fps
        else:
            duration = bad_frames.duration_s(a, b, time_s, fps)
        if duration < min_event_s:
            continue
        present = np.zeros(n, dtype=bool)
        for start in range(a, b + 1, 1000):
            block = active[:, start:min(b + 1, start + 1000)]
            present |= np.any(block & (tree @ block.astype(np.int32) >= min_nearby), axis=1)
        involved = ids[present]
        events.append({"first_frame": int(a), "last_frame": int(b), "start_s": round(float(time_s[a]), 6),
                       "duration_s": round(duration, 6), "components": involved.tolist()})
    share = sum(e["last_frame"] - e["first_frame"] + 1 for e in events) / frames
    return {"events": events, "frame_fraction": round(share, 6), "severe": share >= severe_share}


def _read(path):
    with path.open("rb") as item:
        return pickle.load(item)


def _overlay(est, sid, output):
    from caiman.utils.visualization import get_contours
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure

    ids = accepted(est)
    fig = Figure(figsize=(8, 8), layout="constrained")
    FigureCanvasAgg(fig)
    axis = fig.subplots()
    axis.imshow(est.imax, cmap="gray")
    if len(ids):
        curves = get_contours(est.A[:, ids], dims=est.imax.shape, thr=.3)
        lines = [curve["coordinates"] for curve in curves if len(curve["coordinates"])]
        axis.add_collection(LineCollection(lines, colors="#46f0b5", linewidths=.65))
    axis.set_title(f"{sid} · after manual inspection · {len(ids)} components")
    axis.set_axis_off()
    fig.savefig(output, dpi=130, format="png")


def _candidate_summary(paths, rec):
    original = _read(gate._estimates_file(paths, rec))
    ids = accepted(original)
    spatial, centres = _spatial(original, ids)
    try:
        lit, activity = _support(paths, rec, original.imax.shape)
    except FileNotFoundError:
        lit = activity = None
    times = session_time_s(paths, rec)
    return {"sid": rec.sid, "waves": wave_candidates(original, ids, centres, times, rec.fps),
            "edges": edge_candidates(spatial, ids, lit, activity)}


def make_gallery(series, output, sids=(), candidates=False):
    paths, output = SeriesPaths(series), Path(output)
    records = {r.sid: r for r in registry.all_records(paths)}
    missing = set(sids) - records.keys()
    if missing:
        raise ValueError(f"Unknown sessions: {', '.join(sorted(missing))}")
    requested = [records[sid] for sid in (sids or sorted(records))]
    output.mkdir(parents=True, exist_ok=True)
    by_mouse = defaultdict(list)
    failures = []
    summary = []
    for rec in requested:
        try:
            manual = paths.manual_result(rec.sid)
            if not manual.is_file():
                by_mouse[rec.mouse].append((rec.sid, None))
            else:
                path = output / f"{rec.sid}.png"
                est = _read(manual)
                fileio.write_atomic(path, lambda f: _overlay(est, rec.sid, f), binary=True)
                del est
                by_mouse[rec.mouse].append((rec.sid, path.name))
                print(f"[OK] {path.name}", flush=True)
            if candidates:
                summary.append(_candidate_summary(paths, rec))
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            failures.append({"sid": rec.sid, "error": str(exc)})
            print(f"[FAILED] {rec.sid}: {ascii(str(exc))}", flush=True)
    content = [
        '<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width">',
        "<title>После ручной инспекции</title>",
        "<style>body{font:16px system-ui;margin:24px}section{display:grid;grid-template-columns:"
        "repeat(auto-fit,minmax(260px,1fr));gap:12px}figure{margin:0}img{width:100%;height:auto}</style>",
        "<h1>Пространственные компоненты после ручной инспекции</h1>",
    ]
    for mouse, sessions in by_mouse.items():
        content.append(f"<h2>{html.escape(mouse)}</h2><section>")
        for sid, name in sessions:
            sid = html.escape(sid)
            if name is None:
                content.append(f"<p>{sid} · нет сохранённой ручной инспекции</p>")
            else:
                name = html.escape(name, quote=True)
                content.append(f'<figure><a href="{name}"><img loading="lazy" src="{name}" alt="{sid}"></a>'
                               f"<figcaption>{sid}</figcaption></figure>")
        content.append("</section>")
    if failures:
        errors = "".join(f"<li>{html.escape(e['sid'])}: {html.escape(e['error'])}</li>" for e in failures)
        content.append(f"<h2>Неполные результаты</h2><ul>{errors}</ul>")
    fileio.write_atomic(output / "index.html", lambda f: f.write("\n".join(content)))
    if candidates:
        fileio.write_atomic(output / "candidates.json", lambda f: json.dump({
            "status": "review_candidates", "sessions": summary, "failures": failures,
        }, f, indent=2))
    print(f"[OK] Images: {sum(name is not None for group in by_mouse.values() for _, name in group)}; "
          f"errors: {len(failures)}; index: {output / 'index.html'}", flush=True)
    return int(bool(failures))
