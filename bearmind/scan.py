"""Series scan: find sessions (bonsai and legacy UCLA layouts), check video against timestamps, write the registry."""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import re
import statistics
from pathlib import Path

from . import config
from .naming import mouse_id, session_id
from .paths import SeriesPaths
from .registry import (STAGES, Anomaly, RawPart, SessionRecord, StageRecord, load_record,
                       save_record)
from .timestamps import read_timestamps, robust_fps, stitch
from .video import last_frame_readable, video_info

_LEGACY_CHUNK = re.compile(r"^(\d+)\.avi$", re.I)


def _bonsai_chunks(folder: Path) -> list[tuple[int, str]]:
    sid = session_id(folder.name)
    if not sid:
        return []
    rx = re.compile(rf"^{re.escape(sid)}_(\d+)\.avi$", re.I)
    out = []
    for f in folder.iterdir():
        m = rx.match(f.name)
        if m and f.is_file():
            out.append((int(m.group(1)), f.name))
    return sorted(out)


def _legacy_chunks(folder: Path) -> list[tuple[int, str]]:
    out = []
    for f in folder.iterdir():
        m = _LEGACY_CHUNK.match(f.name)
        if m and f.is_file():
            out.append((int(m.group(1)), f.name))
    return sorted(out)


def _legacy_sid(folder: Path):
    """Session id from <SID>/<date>/<HH_MM_SS>/Miniscope (the great-grandparent folder name)."""
    if folder.name != "Miniscope" or len(folder.parents) < 3:
        return None
    return session_id(folder.parents[2].name)


def detect_mode(folder: Path) -> str | None:
    folder = Path(folder)
    if not folder.is_dir():
        return None
    if _bonsai_chunks(folder):
        return "bonsai"
    if _legacy_sid(folder) and _legacy_chunks(folder):
        return "legacy"
    return None


def _parse_rate(value) -> float | None:
    """Nominal frame rate from a number or a string like '30FPS'; None if nothing parses."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    m = re.search(r"\d+(?:\.\d+)?", str(value or ""))
    return float(m.group()) if m and float(m.group()) > 0 else None


def _json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None


def _find_bonsai_ts(paths: SeriesPaths, folder: Path, sid: str) -> Path | None:
    names = (f"{sid}_Mini_TS.csv", f"{sid}_timestamp.csv")
    for base in (folder, paths.timestamps_root):
        for n in names:
            if (base / n).is_file():
                return base / n
    if paths.timestamps_root.is_dir():
        for n in names:
            hit = next(paths.timestamps_root.rglob(n), None)
            if hit is not None:
                return hit
    return None


def _walk(paths: SeriesPaths):
    """Return ({sid: [(mode, folder)]}, [(folder, reason)]) for every folder with AVI files in 1_Raw."""
    found: dict[str, list[tuple[str, Path]]] = {}
    unrecognized: list[tuple[Path, str]] = []
    for dirpath, dirs, files in os.walk(paths.raw_root):
        dirs.sort()
        d = Path(dirpath)
        if not any(f.lower().endswith(".avi") for f in files):
            continue
        mode = detect_mode(d)
        if mode == "bonsai":
            found.setdefault(session_id(d.name), []).append((mode, d))
        elif mode == "legacy":
            found.setdefault(_legacy_sid(d), []).append((mode, d))
        elif d.name == "Miniscope":
            if _legacy_chunks(d):
                unrecognized.append((d, "legacy folder name is not a session id"))
            else:
                unrecognized.append((d, "Miniscope AVI names are not <n>.avi"))
        elif (d.parent / "Miniscope").is_dir():
            continue  # other camera of a legacy recording (e.g. BehavCam), not a miniscope
        elif session_id(d.name) is None:
            unrecognized.append((d, "folder name is not a session id"))
        else:
            unrecognized.append((d, "AVI names do not match <SID>_<n>.avi"))
    return found, unrecognized


class _Unreadable(Exception):
    def __init__(self, chunk, exc):
        super().__init__(f"{chunk}: {exc}")


def _read_part(paths: SeriesPaths, sid: str, mode: str, folder: Path, cfg_scan: dict):
    """Inspect one part: chunks, frame counts, timestamps, per-part drop rules. Returns (RawPart, extras)."""
    chunks = _bonsai_chunks(folder) if mode == "bonsai" else _legacy_chunks(folder)
    names = [n for _, n in chunks]
    infos = []
    for n in names:
        try:
            infos.append(video_info(folder / n))
        except Exception as e:
            raise _Unreadable(n, e) from e
    total = sum(i[0] for i in infos)
    shape = (infos[0][1], infos[0][2])
    anomalies: list[Anomaly] = []
    where = paths.rel(folder)

    meta_file = start_ms = ts_file = None
    nominal = None
    if mode == "legacy":
        meta = folder.parent / "metaData.json"
        data = _json(meta) if meta.is_file() else None
        if data is not None:
            meta_file = paths.rel(meta)
            try:
                start_ms = float(data["recordingStartTime"]["msecSinceEpoch"])
            except (KeyError, TypeError, ValueError):
                start_ms = None
        nominal_data = _json(folder / "metaData.json") if (folder / "metaData.json").is_file() else None
        if nominal_data:
            nominal = _parse_rate(nominal_data.get("frameRate"))
        ts_path = folder / "timeStamps.csv"
        ts_path = ts_path if ts_path.is_file() else None
    else:
        ts_path = _find_bonsai_ts(paths, folder, sid)

    ts = None
    ts_error = None
    if ts_path is not None:
        ts_file = paths.rel(ts_path)
        try:
            ts = read_timestamps(ts_path, start_epoch_ms=start_ms)
        except ValueError as e:
            ts_error = str(e)
    else:
        ts_error = f"no timestamps file for {where}"

    kept = total
    drop_last = drop_tail = False
    try:
        last_ok = last_frame_readable(folder / names[-1])
    except Exception as e:
        raise _Unreadable(names[-1], e) from e
    if not last_ok:
        drop_last = True
        kept -= 1
        anomalies.append(Anomaly("broken_last_frame", "warning",
                                 f"last frame of {names[-1]} in {where} is unreadable; dropped",
                                 auto_fixed=True))
    if ts is not None:
        surplus = kept - len(ts.t_ms)
        if len(names) > 1 and surplus == len(names) - 1:
            drop_tail = True
            kept -= surplus
            anomalies.append(Anomaly("chunk_tail_frames", "warning",
                                     f"{surplus} extra frames in {where}: one at the end of each chunk but the last; dropped",
                                     auto_fixed=True, data={"surplus": surplus}))
    part = RawPart(folder=where, chunks=names, n_frames=kept, shape=shape, timestamps_file=ts_file,
                   ts_format=ts.fmt if ts else None, start_ms=start_ms, metadata_file=meta_file,
                   drop_chunk_tail=drop_tail, drop_last_frame=drop_last)
    return part, ts, ts_error, nominal, anomalies


def _build_record(paths: SeriesPaths, sid: str, items, cfg_scan: dict):
    parts, tss, errors, nominals, anomalies = [], [], [], [], []
    unreadable = []
    for mode, folder in items:
        try:
            p, ts, err, nominal, an = _read_part(paths, sid, mode, folder, cfg_scan)
        except _Unreadable as e:
            unreadable.append(Anomaly("unreadable_video", "error", str(e)))
            continue
        parts.append(p)
        tss.append(ts)
        errors.append(err)
        nominals.append(nominal)
        anomalies += an
    anomalies += unreadable
    mode = items[0][0]
    if not parts:
        return [], mode, 0, None, {}, anomalies, None
    multi = len(parts) > 1

    # order: legacy by recording start, bonsai by first timestamp; folder name when neither is known
    def key(i):
        p, ts = parts[i], tss[i]
        if mode == "legacy" and p.start_ms is not None:
            return (0, p.start_ms, p.folder)
        if mode == "bonsai" and ts is not None:
            return (0, float(ts.t_ms[0]), p.folder)
        return (1, 0.0, p.folder)

    order = sorted(range(len(parts)), key=key)
    parts = [parts[i] for i in order]
    tss = [tss[i] for i in order]
    errors = [errors[i] for i in order]
    nominal = next((n for n in (nominals[i] for i in order) if n), None)

    if mode == "legacy" and any(p.start_ms is None for p in parts):
        names = ", ".join(p.folder for p in parts if p.start_ms is None)
        anomalies.append(Anomaly("no_metadata", "warning", f"no recording start time for: {names}"))

    unorderable = multi and mode == "legacy" and any(p.start_ms is None for p in parts)
    if multi:
        gaps = []
        if not unorderable and all(t is not None for t in tss):
            gaps = [round(float(tss[i + 1].t_ms[0] - tss[i].t_ms[-1]) / 1000.0, 3)
                    for i in range(len(tss) - 1)]
        if unorderable:
            anomalies.append(Anomaly("split_recording", "error",
                                     f"{len(parts)} parts, recording start unknown: cannot order or stitch",
                                     data={"gaps_s": gaps}))
        else:
            anomalies.append(Anomaly("split_recording", "warning",
                                     f"recording split into {len(parts)} parts; stitched by start time",
                                     data={"gaps_s": gaps}))

    total = sum(p.n_frames for p in parts)
    n_frames, fps, fps_info, st = total, None, {}, None
    bad = next((e for e in errors if e), None)
    if bad is not None:
        anomalies.append(Anomaly("no_timestamps", "error", bad))
    elif not unorderable:
        st = stitch(tss) if multi else tss[0]

    if st is not None:
        rows = len(st.t_ms)
        diff = total - rows
        if diff:
            frac = abs(diff) / rows if rows else 1.0
            ok = frac <= cfg_scan["max_mismatch_frac"]
            anomalies.append(Anomaly(
                "frames_ts_mismatch", "warning" if ok else "error",
                f"{total} frames in video, {rows} timestamp rows (diff {diff}, {frac:.1%})",
                auto_fixed=ok, data={"diff": diff, "where": "end"}))
        n_frames = min(total, rows)
        try:
            est = robust_fps(st.t_ms[:n_frames], cfg_scan["gap_factor"])
        except ValueError as e:
            est = None
            anomalies.append(Anomaly("fps_unavailable", "warning", str(e)))
        if est is not None:
            fps = est.fps
            fps_info = {**dataclasses.asdict(est), "source": "timestamps"}
            if est.n_gaps > 0:
                anomalies.append(Anomaly(
                    "dropped_frames", "warning",
                    f"{est.n_gaps} gaps in timestamps (max {est.max_gap_ms:.0f} ms)",
                    data={"n_gaps": est.n_gaps, "max_gap_ms": est.max_gap_ms}))
    return parts, mode, n_frames, fps, fps_info, anomalies, nominal


def scan_series(series_root: Path) -> list[SessionRecord]:
    paths = SeriesPaths(series_root)
    if not paths.series_config.exists():
        config.update_yaml(paths.series_config, {})
    found, unrecognized = _walk(paths)

    built = {}
    for sid in sorted(found):
        cfg_scan = config.resolve(paths, sid=sid)["scan"]
        built[sid] = _build_record(paths, sid, found[sid], cfg_scan)

    # series-level FPS checks
    fps_list = [b[3] for b in built.values() if b[3]]
    median = statistics.median(fps_list) if fps_list else None
    dev = config.resolve(paths)["scan"]["fps_series_dev"]
    for sid, (parts, mode, n, fps, info, anomalies, nominal) in built.items():
        if not fps:
            continue
        if median and abs(fps / median - 1) > dev:
            anomalies.append(Anomaly("fps_outlier", "warning",
                                     f"fps {fps:.2f} differs from series median {median:.2f}",
                                     data={"fps": fps, "median": median}))
        if nominal and abs(fps / nominal - 1) > dev:
            anomalies.append(Anomaly("fps_outlier", "warning",
                                     f"fps {fps:.2f} differs from nominal frameRate {nominal:g}",
                                     data={"fps": fps, "nominal": nominal}))

    records = []
    for sid, (parts, mode, n, fps, info, anomalies, _nominal) in built.items():
        f = paths.registry_file(sid)
        if f.is_file():
            rec = load_record(paths, sid)
            rec.parts, rec.n_frames, rec.fps = parts, n, fps
            rec.fps_info, rec.anomalies, rec.mode = info, anomalies, mode
        else:
            rec = SessionRecord(
                sid=sid, mouse=mouse_id(sid), mode=mode, parts=parts, n_frames=n, fps=fps,
                fps_info=info, anomalies=anomalies, decisions=[],
                excluded=any(a.level == "error" for a in anomalies),
                stages={s: StageRecord() for s in STAGES})
        save_record(paths, rec)
        records.append(rec)

    out = paths._service / "scan_unrecognized.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "reason"])
        for d, reason in unrecognized:
            w.writerow([paths.rel(d), reason])
    return sorted(records, key=lambda r: r.sid)
