"""Series scan: find sessions (bonsai and legacy UCLA layouts), check video against timestamps, write the registry."""
from __future__ import annotations

import csv
import dataclasses
import io
import json
import os
import re
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config, fileio
from .naming import mouse_id, session_id
from .paths import SeriesPaths
from .registry import (STAGES, Anomaly, RawPart, SessionRecord, StageRecord, replace_broken, save_record,
                       update_record)
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


def _head(ts, n):
    """The first n rows of a Timestamps."""
    return dataclasses.replace(ts, raw=ts.raw[:n], t_ms=ts.t_ms[:n],
                               buffer=None if ts.buffer is None else ts.buffer[:n])


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
                   drop_chunk_tail=drop_tail, drop_last_frame=drop_last,
                   chunk_frames=[int(i[0]) for i in infos])
    return part, ts, ts_error, nominal, anomalies


def _build_record(paths: SeriesPaths, sid: str, items, cfg_scan: dict):
    parts, tss, errors, nominals, anomalies = [], [], [], [], []
    unreadable = []
    for mode, folder in items:
        try:
            p, ts, err, nominal, an = _read_part(paths, sid, mode, folder, cfg_scan)
        except _Unreadable as e:
            # the folder: a session whose only part cannot be read has no parts, and Open folder needs it
            unreadable.append(Anomaly("unreadable_video", "error", str(e), data={"folder": paths.rel(folder)}))
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

    # Each part is cut to its own timestamp rows: a surplus at the end of one part must not shift
    # the frames of the next part against their timestamps.
    for k, (p, ts) in enumerate(zip(parts, tss)):
        if ts is None:
            continue
        rows = len(ts.t_ms)
        diff = p.n_frames - rows
        if diff:
            frac = abs(diff) / rows if rows else 1.0
            ok = frac <= cfg_scan["max_mismatch_frac"]
            label, data = (f"part {k + 1} ({p.folder}): ", {"where": f"part {k} end", "part": k}) if multi \
                else ("", {"where": "end"})
            anomalies.append(Anomaly(
                "frames_ts_mismatch", "warning" if ok else "error",
                f"{label}{p.n_frames} frames in video, {rows} timestamp rows (diff {diff}, {frac:.1%})",
                auto_fixed=ok, data={"diff": diff, **data}))
            p.n_frames = min(p.n_frames, rows)
            tss[k] = _head(ts, p.n_frames)

    n_frames, fps, fps_info, st = sum(p.n_frames for p in parts), None, {}, None
    bad = next((e for e in errors if e), None)
    if bad is not None:
        anomalies.append(Anomaly("no_timestamps", "error", bad))
    elif not unorderable:
        st = stitch(tss) if multi else tss[0]

    if st is not None:
        try:
            est = robust_fps(st.t_ms, cfg_scan["gap_factor"])
        except ValueError as e:
            est = None
            anomalies.append(Anomaly("fps_unavailable", "error", f"{e}; {NO_FPS_MESSAGE}"))
        if est is not None:
            fps = est.fps
            fps_info = {**dataclasses.asdict(est), "source": "timestamps"}
            # Losses are judged per part; FPS above still counts the gaps (it is in the MC stage hash).
            anomalies += _gap_anomalies(tss, cfg_scan["gap_factor"])
    return parts, mode, n_frames, fps, fps_info, anomalies, nominal


# Intervals on each side of a gap that the window test sums (the window is 2 * K + 1 intervals).
WINDOW_K = 30
CERTAIN_INTERVAL = 2.5  # an interval above this many medians is a lost frame whatever the window says
LOSS_MIN_EXCESS = 0.5   # window excess below this is timing jitter
CERTAIN_EXCESS = 1.5
MAX_GAP_LINES = 20  # places (clusters of gaps) listed in the message; data["gaps"] has all of them


@dataclasses.dataclass
class GapLoss:
    """A cluster of gaps (one gap, or several whose windows overlap) and the frames lost in it."""
    index: int          # the first gap is the interval between frames index and index + 1 of the part
    at_ms: float        # where the first gap begins, from the part's first frame
    interval_ms: float  # the longest interval of the cluster
    excess: float       # the window's time in periods minus the number of intervals
    frames: int         # round(excess), at least 1
    lost_ms: float      # frames * period
    certain: bool
    n_gaps: int = 1


@dataclasses.dataclass
class PartGaps:
    gaps: list[GapLoss]   # the clusters with at least one lost frame
    n_jitter: int         # gaps (interval above gap_factor * median) in clusters with no frame lost
    damaged: bool         # a non-positive interval: the test is not run


def part_gap_losses(t_ms, gap_factor: float = 2.0, k: int = WINDOW_K) -> PartGaps:
    """Frames lost at the gaps of one part's timestamps by the window test.

    A late frame makes one interval long and the next ones short, and the frame clock goes on: judged by its own
    length a gap of 2.1 medians is mostly jitter (85 % of the 324 gaps of LT1). Around a gap the intervals of the
    window, in units of the period u, count the frames the clock produced; the excess over the number of intervals
    is the number of frames missing. u is the mean of the intervals that are not gaps: a gap must not stretch its
    own period (span / (N - 1) would, in a short part). The window is the 2 * k + 1 intervals around the gap
    interval; near the start or the end of the part it is cut to the intervals that exist. Gaps closer than 2 * k
    intervals share their windows, so they form one cluster with one window (k intervals before its first gap to k
    after its last) and one excess: the same time is not counted once per gap. A part is judged alone: timestamps
    restart per part, and the pause between parts is not a gap.
    """
    t = np.asarray(t_ms, dtype=np.float64)
    if len(t) < 2:
        return PartGaps([], 0, False)
    d = np.diff(t)
    if np.any(d <= 0):
        return PartGaps([], 0, True)
    m = float(np.median(d))
    is_gap = d > gap_factor * m
    u = float(d[~is_gap].mean())
    clusters: list[list[int]] = []
    for i in np.flatnonzero(is_gap):
        if clusters and i - clusters[-1][-1] <= 2 * k:
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    gaps, n_jitter = [], 0
    for c in clusters:
        window = d[max(0, c[0] - k):c[-1] + k + 1]
        excess = float(window.sum() / u - len(window))
        if excess < LOSS_MIN_EXCESS:
            n_jitter += len(c)
            continue
        frames = max(int(round(excess)), 1)
        longest = float(d[c].max())
        gaps.append(GapLoss(
            index=c[0], at_ms=float(t[c[0]] - t[0]), interval_ms=longest, excess=excess, frames=frames,
            lost_ms=frames * u, certain=bool(longest > CERTAIN_INTERVAL * m or excess >= CERTAIN_EXCESS),
            n_gaps=len(c)))
    return PartGaps(gaps, n_jitter, False)


def position_text(ms: float) -> str:
    """A place in a part for a person: "12:03.4" (minutes:seconds), "1:01:02.0" from one hour."""
    tenths = int(round(ms / 100.0))
    s, tenth = divmod(tenths, 10)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}.{tenth}" if h else f"{m}:{s:02d}.{tenth}"


def _frames_ms_text(frames: int, ms: float) -> str:
    return f"{plural(frames, 'frame')} lost ({duration_text(ms)})"


def _gap_anomalies(tss, gap_factor: float) -> list[Anomaly]:
    """dropped_frames, timing_jitter and gap_test_skipped from the window test of each part."""
    multi = len(tss) > 1
    found: list[tuple[int, GapLoss]] = []
    n_jitter, damaged = 0, []
    for k, ts in enumerate(tss):
        r = part_gap_losses(ts.t_ms, gap_factor)
        found += [(k, g) for g in r.gaps]
        n_jitter += r.n_jitter
        if r.damaged:
            damaged.append(k)
    out = []
    if found:
        rows = [{"part": k, "index": g.index, "at_ms": round(g.at_ms, 1), "interval_ms": round(g.interval_ms, 1),
                 "excess": round(g.excess, 2), "frames": g.frames, "lost_ms": round(g.lost_ms, 1),
                 "certain": g.certain, "n_gaps": g.n_gaps} for k, g in found]
        frames = sum(r["frames"] for r in rows)
        lost = sum(r["lost_ms"] for r in rows)
        n_gaps = sum(r["n_gaps"] for r in rows)
        data = {"rule": "window", "n_gaps": n_gaps, "n_clusters": len(rows), "n_dropped": frames, "lost_ms": round(lost, 1),
                "max_gap_ms": max(r["interval_ms"] for r in rows), "gaps": rows}
        lines = [f"{_frames_ms_text(frames, lost)} in {plural(n_gaps, 'gap')}"]
        parts = []
        for name, sel in (("certain", True), ("probable", False)):
            sub = [r for r in rows if r["certain"] is sel]
            data[f"n_{name}"] = len(sub)
            data[f"frames_{name}"] = sum(r["frames"] for r in sub)
            data[f"lost_ms_{name}"] = round(sum(r["lost_ms"] for r in sub), 1)
            if sub:
                parts.append(f"{name} {_frames_ms_text(data[f'frames_{name}'], data[f'lost_ms_{name}'])}")
        lines[0] += ": " + ", ".join(parts)
        shown = rows[:MAX_GAP_LINES]
        for r in shown:
            place = f"{'part ' + str(r['part'] + 1) + ' ' if multi else ''}at {position_text(r['at_ms'])}"
            group = f"{plural(r['n_gaps'], 'gap')}, " if r["n_gaps"] > 1 else ""
            lines.append(f"{group}{_frames_ms_text(r['frames'], r['lost_ms'])} {place}, "
                         f"{'certain' if r['certain'] else 'probable'}")
        if len(rows) > len(shown):
            lines.append(f"and {len(rows) - len(shown)} more places")
        out.append(Anomaly("dropped_frames", "warning", "\n".join(lines), data=data))
    if n_jitter:
        out.append(Anomaly("timing_jitter", "info", f"timing jitter: {plural(n_jitter, 'gap')}, no frames lost",
                           data={"n_gaps": n_jitter}))
    if damaged:
        names = ", ".join(str(k + 1) for k in damaged) if multi else ""
        out.append(Anomaly("gap_test_skipped", "info",
                           "frame loss test skipped" + (f" for part {names}" if multi else "") +
                           ": the timestamps do not increase (a damaged row)", data={"parts": damaged}))
    return out


def rates_text(rates) -> str:
    """Distinct frame rates as a person reads them: "19.8, 29.8"."""
    return ", ".join(dict.fromkeys(f"{r:.1f}" for r in sorted(rates)))


def plural(n: int, word: str) -> str:
    """"1 gap", "3 gaps"."""
    return f"{n} {word}" + ("" if n == 1 else "s")


def duration_text(ms: float) -> str:
    """A gap or pause for a person: "72 ms", "1.5 s", "38 s", "59 min 58 s", "2 h 3 min"."""
    if ms < 1000:
        return f"{ms:.0f} ms"
    s = ms / 1000.0
    if s < 10:
        return f"{s:.1f} s"
    s = round(s)
    if s < 60:
        return f"{s} s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m} min {s} s"
    h, m = divmod(m, 60)
    return f"{h} h {m} min"


def latest_decision(rec: SessionRecord, what: str):
    """Value of the newest decision of that kind, or None: the latest decision wins."""
    found = [d for d in rec.decisions if d.what == what]
    return found[-1].value if found else None


_latest_decision = latest_decision  # old name


def stitch_fingerprint(parts) -> str:
    """What a stitch confirmation is about: the part folders in stitch order. A part added, removed or put in
    another order gives another value, so the old confirmation does not cover the new stitch."""
    return "|".join(p.folder for p in parts)


def stitch_confirmed(rec: SessionRecord) -> bool:
    """A person confirmed exactly the current stitch. A decision written before the fingerprint ("true") does not
    say which stitch it saw, so it does not count and the person confirms again."""
    fingerprint = stitch_fingerprint(rec.parts)
    return any(d.what == "stitch_confirmed" and d.value == fingerprint for d in rec.decisions)


NO_FPS_MESSAGE = "no frame rate: the stages need it (MC video rate, CNMF-E fr); the session stays excluded"
NOT_FOUND_MESSAGE = "session folder not found by the last scan"


def _mark_no_fps(rec: SessionRecord) -> None:
    """One error-level scan anomaly fps_unavailable (a warning from the fps estimate is raised to error)."""
    found = [a for a in rec.anomalies if a.code == "fps_unavailable" and not a.data.get("stage")]
    if found:
        found[0].level = "error"
    else:
        rec.anomalies.append(Anomaly("fps_unavailable", "error", NO_FPS_MESSAGE))


def evaluate_excluded(rec: SessionRecord) -> bool:
    """Re-evaluated on every scan: the user's latest exclude decision, else any error anomaly of the scan.

    A session without a frame rate is always excluded (fps_unavailable): no stage can run without it, so an
    include decision cannot help; nor can it for a session whose folder the last scan did not find
    (not_found_in_scan): there are no files to process. An accepted mismatch stays in the record as a
    warning; n_frames is already the common length. Anomalies tagged with a stage (data["stage"]) belong to
    that stage and do not exclude the session.
    """
    if latest_decision(rec, "accept_mismatch") == "true":
        for a in rec.anomalies:
            if a.code == "frames_ts_mismatch" and a.level == "error":
                a.level, a.auto_fixed = "warning", True
    if rec.fps is None:
        _mark_no_fps(rec)
        return True
    if any(a.code == "not_found_in_scan" for a in rec.anomalies):
        return True
    exclude = latest_decision(rec, "exclude")
    if exclude is not None:
        return exclude == "true"
    return any(a.level == "error" for a in rec.anomalies if not a.data.get("stage"))


_excluded = evaluate_excluded  # old name


UNREADABLE = (ValueError, KeyError, TypeError, AttributeError)  # a registry file that is not a record


def _recreate(paths: SeriesPaths, f: Path, error: Exception, new_record):
    """A fresh record in place of a registry file that cannot be read; the old file is kept beside it.

    Its decisions and stage states are lost with the file, so every stage of the session runs again.
    Returns None when the file became readable meanwhile (it is then rescanned as usual).
    """
    kept = f.with_name(f"{f.name}.broken-{datetime.now():%Y%m%d-%H%M%S}")
    why = " ".join(f"{type(error).__name__}: {error}".split())
    rec = new_record([Anomaly("registry_recreated", "warning",
                              f"registry file was unreadable ({why}); recreated, the old file kept as {kept}; "
                              f"all its stages run again")])
    return rec if replace_broken(paths, rec, kept) else None


def _mark_not_found(rec: SessionRecord) -> None:
    """The scan did not find the session's folder (removed or renamed in Explorer). Its stages and decisions
    stay; it is excluded until a scan finds the folder again, which replaces the scan's anomalies."""
    if not any(a.code == "not_found_in_scan" and not a.data.get("stage") for a in rec.anomalies):
        rec.anomalies.append(Anomaly("not_found_in_scan", "error", NOT_FOUND_MESSAGE))
    rec.excluded = evaluate_excluded(rec)


# (done, total, sid): "" right after the folder walk, then the sid being read (done sessions before it),
# and a last call (total, total, last sid) once everything is written
ProgressFn = Callable[[int, int, str], None]


def scan_series(series_root: Path, progress: ProgressFn | None = None) -> list[SessionRecord]:
    """Scan the series and write its registry; returns the records of the sessions found.

    Raises FileNotFoundError before writing anything when 1_Raw is not a folder (renamed, a junction to a
    disconnected share): an empty walk would otherwise look like a series without sessions.
    """
    paths = SeriesPaths(series_root)
    if not paths.raw_root.is_dir():
        raise FileNotFoundError(f"1_Raw not found: {paths.raw_root}")
    if not paths.series_config.exists():
        config.update_yaml(paths.series_config, {})
    found, unrecognized = _walk(paths)

    total = len(found)
    if progress:
        progress(0, total, "")
    built = {}
    for i, sid in enumerate(sorted(found)):
        if progress:  # reading the videos and timestamps is the slow part: name the session being read
            progress(i, total, sid)
        cfg_scan = config.resolve(paths, sid=sid)["scan"]
        built[sid] = _build_record(paths, sid, found[sid], cfg_scan)

    # FPS checks against the session's own mouse: mice of one series may be recorded at different rates (LT1:
    # six mice at 19.76 fps, the rest at 30.59), and one mouse may switch rates between days (NOF: 19.8 and
    # 29.8, whose median 24.8 is neither), which is no anomaly. A rate is an outlier only when no other session
    # of the mouse has a rate within fps_series_dev of it. A mouse with fewer sessions with a rate than
    # checks.min_sessions_for_relative has no reference.
    series_cfg = config.resolve(paths)
    dev = series_cfg["scan"]["fps_series_dev"]
    min_n = series_cfg["checks"].get("min_sessions_for_relative", 3)
    rates: dict[str, list[tuple[str, float]]] = {}
    for sid, b in built.items():
        if b[3]:
            rates.setdefault(mouse_id(sid), []).append((sid, b[3]))
    for sid, (parts, mode, n, fps, info, anomalies, nominal) in built.items():
        if not fps:
            continue
        own = rates.get(mouse_id(sid), [])
        others = sorted(r for s, r in own if s != sid)
        if others and len(own) >= min_n and not any(abs(fps / o - 1) <= dev for o in others):
            anomalies.append(Anomaly("fps_outlier", "warning",
                                     f"fps {fps:.1f}; other sessions of the mouse: {rates_text(others)}",
                                     data={"fps": fps, "nearest": min(others, key=lambda o: abs(o - fps)),
                                           "others": others, "mouse": mouse_id(sid)}))
        if nominal and abs(fps / nominal - 1) > dev:
            anomalies.append(Anomaly("fps_outlier", "warning",
                                     f"fps {fps:.2f} differs from nominal frameRate {nominal:g}",
                                     data={"fps": fps, "nominal": nominal}))

    records = []
    for sid, (parts, mode, n, fps, info, anomalies, _nominal) in built.items():  # built is in sorted sid order
        f = paths.registry_file(sid)

        def new_record(extra=(), anomalies=anomalies, sid=sid, mode=mode, parts=parts, n=n, fps=fps, info=info):
            rec = SessionRecord(
                sid=sid, mouse=mouse_id(sid), mode=mode, parts=parts, n_frames=n, fps=fps,
                fps_info=info, anomalies=[*anomalies, *extra], decisions=[], excluded=False,
                stages={s: StageRecord() for s in STAGES})
            rec.excluded = evaluate_excluded(rec)
            return rec

        if f.is_file():
            # under the record lock: the runner or a person may be writing this record meanwhile
            def rescan(rec, parts=parts, n=n, fps=fps, info=info, anomalies=anomalies, mode=mode):
                rec.parts, rec.n_frames, rec.fps, rec.fps_info, rec.mode = parts, n, fps, info, mode
                # the scan's own anomalies are replaced; those of the stages stay with their stage
                rec.anomalies = [a for a in rec.anomalies if a.data.get("stage")] + anomalies
                rec.excluded = evaluate_excluded(rec)

            try:
                rec = update_record(paths, sid, rescan)
            except UNREADABLE as e:
                rec = _recreate(paths, f, e, new_record) or update_record(paths, sid, rescan)
        else:
            rec = new_record()
            save_record(paths, rec)
        records.append(rec)

    for f in sorted(paths.registry_file("x").parent.glob("*.json")):
        if f.stem in found:
            continue
        try:
            update_record(paths, f.stem, _mark_not_found)
        except UNREADABLE:  # listed as an unreadable registry file; there is no record to mark
            pass
        except FileNotFoundError:  # removed since the listing
            pass

    def write_unrecognized(fh):
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["path", "reason"])
        for d, reason in unrecognized:
            w.writerow([paths.rel(d), reason])

    # atomic and retried: the file may be open in Excel, and a reader must never see half of it
    fileio.write_atomic(paths.scan_unrecognized_csv, write_unrecognized, newline="")
    if progress:
        progress(total, total, max(found, default=""))
    return sorted(records, key=lambda r: r.sid)


def read_unrecognized(paths: SeriesPaths) -> list[tuple[str, str]]:
    """(path, reason) of every folder the last scan could not read as a session ([] before any scan).

    Read with fileio.read_text: on Windows the file can be refused for a moment while a scan replaces it.
    """
    try:
        text = fileio.read_text(paths.scan_unrecognized_csv)
    except FileNotFoundError:
        return []
    return [(row["path"], row["reason"]) for row in csv.DictReader(io.StringIO(text))]
