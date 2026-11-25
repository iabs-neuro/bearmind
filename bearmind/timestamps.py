"""Timestamp files of miniscope recordings: reading, FPS estimate, stitching, writing."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BONSAI = "bonsai_ticks"
UCLA = "ucla_ms"
UCLA_HEADER = "Frame Number,Time Stamp (ms),Buffer Index"


@dataclass(frozen=True)
class ClockStep:
    frame: int
    jump_ms: int
    offset_ms: int
    interval_ms: int


@dataclass
class Timestamps:
    raw: np.ndarray
    t_ms: np.ndarray
    fmt: str
    start_epoch_ms: float | None = None
    buffer: np.ndarray | None = None
    dropped_row: str | None = None  # text of an incomplete last row that was left out (Bonsai only)
    clock_steps: tuple[ClockStep, ...] = ()  # repairs applied to raw and t_ms (UCLA only)


@dataclass
class FpsEstimate:
    fps: float
    n_outliers: int  # uneven intervals, outside [0.5 m, 1.5 m] of the median m; counted in fps (information only)
    n_gaps: int  # intervals above gap_factor * m: pauses, dropped frames; left out of fps
    max_gap_ms: float  # the longest interval, a gap or not
    median_interval_ms: float


def _repair_ucla_clock_steps(raw: np.ndarray, rows: list[list[str]]) -> tuple[np.ndarray, tuple[ClockStep, ...]]:
    """Undo large backward clock steps between runs of normally spaced, consecutive frames.

    Estimate only the interval across each step. Shift the later timestamps by a constant so their
    measured intervals are preserved. A duplicate, isolated bad row or unclear change is left for the
    scanner's damaged-timestamps check.
    """
    deltas = np.diff(raw)
    jumps = np.flatnonzero(deltas < 0)
    if not len(jumps):
        return raw, ()
    window = 5  # ordinary intervals required on either side of the jump
    offsets = np.zeros_like(raw)
    steps = []
    for i in jumps:
        if i < window or i + window >= len(deltas):
            continue
        neighbors = np.concatenate((deltas[i - window:i], deltas[i + 1:i + 1 + window]))
        period = float(np.median(neighbors))
        if period <= 0 or np.any((neighbors < 0.5 * period) | (neighbors > 1.5 * period)):
            continue
        jump = int(deltas[i])
        if -jump < 2 * period:
            continue
        try:
            frames = [int(row[0]) for row in rows[i - window:i + window + 2]]
        except ValueError:
            continue
        if any(b != a + 1 for a, b in zip(frames, frames[1:])):
            continue
        interval = int(period + 0.5)  # round the positive estimate to the nearest millisecond
        offset = interval - jump
        offsets[i + 1] = offset
        steps.append(ClockStep(frames[window + 1], jump, offset, interval))
    return (raw + offsets.cumsum(), tuple(steps)) if steps else (raw, ())


def read_timestamps(path: Path, start_epoch_ms: float | None = None) -> Timestamps:
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"unknown timestamp format: {path}")
    first = lines[0]
    if first.replace(" ", "") == UCLA_HEADER.replace(" ", ""):
        try:
            rows = [ln.split(",") for ln in lines[1:]]
            raw = np.array([int(r[1]) for r in rows], dtype=np.int64)
            buf = np.array([int(r[2]) for r in rows], dtype=np.int64)
        except (ValueError, IndexError):
            raise ValueError(f"unknown timestamp format: {path}") from None
        raw, clock_steps = _repair_ucla_clock_steps(raw, rows)
        t = raw.astype(np.float64)
        if start_epoch_ms is not None:
            t = t + start_epoch_ms
        return Timestamps(raw, t, UCLA, start_epoch_ms, buf, clock_steps=clock_steps)
    try:
        raw = np.array([int(ln) for ln in lines], dtype=np.int64)
    except ValueError:
        raise ValueError(f"unknown timestamp format: {path}") from None
    dropped = None
    # A Bonsai file can end in the middle of a row: the last buffer of the file was not written out when the
    # recording stopped (LT1_J05_1D: the file is exactly 231 x 4096 bytes). A cut number is always smaller than
    # the previous full one, so "no line end, not increasing and shorter" finds it exactly; a whole row without a line
    # end stays. The row is not restored: a guessed value would be invented data in the export.
    if len(raw) >= 2 and not text.rstrip(" \t").endswith(("\n", "\r")) and raw[-1] <= raw[-2] and len(lines[-1]) < len(lines[-2]):
        dropped = lines[-1]
        raw = raw[:-1]
    return Timestamps(raw, raw / 1e4, BONSAI, dropped_row=dropped)


def robust_fps(t_ms: np.ndarray, gap_factor: float = 2.0) -> FpsEstimate:
    """Frames per second over the intervals up to gap_factor * median: only gaps (pauses between parts,
    dropped frames) are left out.

    A late frame makes one interval long and the next ones short, and the frame clock goes on. Leaving out
    only the long one biases the rate up: LT1_J06 (long intervals of 1.75 m followed by short ones above
    0.5 m) gave 30.24-31.57 fps instead of 29.78 with the intervals in [0.5 m, 1.5 m]. Counted in full, the
    jitter cancels in the sum.
    """
    d = np.diff(np.asarray(t_ms, dtype=np.float64))
    d = d[d > 0]
    if len(d) < 2:
        raise ValueError("not enough timestamps to estimate fps")
    m = float(np.median(d))
    keep = d[d <= gap_factor * m]
    return FpsEstimate(
        fps=float(1000.0 * len(keep) / keep.sum()),
        n_outliers=int(np.sum((d < 0.5 * m) | (d > 1.5 * m))),
        n_gaps=int(len(d) - len(keep)),
        max_gap_ms=float(d.max()),
        median_interval_ms=m,
    )


def stitch(parts: list[Timestamps]) -> Timestamps:
    if not parts:
        raise ValueError("no timestamp parts to stitch")
    fmts = {p.fmt for p in parts}
    if len(fmts) != 1:
        raise ValueError(f"cannot stitch mixed timestamp formats: {sorted(fmts)}")
    fmt = fmts.pop()
    parts = sorted(parts, key=lambda p: p.t_ms[0])
    t = np.concatenate([p.t_ms for p in parts])
    if fmt == BONSAI:
        return Timestamps(np.concatenate([p.raw for p in parts]), t, fmt)
    if any(p.start_epoch_ms is None for p in parts):
        raise ValueError("cannot stitch UCLA timestamps without recording start")
    start = parts[0].start_epoch_ms
    raw = np.rint(t - start).astype(np.int64)
    buf = np.concatenate([
        p.buffer if p.buffer is not None else np.zeros(len(p.raw), dtype=np.int64)
        for p in parts
    ])
    return Timestamps(raw, t, fmt, start, buf)


def _truncate(ts: Timestamps, n: int) -> Timestamps:
    return dataclasses.replace(ts, raw=ts.raw[:n], t_ms=ts.t_ms[:n],
                               buffer=None if ts.buffer is None else ts.buffer[:n])


def session_timestamps(paths, rec) -> Timestamps:
    """The session's timestamps from its registry record: each part's file cut to the part's frames, the
    parts stitched in time order."""
    parts = []
    for i, p in enumerate(rec.parts):
        ts = read_timestamps(paths.root / p.timestamps_file, start_epoch_ms=p.start_ms)
        if i < len(rec.parts) - 1 and len(ts.raw) < p.n_frames:
            # the video of this part is longer than its timestamps: every later frame would get a
            # timestamp from the wrong position (registries scanned before parts were cut per part)
            raise ValueError(f"{p.folder}: {p.n_frames} frames but {len(ts.raw)} timestamp rows; "
                             f"rescan the series")
        parts.append(_truncate(ts, p.n_frames))
    return stitch(parts)


def write_timestamps(ts: Timestamps, frame_idx: np.ndarray, path: Path) -> None:
    idx = np.asarray(frame_idx, dtype=np.int64)
    raw = ts.raw[idx]
    if ts.fmt == BONSAI:
        body = "".join(f"{int(v)}\n" for v in raw)
    else:
        buf = ts.buffer[idx] if ts.buffer is not None else np.zeros(len(idx), dtype=np.int64)
        body = UCLA_HEADER + "\n" + "".join(
            f"{i},{int(r)},{int(b)}\n" for i, (r, b) in enumerate(zip(raw, buf))
        )
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(body)


def analysis_frame_count(rec) -> int:
    mc = rec.stages.get("mc")
    return int(mc.metrics.get("n_frames", rec.n_frames)) if mc is not None else rec.n_frames


def analysis_timestamps(paths, rec) -> Timestamps:
    return _after_mc(rec, session_timestamps(paths, rec))


def _after_mc(rec, timestamps) -> Timestamps:
    mc = rec.stages.get("mc")
    removed = (mc.metrics or {}).get("dropped_ranges") if mc is not None else None
    if not removed:
        return timestamps
    from bearmind.bad_frames import frame_mask

    keep = frame_mask(rec.n_frames, removed)[:len(timestamps.raw)]
    return dataclasses.replace(timestamps, raw=timestamps.raw[keep], t_ms=timestamps.t_ms[keep],
                               buffer=None if timestamps.buffer is None else timestamps.buffer[keep])


# Frame rates of the sessions processed before the pipeline took the rate from timestamp files ('#;Filename;FPS').
# The processed files of the paper corpus carry no timestamps, so the scripts that work on them (autoinspection
# called without fps, the validation, the research collectors) look the rate up here.
FPS_TABLE = Path(__file__).resolve().parent.parent / "fps_data.csv"


def read_fps_table(path=FPS_TABLE) -> dict[str, float]:
    """Name -> frame rate, as stored (not rounded). A name listed twice is refused: which row wins would be a
    property of the reader, not of the table."""
    import csv

    table = {}
    with open(path, newline="", encoding="utf-8-sig") as f:
        for line, row in enumerate(csv.DictReader(f, delimiter=";"), start=2):
            name = row["Filename"]
            if name in table:
                raise ValueError(f"{path}: {name} is listed twice (line {line})")
            table[name] = float(row["FPS"])
    return table


def fps_from_table(name: str, table: dict[str, float] | None = None) -> float | None:
    """Frame rate of the session `name` (a session ID, a file name or a path) from fps_data.csv; None when absent.

    Looked up as the whole name, then as its session ID (naming.extract_session_id, trial suffix kept), then as the
    ID without the trial suffix, as export_estimates_data's reader did: a trial without a row of its own takes its
    session's rate."""
    from bearmind.naming import extract_base_session, extract_session_id

    table = read_fps_table() if table is None else table
    for key in (name, extract_session_id(name), extract_base_session(name)):
        if key and key in table:
            return table[key]
    return None


def session_time_s(paths, rec, *, after_mc=True) -> np.ndarray:
    """The session's frame times in seconds from its first frame: the stitched timestamps of the registry
    (timestamps.session_timestamps), one per frame; rec.n_frames of them when the timestamps cover every frame."""
    raw = session_timestamps(paths, rec)
    timestamps = _after_mc(rec, raw) if after_mc else raw
    t = np.asarray(timestamps.t_ms, dtype=np.float64)
    return (t - raw.t_ms[0]) / 1000.0 if len(t) else t
