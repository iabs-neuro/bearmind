"""Timestamp files of miniscope recordings: reading, FPS estimate, stitching, writing."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

import numpy as np

BONSAI = "bonsai_ticks"
UCLA = "ucla_ms"
UCLA_HEADER = "Frame Number,Time Stamp (ms),Buffer Index"


@dataclass
class Timestamps:
    raw: np.ndarray
    t_ms: np.ndarray
    fmt: str
    start_epoch_ms: float | None = None
    buffer: np.ndarray | None = None


@dataclass
class FpsEstimate:
    fps: float
    n_outliers: int  # uneven intervals, outside [0.5 m, 1.5 m] of the median m; counted in fps (information only)
    n_gaps: int  # intervals above gap_factor * m: pauses, dropped frames; left out of fps
    max_gap_ms: float  # the longest interval, a gap or not
    median_interval_ms: float


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
        t = raw.astype(np.float64)
        if start_epoch_ms is not None:
            t = t + start_epoch_ms
        return Timestamps(raw, t, UCLA, start_epoch_ms, buf)
    try:
        raw = np.array([int(ln) for ln in lines], dtype=np.int64)
    except ValueError:
        raise ValueError(f"unknown timestamp format: {path}") from None
    return Timestamps(raw, raw / 1e4, BONSAI)


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
