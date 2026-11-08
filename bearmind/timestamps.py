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
    n_outliers: int
    n_gaps: int
    max_gap_ms: float
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
    d = np.diff(np.asarray(t_ms, dtype=np.float64))
    d = d[d > 0]
    if len(d) < 2:
        raise ValueError("not enough timestamps to estimate fps")
    m = float(np.median(d))
    keep = d[(d >= 0.5 * m) & (d <= 1.5 * m)]
    return FpsEstimate(
        fps=float(1000.0 / keep.mean()),
        n_outliers=int(len(d) - len(keep)),
        n_gaps=int(np.sum(d > gap_factor * m)),
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
