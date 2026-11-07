"""What the Scan tab shows, built from the registry without Qt. Reads only; writes nothing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import scan
from .paths import SeriesPaths
from .registry import SessionRecord, read_all

ANOMALY_TITLE = {
    "frames_ts_mismatch": "frames ≠ timestamps",
    "split_recording": "split recording",
    "no_timestamps": "no timestamps found",
    "fps_outlier": "FPS outlier",
    "dropped_frames": "dropped frames",
    "broken_last_frame": "broken last frame",
    "chunk_tail_frames": "extra chunk frames",
    "no_metadata": "no recording start time",
    "unreadable_video": "unreadable video",
    "fps_unavailable": "no frame rate",
    "registry_recreated": "registry file recreated",
    "not_found_in_scan": "not found by the last scan",
}

_RANK = {"ok": 0, "warning": 1, "error": 2}
_SUMMARY_LINES = 3


@dataclass
class AnomalyLine:
    code: str
    level: str
    text: str
    short: str
    auto_fixed: bool
    reviewed: bool


@dataclass
class ScanRow:
    sid: str
    mouse: str
    mode: str
    n_parts: int
    n_chunks: int
    n_video_frames: int
    n_frames: int
    fps: float | None
    duration_min: float | None
    level: str
    excluded: bool
    excluded_by: str
    summary: str
    anomalies: list[AnomalyLine]
    folders: list[str]
    fps_text: str
    decisions: list[str]


@dataclass
class ScanView:
    series: Path
    scanned: bool
    rows: list[ScanRow]
    unrecognized: list[tuple[str, str]]
    broken: list[tuple[Path, str]]
    n_mice: int
    n_excluded: int
    n_problems: int
    modes: dict[str, int]

    def summary_text(self) -> str:
        n, m = len(self.rows), self.n_mice
        return (f"Found {n} {'session' if n == 1 else 'sessions'} of {m} {'mouse' if m == 1 else 'mice'}"
                f" · bonsai {self.modes.get('bonsai', 0)} · legacy {self.modes.get('legacy', 0)}"
                f" · excluded {self.n_excluded} · with problems {self.n_problems}"
                f" · unrecognized folders {len(self.unrecognized)}")


def _pause_text(seconds: float) -> str:
    return f"{seconds:.0f} s" if abs(seconds) < 120 else f"{seconds / 60:.0f} min"


def _short(a, n_parts: int) -> str:
    """Short name of the code with the numbers the scan stored in a.data."""
    title = ANOMALY_TITLE.get(a.code, a.code)
    d = a.data
    if a.code == "frames_ts_mismatch" and "diff" in d:
        where = str(d.get("where", "end"))
        if where.startswith("part ") and where.endswith(" end"):
            where = f"at the end of part {where[5:-4]}"
        else:
            where = "at the end"
        return f"{title}: {d['diff']:+d} {where}"
    if a.code == "split_recording":
        gaps = [g for g in d.get("gaps_s", []) if g is not None]
        if a.level == "error":
            return f"{title}: {n_parts} parts, cannot stitch"
        return f"{title}: {n_parts} parts" + (f", pause {_pause_text(max(gaps))}" if gaps else "")
    if a.code == "fps_outlier" and "fps" in d:
        ref = (f"mouse median {d['median']:.1f}" if "median" in d
               else f"nominal {d['nominal']:g}" if "nominal" in d else "")
        return f"{title}: {d['fps']:.1f}" + (f" ({ref})" if ref else "")
    if a.code == "dropped_frames" and "n_gaps" in d:
        return f"{title}: {d['n_gaps']} gaps, longest {d.get('max_gap_ms', 0):.0f} ms"
    if a.code == "chunk_tail_frames" and "surplus" in d:
        return f"{title}: {d['surplus']}"
    return title


def _fps_text(rec: SessionRecord) -> str:
    info = rec.fps_info or {}
    fps = info.get("fps", rec.fps)
    if not fps:
        return "no frame rate"
    return (f"{fps:.1f} from {info.get('source', 'timestamps')} · {info.get('n_outliers', 0)} intervals dropped"
            f" · longest pause {info.get('max_gap_ms', 0.0) / 1000.0:.2f} s")


def _is_split_pause(a, gaps: list[float]) -> bool:
    """A dropped_frames line that only reports the pause between the stitched parts.

    The estimate counts each pause as a timestamp gap, and every pause exceeds the dropped-frames threshold,
    so a count above the number of pauses means a real drop is in the line and it stays unreviewed.
    """
    if a.code != "dropped_frames" or not gaps or a.data.get("n_gaps", 0) != len(gaps):
        return False
    return abs(a.data.get("max_gap_ms", 0.0) - max(gaps) * 1000.0) <= 0.01 * max(gaps) * 1000.0


def _row(rec: SessionRecord) -> ScanRow:
    n_parts = len(rec.parts)
    confirmed = any(d.what == "stitch_confirmed" and d.value == "true" for d in rec.decisions)
    mine = [a for a in rec.anomalies if not a.data.get("stage")]
    split = next((a for a in mine if a.code == "split_recording" and a.level == "warning"), None)
    gaps = [g for g in (split.data.get("gaps_s", []) if split else []) if g is not None]
    lines = [AnomalyLine(a.code, a.level, a.message, _short(a, n_parts), a.auto_fixed,
                         confirmed and (a is split or _is_split_pause(a, gaps)))
             for a in mine]
    level = max((x.level for x in lines if not x.auto_fixed and not x.reviewed),
                key=lambda lv: _RANK.get(lv, 1), default="ok")
    shown = [x.short for x in lines[:_SUMMARY_LINES]]
    if len(lines) > _SUMMARY_LINES:
        shown.append(f"and {len(lines) - _SUMMARY_LINES} more")
    exclude = scan.latest_decision(rec, "exclude")
    excluded_by = "" if not rec.excluded else "you" if exclude == "true" else "anomaly"
    return ScanRow(
        sid=rec.sid, mouse=rec.mouse, mode=rec.mode, n_parts=n_parts,
        n_chunks=sum(len(p.chunks) for p in rec.parts),
        n_video_frames=sum(sum(p.chunk_frames) for p in rec.parts),
        n_frames=rec.n_frames, fps=rec.fps,
        duration_min=rec.n_frames / rec.fps / 60 if rec.fps else None,
        level=level, excluded=rec.excluded, excluded_by=excluded_by,
        summary="; ".join(shown) if shown else "–", anomalies=lines,
        folders=[p.folder for p in rec.parts], fps_text=_fps_text(rec),
        decisions=[f"{d.when} {d.who}: {d.what} = {d.value}" for d in rec.decisions])


def scan_view(paths: SeriesPaths) -> ScanView:
    records, broken = read_all(paths)
    unrecognized = scan.read_unrecognized(paths)
    rows = sorted((_row(r) for r in records), key=lambda r: (r.mouse, r.sid))
    modes = {"bonsai": 0, "legacy": 0}
    for r in rows:
        modes[r.mode] = modes.get(r.mode, 0) + 1
    return ScanView(
        series=paths.root, scanned=bool(records or broken) or paths.scan_unrecognized_csv.exists(),
        rows=rows, unrecognized=unrecognized, broken=broken,
        n_mice=len({r.mouse for r in rows}), n_excluded=sum(r.excluded for r in rows),
        n_problems=sum(r.level != "ok" for r in rows), modes=modes)
