"""What the Scan tab shows, built from the registry without Qt. Reads only; writes nothing."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import config, scan
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
    text: str           # Anomaly.message as the scan wrote it
    short: str
    auto_fixed: bool
    reviewed: bool      # split_recording (and its pause) whose current stitch a person confirmed
    info: bool = False  # dropped_frames below scan.dropped_frames_warn: shown, not a problem
    note: str = ""      # said after the text in the details: "fixed automatically", "checked by you", ...
    data: dict = field(default_factory=dict)  # Anomaly.data (the sign of a mismatch for its confirmation)

    @property
    def counts(self) -> bool:
        """A problem of the row: colours it and counts in "with problems"."""
        return not (self.auto_fixed or self.reviewed or self.info)


@dataclass
class ScanRow:
    sid: str
    mouse: str
    mode: str
    n_parts: int | None  # None (and the other numbers): a record the view cannot read ("cannot show: ...")
    n_chunks: int | None
    n_video_frames: int | None
    n_frames: int | None
    fps: float | None
    duration_min: float | None
    level: str
    excluded: bool
    excluded_by: str  # "you", "anomaly"; "included": a person's include overrides an error; "" otherwise
    summary: str
    anomalies: list[AnomalyLine]
    folders: list[str]
    fps_text: str
    decisions: list[str]
    summary_parts: list[tuple[str, bool]] = field(default_factory=list)  # (text, greyed): the Anomalies cell
    folder: str = ""  # what Open folder opens: the first part, else the folder of an unreadable session


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
    n_not_found: int = 0  # rows of sessions whose folders the last scan did not find

    def summary_text(self) -> str:
        n, m = len(self.rows) - self.n_not_found, self.n_mice
        text = (f"Found {n} {'session' if n == 1 else 'sessions'} of {m} {'mouse' if m == 1 else 'mice'}"
                f" · bonsai {self.modes.get('bonsai', 0)} · legacy {self.modes.get('legacy', 0)}"
                f" · excluded {self.n_excluded} · with problems {self.n_problems}"
                f" · unrecognized folders {len(self.unrecognized)}")
        if self.n_not_found:
            text += f" · not found by the last scan {self.n_not_found}"
        if self.broken:
            text += f" · unreadable registry files {len(self.broken)}"
        return text


def dropped_rule(paths: SeriesPaths) -> dict:
    """scan.dropped_frames_warn of the series (defaults < series config).

    A series config that cannot be parsed gives the defaults here: the scan itself stops on it with the file
    named, and this rule only decides which lines are counted, not what the registry holds.
    """
    try:
        cfg = config.resolve(paths)
    except config.ConfigError:
        cfg = config.load_defaults()
    return cfg["scan"]["dropped_frames_warn"]


def _part_text(d: dict) -> str:
    """"at the end of part 1": the scan stores the 0-based part index, a person counts from 1."""
    if isinstance(d.get("part"), int):
        return f"at the end of part {d['part'] + 1}"
    where = str(d.get("where", "end"))
    if where.startswith("part ") and where.endswith(" end"):
        return f"at the end of part {int(where[5:-4]) + 1}"
    return "at the end"


def _short(a, n_parts: int, split_pause: bool) -> str:
    """Short name of the code with the numbers the scan stored in a.data."""
    title = ANOMALY_TITLE.get(a.code, a.code)
    d = a.data
    if a.code == "frames_ts_mismatch" and "diff" in d:
        return f"{title}: {d['diff']:+d} {_part_text(d)}"
    if a.code == "split_recording":
        gaps = [g for g in d.get("gaps_s", []) if g is not None]
        if a.level == "error":
            return f"{title}: {n_parts} parts, cannot stitch"
        return f"{title}: {n_parts} parts" + (f", pause {scan.duration_text(max(gaps) * 1000.0)}" if gaps else "")
    if a.code == "fps_outlier" and "fps" in d:
        # "median": a record scanned before the rule compared with the other sessions of the mouse
        ref = (f"other sessions {scan.rates_text(d['others'])}" if d.get("others")
               else f"mouse median {d['median']:.1f}" if "median" in d
               else f"nominal {d['nominal']:g}" if "nominal" in d else "")
        return f"{title}: {d['fps']:.1f}" + (f" ({ref})" if ref else "")
    if a.code == "dropped_frames" and "n_gaps" in d:
        text = f"{title}: {scan.plural(d['n_gaps'], 'gap')}"
        if d.get("max_gap_ms") is not None:
            text += f", longest {scan.duration_text(d['max_gap_ms'])}"
        return text + (" (the pause between parts)" if split_pause else "")
    if a.code == "chunk_tail_frames" and "surplus" in d:
        return f"{title}: {d['surplus']}"
    return title


def _dropped_counts(d: dict, n_frames: int, rule: dict) -> tuple[bool, str]:
    """(counts as a problem, the note of an information line) for a dropped_frames line.

    A problem when the longest gap exceeds rule max_gap_s or the frames lost exceed rule share of the session's
    frames. A record scanned before the scan estimated the lost frames (n_dropped) has the gap count only: each
    gap loses at least one frame, so a gap count above the share is enough, but below it the share is unknown.
    """
    gap_s, share = float(rule["max_gap_s"]), float(rule["share"])
    longest = d.get("max_gap_ms")
    if longest is None or longest > gap_s * 1000.0:  # no longest gap stored: keep the scan's warning
        return True, ""
    if "n_dropped" in d and n_frames:
        if d["n_dropped"] / n_frames > share:
            return True, ""
        return False, f"information: no gap longer than {gap_s:g} s, at most {share:.0%} of frames lost"
    if n_frames and d.get("n_gaps", 0) / n_frames > share:
        return True, ""
    return False, f"information: no gap longer than {gap_s:g} s; the share of lost frames is known after a rescan"


def _fps_text(rec: SessionRecord) -> str:
    info = rec.fps_info or {}
    fps = info.get("fps", rec.fps)
    if not fps:
        return "no frame rate"
    return (f"{fps:.1f} from {info.get('source', 'timestamps')}"
            f" · {scan.plural(info.get('n_outliers', 0), 'interval')} ignored in the FPS estimate"
            f" · longest pause {scan.duration_text(info.get('max_gap_ms', 0.0))}")


def _is_split_pause(a, gaps: list[float]) -> bool:
    """A dropped_frames line that only reports the pause between the stitched parts.

    The estimate counts each pause as a timestamp gap, and every pause exceeds the dropped-frames threshold,
    so a count above the number of pauses means a real drop is in the line and it stays unreviewed.
    """
    if a.code != "dropped_frames" or not gaps or a.data.get("n_gaps", 0) != len(gaps):
        return False
    return abs(a.data.get("max_gap_ms", 0.0) - max(gaps) * 1000.0) <= 0.01 * max(gaps) * 1000.0


def excluded_by(rec: SessionRecord) -> str:
    """Who excluded the session: "you" (a person's latest exclude decision), "anomaly" (an error of the scan, no
    fps, a folder not found); "included": a person's include overrides an error; "" otherwise. The Scan tab's
    Include and Monitor's Include (actions.include_targets) lift only "you"."""
    exclude = scan.latest_decision(rec, "exclude")
    if rec.excluded:
        return "you" if exclude == "true" else "anomaly"
    if exclude == "false" and any(a.level == "error" for a in rec.anomalies if not a.data.get("stage")):
        return "included"
    return ""


def _line(a, rec: SessionRecord, n_parts: int, split, gaps, confirmed: bool, rule: dict) -> AnomalyLine:
    pause = _is_split_pause(a, gaps)
    reviewed = confirmed and (a is split or pause)
    info, note = False, ""
    if a.auto_fixed:
        accepted = a.code == "frames_ts_mismatch" and scan.latest_decision(rec, "accept_mismatch") == "true"
        note = "accepted by you" if accepted else "fixed automatically"
    elif reviewed:
        note = "checked by you"
    elif a.code == "dropped_frames" and not pause:
        counts, note = _dropped_counts(a.data, rec.n_frames, rule)
        info = not counts
    return AnomalyLine(a.code, a.level, a.message, _short(a, n_parts, pause), a.auto_fixed, reviewed, info, note,
                       dict(a.data))


def _summary_parts(lines: list[AnomalyLine]) -> list[tuple[str, bool]]:
    """The Anomalies cell: problems first (errors before warnings), then information, then the lines fixed or
    checked, which are marked and greyed so they do not read as errors; at most three and "and N more"."""
    order = sorted(lines, key=lambda x: (0 if x.counts else 1 if x.info else 2, -_RANK.get(x.level, 1)))
    parts = []
    for x in order[:_SUMMARY_LINES]:
        if x.auto_fixed:
            parts.append((f"{x.short} ({'accepted' if x.note == 'accepted by you' else 'fixed'})", True))
        elif x.reviewed:
            parts.append((f"{x.short} (checked)", True))
        else:
            parts.append((x.short, False))
    if len(order) > _SUMMARY_LINES:
        parts.append((f"and {len(order) - _SUMMARY_LINES} more", False))
    return parts


def _build_row(rec: SessionRecord, rule: dict) -> ScanRow:
    n_parts = len(rec.parts)
    confirmed = scan.stitch_confirmed(rec)
    mine = [a for a in rec.anomalies if not a.data.get("stage")]
    split = next((a for a in mine if a.code == "split_recording" and a.level == "warning"), None)
    gaps = [g for g in (split.data.get("gaps_s", []) if split else []) if g is not None]
    lines = [_line(a, rec, n_parts, split, gaps, confirmed, rule) for a in mine]
    level = max((x.level for x in lines if x.counts), key=lambda lv: _RANK.get(lv, 1), default="ok")
    parts = _summary_parts(lines)
    unreadable = next((a.data["folder"] for a in mine if a.code == "unreadable_video" and a.data.get("folder")), "")
    return ScanRow(
        sid=rec.sid, mouse=rec.mouse, mode=rec.mode, n_parts=n_parts,
        n_chunks=sum(len(p.chunks) for p in rec.parts),
        n_video_frames=sum(sum(p.chunk_frames) for p in rec.parts),
        n_frames=rec.n_frames, fps=rec.fps,
        duration_min=rec.n_frames / rec.fps / 60 if rec.fps else None,
        level=level, excluded=rec.excluded, excluded_by=excluded_by(rec),
        summary="; ".join(t for t, _ in parts) if parts else "–", anomalies=lines,
        folders=[p.folder for p in rec.parts], fps_text=_fps_text(rec),
        decisions=[f"{d.when} {d.who}: {d.what} = {d.value}" for d in rec.decisions],
        summary_parts=parts, folder=rec.parts[0].folder if rec.parts else unreadable)


def _row(rec: SessionRecord, rule: dict) -> ScanRow:
    try:
        return _build_row(rec, rule)
    except Exception as e:  # one odd record (hand-edited, an older version) must not hide the whole table
        why = " ".join(f"{type(e).__name__}: {e}".split())
        try:
            who = excluded_by(rec)
        except Exception:  # the decisions are odd too: the Excluded column cannot say why
            who = ""
        text = f"cannot show: {why}"
        return ScanRow(
            sid=str(rec.sid), mouse=str(rec.mouse), mode=str(rec.mode), n_parts=None, n_chunks=None,
            n_video_frames=None, n_frames=None, fps=None, duration_min=None, level="error",
            excluded=bool(rec.excluded), excluded_by=who, summary=text, anomalies=[], folders=[],
            fps_text="", decisions=[], summary_parts=[(text, False)])


def scan_view(paths: SeriesPaths, cache=None) -> ScanView:
    """The tab's model. `cache`: a monitor.RecordCache, which rereads only the registry files that changed (the
    tab refreshes on a timer)."""
    records, broken = cache.records(paths) if cache is not None else read_all(paths)
    unrecognized = scan.read_unrecognized(paths)
    rule = dropped_rule(paths)
    rows = sorted((_row(r, rule) for r in records), key=lambda r: (r.mouse, r.sid))
    lost = {r.sid for r in rows if any(x.code == "not_found_in_scan" for x in r.anomalies)}
    found = [r for r in rows if r.sid not in lost]
    modes = {"bonsai": 0, "legacy": 0}
    for r in found:
        modes[r.mode] = modes.get(r.mode, 0) + 1
    return ScanView(
        series=paths.root, scanned=bool(records or broken) or paths.scan_unrecognized_csv.exists(),
        rows=rows, unrecognized=unrecognized, broken=broken,
        n_mice=len({r.mouse for r in found}), n_excluded=sum(r.excluded for r in rows),
        n_problems=sum(r.level != "ok" for r in rows), modes=modes, n_not_found=len(lost))
