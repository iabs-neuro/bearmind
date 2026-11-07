"""Monitor model: what the window shows, built from the registry, heartbeat and configs. No Qt.

Read-only: it never writes. A record that cannot be read is a row marked unreadable, never an exception,
because the batch rewrites registry files while the window reads them.
"""
import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psutil

from bearmind import config, fileio, registry
from bearmind.registry import STAGES, SessionRecord, StageRecord, State
from bearmind.runner import STAGE_KEYS, wait_reason

STAGE_TITLE = {"crop": "Crop", "mc": "MC", "thresholds": "Thresholds", "cnmf": "CNMF-E",
               "gate": "Inspection", "autoinspect": "Autoinspect", "export": "Export"}

HINTS = {
    "mc_shift": "Check the MC preview.",
    "stage_failed": "Other sessions continue.",
    "no_progress": "stalled: no CPU progress",
    "pool_worker_died": "a CaImAn worker died",
}

# check name -> panel kind; every other check and anomaly only colours its cell
_CHECK_KIND = {"mc_max_shift_px": "mc_shift", "cnmf_n_components_vs_median": "cnmf_count",
               "auto_deleted_frac": "autoinspect_deleted", "export_n_neurons": "export_count"}
_SCAN_CODES = ("frames_ts_mismatch", "fps_outlier")
_LEVEL = {"ok": 0, "warning": 1, "error": 2}
_CREATE_TIME_TOL_S = 1.0


@dataclass
class RunnerState:
    alive: bool
    status: str  # "running" | "waiting" | "stopped" | "dead"
    pid: int | None
    age_s: float | None
    current: tuple[str, str] | None
    stage_elapsed_s: float | None


@dataclass
class Cell:
    state: str  # a State value or "unreadable"
    level: str  # "ok" | "warning" | "error" | "running" | "waiting" | "none" | "excluded"
    text: str
    tooltip: str


@dataclass
class Row:
    sid: str
    mouse: str
    excluded: bool
    cells: dict[str, Cell]
    unreadable: bool = False


@dataclass
class StageBar:
    done: int
    running: int
    waiting: int  # waits for a person: thresholds and gate only
    failed: int
    pending: int  # not started, skipped, or waiting for the core (crop_finalize)
    total: int


@dataclass
class WarningItem:
    level: str
    kind: str
    sid: str
    stage: str
    text: str
    retry: bool


@dataclass
class SeriesView:
    series: Path
    scanned: bool
    runner: RunnerState
    rows: list[Row]
    bars: dict[str, StageBar]
    eta_s: float | None
    waiting: dict[str, int]
    warnings: list[WarningItem]
    n_warning_sessions: int
    unreadable_files: list[Path] = field(default_factory=list)


# --- reading with a cache ---

def _sig(f: Path):
    try:
        st = f.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


class RecordCache:
    """Rereads only the registry files whose mtime or size changed, and resolves a session's thresholds
    only when its record or one of its three config files changed: resolving reparses defaults and merges
    four layers, too slow for every session on every refresh."""

    def __init__(self):
        self._files: dict[Path, tuple] = {}  # file -> (sig, record or None, reason)
        self._th: dict[str, tuple] = {}  # sid -> (key, (gSig, min_corr, min_pnr))

    def records(self, paths) -> tuple[list[SessionRecord], list[tuple[Path, str]]]:
        folder = paths.registry_file("x").parent
        if not folder.is_dir():
            self._files.clear()
            return [], []
        out, broken, seen = [], [], set()
        for f in sorted(folder.glob("*.json")):
            sig = _sig(f)
            if sig is None:  # removed since the listing
                continue
            seen.add(f)
            hit = self._files.get(f)
            if hit is None or hit[0] != sig:
                try:
                    hit = (sig, registry.read_file(f), None)
                except (OSError, ValueError, KeyError, TypeError, AttributeError) as e:
                    hit = (sig, None, " ".join(f"{type(e).__name__}: {e}".split()))
                self._files[f] = hit
            if hit[1] is not None:
                out.append(hit[1])
            else:
                broken.append((f, hit[2]))
        for f in set(self._files) - seen:
            del self._files[f]
        return sorted(out, key=lambda r: r.sid), broken

    def _thresholds(self, paths, rec) -> tuple:
        # id(rec): the cached record object is replaced whenever its file is reread
        key = (id(rec), _sig(paths.series_config), _sig(paths.mouse_config(rec.mouse)),
               _sig(paths.session_config(rec.sid)))
        hit = self._th.get(rec.sid)
        if hit is None or hit[0] != key:
            try:
                c = config.resolve(paths, sid=rec.sid, mouse=rec.mouse)["cnmf"]
                val = (c.get("gSig"), c.get("min_corr"), c.get("min_pnr"))
            except Exception:  # a broken config is the core's "cannot be scheduled", not the window's crash
                val = (None, None, None)
            hit = (key, val)
            self._th[rec.sid] = hit
        return hit[1]

    def gsig(self, paths, rec) -> int | None:
        return self._thresholds(paths, rec)[0]

    def thresholds_unset(self, paths, rec) -> bool:
        return any(v is None for v in self._thresholds(paths, rec))


# --- runner state ---

def _iso(s):
    try:
        return datetime.fromisoformat(s)
    except (TypeError, ValueError):
        return None


def _read_json(f: Path):
    try:
        d = json.loads(fileio.read_text(f))  # retried: replaced while read
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def _holds_lock(lock: dict | None) -> int | None:
    """PID of the lock's process when it is alive and is the process that wrote the lock (a reused PID has
    another start time)."""
    if not lock or not isinstance(lock.get("pid"), int):
        return None
    try:
        p = psutil.Process(lock["pid"])
        if not p.is_running() or p.status() == psutil.STATUS_ZOMBIE:
            return None
        created = lock.get("create_time")
        if created is not None and abs(p.create_time() - float(created)) > _CREATE_TIME_TOL_S:
            return None
    except (psutil.Error, TypeError, ValueError):
        return None
    return lock["pid"]


def _stage_elapsed(paths, cur, now):
    try:
        st = registry.read_file(paths.registry_file(cur[0])).stages.get(cur[1])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    t = _iso(st.started) if st is not None else None
    return None if t is None else max(0.0, (now - t).total_seconds())


def runner_state(paths, now: datetime | None = None) -> RunnerState:
    now = now or datetime.now()
    limit = config.load_defaults().get("monitor", {}).get("heartbeat_dead_s", 120)
    hb = _read_json(paths.heartbeat)
    t = _iso(hb.get("time")) if hb else None
    age = None if t is None else (now - t).total_seconds()
    pid = _holds_lock(_read_json(paths.lock))
    pulse = (hb or {}).get("status")
    if pid is not None and age is not None and age < limit and pulse in ("running", "waiting"):
        cur = hb.get("current")
        cur = (cur["sid"], cur["stage"]) if isinstance(cur, dict) and "sid" in cur and "stage" in cur else None
        elapsed = _stage_elapsed(paths, cur, now) if cur else None
        return RunnerState(True, pulse, pid, age, cur, elapsed)
    # a batch that is not alive but left a running/waiting pulse died without closing it
    status = "dead" if pulse in ("running", "waiting") else "stopped"
    return RunnerState(False, status, pid, age, None, None)


# --- cells ---

def _num(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _cell_text(rec, stage, st, cache, paths) -> str:
    if stage == "crop":
        box = STAGE_KEYS["crop"][0][1](rec)
        if box is not None:
            return box if box == "uncropped" else box.split("@")[0].replace("x", "×")
        b = st.metrics.get("box")  # [top, left, height, width] as the crop stage records it
        return f"{b[3]}×{b[2]}" if b and len(b) == 4 else "done"
    m = st.metrics
    if stage == "mc":
        v = m.get("max_shift_px")
        return "done" if v is None else f"{v:.1f} px"
    if stage == "thresholds":
        g = cache.gsig(paths, rec)
        return "done" if g is None else f"gSig {_num(g)}"
    if stage == "cnmf":
        v = m.get("n_components")
        return "done" if v is None else f"{_num(v)} cells"
    if stage == "gate":
        return {"cleaned": "cleaned", "not_needed": "not needed", "disabled": "off"}.get(m.get("gate"), "done")
    if stage == "autoinspect":
        v = m.get("n_kept")
        return "done" if v is None else f"{_num(v)} kept"
    v = m.get("n_neurons")
    return "done" if v is None else f"{_num(v)} neurons"


def _live_anomalies(rec, stage):
    return [a for a in rec.anomalies if a.data.get("stage") == stage and not a.auto_fixed]


def _worse(levels) -> str:
    levels = [x if x in _LEVEL else "warning" for x in levels]
    return max(levels, key=_LEVEL.get) if levels else "ok"


def _last_line(text) -> str:
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    return lines[-1] if lines else ""


def _cell(rec, stage, cache, paths, human_thresholds) -> Cell:
    title = STAGE_TITLE[stage]
    st = rec.stages.get(stage) or StageRecord()
    if rec.excluded:
        return Cell(st.state.value, "excluded", "–", f"{title}: session excluded")
    lines = [c.message for c in st.checks] if st.state is State.DONE else []
    lines += [a.message for a in _live_anomalies(rec, stage)]
    if st.state is State.DONE:
        level = _worse([c.level for c in st.checks] + [a.level for a in _live_anomalies(rec, stage)])
        text = _cell_text(rec, stage, st, cache, paths)
    elif st.state is State.FAILED:
        level, text = "error", "failed"
        lines.insert(0, _last_line(st.error) or "failed")
    elif st.state is State.WAITING:
        level, text = "waiting", "waits"
        lines.insert(0, f"waiting for {wait_reason(st) or 'input'}")
    elif st.state is State.RUNNING:
        level, text = "running", "running"
    else:
        level, text = "none", "–"
    if stage == "thresholds" and human_thresholds:
        level, text = "waiting", "waits"
        lines.append("waiting for your thresholds")
    return Cell(st.state.value, level, text, "\n".join([f"{title}: {st.state.value}", *lines]))


def _unreadable_row(f: Path, why: str) -> Row:
    tip = f"registry file cannot be read: {f}: {why}"
    return Row(f.stem, "", False, {s: Cell("unreadable", "error", "?", tip) for s in STAGES}, unreadable=True)


# --- waits, warnings, bars, eta ---

def _human_thresholds(paths, rec, cache) -> bool:
    cnmf = rec.stages.get("cnmf") or StageRecord()
    if cnmf.state is State.WAITING and wait_reason(cnmf) == "thresholds":
        return True
    th = rec.stages.get("thresholds") or StageRecord()
    # run --until thresholds leaves CNMF-E PENDING with a suggestion and no thresholds set
    return (cnmf.state is State.PENDING and th.state is State.DONE and bool(th.metrics.get("suggested"))
            and cache.thresholds_unset(paths, rec))


def _human_gate(rec) -> bool:
    g = rec.stages.get("gate") or StageRecord()
    return g.state is State.WAITING and wait_reason(g) != "crop_finalize"


def _warnings(rec) -> list[WarningItem]:
    out = []
    scan = [a for a in rec.anomalies if a.code in _SCAN_CODES and not a.auto_fixed]
    if scan:
        out.append(WarningItem(_worse([a.level for a in scan]), "scan_data", rec.sid, "",
                               " ; ".join(a.message for a in scan), False))
    if rec.excluded:  # the scan anomalies explain the exclusion; nothing else is shown
        return out
    for stage in STAGES:
        st = rec.stages.get(stage) or StageRecord()
        title = STAGE_TITLE[stage]
        if st.state is State.FAILED:
            crash = next((a for a in rec.anomalies if a.code == "stage_crashed"
                          and a.data.get("stage") == stage and a.data.get("cause") in HINTS), None)
            what = HINTS[crash.data["cause"]] if crash else (_last_line(st.error) or "failed")
            out.append(WarningItem("error", "stage_failed", rec.sid, stage,
                                   f"{title}: {what}. {HINTS['stage_failed']}", True))
        elif st.state is State.DONE:
            bad = [c for c in st.checks if c.name in _CHECK_KIND and c.level != "ok"]
            for kind in dict.fromkeys(_CHECK_KIND[c.name] for c in bad):
                cs = [c for c in bad if _CHECK_KIND[c.name] == kind]
                text = f"{title}: " + " ; ".join(c.message for c in cs)
                if kind in HINTS:
                    text += f". {HINTS[kind]}"
                out.append(WarningItem(_worse([c.level for c in cs]), kind, rec.sid, stage, text, False))
    return out


def _duration(st):
    a, b = _iso(st.started), _iso(st.finished)
    return None if a is None or b is None else (b - a).total_seconds()


def _s_per_frame(records, defaults) -> dict:
    out = {}
    for stage in STAGES:
        hist = []
        for r in records:
            st = r.stages.get(stage)
            d = _duration(st) if st is not None and st.state is State.DONE else None
            if d is not None and r.n_frames > 0:
                hist.append(d / r.n_frames)
        if len(hist) >= 2:
            out[stage] = statistics.median(hist)
        elif stage in defaults:
            out[stage] = defaults[stage]
    return out


def _eta(records, human, now, per_frame):
    total, any_est = 0.0, False
    for r in records:
        if r.excluded:
            continue
        cut = len(STAGES)  # stages from a person's wait on cannot start until the person decides
        if human[r.sid][0]:
            cut = min(cut, STAGES.index("cnmf"))
        if human[r.sid][1]:
            cut = min(cut, STAGES.index("gate"))
        for stage in STAGES[:cut]:
            st = r.stages.get(stage) or StageRecord()
            # WAITING here is the core's own wait (crop_finalize): it ends without a person
            if st.state not in (State.PENDING, State.RUNNING, State.WAITING) or stage not in per_frame:
                continue
            t = per_frame[stage] * r.n_frames
            started = _iso(st.started)
            if st.state is State.RUNNING and started is not None:
                t = max(0.0, t - (now - started).total_seconds())
            total += t
            any_est = True
    return total if any_est else None


def _bars(records, human) -> dict:
    live = [r for r in records if not r.excluded]
    bars = {}
    for stage in STAGES:
        b = StageBar(0, 0, 0, 0, 0, len(live))
        for r in live:
            st = (r.stages.get(stage) or StageRecord()).state
            if (stage == "thresholds" and human[r.sid][0]) or (stage == "gate" and human[r.sid][1]):
                b.waiting += 1
            elif st is State.DONE:
                b.done += 1
            elif st is State.RUNNING:
                b.running += 1
            elif st is State.FAILED:
                b.failed += 1
            else:
                b.pending += 1
        bars[stage] = b
    return bars


def series_view(paths, now: datetime | None = None, cache: RecordCache | None = None) -> SeriesView:
    now = now or datetime.now()
    cache = cache or RecordCache()
    runner = runner_state(paths, now)
    records, broken = cache.records(paths)
    if not records and not broken:
        return SeriesView(paths.root, False, runner, [], {s: StageBar(0, 0, 0, 0, 0, 0) for s in STAGES},
                          None, {"thresholds": 0, "manual_inspection": 0}, [], 0, [])
    human = {r.sid: (_human_thresholds(paths, r, cache), _human_gate(r)) for r in records}
    rows = [Row(r.sid, r.mouse, r.excluded,
                {s: _cell(r, s, cache, paths, human[r.sid][0]) for s in STAGES}) for r in records]
    rows += [_unreadable_row(f, why) for f, why in broken]
    rows.sort(key=lambda x: (x.mouse, x.sid))
    live = [r for r in records if not r.excluded]
    waiting = {"thresholds": sum(human[r.sid][0] for r in live),
               "manual_inspection": sum(human[r.sid][1] for r in live)}
    warnings = [w for r in records for w in _warnings(r)]
    order = {s: i for i, s in enumerate(("",) + STAGES)}
    warnings.sort(key=lambda w: (w.level != "error", w.sid, order[w.stage]))
    per_frame = _s_per_frame(records, config.load_defaults().get("monitor", {}).get("eta_s_per_frame", {}))
    return SeriesView(paths.root, True, runner, rows, _bars(records, human),
                      _eta(records, human, now, per_frame), waiting, warnings,
                      len({w.sid for w in warnings}), [f for f, _ in broken])
