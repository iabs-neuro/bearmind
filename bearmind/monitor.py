"""Monitor model: what the window shows, built from the registry, heartbeat and configs. No Qt.

Read-only: it never writes. A record that cannot be read is a row marked unreadable, never an exception,
because the batch rewrites registry files while the window reads them.
"""
import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import psutil

from bearmind import actions, config, fileio, registry, runner
from bearmind.registry import ALL_STAGES, SIDE_STAGES, STAGES, SessionRecord, StageRecord, State
from bearmind.runner import wait_reason

STAGE_TITLE = {"crop": "Crop", "mc": "MC", "thresholds": "Thresholds", "cnmf": "CNMF-E",
               "gate": "Inspection", "autoinspect": "Autoinspect", "export": "Export",
               "mc_preview": "MC preview", "cnmf_preview": "CNMF-E preview"}

HINTS = {
    "mc_shift": "Check the MC preview.",
    "stage_failed": "Other sessions continue.",
    "no_progress": "stalled: no CPU progress",
    "pool_worker_died": "a CaImAn worker died",
    # shown in the CNMF-E cell: the panel keeps its six kinds
    "cnmf_downsampled": "Background downsampled to fit RAM; a tighter crop avoids it.",
}
_SIDE_OF = {main: side for side, main in SIDE_STAGES.items()}  # MC -> its preview, CNMF-E -> its QC video

# check name -> panel kind; every other check and anomaly only colours its cell
_CHECK_KIND = {"mc_max_shift_px": "mc_shift", "cnmf_n_components_vs_median": "cnmf_count",
               "auto_deleted_frac": "autoinspect_deleted", "export_n_neurons": "export_count"}
_SCAN_CODES = ("frames_ts_mismatch", "fps_outlier")
_LEVEL = {"ok": 0, "warning": 1, "error": 2}
_CREATE_TIME_TOL_S = 1.0
_PARSE_ERRORS = (ValueError, KeyError, TypeError, AttributeError)  # a registry file that is not a record


@dataclass
class RunnerState:
    alive: bool
    status: str  # "running" | "waiting" | "stopped" | "dead"
    pid: int | None
    age_s: float | None
    current: tuple[str, str] | None
    stage_elapsed_s: float | None
    exit: str | None = None  # why the last run ended (its final heartbeat); None: unknown or not ended
    stopping: bool = False  # a stop request waits for the running stage to end
    unschedulable: dict | None = None  # sid -> why it cannot be scheduled, from the live runner's heartbeat
    heartbeat_time: datetime | None = None


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
    problem: str | None = None  # why the batch cannot schedule the session; the row is red


@dataclass
class StageBar:
    done: int
    running: int
    waiting: int  # waits for a person: thresholds and gate only
    failed: int  # failed, or not done in a session that cannot be scheduled (both need a person)
    pending: int  # not started, skipped, interrupted, or waiting for the core (crop_finalize)
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
    eta_s: float | None  # 0.0: nothing left for the batch; None: something left that has no estimate
    waiting: dict[str, int]
    warnings: list[WarningItem]
    n_warning_sessions: int
    unreadable_files: list[Path] = field(default_factory=list)
    found_nothing: bool = False  # not scanned: the scan ran (it wrote its unrecognized list) but found no session


# --- reading with a cache ---

def _sig(f: Path):
    try:
        st = f.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _why(e: Exception) -> str:
    return " ".join(f"{type(e).__name__}: {e}".split())


class RecordCache:
    """Rereads only the registry files whose mtime or size changed, and recomputes what a session needs from
    its configs (thresholds, "cannot be scheduled", "waits for manual inspection") only when its record or
    one of its config files changed: resolving reparses defaults and merges four layers, too slow for every
    session on every refresh."""

    def __init__(self):
        self._files: dict[Path, tuple] = {}  # file -> (sig, record or None, reason)
        self._rec_sig: dict[str, tuple] = {}  # sid -> sig of the file its record was read from
        self._th: dict[str, tuple] = {}  # sid -> (key, (gSig, min_corr, min_pnr))
        self._problem: dict[str, tuple] = {}  # sid -> (key, why or None)
        self._gate: dict[str, tuple] = {}  # sid -> (key, waits for manual inspection)

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
                except OSError as e:
                    # busy (the batch replaces it; read_text has retried) or gone: not a broken record, read
                    # again on the next refresh; meanwhile the record read last, if any
                    if hit is not None and hit[1] is not None:
                        out.append(hit[1])
                        self._rec_sig[hit[1].sid] = hit[0]
                    else:
                        broken.append((f, _why(e)))
                    continue
                except _PARSE_ERRORS as e:
                    hit = (sig, None, _why(e))
                self._files[f] = hit
            if hit[1] is not None:
                out.append(hit[1])
                self._rec_sig[hit[1].sid] = hit[0]
            else:
                broken.append((f, hit[2]))
        for f in set(self._files) - seen:
            del self._files[f]
        return sorted(out, key=lambda r: r.sid), broken

    def _config_key(self, paths, rec, sigs=None, record=True) -> tuple:
        # the three config files the session's settings come from (and its record file); an object id would
        # be reused by a later record and serve a stale value. `sigs` shares the stat calls within a refresh.
        sigs = {} if sigs is None else sigs
        files = (paths.series_config, paths.mouse_config(rec.mouse), paths.session_config(rec.sid))
        for f in files:
            if f not in sigs:
                sigs[f] = _sig(f)
        own = (self._rec_sig.get(rec.sid),) if record else ()
        return (rec.sid, rec.mouse) + own + tuple(sigs[f] for f in files)

    def _thresholds(self, paths, rec) -> tuple:
        key = self._config_key(paths, rec, record=False)  # the thresholds live in the configs only
        hit = self._th.get(rec.sid)
        if hit is None or hit[0] != key:
            try:
                cfg = config.resolve(paths, sid=rec.sid, mouse=rec.mouse)
                c = cfg["cnmf"]
                # one flag (preview.enabled) drives both preview stages today, so mc_preview speaks for both
                val = (c.get("gSig"), c.get("min_corr"), c.get("min_pnr"), runner.side_enabled("mc_preview", cfg))
            except Exception:  # a broken config is the core's "cannot be scheduled", not the window's crash
                val = (None, None, None, None)
            hit = (key, val)
            self._th[rec.sid] = hit
        return hit[1]

    def gsig(self, paths, rec) -> int | None:
        return self._thresholds(paths, rec)[0]

    def thresholds_unset(self, paths, rec) -> bool:
        return any(v is None for v in self._thresholds(paths, rec)[:3])

    def preview_on(self, paths, rec) -> bool | None:
        """The session's preview checkbox (preview.enabled); None when its config cannot be read."""
        return self._thresholds(paths, rec)[3]

    def problems(self, paths, records) -> dict:
        """sid -> why the runner cannot schedule the session (runner.schedule_problems, the runner's own
        function), for the non-excluded sessions."""
        sigs, out, live = {}, {}, set()
        for rec in records:
            if rec.excluded:
                continue
            live.add(rec.sid)
            key = self._config_key(paths, rec, sigs)
            hit = self._problem.get(rec.sid)
            if hit is None or hit[0] != key:
                hit = (key, runner.schedule_problems(paths, [rec]).get(rec.sid))
                self._problem[rec.sid] = hit
            if hit[1] is not None:
                out[rec.sid] = hit[1]
        for sid in set(self._problem) - live:
            del self._problem[sid]
        return out

    def gate_waits(self, paths, rec) -> bool:
        """actions.gate_waits: the set "Inspection not needed for all waiting" writes. Its manual result file
        is in the key: saving the cleaned file can end the wait."""
        if not actions.gate_may_wait(rec):
            return False
        key = self._config_key(paths, rec) + (_sig(paths.manual_result(rec.sid)),)
        hit = self._gate.get(rec.sid)
        if hit is None or hit[0] != key:
            try:
                waits = actions.gate_waits(paths, rec)
            except Exception:  # e.g. a decision time that does not parse: the batch reports it, not the window
                waits = False
            hit = (key, waits)
            self._gate[rec.sid] = hit
        return hit[1]


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


# pid -> (psutil.Process, its start time or None): psutil's start time costs tens of ms when it falls back to
# the slow query, and a PID's start time never changes. One entry: the current lock holder.
_LOCK_PROC: dict[int, tuple] = {}
# (pid, create_time) of the lock whose process is gone: it never comes back (a new runner writes a new lock), and
# psutil needs a system-wide query (~60 ms on Windows) to report a dead PID, on every refresh otherwise
_LOCK_GONE: set = set()


def _holds_lock(lock: dict | None) -> int | None:
    """PID of the lock's process when it is alive and is the process that wrote the lock (a reused PID has
    another start time)."""
    if not lock or not isinstance(lock.get("pid"), int):
        _LOCK_PROC.clear()
        return None
    pid = lock["pid"]
    key = (pid, str(lock.get("create_time")))  # str: a hand-edited lock may hold anything
    if key in _LOCK_GONE:
        return None
    _LOCK_GONE.clear()  # another lock now
    for other in [p for p in _LOCK_PROC if p != pid]:  # the lock changed hands
        del _LOCK_PROC[other]
    try:
        hit = _LOCK_PROC.get(pid)
        if hit is None:
            p = psutil.Process(pid)
            try:
                started = p.create_time()
            except psutil.AccessDenied:  # another user's process on a reused PID: never our runner
                started = None
            hit = _LOCK_PROC[pid] = (p, started)
        p, started = hit
        # is_running compares the PID's current start time with the cached process's: False when reused.
        # Zombies exist on POSIX only, and status() costs a system-wide query on Windows.
        if not p.is_running() or (os.name != "nt" and p.status() == psutil.STATUS_ZOMBIE):
            _LOCK_PROC.pop(pid, None)
            _LOCK_GONE.add(key)
            return None
        created = lock.get("create_time")
        if started is None or (created is not None and abs(started - float(created)) > _CREATE_TIME_TOL_S):
            return None
    except psutil.NoSuchProcess:
        _LOCK_PROC.pop(pid, None)
        _LOCK_GONE.add(key)
        return None
    except (psutil.Error, TypeError, ValueError):
        _LOCK_PROC.pop(pid, None)
        return None
    return pid


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
        un = hb.get("unschedulable")
        return RunnerState(True, pulse, pid, age, cur, elapsed, stopping=paths.stop_request.exists(),
                           unschedulable=un if isinstance(un, dict) else None, heartbeat_time=t)
    # a batch that is not alive but left a running/waiting pulse died without closing it
    status = "dead" if pulse in ("running", "waiting") else "stopped"
    why = hb.get("exit") if hb and status == "stopped" else None
    return RunnerState(False, status, pid, age, None, None, exit=why if isinstance(why, str) else None,
                       heartbeat_time=t)


# --- cells ---

def _num(x) -> str:
    return f"{x:g}" if isinstance(x, float) else str(x)


def _cell_text(rec, stage, st, cache, paths) -> str:
    if stage == "crop":
        # width × height (as the mockup's 512×480): the box MC used once MC ran, the placed box before; both
        # are recorded as [top, left, height, width]
        mc = rec.stages.get("mc")
        if mc is not None and mc.state is State.DONE and "crop" in mc.metrics:
            b = mc.metrics["crop"]
            if b is None:
                return "uncropped"
        else:
            b = st.metrics.get("box")
        return f"{b[3]}×{b[2]}" if isinstance(b, (list, tuple)) and len(b) == 4 else "done"
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


INTERRUPTED = "interrupted: the batch is not running; this stage runs again from its start when the batch starts"


def _cell(rec, stage, cache, paths, human, alive) -> Cell:
    """`human`: (waits for thresholds, waits for manual inspection); `alive`: the batch is running."""
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
    elif st.state is State.RUNNING and alive:
        level, text = "running", "running"
    elif st.state is State.RUNNING:  # left by a batch that stopped or died during the stage
        level, text = "warning", "interrupted"
        lines.insert(0, INTERRUPTED)
    else:
        level, text = "none", "–"
    if stage == "thresholds" and human[0]:
        level, text = "waiting", "waits"
        lines.append("waiting for your thresholds")
    if stage == "gate" and human[1] and st.state is not State.WAITING:  # e.g. behind a CNMF-E just done
        level, text = "waiting", "waits"
        lines.append("waiting for manual inspection")
    if stage in _SIDE_OF:
        side = rec.stages.get(_SIDE_OF[stage]) or StageRecord()
        off = side.state is State.PENDING and cache.preview_on(paths, rec) is False
        lines.append(f"preview: {'off' if off else side.state.value}")
    if any(a.code == "cnmf_downsampled" for a in _live_anomalies(rec, stage)):
        lines.append(HINTS["cnmf_downsampled"])
    return Cell(st.state.value, level, text, "\n".join([f"{title}: {st.state.value}", *lines]))


def _blocked(cell: Cell, why: str) -> Cell:
    return Cell(cell.state, "error", cell.text, f"cannot be scheduled: {why}\n{cell.tooltip}")


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


def _problems(paths, records, rs: RunnerState, cache) -> dict:
    """sid -> why the batch cannot schedule the session: the live runner's own list from its heartbeat, else
    computed as the runner does, so a stopped batch shows them too (the runner exits when only they remain)."""
    if rs.alive and rs.unschedulable is not None:
        live = {r.sid for r in records if not r.excluded}  # broken registry files are rows of their own
        return {sid: str(why) for sid, why in rs.unschedulable.items() if sid in live}
    return cache.problems(paths, records)


def _warnings(rec, preview_on) -> list[WarningItem]:
    # the window labels each item "<sid> · <stage>": the text starts with what happened
    out = []
    scan = [a for a in rec.anomalies if a.code in _SCAN_CODES and not a.auto_fixed]
    if scan:
        out.append(WarningItem(_worse([a.level for a in scan]), "scan_data", rec.sid, "",
                               " ; ".join(a.message for a in scan), False))
    if rec.excluded:  # the scan anomalies explain the exclusion; nothing else is shown
        return out
    for stage in ALL_STAGES:
        st = rec.stages.get(stage) or StageRecord()
        # a failed preview its config turned off holds nothing (runner.first_failed skips it too)
        if stage in SIDE_STAGES and preview_on is False:
            continue
        if st.state is State.FAILED:
            crash = next((a for a in rec.anomalies if a.code == "stage_crashed"
                          and a.data.get("stage") == stage and a.data.get("cause") in HINTS), None)
            what = HINTS[crash.data["cause"]] if crash else (_last_line(st.error) or "failed")
            out.append(WarningItem("error", "stage_failed", rec.sid, stage,
                                   f"{what.rstrip('.')}. {HINTS['stage_failed']}", True))
        elif st.state is State.DONE:
            bad = [c for c in st.checks if c.name in _CHECK_KIND and c.level != "ok"]
            for kind in dict.fromkeys(_CHECK_KIND[c.name] for c in bad):
                cs = [c for c in bad if _CHECK_KIND[c.name] == kind]
                text = " ; ".join(c.message for c in cs)
                if kind in HINTS:
                    text += f". {HINTS[kind]}"
                out.append(WarningItem(_worse([c.level for c in cs]), kind, rec.sid, stage, text, False))
    return out


def _preview_if_failed(paths, rec, cache) -> bool | None:
    return cache.preview_on(paths, rec) if any(
        s in rec.stages and rec.stages[s].state is State.FAILED for s in SIDE_STAGES) else None


def _duration(st):
    a, b = _iso(st.started), _iso(st.finished)
    return None if a is None or b is None else (b - a).total_seconds()


PER_SESSION = ("crop",)  # the crop reads three windows and builds seed maps: its time does not follow the length


def _rates(records, cfg) -> dict:
    """stage -> (seconds, per_session): seconds per frame, or per session for PER_SESSION stages; the median
    of finished runs once there are two, else the default."""
    out = {}
    per_frame, per_session = cfg.get("eta_s_per_frame", {}), cfg.get("eta_s_per_session", {})
    for stage in STAGES:
        by_session = stage in PER_SESSION
        defaults = per_session if by_session else per_frame
        hist = []
        for r in records:
            st = r.stages.get(stage)
            d = _duration(st) if st is not None and st.state is State.DONE else None
            if d is not None and r.n_frames > 0:
                hist.append(d if by_session else d / r.n_frames)
        if len(hist) >= 2:
            out[stage] = (statistics.median(hist), by_session)
        elif stage in defaults:
            out[stage] = (defaults[stage], by_session)
    return out


def _eta(records, human, now, rates, problems, alive):
    """Seconds of batch work left: 0.0 when nothing is left, None when what is left has no estimate. Waits for
    a person and sessions that cannot be scheduled are not counted."""
    total, left, any_est = 0.0, False, False
    for r in records:
        if r.excluded or r.sid in problems:
            continue
        cut = len(STAGES)  # stages from a person's wait on cannot start until the person decides
        if human[r.sid][0]:
            cut = min(cut, STAGES.index("cnmf"))
        if human[r.sid][1]:
            cut = min(cut, STAGES.index("gate"))
        for stage in STAGES[:cut]:
            st = r.stages.get(stage) or StageRecord()
            # WAITING here is the core's own wait (crop_finalize): it ends without a person
            if st.state not in (State.PENDING, State.RUNNING, State.WAITING):
                continue
            left = True
            if stage == "gate":  # a check of the decision, no work: a person's wait is cut above
                any_est = True
                continue
            if stage not in rates:
                continue
            sec, by_session = rates[stage]
            t = sec if by_session else sec * r.n_frames
            started = _iso(st.started)
            # an interrupted stage (no live batch) runs again from its start
            if st.state is State.RUNNING and alive and started is not None:
                t = max(0.0, t - (now - started).total_seconds())
            total += t
            any_est = True
    if not left:
        return 0.0
    return total if any_est else None


def _bars(records, human, problems, alive) -> dict:
    live = [r for r in records if not r.excluded]
    bars = {}
    for stage in STAGES:
        b = StageBar(0, 0, 0, 0, 0, len(live))
        for r in live:
            st = (r.stages.get(stage) or StageRecord()).state
            if (stage == "thresholds" and human[r.sid][0]) or (stage == "gate" and human[r.sid][1]):
                b.waiting += 1  # a person can act on it, even in a session that cannot be scheduled
            elif st is State.DONE:
                b.done += 1
            elif r.sid in problems:  # cannot run until a config is fixed: needs a person, like a failure
                b.failed += 1
            elif st is State.RUNNING and alive:
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
    rs = runner_state(paths, now)
    records, broken = cache.records(paths)
    if not records and not broken:
        return SeriesView(paths.root, False, rs, [], {s: StageBar(0, 0, 0, 0, 0, 0) for s in STAGES},
                          None, {"thresholds": 0, "manual_inspection": 0}, [], 0, [],
                          found_nothing=paths.scan_unrecognized_csv.exists())
    problems = _problems(paths, records, rs, cache)
    human = {r.sid: (_human_thresholds(paths, r, cache), cache.gate_waits(paths, r)) for r in records}
    rows = []
    for r in records:
        cells = {s: _cell(r, s, cache, paths, human[r.sid], rs.alive) for s in STAGES}
        why = problems.get(r.sid)
        if why is not None:  # red like an unreadable record; the panel has no seventh kind
            cells = {s: _blocked(c, why) for s, c in cells.items()}
        rows.append(Row(r.sid, r.mouse, r.excluded, cells, problem=why))
    rows += [_unreadable_row(f, why) for f, why in broken]
    rows.sort(key=lambda x: (x.mouse, x.sid))
    live = [r for r in records if not r.excluded]
    waiting = {"thresholds": sum(human[r.sid][0] for r in live),
               "manual_inspection": sum(human[r.sid][1] for r in live)}
    warnings = [w for r in records for w in _warnings(r, _preview_if_failed(paths, r, cache))]
    order = {s: i for i, s in enumerate(("",) + ALL_STAGES)}
    warnings.sort(key=lambda w: (w.level != "error", w.sid, order[w.stage]))
    rates = _rates(records, config.load_defaults().get("monitor", {}))
    return SeriesView(paths.root, True, rs, rows, _bars(records, human, problems, rs.alive),
                      _eta(records, human, now, rates, problems, rs.alive), waiting, warnings,
                      len({w.sid for w in warnings}), [f for f, _ in broken])
