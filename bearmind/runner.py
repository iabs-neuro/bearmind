"""Batch scheduler: drives every session of a series through the stages, resumable and failure-isolated.

One process per series (_bearmind/run.lock). Every loop iteration rereads the registry and the configs, so
what other tools write meanwhile (thresholds, manual-inspection decisions, reset_from, a rescan) is picked
up. A stage is up to date when it is DONE, its parameter hash matches the current configuration and its
required outputs exist; otherwise it and every later stage of that session are recomputed (spec 5.10).
The runner writes only the fields it owns (the stage record and that stage's anomalies) through
registry.update_record, so concurrent writers do not lose their changes. Each stage runs in a child
process of its own (bearmind.isolate): a native crash or a dead CaImAn pool worker fails that stage and the
batch goes on with the other sessions.

Side stages (registry.SIDE_STAGES: the preview videos) run right after their parent main stage while their
checkbox is set. Their hash is their settings plus the parent's hash; the main chain never depends on them,
so a disabled, failed or changed preview holds and re-runs nothing else.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import logging
import os
import shutil
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import psutil

try:
    import msvcrt
except ImportError:  # not Windows
    msvcrt = None
    import fcntl

from bearmind import checks, config, fileio, isolate, naming, provenance, registry
from bearmind.crop import fitted_size, mouse_frame_shape
from bearmind.paths import SeriesPaths
from bearmind.registry import ALL_STAGES, SIDE_STAGES, STAGES, Anomaly, Decision, StageRecord, State
import bearmind.crop
import bearmind.stages
import export_estimates_data
from bearmind.stages import WaitingForInput

log = logging.getLogger("bearmind.runner")

# "module:function" strings: stage modules import CaImAn and friends, so they load only when a stage runs,
# in the stage's own process (runner.isolate_stages). Tests replace entries with callables, which run in
# this process (isolate_stages: false).
STAGE_FUNCS = {
    "crop": "bearmind.stages.crop:detect",
    "mc": "bearmind.stages.mc:run",
    "thresholds": "bearmind.stages.thresholds:run",
    "cnmf": "bearmind.stages.cnmf:run",
    "gate": "bearmind.stages.gate:run",
    "autoinspect": "bearmind.stages.autoinspect:run",
    "export": "bearmind.stages.export:run",
    "mc_preview": "bearmind.stages.previews:mc_previews",
    "cnmf_preview": "bearmind.stages.previews:cnmf_preview",
}
FINALIZE_FUNC = "bearmind.stages.crop:finalize_mouse"

# Checks run in this process before a stage process is started, and before the stage is written RUNNING.
# They raise WaitingForInput without heavy work: a stage waiting for a person is retried every poll_s, maybe
# for days, and must not cost a stage process (0.5-12 s of start-up and imports) or a registry write per
# poll. The gate is all check. Their modules are light (cnmf imports bm_qc only inside its functions).
WAIT_CHECKS = {
    "mc": "bearmind.stages.mc:check",
    "cnmf": "bearmind.stages.cnmf:check",
    "gate": "bearmind.stages.gate:run",
}

# Outputs that must exist for a DONE stage to count as done; the residual-maps picture of cnmf is not one.
REQUIRED = {
    "crop": (),
    "mc": ("mc_video", "mc_shifts"),
    "thresholds": ("threshold_maps",),
    "cnmf": ("estimates",),
    "gate": (),
    "autoinspect": ("processed",),
    "export": ("npz", "json", "timestamps"),
    "mc_preview": ("preview_raw", "preview_raw_dff", "preview_mc", "preview_mc_dff"),
    "cnmf_preview": ("preview_cnmf_qc",),
}

HEARTBEAT_S = 30.0
COUNT_KEYS = ("done", "running", "waiting_thresholds", "waiting_manual", "waiting_crop", "failed", "skipped",
              "pending")
_WAIT_KIND = {"thresholds": "waiting_thresholds", "crop_finalize": "waiting_crop"}  # other waits: a person
_UNSET = "<unset>"  # crop.box_pos not written yet (finalize pending), distinct from null (no FOV: uncropped)
FINALIZE_FAILED = "crop_finalize_failed"  # crop anomaly of the sessions whose mouse box could not be placed
FINALIZE_TRIES = 2  # failed finalize attempts after which the mouse's sessions cannot progress (retried before)
_LEVEL = {"ok": 0, "warning": 1, "error": 2}


def _now() -> str:
    # milliseconds: gate.run compares a manual decision with the CNMF finish time (R28)
    return datetime.now().isoformat(timespec="milliseconds")


def _func(spec):
    return spec if callable(spec) else isolate.resolve(spec)


def _stages_until(until) -> tuple:
    """Main stages up to `until` (a main stage)."""
    if until is None:
        return STAGES
    if until not in STAGES:
        raise ValueError(f"unknown stage {until!r}; stages: {', '.join(STAGES)}")
    return STAGES[:STAGES.index(until) + 1]


def _offered(until) -> tuple:
    """Stages a run looks at, in ALL_STAGES order: the main stages up to `until` and their side stages."""
    main = _stages_until(until)
    return tuple(s for s in ALL_STAGES if s in main or SIDE_STAGES.get(s) in main)


def _sides_of(stage) -> list:
    """Side stages that read the outputs of the main `stage` or of a later main stage."""
    i = STAGES.index(stage)
    return [s for s, parent in SIDE_STAGES.items() if STAGES.index(parent) >= i]


def side_enabled(stage: str, cfg: dict) -> bool:
    """A side stage runs only while its checkbox is set: preview.enabled for both preview stages."""
    if stage not in SIDE_STAGES:
        raise ValueError(f"not a side stage: {stage!r}; side stages: {', '.join(SIDE_STAGES)}")
    pv = cfg.get("preview")
    return isinstance(pv, dict) and bool(pv.get("enabled"))


def _side_off(stage, cfg) -> bool:
    """The resolved config turns the side stage off; a config that cannot be read (None) does not."""
    return cfg is not None and not side_enabled(stage, cfg)


_SIDE_ORDER = tuple(s for s in ALL_STAGES if s in SIDE_STAGES)


def _failed_main(rec) -> str | None:
    return next((s for s in STAGES if s in rec.stages and rec.stages[s].state is State.FAILED), None)


def _failed_sides(rec, cfg) -> list:
    """FAILED side stages that are on. One its config turns off is skipped: its old record is kept but holds
    nothing (cfg None: the config cannot be read, every FAILED side stage counts)."""
    return [s for s in _SIDE_ORDER
            if s in rec.stages and rec.stages[s].state is State.FAILED and not _side_off(s, cfg)]


def first_failed(rec, cfg) -> str | None:
    """The failure to report: the first FAILED main stage; without one, the first FAILED side stage that is on.
    A main failure goes first even after a failed preview (disk full fails both): it is what holds the results,
    and a retry takes the previews along (retry_stages)."""
    main = _failed_main(rec)
    if main is not None:
        return main
    sides = _failed_sides(rec, cfg)
    return sides[0] if sides else None


def retry_stages(rec, cfg) -> list:
    """What one retry resets, in this order: the first FAILED main stage (its reset_from also resets the side
    stages at or after it), then every FAILED side stage that is on and whose parent comes before it. [] when
    nothing failed."""
    main = _failed_main(rec)
    end = len(STAGES) if main is None else STAGES.index(main)
    sides = [s for s in _failed_sides(rec, cfg) if STAGES.index(SIDE_STAGES[s]) < end]
    return ([] if main is None else [main]) + sides


def _what(st) -> str:
    """What a WAITING stage waits for, from its error text 'waiting for <what>[: <detail>]'."""
    e = st.error or ""
    e = e[len("waiting for "):] if e.startswith("waiting for ") else e
    return e.split(":", 1)[0]


wait_reason = _what  # public name for the monitor window


# --- parameters and hashes ---

_SHA_CACHE: dict = {}


def _model_sha(f: Path) -> str | None:
    """sha256 of the model file, cached by size and mtime: hashes are checked for every session each loop."""
    try:
        st = f.stat()
    except OSError:
        return None
    key = (str(f), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = provenance._sha256(f)
    return _SHA_CACHE[key]


def _latest_inspection(rec):
    found = [d for d in rec.decisions if d.what == "manual_inspection"]
    return found[-1] if found else None


def _file_stat(f) -> dict | None:
    try:
        st = Path(f).stat()
    except OSError:
        return None
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


# written by finalize after crop is done: box and box_pos are inputs of MC, box_source of finalize only
_CROP_WRITTEN = ("box", "box_pos", "box_source")


def _side_params(cfg) -> dict:
    from bearmind.stages.previews import preview_params  # bearmind.video imports PyAV

    try:
        return preview_params(cfg)
    except ValueError:
        # A setting the stages refuse (a hand edit) is offered anyway and fails in the stage, which validates
        # the same way: it shows as a FAILED preview, and it never makes the session unschedulable.
        return {"enabled": True, "invalid": copy.deepcopy(cfg.get("preview"))}


def _side_hash(params: dict, parent_hash: str) -> str:
    return config.params_hash({"params": params, "upstream": [parent_hash]})


def stage_params(stage: str, rec, cfg: dict, paths) -> dict:
    """Resolved inputs of a stage (R25): stored in StageRecord.params and hashed; machine settings excluded.

    The crop box and the thresholds are written to the configs after crop / thresholds are done, so they
    are inputs of MC and CNMF only, never of the stage that produced them. A side stage: its settings.
    """
    if stage in SIDE_STAGES:
        return _side_params(cfg)
    if stage == "crop":
        # The field rule is code, not configuration: its version in the hash re-runs crop on sessions
        # detected with another one.
        return {"parts": [{"folder": p.folder, "chunks": list(p.chunks), "n_frames": p.n_frames}
                          for p in rec.parts],
                "n_frames": rec.n_frames,
                "crop": {k: v for k, v in (cfg.get("crop") or {}).items() if k not in _CROP_WRITTEN},
                "fov_rule": bearmind.crop.FOV_RULE}
    if stage == "mc":
        c = cfg.get("crop") or {}
        return {"mc": cfg["mc"], "box": c.get("box"), "box_pos": c.get("box_pos", _UNSET), "fps": rec.fps}
    if stage == "thresholds":
        return {"gsig_range": cfg["thresholds"]["gsig_range"], "tsub": cfg["cnmf"]["tsub"]}
    if stage == "cnmf":
        from bearmind.stages.cnmf import cnmf_movie_shape, cnmf_params  # imports CaImAn

        try:
            return cnmf_params(cfg, rec.fps, cnmf_movie_shape(rec))
        except WaitingForInput as w:
            return {"waiting_for": w.what}
    if stage == "gate":
        if not (cfg.get("gate") or {}).get("enabled", True):
            return {"enabled": False}
        d = _latest_inspection(rec)
        cnmf = rec.stages.get("cnmf")
        # R28: a new decision, a re-saved manual result or a new CNMF output re-runs the gate
        return {"enabled": True,
                "decision": None if d is None else {"what": d.what, "value": d.value, "when": d.when},
                "manual_result": _file_stat(paths.manual_result(rec.sid)),
                "cnmf_finished": None if cnmf is None else cnmf.finished}
    if stage == "autoinspect":
        # The grouping algorithm is code, not configuration: its version in the hash re-runs autoinspection
        # (and export after it) on sessions processed with another one.
        return {"autoinspect": cfg["autoinspect"],
                "merge_grouping": export_estimates_data.MERGE_GROUPING,
                "model_sha256": _model_sha(provenance.model_file(cfg)),
                "fps": round(rec.fps) if rec.fps else None}
    if stage == "export":
        # The threshold the processed pickle carries is the autoinspect.ml_threshold that stage ran with
        # (ae_launch stores the argument in autoinspection_config); its stored params give it without
        # loading a pickle of hundreds of MB per session on every check.
        ran = (rec.stages.get("autoinspect") or StageRecord()).params.get("autoinspect") or {}
        # R31: the exported timestamps file is built from these (a rescan may change them)
        parts = [{"timestamps_file": p.timestamps_file, "ts_format": p.ts_format, "start_ms": p.start_ms,
                  "n_frames": p.n_frames,
                  "file": _file_stat(paths.root / p.timestamps_file) if p.timestamps_file else None}
                 for p in rec.parts]
        return {"ml_threshold": ran.get("ml_threshold", cfg["autoinspect"].get("ml_threshold")),
                "component_rule": bearmind.stages.COMPONENT_RULE,
                "timestamps": parts}
    raise ValueError(f"unknown stage {stage!r}")


def _chain(rec, cfg, paths, upto=None):
    """params and hashes of every main stage up to `upto`; each hash covers the hashes of all earlier main
    stages (never of a side stage)."""
    params, hashes = {}, {}
    for s in STAGES[:STAGES.index(upto) + 1] if upto else STAGES:
        params[s] = stage_params(s, rec, cfg, paths)
        hashes[s] = config.params_hash({"params": params[s], "upstream": list(hashes.values())})
    return params, hashes


def stage_hash(rec, stage: str, cfg: dict, paths) -> str:
    """params_hash of the stage parameters and of all earlier main stages' hashes; a side stage: of its
    settings and its parent's hash."""
    if stage in SIDE_STAGES:
        parent = SIDE_STAGES[stage]
        return _side_hash(stage_params(stage, rec, cfg, paths), _chain(rec, cfg, paths, parent)[1][parent])
    return _chain(rec, cfg, paths, stage)[1][stage]


def _output_ok(paths, outputs: dict, key: str) -> bool:
    return key in outputs and (paths.root / outputs[key]).is_file()


def _one_line(e: BaseException) -> str:
    return " ".join(f"{type(e).__name__}: {e}".split())  # YAML errors span several lines


NO_FPS = "no frame rate (fps_unavailable); run scan again: it excludes the session"


class _View:
    """Resolved configs and stage hashes of one registry snapshot (one loop iteration)."""

    def __init__(self, paths):
        self.paths = paths
        self._cfg, self._hashes, self._side = {}, {}, {}
        self.errors = {}  # sid -> why the session cannot be scheduled now

    def cfg(self, rec):
        if rec.sid not in self._cfg:
            try:
                self._cfg[rec.sid] = config.resolve(self.paths, sid=rec.sid, mouse=rec.mouse)
            except Exception as e:  # a broken session config holds this session only, its states stay
                self._cfg[rec.sid] = None
                self.errors[rec.sid] = f"config: {_one_line(e)}"
        return self._cfg[rec.sid]

    def hashes(self, rec):
        if rec.sid not in self._hashes:
            h, cfg = None, self.cfg(rec)
            if cfg is not None and rec.fps is None:  # a record scanned before such sessions were excluded
                self.errors[rec.sid] = NO_FPS
            elif cfg is not None:
                try:
                    h = _chain(rec, cfg, self.paths)[1]
                except Exception as e:
                    self.errors[rec.sid] = f"stage parameters: {_one_line(e)}"
            self._hashes[rec.sid] = h
        return self._hashes[rec.sid]

    def hash(self, rec, stage):
        """The stage's hash for the current configs; None while the session's hashes cannot be computed."""
        h = self.hashes(rec)
        if h is None or stage not in SIDE_STAGES:
            return None if h is None else h[stage]
        key = (rec.sid, stage)
        if key not in self._side:
            self._side[key] = _side_hash(_side_params(self.cfg(rec)), h[SIDE_STAGES[stage]])
        return self._side[key]

    def side_on(self, rec, stage) -> bool:
        cfg = self.cfg(rec)
        return cfg is not None and side_enabled(stage, cfg)

    def current(self, rec, stage) -> bool:
        st = rec.stages.get(stage)
        if st is None or st.state is not State.DONE:
            return False
        h = self.hash(rec, stage)
        return (h is not None and st.params_hash == h
                and all(_output_ok(self.paths, st.outputs, k) for k in REQUIRED[stage]))

    def blocked(self, rec, stage) -> bool:
        """FAILED / SKIPPED with unchanged parameters: retried only after reset_from or a parameter change."""
        st = rec.stages.get(stage)
        if st is None or st.state not in (State.FAILED, State.SKIPPED):
            return False
        h = self.hash(rec, stage)
        return h is not None and st.params_hash == h


class _Scope:
    """Sessions a run works on. Crop detection also runs for the other sessions of a selected session's
    mouse: the box size is shared by all sessions of a mouse and is fixed once written."""

    def __init__(self, records, sids):
        self.sids = None if sids is None else set(sids)
        self.mice = None if sids is None else {r.mouse for r in records if r.sid in self.sids}

    def has(self, rec, stage) -> bool:
        if self.sids is None or rec.sid in self.sids:
            return True
        return stage == "crop" and rec.mouse in self.mice


def next_ready(records, paths, until=None, sids=None, skip=(), _view=None):
    """(sid, stage) of the next stage to run, or None: lowest index in ALL_STAGES first, then sid order.

    Ready: session not excluded, every earlier main stage up to date, the stage itself not up to date and
    not FAILED/SKIPPED with unchanged parameters. A side stage needs its checkbox set and every main stage up
    to its parent up to date; a main stage never waits for a side stage. With `until`, a side stage is in
    range when its parent is. `skip` holds (sid, stage) pairs not to offer now (waiting stages already tried).
    crop.finalize_mouse is scheduled by run(), not here.
    """
    view = _view or _View(paths)
    scope = _Scope(records, sids)
    recs = sorted((r for r in records if not r.excluded), key=lambda r: r.sid)
    for stage in _offered(until):
        side = stage in SIDE_STAGES
        need = STAGES[:STAGES.index(SIDE_STAGES[stage]) + 1] if side else STAGES[:STAGES.index(stage)]
        for r in recs:
            if (r.sid, stage) in skip or not scope.has(r, stage) or view.hashes(r) is None:
                continue
            if side and not view.side_on(r, stage):
                continue  # disabled: never offered, nothing recorded; an old record stays as it was
            if (all(view.current(r, s) for s in need) and not view.current(r, stage)
                    and not view.blocked(r, stage)):
                return r.sid, stage
    return None


# --- what the series needs ---

def _finalize_failure(rec):
    return next((a for a in rec.anomalies if a.code == FINALIZE_FAILED), None)


def schedule_problems(paths, records, view=None, broken=()) -> dict:
    """sid -> why the session cannot be scheduled now: its config or stage parameters cannot be resolved,
    or the box of its mouse could not be placed FINALIZE_TRIES times (non-excluded sessions); or its registry
    file cannot be read (under the file name)."""
    view = view or _View(paths)
    active = [r for r in records if not r.excluded]
    for r in active:
        view.hashes(r)
    out = {r.sid: view.errors[r.sid] for r in active if r.sid in view.errors}
    for r in active:
        a = _finalize_failure(r)
        if a is not None and a.data.get("attempts", 1) >= FINALIZE_TRIES and r.sid not in out:
            out[r.sid] = f"{a.message} ({a.data['attempts']} attempts)"
    for f, why in broken:
        out[Path(f).stem] = f"broken registry file (run scan to recreate it): {f}: {why}"
    return out


def _waiting(records, scope, stages) -> list:
    return [(r.sid, s, _what(r.stages[s])) for r in records if not r.excluded for s in stages
            if scope.has(r, s) and s in r.stages and r.stages[s].state is State.WAITING]


def _split_waits(waiting, records, problems) -> tuple[list, list]:
    """(waits a person can resolve, waits blocked by a session that cannot be scheduled).

    A wait is blocked when its own session cannot be scheduled, or when it waits for the crop box of a mouse
    with such a session: the box is placed once every session of the mouse is detected. Blocked waits are
    (sid, stage, what, [blocking sids]); only a config fix moves them.
    """
    mouse = {r.sid: r.mouse for r in records}
    stuck = {}
    for r in records:
        if not r.excluded and r.sid in problems:
            stuck.setdefault(r.mouse, []).append(r.sid)
    person, blocked = [], []
    for sid, stage, what in waiting:
        if sid in problems:
            blocked.append((sid, stage, what, [sid]))
        elif what == "crop_finalize" and stuck.get(mouse.get(sid)):
            blocked.append((sid, stage, what, stuck[mouse[sid]]))
        else:
            person.append((sid, stage, what))
    return person, blocked


def _blocked_text(blocked, problems) -> str:
    return "; ".join(f"{sid} {stage} waits for {what}, blocked by "
                     + ", ".join(f"{b} ({problems[b]})" for b in by) for sid, stage, what, by in blocked)


@dataclasses.dataclass
class SeriesState:
    """What the series needs now, over all its sessions (not the counts of one run)."""
    records: list
    broken: list  # (registry file, why it cannot be read)
    problems: dict  # sid -> why it cannot be scheduled; broken registry files under their file name
    waiting: list  # (sid, stage, what): waits a person can resolve
    blocked: list  # (sid, stage, what, [blocking sids]): waits only a config fix moves
    failed: dict  # sid -> (stage, error) of first_failed (a main failure first), non-excluded sessions
    excluded: list
    finalize_failed: dict  # sid -> why the box of its mouse is not placed (the last finalize failed)
    # sid -> {side stage: on in its resolved config}; None when the config cannot be read
    previews: dict = dataclasses.field(default_factory=dict)


def series_state(paths) -> SeriesState:
    records, broken = registry.read_all(paths)
    view = _View(paths)
    problems = schedule_problems(paths, records, view, broken=broken)
    person, blocked = _split_waits(_waiting(records, _Scope(records, None), STAGES), records, problems)
    failed = {}
    for r in records:
        first = first_failed(r, view.cfg(r))
        if first is not None and not r.excluded:
            failed[r.sid] = (first, r.stages[first].error)
    finalize_failed = {r.sid: _finalize_failure(r).message for r in records
                       if not r.excluded and _finalize_failure(r) is not None}
    person = [w for w in person if w[0] not in finalize_failed]  # waiting for a retry, not for a person
    previews = {r.sid: None if view.cfg(r) is None else {s: side_enabled(s, view.cfg(r)) for s in _SIDE_ORDER}
                for r in records}
    return SeriesState(records, broken, problems, person, blocked, failed, [r.sid for r in records if r.excluded],
                       finalize_failed, previews)


# --- registry changes made by the runner ---

def _skip_downstream(d, stage, cfg) -> int:
    """A main stage failed: every later main stage, and the enabled side stages that read its outputs or
    later ones, -> SKIPPED. Returns how many."""
    n = 0
    for s in STAGES[STAGES.index(stage) + 1:]:
        st = d.stages.setdefault(s, StageRecord())
        st.state, st.error = State.SKIPPED, f"skipped: {stage} failed"
        n += 1
    for s in _sides_of(stage):
        if side_enabled(s, cfg):  # a disabled side stage is never recorded
            st = d.stages.setdefault(s, StageRecord())
            st.state, st.error = State.SKIPPED, f"skipped: {stage} failed"
            n += 1
    return n


def _pending_sides(d, stage) -> None:
    """The recorded side stages that read the outputs of the main `stage` or of a later one -> PENDING: they
    show the old outputs, and with unchanged parameters their hash alone would still call them current."""
    for s in _sides_of(stage):
        st = d.stages.get(s)
        if st is not None and st.state is not State.PENDING:
            st.state, st.error = State.PENDING, None


def reset_from(paths, sid: str, stage: str, who: str) -> None:
    """Re-analysis. A main stage: it, every later main stage and the recorded side stages that read their
    outputs -> PENDING. A side stage: only itself. Old outputs stay until the stage replaces them."""
    if stage not in ALL_STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages: {', '.join(ALL_STAGES)}")
    now = _now()

    def reset(d):
        if stage in SIDE_STAGES:
            st = d.stages.get(stage)
            if st is None:  # never recorded: nothing changes, so no decision either
                return
            st.state, st.error = State.PENDING, None
        else:
            for s in STAGES[STAGES.index(stage):]:
                st = d.stages.setdefault(s, StageRecord())
                st.state, st.error = State.PENDING, None
            _pending_sides(d, stage)
        d.decisions.append(Decision("rerun_from", stage, who, now))

    registry.update_record(paths, sid, reset)


def _clean_temp(paths, temp_root, sid) -> None:
    """Remove this series' temp folder of the session after a failed or killed stage (the stages clean it
    on success)."""
    if not temp_root:
        return
    d = paths.temp_dir(temp_root, sid)
    for _ in range(5):
        if not d.exists():
            return
        shutil.rmtree(d, ignore_errors=True)
        if not d.exists():
            return
        time.sleep(0.2)
    log.warning("%s: temp folder not removed: %s", sid, d, extra={"sid": sid})


def _last_line(e: BaseException) -> str:
    return traceback.format_exception_only(type(e), e)[-1].strip()


# --- logs, heartbeat, lock ---

_FMT = logging.Formatter("%(asctime)s %(levelname)s %(message)s")


class _SeriesFilter(logging.Filter):
    def filter(self, record):
        return not getattr(record, "session_only", False)  # tracebacks go to the session log only


class _SessionFilter(logging.Filter):
    def __init__(self, sid):
        super().__init__()
        self.sid, self.thread = sid, threading.get_ident()

    def filter(self, record):
        sid = getattr(record, "sid", None)
        # records without a sid come from code the stage calls (same thread, while the stage runs)
        return sid == self.sid or (sid is None and record.thread == self.thread)


class _ConsoleFilter(logging.Filter):
    def filter(self, record):
        return getattr(record, "console", False)  # the events of the run, not everything the stages log


class _AsciiFormatter(logging.Formatter):
    """'HH:MM:SS message', ASCII only: students read it in a Windows console, and a stage error can carry
    a localized system message (the logs keep the original text)."""

    def format(self, record):
        return super().format(record).encode("ascii", "backslashreplace").decode("ascii")


class _Logs:
    """File handlers on the 'bearmind' logger, and with console=True one line on stdout per event of the
    run (stage started / done / failed, waits); the root logger is left alone."""

    def __init__(self, paths, console=False):
        self.paths = paths
        self.logger = logging.getLogger("bearmind")
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(paths.series_log, encoding="utf-8")
        self.handler.setFormatter(_FMT)
        self.handler.addFilter(_SeriesFilter())
        self.logger.addHandler(self.handler)
        self.console = None
        if console:
            self.console = logging.StreamHandler(sys.stdout)
            self.console.setFormatter(_AsciiFormatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
            self.console.addFilter(_ConsoleFilter())
            self.logger.addHandler(self.console)
        self.level = self.logger.level
        if self.logger.getEffectiveLevel() > logging.INFO:
            self.logger.setLevel(logging.INFO)

    @contextmanager
    def session(self, sid):
        h = logging.FileHandler(self.paths.session_log(sid), encoding="utf-8")
        h.setFormatter(_FMT)
        h.addFilter(_SessionFilter(sid))
        self.logger.addHandler(h)
        try:
            yield
        finally:
            self.logger.removeHandler(h)
            h.close()

    def close(self):
        self.logger.removeHandler(self.handler)
        self.handler.close()
        if self.console is not None:
            self.logger.removeHandler(self.console)
        self.logger.setLevel(self.level)


class _Heartbeat:
    """_bearmind/heartbeat.json: written on every transition and every HEARTBEAT_S by a daemon thread."""

    def __init__(self, paths):
        self.paths = paths
        self.lock = threading.Lock()
        self.payload = {"pid": os.getpid(), "time": None, "current": None,
                        "counts": dict.fromkeys(COUNT_KEYS, 0), "by_stage": {}, "status": "running",
                        "unschedulable": {}, "exit": None}
        self._stop = threading.Event()
        self._thread = None

    def _write(self):
        self.payload["time"] = datetime.now().isoformat(timespec="milliseconds")
        try:
            fileio.write_atomic(self.paths.heartbeat, lambda fh: json.dump(self.payload, fh, indent=2))
        except Exception as e:
            log.warning("heartbeat not written: %s", _last_line(e))

    def update(self, **fields):
        with self.lock:
            self.payload.update(fields)
            self._write()

    def start(self):
        self._thread = threading.Thread(target=self._pulse, name="bearmind-heartbeat", daemon=True)
        self._thread.start()

    def _pulse(self):
        while not self._stop.wait(HEARTBEAT_S):
            with self.lock:
                self._write()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)


_LOCK_BYTE = 1 << 20  # the OS lock covers one byte past the text, so others can still read who holds it


def _os_lock(fh) -> bool:
    fh.seek(_LOCK_BYTE)
    try:
        if msvcrt is not None:
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            fcntl.lockf(fh, fcntl.LOCK_EX | fcntl.LOCK_NB, 1, _LOCK_BYTE)
        return True
    except OSError:
        return False


def _os_unlock(fh) -> None:
    fh.seek(_LOCK_BYTE)
    if msvcrt is not None:
        msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.lockf(fh, fcntl.LOCK_UN, 1, _LOCK_BYTE)


def _lock_holder(lock: Path):
    """PID text of the runner holding the lock, for the message only (the OS lock decides)."""
    try:
        return json.loads(lock.read_text(encoding="utf-8")).get("pid")
    except (OSError, ValueError, AttributeError):
        return None


def _take_lock(lock: Path, extra: dict | None = None):
    """Open `lock`, take the OS lock, write who holds it. Returns the handle, or None when it is held."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    fh = os.fdopen(os.open(lock, os.O_RDWR | os.O_CREAT), "r+b", buffering=0)
    if not _os_lock(fh):
        fh.close()
        return None
    me = psutil.Process()
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps({"pid": me.pid, "create_time": me.create_time(), "time": _now(), **(extra or {})}).encode())
    return fh


def _drop_lock(lock: Path, fh) -> None:
    try:
        _os_unlock(fh)
    except OSError:
        pass
    fh.close()
    try:
        lock.unlink()
    except (FileNotFoundError, PermissionError):
        pass  # a new runner opened it already


def _acquire_lock(paths):
    """OS lock on _bearmind/run.lock, held through an open handle for the whole run.

    The OS releases it when the process dies, so a killed runner never blocks the series; the PID and
    process start time written into the file only tell a second runner who holds it.
    """
    fh = _take_lock(paths.lock)
    if fh is None:
        pid = _lock_holder(paths.lock)
        raise RuntimeError(f"series is already being processed by PID {pid if pid is not None else 'unknown'}")
    return fh


def _release_lock(paths, fh) -> None:
    _drop_lock(paths.lock, fh)


def machine_lock() -> Path:
    return config._home() / "runner.lock"


def _machine_busy_text(series, pid) -> str:
    return (f"another batch is running on this machine: {series if series else 'unknown series'} "
            f"(PID {pid if pid is not None else 'unknown'}); run one batch at a time")


def _lock_info(lock: Path) -> dict:
    try:
        info = json.loads(lock.read_text(encoding="utf-8"))
        return info if isinstance(info, dict) else {}
    except (OSError, ValueError):
        return {}


def machine_busy():
    """(series, PID) of the batch holding the machine lock, or None when no live runner holds it.

    The pools of two series together can exceed the machine's memory budget, so one batch runs at a time.
    """
    lock = machine_lock()
    if not lock.exists():
        return None
    try:
        fh = os.fdopen(os.open(lock, os.O_RDWR), "r+b", buffering=0)
    except OSError:
        return None
    try:
        if _os_lock(fh):
            _os_unlock(fh)
            return None
    finally:
        fh.close()
    info = _lock_info(lock)
    return info.get("series"), info.get("pid")


def busy_message() -> str | None:
    busy = machine_busy()
    return None if busy is None else _machine_busy_text(*busy)


def _acquire_machine_lock(paths):
    fh = _take_lock(machine_lock(), {"series": str(paths.root)})
    if fh is None:
        info = _lock_info(machine_lock())
        raise RuntimeError(_machine_busy_text(info.get("series"), info.get("pid")))
    return fh


# --- the run ---

RESULT_WRITE_TRIES = 7  # x LOCK_TIMEOUT_S: longer than LOCK_STALE_S, so a stuck lock is broken meanwhile
RUNNING_WRITE_TRIES = 2  # nothing has run yet: a stage that cannot start is offered again later


def _session_status(r, scope, stages) -> str | None:
    sts = [r.stages.get(s) or StageRecord() for s in stages if scope.has(r, s)]
    if not sts:
        return None
    states = [st.state for st in sts]
    if State.RUNNING in states:
        return "running"
    waiting = [st for st in sts if st.state is State.WAITING]
    if waiting:
        return _WAIT_KIND.get(_what(waiting[0]), "waiting_manual")
    if State.FAILED in states:
        return "failed"
    if State.SKIPPED in states:
        return "skipped"
    return "done" if all(x is State.DONE for x in states) else "pending"


def _counts(records, scope, stages):
    """Sessions by status (each non-excluded session once) and the number of DONE sessions per stage."""
    c, by_stage = dict.fromkeys(COUNT_KEYS, 0), dict.fromkeys(stages, 0)
    for r in records:
        if r.excluded:
            continue
        status = _session_status(r, scope, stages)
        if status is not None:
            c[status] += 1
        for s in stages:
            if scope.has(r, s) and (r.stages.get(s) or StageRecord()).state is State.DONE:
                by_stage[s] += 1
    return c, by_stage


def _update(paths, sid, change, tries=1):
    """registry.update_record, retried on OSError: a LockTimeout (each try waits up to fileio.LOCK_TIMEOUT_S)
    or a write error such as PermissionError (fileio.replace has retried it for a few seconds already)."""
    for i in range(tries):
        try:
            return registry.update_record(paths, sid, change)
        except OSError as e:
            if i == tries - 1:
                raise
            log.warning("%s: registry not writable now, retrying: %s", sid, _last_line(e), extra={"sid": sid})


class ResultNotRecorded(RuntimeError):
    """A finished stage whose result the registry would not take: the stage is recorded FAILED, its outputs
    stay on disk."""


class _Run:
    """State of one run: outcome counts, the stage being run, logs and heartbeat."""

    def __init__(self, paths, until, sids, once, poll_s, console=False):
        self.paths, self.sids, self.once, self.poll_s, self.console = paths, sids, once, poll_s, console
        self.until, self.stages = until, _stages_until(until)
        self.result = {"done": 0, "failed": 0, "waiting": 0, "skipped": 0, "not_recorded": []}
        self.not_recorded = {}  # (sid, stage) -> why its start or outcome is not in the registry
        self.current = None  # (sid, stage, started, params_hash, temp_root)
        self.logs = self.hb = None
        self.waiting_logged = set()
        self.auto_retries = {}  # (sid, stage) -> automatic retries after a crash used in this run
        # why the run ended, for the window: "finished", "stopped" (stop request), "interrupted" (Ctrl+C) or
        # "error: <exception>"; written with the final heartbeat
        self.exit = None

    def open(self):
        self.logs = _Logs(self.paths, self.console)
        self.hb = _Heartbeat(self.paths)
        self.hb.update(status="running")
        self.hb.start()

    def refresh(self, status="running", records=None, broken=None, problems=None, **extra):
        """Heartbeat counts and summary.csv from the registry as it is now; `extra` goes into the heartbeat."""
        if records is None:
            records, broken = registry.read_all(self.paths)
        if problems is None:
            problems = schedule_problems(self.paths, records, broken=broken or ())
        cur = None if self.current is None else {"sid": self.current[0], "stage": self.current[1]}
        counts, by_stage = _counts(records, _Scope(records, self.sids), self.stages)
        self.hb.update(counts=counts, by_stage=by_stage, status=status, current=cur, unschedulable=problems,
                       **extra)
        summary(self.paths, records, problems=problems, broken=broken)

    def abort(self):
        """KeyboardInterrupt or an error outside a stage: the stage being run goes back to PENDING."""
        if self.current is None:
            return
        sid, stage, started, h, temp_root = self.current

        def back(d):
            st = d.stages.get(stage)
            if st is not None and st.state is State.RUNNING and st.started == started:
                st.state, st.error = State.PENDING, None

        try:
            registry.update_record(self.paths, sid, back)
            log.warning("%s %s: interrupted, set back to pending", sid, stage, extra={"sid": sid})
        except Exception as e:
            log.error("%s %s: could not set back to pending: %s", sid, stage, _last_line(e), extra={"sid": sid})
        _clean_temp(self.paths, temp_root, sid)
        self.current = None

    def close(self):
        if self.hb is not None:
            self.hb.stop()
            try:
                self.refresh(status="idle", exit=self.exit)
            except Exception as e:
                log.warning("final heartbeat / summary failed: %s", _last_line(e))
                # the window tells "stopped" from "not responding" by this pulse: it is written without counts
                self.hb.update(status="idle", current=None, exit=self.exit)
        if self.logs is not None:
            self.logs.close()


def _reset_running(paths) -> None:
    """Stages left RUNNING by a killed process go back to PENDING; their temp folders are removed."""
    temp_root = config.machine_config().get("temp_dir")
    for rec in registry.all_records(paths):
        running = [s for s, st in rec.stages.items() if st.state is State.RUNNING]
        if not running:
            continue

        def back(d, running=running):
            for s in running:
                st = d.stages.get(s)
                if st is not None and st.state is State.RUNNING:
                    st.state, st.error = State.PENDING, None

        try:
            _update(paths, rec.sid, back)
        except OSError as e:  # a RUNNING stage is not up to date anyway: it runs again
            log.warning("%s: registry not writable now, left as running: %s", rec.sid, _last_line(e),
                        extra={"sid": rec.sid})
            continue
        _clean_temp(paths, temp_root, rec.sid)
        log.warning("%s %s: was running at start (process killed?), set to pending", rec.sid,
                    ",".join(running), extra={"sid": rec.sid})


def _box_list(box):
    if box is None:
        return None
    if hasattr(box, "top"):
        return [int(box.top), int(box.left), int(box.height), int(box.width)]
    return [int(x) for x in box]


def _placed(r, cfg, frame_shape) -> bool:
    """The session's box is placed for the current mouse box (a session without FOV has none to place)."""
    m, c = r.stages["crop"].metrics, cfg.get("crop") or {}
    if "box" not in m or "box_pos" not in c:
        return False
    if m["box"] is None:
        return True
    return c.get("box") is not None and list(m["box"][2:]) == fitted_size(c["box"], frame_shape)


def _tuple(x):
    return tuple(x) if isinstance(x, list) else x


def _finalize_due(records, view, scope, attempted, once, poll_s) -> list:
    """Mice whose crop detection is settled for every session and whose boxes are not placed for the
    current mouse box: never placed, crop detected again, or the mouse box removed or changed by hand.

    Settled = up to date, or FAILED with unchanged parameters (one bad session does not hold its mouse).
    A session that cannot be scheduled (unreadable config) is not settled: its mouse waits (M1). Only
    sessions with crop DONE are passed to finalize_mouse. A crop re-run drops metrics['box'], so the box is
    placed again. `attempted` maps a key to (ok, time, failures): a key that worked is not run again; one that
    failed is retried after poll_s, up to FINALIZE_TRIES times per run (once=True: by the next run). The key
    holds the mouse box and the positions, so an edit of either during a run is seen.
    """
    by_mouse = {}
    for r in records:
        if not r.excluded and scope.has(r, "crop"):
            by_mouse.setdefault(r.mouse, []).append(r)
    due = []
    now = time.monotonic()
    for mouse, recs in sorted(by_mouse.items()):
        done = [r for r in recs if view.current(r, "crop")]
        if not done or any(not (view.current(r, "crop") or view.blocked(r, "crop")) for r in recs):
            continue
        frame_shape = mouse_frame_shape(r.stages["crop"].metrics for r in done)
        if all(_placed(r, view.cfg(r), frame_shape) for r in done):
            continue
        crops = [view.cfg(r).get("crop") or {} for r in done]
        key = (mouse, _tuple(crops[0].get("box")),
               tuple((r.sid, r.stages["crop"].params_hash, r.stages["crop"].finished, _tuple(c.get("box_pos")))
                     for r, c in zip(done, crops)))
        tried = attempted.get(key)  # (ok, time, failures in this run)
        if tried is None or (not tried[0] and not once and tried[2] < FINALIZE_TRIES and now - tried[1] >= poll_s):
            due.append((mouse, done, key))
    return due


def _run_finalize(paths, mouse, done, run_: _Run) -> bool:
    """Place the mouse box. An error is logged and retried later; crop stays DONE (M2)."""
    before = {r.sid: len(r.anomalies) for r in done}
    finished = {r.sid: r.stages["crop"].finished for r in done}

    def ours(d):
        st = d.stages.get("crop")
        return st is not None and st.state is State.DONE and st.finished == finished[d.sid]

    try:
        cfg = config.resolve(paths, mouse=mouse)
        boxes = _func(FINALIZE_FUNC)(paths, mouse, done, cfg) or {}
        for r in done:
            new = [dataclasses.replace(a, data={**a.data, "stage": "crop"}) for a in r.anomalies[before[r.sid]:]]

            def apply(d, r=r, new=new):
                if not ours(d):
                    return
                if r.sid in boxes:
                    d.stages["crop"].metrics["box"] = _box_list(boxes[r.sid])
                d.anomalies = [a for a in d.anomalies if a.code != FINALIZE_FAILED]
                have = {a.code for a in d.anomalies if a.data.get("stage") == "crop"}
                d.anomalies.extend(a for a in new if a.code not in have)

            rec = _update(paths, r.sid, apply, RESULT_WRITE_TRIES)
            with run_.logs.session(r.sid):
                if r.sid in boxes:
                    log.info("%s crop: box %s (mouse %s)", r.sid, rec.stages["crop"].metrics.get("box"), mouse,
                             extra={"sid": r.sid})
                else:
                    log.warning("%s crop: finalize placed no box (no crop metrics)", r.sid, extra={"sid": r.sid})
    except Exception as e:
        last, tb = _last_line(e), traceback.format_exc()
        log.error("crop finalize of mouse %s failed (MC of %s waits): %s", mouse, ", ".join(r.sid for r in done),
                  last, extra={"console": True})
        for r in done:
            with run_.logs.session(r.sid):
                log.error("%s crop: finalize of mouse %s failed, retried later (MC waits): %s", r.sid, mouse,
                          last, extra={"sid": r.sid})
                log.error("%s", tb, extra={"sid": r.sid, "session_only": True})
            _record_finalize_failure(paths, r.sid, mouse, last, ours)
        return False
    _write_crop_overview(paths)
    return True


def _write_crop_overview(paths) -> None:
    """Redraw the series crop overview after a placement of boxes. Only a picture for the author: an error
    is logged and costs nothing else."""
    try:
        from bearmind.crop_overview import write_overview  # matplotlib only when there is something to draw

        write_overview(paths, [r for r in registry.read_all(paths)[0] if not r.excluded])
    except Exception as e:
        log.warning("crop overview not written: %s", _last_line(e), extra={"console": True})


def _record_finalize_failure(paths, sid, mouse, why, ours) -> None:
    """The reason the session's box is missing, for status and summary.csv (a crop anomaly with the number
    of failed attempts); a later finalize that works removes it, a crop re-run drops it."""
    def note(d):
        if not ours(d):
            return
        old = _finalize_failure(d)
        attempts = (old.data.get("attempts", 1) if old is not None else 0) + 1
        d.anomalies = [a for a in d.anomalies if a.code != FINALIZE_FAILED]
        d.anomalies.append(Anomaly(FINALIZE_FAILED, "error", f"crop box not placed: finalize of mouse {mouse} "
                                   f"failed: {why}", data={"stage": "crop", "attempts": attempts}))

    try:
        _update(paths, sid, note, RESULT_WRITE_TRIES)
    except OSError as e:
        log.error("%s crop: finalize failure not recorded, registry not writable: %s", sid, _last_line(e),
                  extra={"sid": sid})


def _wait_check(paths, stage, rec, cfg):
    """(checked, WaitingForInput or None): the stage's wait check, run before anything is written.

    Only for isolated stages (an in-process stage raises WaitingForInput itself). Another error of the
    check is not handled here: _call_stage runs the check again where the stage's failures are recorded.
    """
    if not cfg["runner"]["isolate_stages"] or stage not in WAIT_CHECKS:
        return False, None
    try:
        _func(WAIT_CHECKS[stage])(paths, rec, cfg)
    except WaitingForInput as w:
        return True, w
    except Exception:
        return False, None
    return True, None


def _reset_downstream(d, stage) -> None:
    for s in STAGES[STAGES.index(stage) + 1:]:  # recomputed after this stage, whatever their hash
        st = d.stages.setdefault(s, StageRecord())
        if st.state is not State.PENDING:
            st.state, st.error = State.PENDING, None
    _pending_sides(d, stage)


def _record_wait(paths, sid, stage, rec, h, params, w, run_: _Run) -> str:
    """A stage whose check says it waits. Its record is written only when it changes (first wait, another
    reason, other parameters): a waiting stage is checked every poll_s, maybe for days, and a write per
    check rewrote every record and summary.csv on each pass. Returns 'waiting'; 'retry' when a decision or
    a reset landed after the check (checked again on the next loop); 'none' when the registry cannot be
    written (checked again after poll_s)."""
    st = rec.stages.get(stage)
    if st is not None and st.state is State.WAITING and st.error == str(w) and st.params_hash == h:
        return "waiting"
    seen, decisions = copy.deepcopy(st), copy.deepcopy(rec.decisions)
    applied = []

    def waiting(d):
        if d.stages.get(stage) != seen or d.decisions != decisions:
            return
        applied.append(True)
        d.stages[stage] = StageRecord(state=State.WAITING, params_hash=h, params=params, error=str(w))
        d.anomalies = [a for a in d.anomalies if a.data.get("stage") != stage]
        _reset_downstream(d, stage)

    try:
        _update(paths, sid, waiting, RESULT_WRITE_TRIES)
    except OSError as e:  # checked again after poll_s
        log.error("%s %s: wait not recorded, registry not writable: %s", sid, stage, _last_line(e),
                  extra={"sid": sid})
        run_.not_recorded[(sid, stage)] = f"wait not recorded: {_last_line(e)}"
        return "none"
    run_.not_recorded.pop((sid, stage), None)
    if not applied:
        return "retry"
    run_.waiting_logged.add((sid, stage))
    with run_.logs.session(sid):
        log.info("%s %s: %s", sid, stage, w, extra={"sid": sid, "console": True})
    run_.refresh()
    return "waiting"


def _call_stage(paths, sid, stage, rec, cfg, temp_root, run_: _Run):
    """The stage's StageResult, from its own process unless runner.isolate_stages is false.

    A stage process that crashed (StageCrashed) is started again up to runner.max_auto_retries times per
    run; any other failure is final (spec 5.10: FAILED, retried after a reset or a parameter change).
    """
    rc = cfg["runner"]  # defaults.yaml has both keys: no second set of defaults here
    spec = STAGE_FUNCS[stage]
    if not rc["isolate_stages"]:
        return _func(spec)(paths, rec, cfg)
    if stage in WAIT_CHECKS:
        # again, on the record as written with RUNNING: a decision may have changed since _wait_check
        _func(WAIT_CHECKS[stage])(paths, rec, cfg)
    limit = int(rc["max_auto_retries"])
    while True:
        try:
            return isolate.run_isolated(spec, paths, rec, cfg, paths.session_log(sid),
                                        stall_timeout_min=rc["stall_timeout_min"])
        except isolate.StageCrashed as e:
            used = run_.auto_retries.get((sid, stage), 0)
            if used >= limit:
                raise
            run_.auto_retries[(sid, stage)] = used + 1
            log.warning("%s %s: %s; automatic retry %d of %d", sid, stage, e, used + 1, limit,
                        extra={"sid": sid, "console": True})
            _clean_temp(paths, temp_root, sid)


def _run_stage(paths, sid, stage, run_: _Run) -> str:
    """Run one stage of one session.

    Returns 'done' | 'waiting' | 'failed', 'none' (could not start: config or record changed) or 'retry'
    (the registry lock was busy, or a side stage was turned off since it was offered: the next loop decides
    again, the batch goes on).
    """
    side = stage in SIDE_STAGES
    try:
        rec = registry.load_record(paths, sid)
        cfg = config.resolve(paths, sid=sid, mouse=rec.mouse)  # reread before every stage
        if side:
            if not side_enabled(stage, cfg):  # turned off since next_ready looked: never called, not recorded
                log.info("%s %s: previews turned off, not started", sid, stage, extra={"sid": sid})
                return "retry"
            parent = SIDE_STAGES[stage]
            params = {stage: stage_params(stage, rec, cfg, paths)}
            h = _side_hash(params[stage], _chain(rec, cfg, paths, parent)[1][parent])
        else:
            params, hashes = _chain(rec, cfg, paths, stage)
            h = hashes[stage]
    except Exception as e:  # changed since next_ready looked: not started, retried later
        log.error("%s %s: not started: %s", sid, stage, _last_line(e), extra={"sid": sid})
        return "none"
    checked, w = _wait_check(paths, stage, rec, cfg)
    if w is not None:
        return _record_wait(paths, sid, stage, rec, h, params[stage], w, run_)
    if checked:
        run_.waiting_logged.discard((sid, stage))  # its check passed: this is a real start
    started = _now()
    temp_root = (cfg.get("machine") or {}).get("temp_dir")

    def running(d):
        d.stages[stage] = StageRecord(state=State.RUNNING, params_hash=h, params=params[stage], started=started)
        d.anomalies = [a for a in d.anomalies if a.data.get("stage") != stage]
        if not side:  # nothing reads a side stage's outputs
            _reset_downstream(d, stage)

    def ours(d):
        st = d.stages.get(stage)
        # reset_from or another writer changed the stage meanwhile: its decision wins over this result
        return st is not None and st.state is State.RUNNING and st.started == started and st.params_hash == h

    try:
        rec = _update(paths, sid, running, RUNNING_WRITE_TRIES)
    except fileio.LockTimeout as e:
        log.warning("%s %s: not started, registry lock busy: %s", sid, stage, e, extra={"sid": sid})
        return "retry"
    except OSError as e:  # offered again after poll_s; nothing ran, nothing is lost
        log.error("%s %s: not started, registry not writable: %s", sid, stage, _last_line(e),
                  extra={"sid": sid, "console": True})
        run_.not_recorded[(sid, stage)] = f"not started, registry not writable: {_last_line(e)}"
        return "none"
    run_.not_recorded.pop((sid, stage), None)
    run_.current = (sid, stage, started, h, temp_root)
    run_.refresh()
    with run_.logs.session(sid):
        if (sid, stage) not in run_.waiting_logged:
            log.info("%s %s: started", sid, stage, extra={"sid": sid, "console": True})
        applied = []  # the result was written: nobody changed the stage while it ran
        try:
            res = _call_stage(paths, sid, stage, rec, cfg, temp_root, run_)
            missing = [k for k in REQUIRED[stage] if not _output_ok(paths, res.outputs, k)]
            if missing:
                raise FileNotFoundError(f"required output missing: {', '.join(missing)}")
            anomalies = [dataclasses.replace(a, data={**a.data, "stage": stage}) for a in res.anomalies]
            if side:
                found = []  # preview videos are for looking at: there is nothing to check
            else:
                series = [r.stages[stage].metrics for r in registry.all_records(paths)
                          if r.sid != sid and not r.excluded and r.stages.get(stage) is not None
                          and r.stages[stage].state is State.DONE]
                found = checks.evaluate(stage, res.metrics, series + [res.metrics], cfg)

            def done(d):
                if ours(d):
                    applied.append(True)
                    st = d.stages[stage]
                    st.state, st.finished, st.error = State.DONE, _now(), None
                    st.outputs, st.metrics, st.checks = dict(res.outputs), dict(res.metrics), found
                    d.anomalies = [a for a in d.anomalies if a.data.get("stage") != stage] + anomalies

            # inside the try: metrics the registry cannot store fail this stage, not the batch
            try:
                _update(paths, sid, done, RESULT_WRITE_TRIES)
            except OSError as e:
                raise ResultNotRecorded(f"result not recorded in the registry: {_last_line(e)}; its outputs "
                                        f"are on disk") from e
        except WaitingForInput as w:
            def waiting(d):
                if ours(d):
                    applied.append(True)
                    d.stages[stage].state, d.stages[stage].error = State.WAITING, str(w)

            outcome = _record_outcome(paths, sid, stage, waiting, "waiting", applied, run_)
            if outcome == "waiting" and (sid, stage) not in run_.waiting_logged:  # retried every poll
                run_.waiting_logged.add((sid, stage))
                log.info("%s %s: %s", sid, stage, w, extra={"sid": sid, "console": True})
        except Exception as e:
            last = _last_line(e)
            log.error("%s %s: failed: %s", sid, stage, last, extra={"sid": sid, "console": True})
            log.error("%s", traceback.format_exc(), extra={"sid": sid, "session_only": True})
            _clean_temp(paths, temp_root, sid)
            skipped = []
            crash = ([Anomaly("stage_crashed", "error", last,
                              data={"exit_code": e.exit_code, "stage": stage, "cause": e.cause})]
                     if isinstance(e, isolate.StageCrashed) else [])

            def failed(d):
                if ours(d):
                    applied.append(True)
                    st = d.stages[stage]
                    st.state, st.error, st.finished = State.FAILED, last, _now()
                    d.anomalies.extend(crash)
                    # nothing reads a side stage's outputs: its failure skips nothing
                    skipped.append(0 if side else _skip_downstream(d, stage, cfg))

            outcome = _record_outcome(paths, sid, stage, failed, "failed", applied, run_)
            if outcome == "failed":
                run_.result["failed"] += 1
                run_.result["skipped"] += sum(skipped)
        else:
            run_.waiting_logged.discard((sid, stage))
            if applied:
                worst = max((c.level for c in found), key=_LEVEL.get, default="ok")
                log.info("%s %s: done (checks: %s)", sid, stage, worst, extra={"sid": sid, "console": True})
                run_.result["done"] += 1
                outcome = "done"
            else:
                outcome = _superseded(sid, stage)
    run_.current = None
    run_.refresh()
    return outcome


def _record_outcome(paths, sid, stage, change, outcome, applied, run_: _Run) -> str:
    try:
        _update(paths, sid, change, RESULT_WRITE_TRIES)
    except OSError as e:  # the batch goes on; the stage stays RUNNING on disk and is offered after poll_s
        log.error("%s %s: %s not recorded, registry not writable: %s", sid, stage, outcome, _last_line(e),
                  extra={"sid": sid, "console": True})
        run_.not_recorded[(sid, stage)] = f"{outcome} not recorded, registry not writable: {_last_line(e)}"
        return "none"
    return outcome if applied else _superseded(sid, stage)


def _superseded(sid, stage) -> str:
    """The stage was changed while it ran (reset_from, a new decision): its result is dropped, it runs again."""
    log.info("%s %s: changed while running (reset or new decision); result dropped, runs again", sid, stage,
             extra={"sid": sid})
    return "retry"


def _skip(tried, not_started, records, once, poll_s) -> set:
    """(sid, stage) pairs not to offer now.

    A waiting stage tried less than poll_s ago (once=True: tried in this run) is skipped only while it is
    still WAITING: a decision written meanwhile makes it ready at once (C1). A stage that could not start
    is skipped for poll_s (once=True: for the run).
    """
    now = time.monotonic()
    still_waiting = {(r.sid, s) for r in records for s, st in r.stages.items() if st.state is State.WAITING}
    skip = {k for k, t in tried.items() if k in still_waiting and (once or now - t < poll_s)}
    return skip | {k for k, t in not_started.items() if once or now - t < poll_s}


def _clear_stop_request(paths, tries: int = 5) -> None:
    """Remove stop.request; on Windows a monitor reading it at that moment makes the removal fail briefly."""
    for i in range(tries):
        try:
            paths.stop_request.unlink(missing_ok=True)
            return
        except PermissionError:
            if i == tries - 1:
                raise
            time.sleep(0.05)


def _sleep_unless_stopped(paths, seconds: float) -> None:
    """Sleep in slices of 1 s: a stop request ends a wait that lasts poll_s (30 s) within a second."""
    end = time.monotonic() + seconds
    while not paths.stop_request.exists():
        left = end - time.monotonic()
        if left <= 0:
            return
        time.sleep(min(left, 1.0))


def _loop(paths, run_: _Run) -> dict:
    tried = {}  # (sid, stage) -> monotonic time of the last attempt that ended WAITING
    not_started = {}  # (sid, stage) -> monotonic time of the last attempt that could not start
    attempted = {}  # finalize key -> (ok, monotonic time)
    reported = set()
    last_msg = None
    once, poll_s = run_.once, run_.poll_s

    def ready(records, blocked, view=None):
        # a blocked wait is not checked again: only a config fix (or a working finalize) moves it
        skip = _skip(tried, not_started, records, once, poll_s) | {(sid, stage) for sid, stage, _, _ in blocked}
        return next_ready(records, paths, until=run_.until, sids=run_.sids, skip=skip, _view=view)

    def person_waits(records, broken, view=None):
        problems = schedule_problems(paths, records, view, broken)
        waiting = _waiting(records, _Scope(records, run_.sids), run_.stages)
        return (*_split_waits(waiting, records, problems), problems)

    while True:
        # top of the iteration: a stage that is running has finished, the next one is not chosen yet
        if paths.stop_request.exists():
            log.info("stop requested: the runner stops before its next stage", extra={"console": True})
            run_.result["stopped"] = True
            return dict(run_.result)
        records, broken = registry.read_all(paths)
        view = _View(paths)
        due = _finalize_due(records, view, _Scope(records, run_.sids), attempted, once, poll_s)
        if due:
            for mouse, done, key in due:
                ok = _run_finalize(paths, mouse, done, run_)
                fails = 0 if ok else (attempted[key][2] if key in attempted else 0) + 1
                attempted[key] = (ok, time.monotonic(), fails)
            run_.refresh()
            continue
        person, blocked, problems = person_waits(records, broken, view)
        job = ready(records, blocked, view)
        for sid, err in problems.items():
            if (sid, err) not in reported:
                reported.add((sid, err))
                log.error("%s: cannot be scheduled: %s", sid, err, extra={"sid": sid, "console": True})
        if job is not None:
            outcome = _run_stage(paths, *job, run_)
            tried.pop(job, None)
            not_started.pop(job, None)
            if outcome == "waiting":
                tried[job] = time.monotonic()
            elif outcome == "none":
                not_started[job] = time.monotonic()
            continue  # 'retry': offered again on the next loop
        if not once and person:  # only a wait a person can resolve keeps the runner polling
            msg = _wait_message(person, blocked)
            if msg != last_msg:
                log.info(msg, extra={"console": True})
                last_msg = msg
            run_.refresh(status="waiting", records=records, broken=broken, problems=problems)
            now = time.monotonic()
            # next retry of a waiting stage; one that is due but was not offered (its session cannot be
            # scheduled now) must not turn this into a busy loop, so it waits a full poll
            due_at = [t + poll_s for t in list(tried.values()) + list(not_started.values()) if t + poll_s > now]
            _sleep_unless_stopped(paths, max(0.01, min(due_at) - now if due_at else poll_s))
            continue
        # about to return: a decision may have landed after this snapshot was read (C1)
        fresh, fresh_broken = registry.read_all(paths)
        fresh_person, fresh_blocked, _ = person_waits(fresh, fresh_broken)
        if ready(fresh, fresh_blocked) is not None or (not once and fresh_person):
            continue
        if blocked and not person:
            log.error("nothing can progress until a config is fixed: %s", _blocked_text(blocked, problems),
                      extra={"console": True})
        run_.result["waiting"] = len(_waiting(fresh, _Scope(fresh, run_.sids), run_.stages))
        run_.result["not_recorded"] = _not_recorded(fresh, run_)
        return dict(run_.result)


def _not_recorded(records, run_: _Run) -> list:
    """(sid, stage, why) of the stages whose start or outcome the registry did not take in this run, and of
    any stage left RUNNING on disk (no stage runs when the loop returns)."""
    out = dict(run_.not_recorded)
    for r in records:
        for s, st in r.stages.items():
            if st.state is State.RUNNING:
                out.setdefault((r.sid, s), "left RUNNING in the registry: its outcome was not recorded")
    return [(sid, stage, why) for (sid, stage), why in sorted(out.items())]


def _wait_message(person, blocked) -> str:
    whats = [w for _, _, w in person]
    msg = (f"waiting: thresholds {whats.count('thresholds')}, "
           f"manual inspection {sum(w not in _WAIT_KIND for w in whats)}")
    if whats.count("crop_finalize"):
        msg += f", crop finalize {whats.count('crop_finalize')}"
    if blocked:
        msg += "; blocked by a session that cannot be scheduled: " + ", ".join(
            f"{sid} {stage} <- {','.join(by)}" for sid, stage, _, by in blocked)
    return msg


def run(series_root: Path, until: str | None = None, sids: list[str] | None = None, once: bool = False,
        poll_s: float = 30, console: bool = False) -> dict:
    """Process the series until no stage is ready; returns {"done", "failed", "waiting", "skipped",
    "not_recorded"}.

    done / failed / skipped count stage outcomes of this run; waiting counts stages left WAITING at exit;
    not_recorded lists (sid, stage, why) of stages the registry did not take (start or outcome) or left
    RUNNING.
    While sessions wait for a person (thresholds, manual inspection) the process polls every `poll_s`
    and continues when the decision arrives; once=True returns as soon as no stage is ready. console=True
    prints one ASCII line per event (stage started / done / failed, waits) on stdout.
    """
    paths = SeriesPaths(Path(series_root).resolve())  # absolute before the chdir below
    run_ = _Run(paths, until, sids, once, poll_s, console)
    config.resolve(paths)  # a broken series config stops here, before any state changes
    lock = _acquire_lock(paths)
    try:
        mlock = _acquire_machine_lock(paths)
    except BaseException:
        _release_lock(paths, lock)
        raise
    cwd = os.getcwd()
    try:
        os.chdir(provenance.REPO_ROOT)  # the old config.py creates config.json in the current folder
        run_.open()
        log.info("run started: pid %d, until %s, sessions %s, once %s", os.getpid(), until or "export",
                 "all" if sids is None else ",".join(sids), once)
        # a request left by an earlier run (a killed runner never removed it) must not stop this one
        _clear_stop_request(paths)
        known = {r.sid for r in registry.all_records(paths)}
        for sid in sids or ():
            if sid not in known:
                log.warning("%s: no such session in the registry", sid)
        _reset_running(paths)
        result = _loop(paths, run_)
        if result.get("stopped"):
            _clear_stop_request(paths)
        run_.exit = "stopped" if result.get("stopped") else "finished"
        log.info("run finished: %s", ", ".join(f"{k} {v}" for k, v in result.items()))
        return result
    except BaseException as e:
        run_.exit = ("interrupted" if isinstance(e, KeyboardInterrupt)
                     else "error: " + " ".join(_last_line(e).split()))
        run_.abort()
        log.error("run stopped: %s", _last_line(e))
        raise
    finally:
        run_.close()
        os.chdir(cwd)
        _drop_lock(machine_lock(), mlock)
        _release_lock(paths, lock)


# --- summary table ---

def _crop_box(rec) -> str | None:
    """The box MC used (its metrics), not the one finalize placed: a box changed since shows once MC re-ran."""
    mc = rec.stages.get("mc")
    if mc is None or mc.state is not State.DONE or "crop" not in mc.metrics:
        return None
    b = mc.metrics["crop"]
    return "uncropped" if b is None else f"{b[2]}x{b[3]}@{b[0]},{b[1]}"


_KEYS = {  # stage -> [(column, metric of that stage, or a function of the record)]; spec 5.11 key numbers
    "crop": [("crop_box", _crop_box)],
    "mc": [("mc_max_shift_px", "max_shift_px")],
    "thresholds": [],
    "cnmf": [("cnmf_n_components", "n_components")],
    "gate": [("gate_decision", "gate")],
    "autoinspect": [("autoinspect_n_kept", "n_kept"), ("autoinspect_deleted_frac", "deleted_frac")],
    "export": [("export_n_neurons", "n_neurons"), ("export_median_snr", "median_snr")],
}
STAGE_KEYS = _KEYS  # public name for the monitor window
# a side stage's column follows its parent's key numbers
SUMMARY_COLUMNS = (["sid", "mouse", "excluded"]
                   + [c for s in STAGES
                      for c in [s] + [k for k, _ in _KEYS[s]] + [x for x, p in SIDE_STAGES.items() if p == s]]
                   + ["gSig", "min_corr", "min_pnr", "worst_check", "waiting_for", "error", "anomalies"])


def _summary_row(paths, rec, problem=None) -> dict:
    row = {"sid": rec.sid, "mouse": rec.mouse, "excluded": rec.excluded}
    levels, waiting_for, error = [], None, None
    try:
        cfg = config.resolve(paths, sid=rec.sid, mouse=rec.mouse)
    except Exception:
        cfg = None
    for s in ALL_STAGES:
        st = rec.stages.get(s) or StageRecord()
        if s in SIDE_STAGES and _side_off(s, cfg):
            row[s] = "off"  # its old record, if any, holds nothing
            continue
        row[s] = st.state.value
        for col, key in _KEYS.get(s, ()):
            if callable(key):
                row[col] = key(rec)
            else:
                row[col] = st.metrics.get(key) if st.state is State.DONE else None
        if st.state is State.DONE:
            levels += [c.level for c in st.checks]
        elif st.state is State.FAILED:
            levels.append("error")
        elif st.state is State.WAITING:
            waiting_for = waiting_for or _what(st)
    first = first_failed(rec, cfg)  # a main failure is named before a failed preview
    if first is not None:
        error = f"{first}: {rec.stages[first].error}"
    c = cfg["cnmf"] if cfg is not None else (rec.stages.get("cnmf") or StageRecord()).metrics
    if problem is not None:  # what holds the session now goes before the error of an earlier failure
        error = f"cannot be scheduled: {problem}"
    elif error is None and _finalize_failure(rec) is not None:
        error = _finalize_failure(rec).message
    row.update(gSig=c.get("gSig"), min_corr=c.get("min_corr"), min_pnr=c.get("min_pnr"),
               worst_check=max(levels, key=_LEVEL.get) if levels else "", waiting_for=waiting_for,
               error=error, anomalies=",".join(a.code for a in rec.anomalies))
    return row


def _broken_row(f, problem) -> dict:
    sid = Path(f).stem
    return {"sid": sid, "mouse": naming.mouse_id(sid) or None, "error": problem}


_SUMMARY_WARNED: set = set()  # summary files whose last write failed (warned once until a write works)


def summary(paths, records=None, problems=None, broken=None):
    """One row per session (state and key number of each stage, thresholds, worst check); writes summary.csv.

    A session that cannot be scheduled has the reason in `error`; a registry file that cannot be read gets a
    row of its own with the file and the parse error.

    Written only when its content changes (a waiting runner refreshes it every poll). One write attempt, no
    retry loop (it runs after every transition and must not hold the batch); a failure (file open in Excel)
    is a warning, given once until a write succeeds again.
    """
    import pandas as pd

    if records is None or broken is None:
        everything, found = registry.read_all(paths)
        records = everything if records is None else records
        broken = found if broken is None else broken
    if problems is None:
        problems = schedule_problems(paths, records, broken=broken)
    rows = ([_summary_row(paths, r, problems.get(r.sid)) for r in records]
            + [_broken_row(f, problems[Path(f).stem]) for f, _ in broken])
    df = pd.DataFrame(sorted(rows, key=lambda x: x["sid"]), columns=SUMMARY_COLUMNS)
    text = df.to_csv(index=False, lineterminator="\n")
    key = str(paths.summary_csv)
    try:
        if fileio.read_text(paths.summary_csv) == text:
            _SUMMARY_WARNED.discard(key)
            return df
    except (OSError, UnicodeDecodeError):
        pass  # absent or unreadable now: written below
    try:
        fileio.write_atomic(paths.summary_csv, lambda fh: fh.write(text), newline="", replace_tries=1)
    except Exception as e:
        if key not in _SUMMARY_WARNED:
            _SUMMARY_WARNED.add(key)
            log.warning("summary.csv not written (warned once until it works again): %s", _last_line(e))
    else:
        _SUMMARY_WARNED.discard(key)
    return df
