"""Batch scheduler: drives every session of a series through the stages, resumable and failure-isolated.

One process per series (_bearmind/run.lock). Every loop iteration rereads the registry and the configs, so
what other tools write meanwhile (thresholds, manual-inspection decisions, reset_from, a rescan) is picked
up. A stage is up to date when it is DONE, its parameter hash matches the current configuration and its
required outputs exist; otherwise it and every later stage of that session are recomputed (spec 5.10).
The runner writes only the fields it owns (the stage record and that stage's anomalies) through
registry.update_record, so concurrent writers do not lose their changes.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import logging
import os
import shutil
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import psutil

from bearmind import checks, config, fileio, provenance, registry
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, Decision, StageRecord, State
from bearmind.stages import WaitingForInput

log = logging.getLogger("bearmind.runner")

# "module:function" strings: stage modules import CaImAn and friends, so they load only when a stage runs.
# Tests replace entries with callables.
STAGE_FUNCS = {
    "crop": "bearmind.stages.crop:detect",
    "mc": "bearmind.stages.mc:run",
    "thresholds": "bearmind.stages.thresholds:run",
    "cnmf": "bearmind.stages.cnmf:run",
    "gate": "bearmind.stages.gate:run",
    "autoinspect": "bearmind.stages.autoinspect:run",
    "export": "bearmind.stages.export:run",
}
FINALIZE_FUNC = "bearmind.stages.crop:finalize_mouse"

# Outputs that must exist for a DONE stage to count as done; previews are optional.
REQUIRED = {
    "crop": (),
    "mc": ("mc_video", "mc_shifts"),
    "thresholds": ("threshold_maps",),
    "cnmf": ("estimates",),
    "gate": (),
    "autoinspect": ("processed",),
    "export": ("npz", "json", "timestamps"),
}

HEARTBEAT_S = 30.0
COUNT_KEYS = ("done", "running", "waiting_thresholds", "waiting_manual", "waiting_crop", "failed", "skipped",
              "pending")
_WAIT_KIND = {"thresholds": "waiting_thresholds", "crop_finalize": "waiting_crop"}  # other waits: a person
_UNSET = "<unset>"  # crop.box_pos not written yet (finalize pending), distinct from null (no FOV: uncropped)
_LEVEL = {"ok": 0, "warning": 1, "error": 2}


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _func(spec):
    if callable(spec):
        return spec
    module, name = spec.split(":")
    return getattr(importlib.import_module(module), name)


def _stages_until(until) -> tuple:
    if until is None:
        return STAGES
    if until not in STAGES:
        raise ValueError(f"unknown stage {until!r}; stages: {', '.join(STAGES)}")
    return STAGES[:STAGES.index(until) + 1]


def _what(st) -> str:
    """What a WAITING stage waits for, from its error text 'waiting for <what>'."""
    e = st.error or ""
    return e[len("waiting for "):] if e.startswith("waiting for ") else e


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


def _latest_inspection(rec) -> str | None:
    found = [d for d in rec.decisions if d.what == "manual_inspection"]
    return found[-1].value if found else None


def stage_params(stage: str, rec, cfg: dict, paths) -> dict:
    """Resolved inputs of a stage (R25): stored in StageRecord.params and hashed; machine settings excluded.

    The crop box and the thresholds are written to the configs after crop / thresholds are done, so they
    are inputs of MC and CNMF only, never of the stage that produced them.
    """
    if stage == "crop":
        return {"parts": [{"folder": p.folder, "chunks": list(p.chunks), "n_frames": p.n_frames}
                          for p in rec.parts],
                "n_frames": rec.n_frames}
    if stage == "mc":
        c = cfg.get("crop") or {}
        return {"mc": cfg["mc"], "box": c.get("box"), "box_pos": c.get("box_pos", _UNSET), "fps": rec.fps}
    if stage == "thresholds":
        return {"gsig_range": cfg["thresholds"]["gsig_range"], "tsub": cfg["cnmf"]["tsub"]}
    if stage == "cnmf":
        from bearmind.stages.cnmf import cnmf_params  # imports CaImAn

        try:
            return cnmf_params(cfg, rec.fps)
        except WaitingForInput as w:
            return {"waiting_for": w.what}
    if stage == "gate":
        if not (cfg.get("gate") or {}).get("enabled", True):
            return {"enabled": False}
        decision = _latest_inspection(rec)
        out = {"enabled": True, "decision": decision, "manual_result": None}
        f = paths.manual_result(rec.sid)
        if decision == "cleaned" and f.is_file():  # a re-saved inspection result re-runs what follows
            st = f.stat()
            out["manual_result"] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
        return out
    if stage == "autoinspect":
        return {"autoinspect": cfg["autoinspect"],
                "model_sha256": _model_sha(provenance.model_file(cfg)),
                "fps": round(rec.fps) if rec.fps else None}
    if stage == "export":
        # The threshold the processed pickle carries is the autoinspect.ml_threshold that stage ran with
        # (ae_launch stores the argument in autoinspection_config); its stored params give it without
        # loading a pickle of hundreds of MB per session on every check.
        ran = (rec.stages.get("autoinspect") or StageRecord()).params.get("autoinspect") or {}
        return {"ml_threshold": ran.get("ml_threshold", cfg["autoinspect"].get("ml_threshold"))}
    raise ValueError(f"unknown stage {stage!r}")


def _chain(rec, cfg, paths, upto=None):
    """params and hashes of every stage up to `upto`; each hash covers the hashes of all earlier stages."""
    params, hashes = {}, {}
    for s in STAGES[:STAGES.index(upto) + 1] if upto else STAGES:
        params[s] = stage_params(s, rec, cfg, paths)
        hashes[s] = config.params_hash({"params": params[s], "upstream": list(hashes.values())})
    return params, hashes


def stage_hash(rec, stage: str, cfg: dict, paths) -> str:
    """params_hash of the stage parameters and of all earlier stages' hashes."""
    return _chain(rec, cfg, paths, stage)[1][stage]


def _output_ok(paths, outputs: dict, key: str) -> bool:
    return key in outputs and (paths.root / outputs[key]).is_file()


class _View:
    """Resolved configs and stage hashes of one registry snapshot (one loop iteration)."""

    def __init__(self, paths):
        self.paths = paths
        self._cfg, self._hashes = {}, {}
        self.errors = {}  # sid -> why the session cannot be scheduled now

    def cfg(self, rec):
        if rec.sid not in self._cfg:
            try:
                self._cfg[rec.sid] = config.resolve(self.paths, sid=rec.sid, mouse=rec.mouse)
            except Exception as e:  # a broken session config holds this session only, its states stay
                self._cfg[rec.sid] = None
                self.errors[rec.sid] = f"config: {type(e).__name__}: {e}"
        return self._cfg[rec.sid]

    def hashes(self, rec):
        if rec.sid not in self._hashes:
            h, cfg = None, self.cfg(rec)
            if cfg is not None:
                try:
                    h = _chain(rec, cfg, self.paths)[1]
                except Exception as e:
                    self.errors[rec.sid] = f"stage parameters: {type(e).__name__}: {e}"
            self._hashes[rec.sid] = h
        return self._hashes[rec.sid]

    def current(self, rec, stage) -> bool:
        st, h = rec.stages.get(stage), self.hashes(rec)
        return (h is not None and st is not None and st.state is State.DONE and st.params_hash == h[stage]
                and all(_output_ok(self.paths, st.outputs, k) for k in REQUIRED[stage]))

    def blocked(self, rec, stage) -> bool:
        """FAILED / SKIPPED with unchanged parameters: retried only after reset_from or a parameter change."""
        st, h = rec.stages.get(stage), self.hashes(rec)
        return (st is not None and st.state in (State.FAILED, State.SKIPPED) and h is not None
                and st.params_hash == h[stage])


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
    """(sid, stage) of the next stage to run, or None: lowest stage index first, then sid order.

    Ready: session not excluded, every earlier stage up to date, the stage itself not up to date and not
    FAILED/SKIPPED with unchanged parameters. `skip` holds (sid, stage) pairs not to offer now (waiting
    stages already tried). crop.finalize_mouse is scheduled by run(), not here.
    """
    view = _view or _View(paths)
    scope = _Scope(records, sids)
    recs = sorted((r for r in records if not r.excluded), key=lambda r: r.sid)
    for i, stage in enumerate(_stages_until(until)):
        for r in recs:
            if (r.sid, stage) in skip or not scope.has(r, stage) or view.hashes(r) is None:
                continue
            if (all(view.current(r, s) for s in STAGES[:i]) and not view.current(r, stage)
                    and not view.blocked(r, stage)):
                return r.sid, stage
    return None


# --- registry changes made by the runner ---

def _skip_downstream(d, stage) -> int:
    n = 0
    for s in STAGES[STAGES.index(stage) + 1:]:
        st = d.stages.setdefault(s, StageRecord())
        st.state, st.error = State.SKIPPED, f"skipped: {stage} failed"
        n += 1
    return n


def reset_from(paths, sid: str, stage: str, who: str) -> None:
    """Re-analysis: the stage and all later ones -> PENDING. Old outputs stay until the stage replaces them."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages: {', '.join(STAGES)}")
    now = _now()

    def reset(d):
        for s in STAGES[STAGES.index(stage):]:
            st = d.stages.setdefault(s, StageRecord())
            st.state, st.error = State.PENDING, None
        d.decisions.append(Decision("rerun_from", stage, who, now))

    registry.update_record(paths, sid, reset)


def _clean_temp(temp_root, sid) -> None:
    """Remove <machine temp>/<sid> after a failed or killed stage (the stages clean it on success)."""
    if not temp_root:
        return
    d = Path(temp_root) / sid
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


class _Logs:
    """File handlers on the 'bearmind' logger; the root logger is left alone."""

    def __init__(self, paths):
        self.paths = paths
        self.logger = logging.getLogger("bearmind")
        paths.logs_dir.mkdir(parents=True, exist_ok=True)
        self.handler = logging.FileHandler(paths.series_log, encoding="utf-8")
        self.handler.setFormatter(_FMT)
        self.handler.addFilter(_SeriesFilter())
        self.logger.addHandler(self.handler)
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
        self.logger.setLevel(self.level)


class _Heartbeat:
    """_bearmind/heartbeat.json: written on every transition and every HEARTBEAT_S by a daemon thread."""

    def __init__(self, paths):
        self.paths = paths
        self.lock = threading.Lock()
        self.payload = {"pid": os.getpid(), "time": None, "current": None,
                        "counts": dict.fromkeys(COUNT_KEYS, 0), "status": "running"}
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


def _read_lock(lock: Path, tries: int = 10):
    for i in range(tries):
        try:
            return json.loads(lock.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            if i == tries - 1:
                return None
            time.sleep(0.1)  # the holder may be writing it right now


def _alive(holder) -> bool:
    """The lock holder still runs: same PID and same process start time (a PID can be reused)."""
    try:
        p = psutil.Process(int(holder["pid"]))
        if holder.get("create_time") is None:
            return p.is_running()
        return abs(p.create_time() - float(holder["create_time"])) < 1.0
    except (psutil.NoSuchProcess, psutil.ZombieProcess, KeyError, TypeError, ValueError):
        return False
    except psutil.AccessDenied:
        return True  # cannot tell: do not take the series from a running process


def _acquire_lock(paths) -> dict:
    lock = paths.lock
    lock.parent.mkdir(parents=True, exist_ok=True)
    me = psutil.Process()
    mine = {"pid": me.pid, "create_time": me.create_time(), "time": _now()}
    for _ in range(5):
        try:
            fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = _read_lock(lock)
            if holder is not None and _alive(holder):
                raise RuntimeError(f"series is already being processed by PID {holder['pid']}")
            log.warning("run lock of a process that is gone taken over: %s", holder)
            try:
                lock.unlink()
            except FileNotFoundError:
                pass
            continue
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(mine, fh)
        return mine
    raise RuntimeError(f"cannot take the run lock {lock}")


def _release_lock(paths, mine) -> None:
    holder = _read_lock(paths.lock, tries=3)
    if holder and holder.get("pid") == mine["pid"] and holder.get("create_time") == mine["create_time"]:
        paths.lock.unlink(missing_ok=True)


# --- the run ---

def _counts(records, scope, stages) -> dict:
    c = dict.fromkeys(COUNT_KEYS, 0)
    for r in records:
        if r.excluded:
            continue
        for s in stages:
            if not scope.has(r, s):
                continue
            st = r.stages.get(s) or StageRecord()
            if st.state is State.WAITING:
                c[_WAIT_KIND.get(_what(st), "waiting_manual")] += 1
            else:
                c[st.state.value] += 1
    return c


class _Run:
    """State of one run: outcome counts, the stage being run, logs and heartbeat."""

    def __init__(self, paths, until, sids, once, poll_s):
        self.paths, self.sids, self.once, self.poll_s = paths, sids, once, poll_s
        self.until, self.stages = until, _stages_until(until)
        self.result = {"done": 0, "failed": 0, "waiting": 0, "skipped": 0}
        self.current = None  # (sid, stage, started, params_hash, temp_root)
        self.logs = self.hb = None
        self.waiting_logged = set()

    def open(self):
        self.logs = _Logs(self.paths)
        self.hb = _Heartbeat(self.paths)
        self.hb.update(status="running")
        self.hb.start()

    def refresh(self, status="running", records=None):
        """Heartbeat counts and summary.csv from the registry as it is now."""
        records = registry.all_records(self.paths) if records is None else records
        cur = None if self.current is None else {"sid": self.current[0], "stage": self.current[1]}
        self.hb.update(counts=_counts(records, _Scope(records, self.sids), self.stages), status=status,
                       current=cur)
        summary(self.paths, records)

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
        _clean_temp(temp_root, sid)
        self.current = None

    def close(self):
        if self.hb is not None:
            self.hb.stop()
            try:
                self.refresh(status="idle")
            except Exception as e:
                log.warning("final heartbeat / summary failed: %s", _last_line(e))
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

        registry.update_record(paths, rec.sid, back)
        _clean_temp(temp_root, rec.sid)
        log.warning("%s %s: was running at start (process killed?), set to pending", rec.sid,
                    ",".join(running), extra={"sid": rec.sid})


def _box_list(box):
    if box is None:
        return None
    if hasattr(box, "top"):
        return [int(box.top), int(box.left), int(box.height), int(box.width)]
    return [int(x) for x in box]


def _finalize_due(records, view, scope, attempted) -> list:
    """Mice whose crop detection is settled for every session and whose box is not placed yet.

    Settled = up to date, or FAILED with unchanged parameters (one bad session does not hold its mouse).
    Only sessions with crop DONE are passed to finalize_mouse. A crop re-run drops metrics['box'], so the
    box is placed again; `attempted` stops a retry loop when finalize leaves a session without a box.
    """
    by_mouse = {}
    for r in records:
        if not r.excluded and scope.has(r, "crop"):
            by_mouse.setdefault(r.mouse, []).append(r)
    due = []
    for mouse, recs in sorted(by_mouse.items()):
        done = [r for r in recs if view.current(r, "crop")]
        if not done or any(not (view.current(r, "crop") or view.blocked(r, "crop") or view.hashes(r) is None)
                           for r in recs):
            continue
        if all("box" in r.stages["crop"].metrics and "box_pos" in (view.cfg(r).get("crop") or {})
               for r in done):
            continue
        key = (mouse, tuple((r.sid, r.stages["crop"].params_hash, r.stages["crop"].finished) for r in done))
        if key not in attempted:
            due.append((mouse, done, key))
    return due


def _run_finalize(paths, mouse, done, run_: _Run) -> None:
    before = {r.sid: len(r.anomalies) for r in done}
    finished = {r.sid: r.stages["crop"].finished for r in done}

    def ours(d):
        st = d.stages.get("crop")
        return st is not None and st.state is State.DONE and st.finished == finished[d.sid]

    try:
        cfg = config.resolve(paths, mouse=mouse)
        boxes = _func(FINALIZE_FUNC)(paths, mouse, done, cfg) or {}
    except Exception as e:
        last, tb = _last_line(e), traceback.format_exc()
        for r in done:
            with run_.logs.session(r.sid):
                log.error("%s crop: finalize of mouse %s failed: %s", r.sid, mouse, last, extra={"sid": r.sid})
                log.error("%s", tb, extra={"sid": r.sid, "session_only": True})

            def fail(d):
                if ours(d):
                    d.stages["crop"].state = State.FAILED
                    d.stages["crop"].error = f"crop finalize failed: {last}"
                    run_.result["skipped"] += _skip_downstream(d, "crop")
                    run_.result["failed"] += 1

            registry.update_record(paths, r.sid, fail)
        return

    for r in done:
        new = [dataclasses.replace(a, data={**a.data, "stage": "crop"}) for a in r.anomalies[before[r.sid]:]]

        def apply(d, r=r, new=new):
            if not ours(d):
                return
            if r.sid in boxes:
                d.stages["crop"].metrics["box"] = _box_list(boxes[r.sid])
            have = {a.code for a in d.anomalies if a.data.get("stage") == "crop"}
            d.anomalies.extend(a for a in new if a.code not in have)

        rec = registry.update_record(paths, r.sid, apply)
        with run_.logs.session(r.sid):
            if r.sid in boxes:
                log.info("%s crop: box %s (mouse %s)", r.sid, rec.stages["crop"].metrics.get("box"), mouse,
                         extra={"sid": r.sid})
            else:
                log.warning("%s crop: finalize placed no box (no crop metrics)", r.sid, extra={"sid": r.sid})


def _run_stage(paths, sid, stage, run_: _Run) -> str:
    """Run one stage of one session; returns 'done' | 'waiting' | 'failed' (or 'none': not started)."""
    try:
        rec = registry.load_record(paths, sid)
        cfg = config.resolve(paths, sid=sid, mouse=rec.mouse)  # reread before every stage
        params, hashes = _chain(rec, cfg, paths, stage)
    except Exception as e:  # changed since next_ready looked: not started, retried later
        log.error("%s %s: not started: %s", sid, stage, _last_line(e), extra={"sid": sid})
        return "none"
    h = hashes[stage]
    started = _now()
    temp_root = (cfg.get("machine") or {}).get("temp_dir")

    def running(d):
        d.stages[stage] = StageRecord(state=State.RUNNING, params_hash=h, params=params[stage], started=started)
        d.anomalies = [a for a in d.anomalies if a.data.get("stage") != stage]
        for s in STAGES[STAGES.index(stage) + 1:]:  # recomputed after this stage, whatever their hash
            st = d.stages.setdefault(s, StageRecord())
            if st.state is not State.PENDING:
                st.state, st.error = State.PENDING, None

    def ours(d):
        st = d.stages.get(stage)
        # reset_from or another writer changed the stage meanwhile: its decision wins over this result
        return st is not None and st.state is State.RUNNING and st.started == started and st.params_hash == h

    rec = registry.update_record(paths, sid, running)
    run_.current = (sid, stage, started, h, temp_root)
    run_.refresh()
    with run_.logs.session(sid):
        if (sid, stage) not in run_.waiting_logged:
            log.info("%s %s: started", sid, stage, extra={"sid": sid})
        try:
            res = _func(STAGE_FUNCS[stage])(paths, rec, cfg)
            missing = [k for k in REQUIRED[stage] if not _output_ok(paths, res.outputs, k)]
            if missing:
                raise FileNotFoundError(f"required output missing: {', '.join(missing)}")
            anomalies = [dataclasses.replace(a, data={**a.data, "stage": stage}) for a in res.anomalies]
            series = [r.stages[stage].metrics for r in registry.all_records(paths)
                      if r.sid != sid and not r.excluded and r.stages.get(stage) is not None
                      and r.stages[stage].state is State.DONE]
            found = checks.evaluate(stage, res.metrics, series + [res.metrics], cfg)

            def done(d):
                if ours(d):
                    st = d.stages[stage]
                    st.state, st.finished, st.error = State.DONE, _now(), None
                    st.outputs, st.metrics, st.checks = dict(res.outputs), dict(res.metrics), found
                    d.anomalies = [a for a in d.anomalies if a.data.get("stage") != stage] + anomalies

            # inside the try: metrics the registry cannot store fail this stage, not the batch
            registry.update_record(paths, sid, done)
        except WaitingForInput as w:
            def waiting(d):
                if ours(d):
                    d.stages[stage].state, d.stages[stage].error = State.WAITING, str(w)

            registry.update_record(paths, sid, waiting)
            if (sid, stage) not in run_.waiting_logged:  # retried every poll; log once
                run_.waiting_logged.add((sid, stage))
                log.info("%s %s: %s", sid, stage, w, extra={"sid": sid})
            outcome = "waiting"
        except Exception as e:
            last = _last_line(e)
            log.error("%s %s: failed: %s", sid, stage, last, extra={"sid": sid})
            log.error("%s", traceback.format_exc(), extra={"sid": sid, "session_only": True})
            _clean_temp(temp_root, sid)
            skipped = []

            def failed(d):
                if ours(d):
                    st = d.stages[stage]
                    st.state, st.error, st.finished = State.FAILED, last, _now()
                    skipped.append(_skip_downstream(d, stage))

            registry.update_record(paths, sid, failed)
            run_.result["failed"] += 1
            run_.result["skipped"] += sum(skipped)
            outcome = "failed"
        else:
            run_.waiting_logged.discard((sid, stage))
            worst = max((c.level for c in found), key=_LEVEL.get, default="ok")
            log.info("%s %s: done (checks: %s)", sid, stage, worst, extra={"sid": sid})
            run_.result["done"] += 1
            outcome = "done"
    run_.current = None
    run_.refresh()
    return outcome


def _waiting(records, scope, stages) -> list:
    return [(r.sid, s, _what(r.stages[s])) for r in records if not r.excluded for s in stages
            if scope.has(r, s) and s in r.stages and r.stages[s].state is State.WAITING]


def _loop(paths, run_: _Run) -> dict:
    tried = {}  # (sid, stage) -> monotonic time of the last attempt that ended WAITING
    attempted = set()
    reported = set()
    last_msg = None
    while True:
        records = registry.all_records(paths)
        scope = _Scope(records, run_.sids)
        view = _View(paths)
        due = _finalize_due(records, view, scope, attempted)
        if due:
            for mouse, done, key in due:
                attempted.add(key)
                _run_finalize(paths, mouse, done, run_)
            run_.refresh()
            continue
        now = time.monotonic()
        skip = {k for k, t in tried.items() if run_.once or now - t < run_.poll_s}
        job = next_ready(records, paths, until=run_.until, sids=run_.sids, skip=skip, _view=view)
        for sid, err in view.errors.items():
            if (sid, err) not in reported:
                reported.add((sid, err))
                log.error("%s: cannot be scheduled: %s", sid, err, extra={"sid": sid})
        if job is not None:
            outcome = _run_stage(paths, *job, run_)
            if outcome in ("waiting", "none"):
                tried[job] = time.monotonic()
            else:
                tried.pop(job, None)
            continue
        waiting = _waiting(records, scope, run_.stages)
        if run_.once or not waiting:
            run_.result["waiting"] = len(waiting)
            return dict(run_.result)
        whats = [w for _, _, w in waiting]
        msg = (f"waiting: thresholds {whats.count('thresholds')}, "
               f"manual inspection {sum(w not in _WAIT_KIND for w in whats)}")
        if whats.count("crop_finalize"):
            msg += f", crop finalize {whats.count('crop_finalize')}"
        if msg != last_msg:
            log.info(msg)
            last_msg = msg
        run_.refresh(status="waiting", records=records)
        now = time.monotonic()
        # next retry of a waiting stage; one that is due but was not offered (its session cannot be
        # scheduled now) must not turn this into a busy loop, so it waits a full poll
        due_at = [tried[(s, st)] + run_.poll_s for s, st, _ in waiting
                  if (s, st) in tried and tried[(s, st)] + run_.poll_s > now]
        time.sleep(max(0.01, min(due_at) - now if due_at else run_.poll_s))


def run(series_root: Path, until: str | None = None, sids: list[str] | None = None, once: bool = False,
        poll_s: float = 30) -> dict:
    """Process the series until no stage is ready; returns {"done", "failed", "waiting", "skipped"}.

    done / failed / skipped count stage outcomes of this run; waiting counts stages left WAITING at exit.
    While sessions wait for a person (thresholds, manual inspection) the process polls every `poll_s`
    and continues when the decision arrives; once=True returns as soon as no stage is ready.
    """
    paths = SeriesPaths(Path(series_root).resolve())  # absolute before the chdir below
    run_ = _Run(paths, until, sids, once, poll_s)
    config.resolve(paths)  # a broken series config stops here, before any state changes
    mine = _acquire_lock(paths)
    cwd = os.getcwd()
    try:
        os.chdir(provenance.REPO_ROOT)  # the old config.py creates config.json in the current folder
        run_.open()
        log.info("run started: pid %d, until %s, sessions %s, once %s", os.getpid(), until or "export",
                 "all" if sids is None else ",".join(sids), once)
        known = {r.sid for r in registry.all_records(paths)}
        for sid in sids or ():
            if sid not in known:
                log.warning("%s: no such session in the registry", sid)
        _reset_running(paths)
        result = _loop(paths, run_)
        log.info("run finished: %s", ", ".join(f"{k} {v}" for k, v in result.items()))
        return result
    except BaseException as e:
        run_.abort()
        log.error("run stopped: %s", _last_line(e))
        raise
    finally:
        run_.close()
        os.chdir(cwd)
        _release_lock(paths, mine)


# --- summary table ---

def _box_text(metrics) -> str | None:
    if "box" not in metrics:
        return None
    b = metrics["box"]
    return "uncropped" if b is None else f"{b[2]}x{b[3]}@{b[0]},{b[1]}"


_KEYS = {  # stage -> [(column, metric)]; spec 5.11 key numbers
    "crop": [("crop_box", _box_text)],
    "mc": [("mc_max_shift_px", "max_shift_px")],
    "thresholds": [],
    "cnmf": [("cnmf_n_components", "n_components")],
    "gate": [("gate_decision", "gate")],
    "autoinspect": [("autoinspect_n_final", "n_final"), ("autoinspect_deleted_frac", "deleted_frac")],
    "export": [("export_n_neurons", "n_neurons"), ("export_median_snr", "median_snr")],
}
SUMMARY_COLUMNS = (["sid", "mouse", "excluded"]
                   + [c for s in STAGES for c in [s] + [k for k, _ in _KEYS[s]]]
                   + ["gSig", "min_corr", "min_pnr", "worst_check", "waiting_for", "error", "anomalies"])


def _summary_row(paths, rec) -> dict:
    row = {"sid": rec.sid, "mouse": rec.mouse, "excluded": rec.excluded}
    levels, waiting_for, error = [], None, None
    for s in STAGES:
        st = rec.stages.get(s) or StageRecord()
        row[s] = st.state.value
        for col, key in _KEYS[s]:
            if st.state is not State.DONE:
                row[col] = None
            else:
                row[col] = key(st.metrics) if callable(key) else st.metrics.get(key)
        if st.state is State.DONE:
            levels += [c.level for c in st.checks]
        elif st.state is State.FAILED:
            levels.append("error")
            error = error or f"{s}: {st.error}"
        elif st.state is State.WAITING:
            waiting_for = waiting_for or _what(st)
    try:
        c = config.resolve(paths, sid=rec.sid, mouse=rec.mouse)["cnmf"]
    except Exception:
        c = (rec.stages.get("cnmf") or StageRecord()).metrics
    row.update(gSig=c.get("gSig"), min_corr=c.get("min_corr"), min_pnr=c.get("min_pnr"),
               worst_check=max(levels, key=_LEVEL.get) if levels else "", waiting_for=waiting_for,
               error=error, anomalies=",".join(a.code for a in rec.anomalies))
    return row


def summary(paths, records=None):
    """One row per session (state and key number of each stage, thresholds, worst check); writes summary.csv.

    A failed write (file open in Excel) is a warning: the table is still returned.
    """
    import pandas as pd

    records = registry.all_records(paths) if records is None else records
    df = pd.DataFrame([_summary_row(paths, r) for r in records], columns=SUMMARY_COLUMNS)
    try:
        fileio.write_atomic(paths.summary_csv, lambda fh: df.to_csv(fh, index=False, lineterminator="\n"),
                            newline="")
    except Exception as e:
        log.warning("summary.csv not written: %s", _last_line(e))
    return df
