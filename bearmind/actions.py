"""Actions that change a series: one implementation for the command line and the monitor window.

Every function returns an ActionResult with codes; the callers turn the codes into their own texts (the CLI
prints its console messages, the window uses failure_text). Unknown sessions or stages raise before anything is
written, so a typo never leaves a half-applied batch. Every write goes through registry.update_record /
config.update_yaml, which keeps it safe next to a running runner.
"""
import csv
import getpass
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from bearmind import config, fileio, registry, runner, scan, scan_view
from bearmind.registry import SIDE_STAGES, STAGES, Decision, State
from bearmind.stages import WaitingForInput, gate

if TYPE_CHECKING:  # threshold_model imports this module
    from bearmind.threshold_model import Counters

log = logging.getLogger("bearmind.actions")


@dataclass
class ActionResult:
    ok: list = field(default_factory=list)        # sid the action was applied to
    failed: dict = field(default_factory=dict)    # sid -> (code, detail)
    warnings: list = field(default_factory=list)  # (sid, code)


class UnknownSession(ValueError):
    """Sessions that are not registered in the series; raised before any write."""

    def __init__(self, unknown, known=()):
        self.unknown = list(unknown)
        self.known = sorted(known)
        super().__init__(f"unknown session {', '.join(self.unknown)}")


_FAILURE_TEXT = {
    "no_suggestion": "No threshold suggestion yet: run the thresholds stage first, or enter the values by hand.",
    "invalid_suggestion": "The suggested thresholds are invalid: {detail}",
    "manual_result_missing": "The manual inspection result is not found. Save the cleaned file at {detail} "
                             "and repeat.",
    "no_split": "The session is not a stitched split recording, so there is nothing to confirm.",
    "no_failed_stage": "The session has no failed stage, so nothing was reset.",
    # the window's text (the CLI prints its own): the window has no force path, so it asks for nothing
    "already_passed": "Manual inspection already passed, so nothing was recorded. A new decision would make "
                      "autoinspection and export run again, and autoinspection is not reproducible run to run, "
                      "so the exported neurons could change.",
    "write_failed": "The write failed: {detail}",
}


def failure_text(code: str, detail: str) -> str:
    template = _FAILURE_TEXT.get(code, "The action failed ({code}): {detail}")
    return template.format(code=code, detail=detail)


def current_user() -> str:
    return getpass.getuser()


# --- validators (the CLI uses the same functions for its argument parser) ---

def gsig_error(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x != int(x) or x < 1:
        return f"--gsig must be a whole number >= 1, got {x}"
    return None


def min_corr_error(x):
    if not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 < x <= 1:
        return f"--min-corr must be a number above 0 and at most 1, got {x}"
    return None


def min_pnr_error(x):
    if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        return f"--min-pnr must be a finite number above 0, got {x}"
    return None


def threshold_error(gsig, min_corr, min_pnr):
    return gsig_error(gsig) or min_corr_error(min_corr) or min_pnr_error(min_pnr)


def preview_setting_error(x, name):
    """preview.tbin / preview.ssub: a whole number of frames / pixels averaged into one."""
    if isinstance(x, bool) or not isinstance(x, int) or x < 1:
        return f"{name} must be a whole number >= 1, got {x}"
    return None


# --- state helpers ---

def _records(paths) -> dict:
    return {r.sid: r for r in registry.all_records(paths)}


def check_sids(recs: dict, sids) -> None:
    unknown = [s for s in sids if s not in recs]
    if unknown:
        raise UnknownSession(unknown, recs)


THRESHOLD_KEYS = ("gSig", "min_corr", "min_pnr")


def waits_for_thresholds(view, rec) -> bool:
    """The batch waits for a person's thresholds for this session: not excluded, no thresholds in the resolved
    config, and crop, MC and the maps up to date (view: runner.StageView). Then next_ready offers CNMF-E and its
    check waits; with an earlier stage out of date the batch redoes that one first, so a choice now would apply
    to maps nobody has seen. The one rule for the Thresholds tab, Monitor and the CLI: build one view per
    refresh or command and ask it about every session.
    With a suggestion or without one: the batch waits either way."""
    if rec.excluded:
        return False
    cfg = view.cfg(rec)
    cnmf = cfg.get("cnmf") if cfg is not None else None
    if not isinstance(cnmf, dict):  # cannot be read: the session is "cannot be scheduled", not waiting
        return False
    return any(cnmf.get(k) is None for k in THRESHOLD_KEYS) and view.current_through(rec, "thresholds")


def _first(text) -> str:
    return (str(text).splitlines() or [""])[0]


def config_problem(paths, e: BaseException) -> str:
    """'<file>: <first line of the error>' for a config that cannot be read; the file relative to the series."""
    f = getattr(e, "path", None) or getattr(e, "filename", None)
    if isinstance(e, config.ConfigError) and not e.first_line:
        return _first(e)  # rebuilt from its message, which starts with the file
    if isinstance(e, config.ConfigError):
        first = e.first_line
    elif isinstance(e, OSError) and e.strerror:
        first = e.strerror
    else:
        first = f"{type(e).__name__}: {_first(e)}"
    if f is None:
        return first
    try:
        name = paths.rel(f)
    except ValueError:  # outside the series (machine.yaml)
        name = str(f)
    return f"{name}: {first}"


def thresholds_choice(paths, rec) -> tuple[dict | None, str | None]:
    """(the session's thresholds, None); (None, None) when it has none; (None, why) when its config cannot be
    read. "Cannot tell" is never "no choice": a write meant for sessions without thresholds must leave such a
    session alone, or a typo in the series or mouse config would let it overwrite the session's own thresholds
    (its session file alone can still be read and written)."""
    try:
        cnmf = config.resolve(paths, sid=rec.sid, mouse=rec.mouse)["cnmf"]
        values = {k: cnmf.get(k) for k in THRESHOLD_KEYS}
    except Exception as e:  # not only ConfigError: a cnmf section that is not a mapping cannot be told either
        return None, config_problem(paths, e)
    return (values if all(v is not None for v in values.values()) else None), None


def gate_may_wait(rec) -> bool:
    """The part of gate_waits that needs the record only (no file read): not excluded, not passed, and
    WAITING or behind a finished CNMF-E. The monitor window checks it first on every refresh."""
    cnmf, g = rec.stages.get("cnmf"), rec.stages.get("gate")
    if rec.excluded or (g is not None and g.state is State.DONE):
        return False
    return (g is not None and g.state is State.WAITING) or (cnmf is not None and cnmf.state is State.DONE)


def gate_waits(paths, rec) -> bool:
    """The gate waits for a person now: WAITING, or (CNMF done) any state but DONE, since the runner may be
    checking it at this moment. A passed gate, or one holding a decision that counts, is not taken. The one
    definition of "waits for manual inspection": the window's count and its "for all waiting" action use it."""
    if not gate_may_wait(rec):
        return False
    try:
        cfg = config.resolve(paths, sid=rec.sid, mouse=rec.mouse)
    except Exception:  # a broken session config is shown by status; the decision counts once it is fixed
        cfg = {"gate": {"enabled": True}}
    try:
        gate.run(paths, rec, cfg)
    except WaitingForInput as w:
        return w.what == "manual_inspection"
    return False


def gate_passed(rec) -> bool:
    g = rec.stages.get("gate")
    return g is not None and g.state is State.DONE


def _side_cfg(rec, paths):
    """The session's config when it has a FAILED side stage (is that preview still on?); else None, not read."""
    if not any(rec.stages.get(s) and rec.stages[s].state is State.FAILED for s in SIDE_STAGES):
        return None
    try:
        return config.resolve(paths, sid=rec.sid, mouse=rec.mouse)
    except Exception:  # a broken config is shown by status; every failed preview counts meanwhile
        return None


def failed_stage(rec, paths):
    """runner.first_failed: the first failed main stage, else the first failed preview that is on."""
    return runner.first_failed(rec, _side_cfg(rec, paths))


def retry_stages(rec, paths) -> list:
    """runner.retry_stages: what retry_failed resets for the session ([] when nothing failed)."""
    return runner.retry_stages(rec, _side_cfg(rec, paths))


def retry_plan(paths, sids) -> dict:
    """sid -> retry_stages, for a confirmation that says what will be reset; an unknown sid gets []."""
    recs = _records(paths)
    return {sid: retry_stages(recs[sid], paths) if sid in recs else [] for sid in sids}


def waiting_gate_sids(paths) -> list:
    return [s for s, r in sorted(_records(paths).items()) if gate_waits(paths, r)]


def waiting_threshold_sids(paths) -> list:
    view = runner.StageView(paths)
    return [s for s, r in sorted(_records(paths).items()) if waits_for_thresholds(view, r)]


def failed_sids(paths) -> list:
    return [s for s, r in sorted(_records(paths).items()) if not r.excluded and failed_stage(r, paths) is not None]


CHOICE_COLUMNS = ("sid", "mouse", "when", "who", "source", "suggested_gSig", "suggested_min_corr",
                  "suggested_min_pnr", "suggested_method", "gSig", "min_corr", "min_pnr", "n_blobs",
                  "single_frac", "n_mirror_blobs")


PENDING_CHOICES = "threshold_choices.pending.csv"  # next to the log: rows the log would not take yet
CHOICE_NOT_LOGGED = "choice_log_not_written"  # warning: the row waits in PENDING_CHOICES
CHOICE_ONLY_IN_LOG = "choice_log_only_in_log"  # warning: the pending file failed too; the row is a log WARNING


def _csv_text(rows, header: bool) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, CHOICE_COLUMNS, lineterminator="\n")
    if header:
        w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def _empty(f) -> bool:
    return not f.exists() or f.stat().st_size == 0


def _pending_rows(pending) -> list:
    if not pending.exists():
        return []
    with io.StringIO(fileio.read_text(pending), newline="") as f:
        return [{k: r.get(k) or "" for k in CHOICE_COLUMNS} for r in csv.DictReader(f)]


def _log_choice(paths, rec, who: str, source: str, gsig, min_corr, min_pnr, counters, when: str) -> str | None:
    """One row per choice, next to the suggestion it answers: the decision log keeps only the chosen values,
    and the suggestion in the record is replaced when the thresholds stage re-runs.

    The thresholds are written by then, so a log that cannot be written (open in Excel) must not fail the choice:
    the row waits in PENDING_CHOICES, and the next append that works writes the waiting rows first, in their
    order. Returns None, CHOICE_NOT_LOGGED, or CHOICE_ONLY_IN_LOG (the pending file failed too: the row is in
    the log as a WARNING)."""
    th = rec.stages.get("thresholds")
    sug = (th.metrics.get("suggested") if th is not None else None) or {}
    row = {"sid": rec.sid, "mouse": rec.mouse, "when": when, "who": who, "source": source,
           "suggested_gSig": sug.get("gSig"), "suggested_min_corr": sug.get("min_corr"),
           "suggested_min_pnr": sug.get("min_pnr"), "suggested_method": sug.get("method"),
           "gSig": gsig, "min_corr": min_corr, "min_pnr": min_pnr}
    if counters is not None:
        row.update(n_blobs=counters.n_blobs, single_frac=counters.single_frac,
                   n_mirror_blobs=counters.n_mirror_blobs)
    row = {k: "" if row.get(k) is None else row[k] for k in CHOICE_COLUMNS}
    target = paths.threshold_choices_csv
    pending = target.with_name(PENDING_CHOICES)
    written = False
    try:
        # both locks: a writer that keeps its row in the pending file meanwhile must not be flushed half-way
        with fileio.file_lock(target), fileio.file_lock(pending):
            waiting = _pending_rows(pending)
            fileio.append_text(target, _csv_text(waiting + [row], header=_empty(target)))
            written = True
            if waiting:
                try:
                    pending.unlink()
                except OSError as e:
                    log.warning("the rows of %s are in %s now, but the file cannot be removed (remove it, or its "
                                "rows are written again): %s", pending, target, runner._one_line(e))
        return None
    except OSError as e:
        if written:  # the row is in the log; only a lock could not be released
            return None
        why = runner._one_line(e)
    try:
        with fileio.file_lock(pending):
            fileio.append_text(pending, _csv_text([row], header=_empty(pending)))
    except OSError as e:
        log.warning("threshold choice not logged: %s and %s cannot be written (%s; %s); the row: %s", target,
                    pending, why, runner._one_line(e), row)
        return CHOICE_ONLY_IN_LOG
    log.warning("threshold choice log %s cannot be written (%s); the row was kept in %s", target, why, pending)
    return CHOICE_NOT_LOGGED


def _write_thresholds(paths, rec, gsig, min_corr, min_pnr, who: str, source: str, counters) -> str | None:
    """The thresholds, the decision, then the choice log; returns the choice log's warning code, if any."""
    now = datetime.now().isoformat(timespec="milliseconds")
    config.update_yaml(paths.session_config(rec.sid), {"cnmf": {"gSig": gsig, "min_corr": min_corr,
                                                                "min_pnr": min_pnr}})
    value = f"gSig={gsig} min_corr={min_corr} min_pnr={min_pnr}"
    registry.update_record(paths, rec.sid, lambda r: r.decisions.append(Decision("thresholds", value, who, now)))
    # the CSV log comes last: if a write above fails, the log never claims a choice the config does not hold
    return _log_choice(paths, rec, who, source, gsig, min_corr, min_pnr, counters, now)


def _warn_excluded(res: ActionResult, rec) -> None:
    if rec.excluded:
        res.warnings.append((rec.sid, "excluded_session"))


# --- actions ---

def _decide(paths, sids, what: str, value: str, who: str) -> None:
    now = datetime.now().isoformat(timespec="milliseconds")  # same format as gate.decide

    def change(rec):
        rec.decisions.append(Decision(what, value, who, now))
        rec.excluded = scan.evaluate_excluded(rec)  # the same rule as the scan: the latest decision wins

    for sid in sids:
        registry.update_record(paths, sid, change)


def set_excluded(paths, sids, excluded: bool, who: str) -> ActionResult:
    recs = _records(paths)
    sids = list(sids)
    check_sids(recs, sids)
    res = ActionResult()
    for sid in sids:
        if not excluded and recs[sid].fps is None:
            res.warnings.append((sid, "no_fps_stays_excluded"))
        elif not excluded and any(a.code == "not_found_in_scan" for a in recs[sid].anomalies):
            res.warnings.append((sid, "not_found_stays_excluded"))  # scan.evaluate_excluded always excludes it
    _decide(paths, sids, "exclude", "true" if excluded else "false", who)
    res.ok = sids
    return res


def accept_mismatch(paths, sids, who: str) -> ActionResult:
    recs = _records(paths)
    sids = list(sids)
    check_sids(recs, sids)
    res = ActionResult()
    for sid in sids:
        if not any(a.code == "frames_ts_mismatch" and a.level == "error" for a in recs[sid].anomalies):
            res.warnings.append((sid, "no_mismatch_error"))
    _decide(paths, sids, "accept_mismatch", "true", who)
    res.ok = sids
    return res


def confirm_stitch(paths, sids, who: str) -> ActionResult:
    """Mark the stitch of a split recording as reviewed (only for the warning level: an error-level split
    cannot be stitched, so there is nothing to confirm). The decision's value is the stitch it confirms
    (scan.stitch_fingerprint), so a part added or removed later needs a new confirmation. A stitch that is
    already confirmed gets the warning already_confirmed and no second decision."""
    recs = _records(paths)
    sids = list(sids)
    check_sids(recs, sids)
    res = ActionResult()
    todo = []
    for sid in sids:
        if scan.stitch_confirmed(recs[sid]):
            res.warnings.append((sid, "already_confirmed"))
        elif any(a.code == "split_recording" and a.level == "warning" and not a.data.get("stage")
               for a in recs[sid].anomalies):
            todo.append(sid)
        else:
            res.failed[sid] = ("no_split", "")
    # not _decide: this decision does not touch the exclusion, which the scan re-evaluates anyway
    now = datetime.now().isoformat(timespec="milliseconds")
    for sid in todo:
        registry.update_record(paths, sid, lambda rec: rec.decisions.append(
            Decision("stitch_confirmed", scan.stitch_fingerprint(rec.parts), who, now)))
        res.ok.append(sid)
    return res


def rescan(paths, progress=None):
    """Scan the series again and return the Scan tab's model. Manual decisions stay; the scan re-evaluates
    the exclusion, so an anomaly-based one clears once the files are fixed.
    Next to a running batch a rescan can flip a session's `excluded` (fixed files re-include it, new errors
    exclude it); the tab asks for confirmation then."""
    scan.scan_series(paths.root, progress)
    return scan_view.scan_view(paths)


def gate_not_needed(paths, sids, who: str, force: bool = False) -> ActionResult:
    recs = _records(paths)
    sids = waiting_gate_sids(paths) if sids is None else list(sids)
    check_sids(recs, sids)
    res = ActionResult()
    for sid in sids:
        _warn_excluded(res, recs[sid])
        if gate_passed(recs[sid]) and not force:
            res.failed[sid] = ("already_passed", "")
            continue
        gate.decide(paths, sid, "not_needed", who)
        res.ok.append(sid)
    return res


def gate_cleaned(paths, sid: str, who: str) -> ActionResult:
    recs = _records(paths)
    check_sids(recs, [sid])
    res = ActionResult()
    _warn_excluded(res, recs[sid])
    manual = paths.manual_result(sid)
    if not manual.exists():
        res.failed[sid] = ("manual_result_missing", str(manual))
        return res
    gate.decide(paths, sid, "cleaned", who)
    res.ok.append(sid)
    return res


def rerun_from(paths, sids, stage: str, who: str) -> ActionResult:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; stages: {', '.join(STAGES)}")
    sids = list(sids)
    check_sids(_records(paths), sids)
    res = ActionResult()
    for sid in sids:
        runner.reset_from(paths, sid, stage, who)
        res.ok.append(sid)
    return res


def retry_failed(paths, sids, who: str) -> ActionResult:
    recs = _records(paths)
    sids = failed_sids(paths) if sids is None else list(sids)
    check_sids(recs, sids)
    res = ActionResult()
    for sid in sids:
        stages = retry_stages(recs[sid], paths)  # a failed preview alone: only the preview is reset
        if not stages:
            res.failed[sid] = ("no_failed_stage", "")
            continue
        for stage in stages:
            runner.reset_from(paths, sid, stage, who)
        res.ok.append(sid)
    return res


def set_previews(paths, enabled: bool, tbin: int | None = None, ssub: int | None = None) -> None:
    """The preview checkbox and its downsampling, in the series config (section preview); None keeps a setting.
    The runner rereads the configs on every loop, so a running batch makes the previews of every session whose
    MC / CNMF-E is done, without running those again. Bad values raise ValueError before anything is written."""
    patch = {"enabled": bool(enabled)}
    for name, value in (("tbin", tbin), ("ssub", ssub)):
        if value is None:
            continue
        bad = preview_setting_error(value, name)
        if bad:
            raise ValueError(bad)
        patch[name] = int(value)
    config.update_yaml(paths.series_config, {"preview": patch})


def set_thresholds(paths, sid: str, gsig, min_corr, min_pnr, *, who: str = "", source: str = "cli",
                   counters: "Counters | None" = None) -> ActionResult:
    bad = threshold_error(gsig, min_corr, min_pnr)
    if bad:
        raise ValueError(bad)
    recs = _records(paths)
    check_sids(recs, [sid])
    res = ActionResult()
    _warn_excluded(res, recs[sid])
    logged = _write_thresholds(paths, recs[sid], int(gsig), min_corr, min_pnr, who, source, counters)
    res.ok.append(sid)
    if logged:
        res.warnings.append((sid, logged))
    return res


def accept_suggested(paths, sids, *, who: str = "") -> ActionResult:
    recs = _records(paths)
    sids = waiting_threshold_sids(paths) if sids is None else list(sids)
    check_sids(recs, sids)
    res = ActionResult()
    for sid in sids:
        th = recs[sid].stages["thresholds"]
        sug = th.metrics.get("suggested") if th.state is State.DONE else None
        if not sug:
            res.failed[sid] = ("no_suggestion", "")
            continue
        bad = threshold_error(sug.get("gSig"), sug.get("min_corr"), sug.get("min_pnr"))
        if bad:
            res.failed[sid] = ("invalid_suggestion", bad)
            continue
        _warn_excluded(res, recs[sid])
        logged = _write_thresholds(paths, recs[sid], int(sug["gSig"]), sug["min_corr"], sug["min_pnr"], who,
                                   "accept_suggested", None)
        res.ok.append(sid)
        if logged:
            res.warnings.append((sid, logged))
    return res


@dataclass
class MousePlan:
    """What apply_to_mouse does to each session of a mouse (sids in order); the window lists it before the write."""
    will_get: list = field(default_factory=list)
    keep_own: list = field(default_factory=list)   # has thresholds: never overwritten, its CNMF-E would re-run
    excluded: list = field(default_factory=list)   # left alone
    cnmf_done: list = field(default_factory=list)  # in will_get with CNMF-E done (config lost them): it re-runs
    unreadable: list = field(default_factory=list)  # (sid, why): its config cannot be read, so it is left alone


def _mouse_plan(paths, mouse: str) -> tuple[MousePlan, dict]:
    recs = _records(paths)
    mine = {sid: rec for sid, rec in sorted(recs.items()) if rec.mouse == mouse}
    if not mine:  # a typo or a mouse whose sessions are gone: raises before any write, as an unknown session
        raise UnknownSession([mouse], {r.mouse for r in recs.values()})
    plan = MousePlan()
    for sid, rec in mine.items():
        # excluded first: the runner ignores an excluded session, so that is its warning even when it has a choice
        if rec.excluded:
            plan.excluded.append(sid)
            continue
        chosen, why = thresholds_choice(paths, rec)
        if why is not None:
            plan.unreadable.append((sid, why))
        elif chosen is not None:
            plan.keep_own.append(sid)
        else:
            plan.will_get.append(sid)
            cnmf = rec.stages.get("cnmf")
            if cnmf is not None and cnmf.state is State.DONE:
                plan.cnmf_done.append(sid)
    return plan, mine


def mouse_plan(paths, mouse: str) -> MousePlan:
    return _mouse_plan(paths, mouse)[0]


def apply_to_mouse(paths, mouse: str, gsig: int, min_corr: float, min_pnr: float, who: str) -> ActionResult:
    """The same thresholds for every session of the mouse that has none (mouse_plan). A session with a choice is
    never overwritten (its CNMF-E would re-run), and an excluded one or one whose config cannot be read is left
    alone: each gets a warning. A session that cannot be written goes into `failed` (write_failed) and the others
    are still written: stopping half-way would leave the mouse partly written without saying which part."""
    bad = threshold_error(gsig, min_corr, min_pnr)
    if bad:
        raise ValueError(bad)
    plan, mine = _mouse_plan(paths, mouse)
    unreadable = dict(plan.unreadable)
    res = ActionResult()
    for sid, rec in mine.items():
        if sid in plan.excluded:
            res.warnings.append((sid, "excluded_session"))
        elif sid in unreadable:
            res.warnings.append((sid, "config_unreadable"))
        elif sid in plan.keep_own:
            res.warnings.append((sid, "already_chosen"))
        else:
            try:
                logged = _write_thresholds(paths, rec, int(gsig), min_corr, min_pnr, who, "window_mouse", None)
            except Exception as e:  # a locked file, a session config broken since the plan: this session only
                res.failed[sid] = ("write_failed", runner._one_line(e))
                continue
            res.ok.append(sid)
            if logged:
                res.warnings.append((sid, logged))
    return res


def export_summary(paths, dest: Path) -> Path:
    """The same table `run_bearmind status` writes to summary.csv (same records, problems, broken files)."""
    state = runner.series_state(paths)
    df = runner.summary(paths, state.records, problems=state.problems, broken=state.broken)
    df.to_csv(dest, index=False, encoding="utf-8", lineterminator="\n")
    return dest
