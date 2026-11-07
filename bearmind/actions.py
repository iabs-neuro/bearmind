"""Actions that change a series: one implementation for the command line and the monitor window.

Every function returns an ActionResult with codes; the callers turn the codes into their own texts (the CLI
prints its console messages, the window uses failure_text). Unknown sessions or stages raise before anything is
written, so a typo never leaves a half-applied batch. Every write goes through registry.update_record /
config.update_yaml, which keeps it safe next to a running runner.
"""
import getpass
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from bearmind import config, registry, runner, scan
from bearmind.registry import SIDE_STAGES, STAGES, Decision, State
from bearmind.stages import WaitingForInput, gate


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
    "no_failed_stage": "The session has no failed stage, so nothing was reset.",
    # the window's text (the CLI prints its own): the window has no force path, so it asks for nothing
    "already_passed": "Manual inspection already passed, so nothing was recorded. A new decision would make "
                      "autoinspection and export run again, and autoinspection is not reproducible run to run, "
                      "so the exported neurons could change.",
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


def waiting_for(rec, stage: str):
    st = rec.stages.get(stage)
    if st is None or st.state is not State.WAITING:
        return None
    return runner._what(st)


def suggested_unset(paths, rec) -> bool:
    """The thresholds stage is done with a suggestion and the session has no thresholds yet: after
    run --until thresholds CNMF is PENDING, not WAITING."""
    th = rec.stages.get("thresholds")
    if th is None or th.state is not State.DONE or not th.metrics.get("suggested"):
        return False
    try:
        c = config.resolve(paths, sid=rec.sid, mouse=rec.mouse)["cnmf"]
    except Exception:  # a broken config is shown by status as "cannot be scheduled"
        return False
    return any(c.get(k) is None for k in ("gSig", "min_corr", "min_pnr"))


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


def failed_stage(rec, paths=None):
    """The first FAILED stage in ALL_STAGES order. With `paths`, a failed preview the session's config turns off
    holds nothing and is not offered (the config is read only for a record with a failed side stage)."""
    cfg = None
    if paths is not None and any(rec.stages.get(s) and rec.stages[s].state is State.FAILED for s in SIDE_STAGES):
        try:
            cfg = config.resolve(paths, sid=rec.sid, mouse=rec.mouse)
        except Exception:  # a broken config is shown by status; every failed stage counts meanwhile
            cfg = None
    return runner.first_failed(rec, cfg)


def waiting_gate_sids(paths) -> list:
    return [s for s, r in sorted(_records(paths).items()) if gate_waits(paths, r)]


def waiting_threshold_sids(paths) -> list:
    return [s for s, r in sorted(_records(paths).items()) if not r.excluded
            and (waiting_for(r, "cnmf") == "thresholds" or suggested_unset(paths, r))]


def failed_sids(paths) -> list:
    return [s for s, r in sorted(_records(paths).items()) if not r.excluded and failed_stage(r, paths) is not None]


def _write_thresholds(paths, sid: str, gsig, min_corr, min_pnr) -> None:
    config.update_yaml(paths.session_config(sid), {"cnmf": {"gSig": gsig, "min_corr": min_corr,
                                                            "min_pnr": min_pnr}})


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
        first = failed_stage(recs[sid], paths)  # a failed preview: only the preview is reset
        if first is None:
            res.failed[sid] = ("no_failed_stage", "")
            continue
        runner.reset_from(paths, sid, first, who)
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


def set_thresholds(paths, sid: str, gsig, min_corr, min_pnr) -> ActionResult:
    bad = threshold_error(gsig, min_corr, min_pnr)
    if bad:
        raise ValueError(bad)
    recs = _records(paths)
    check_sids(recs, [sid])
    res = ActionResult()
    _warn_excluded(res, recs[sid])
    _write_thresholds(paths, sid, int(gsig), min_corr, min_pnr)
    res.ok.append(sid)
    return res


def accept_suggested(paths, sids) -> ActionResult:
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
        _write_thresholds(paths, sid, sug["gSig"], sug["min_corr"], sug["min_pnr"])
        res.ok.append(sid)
    return res


def export_summary(paths, dest: Path) -> Path:
    """The same table `run_bearmind status` writes to summary.csv (same records, problems, broken files)."""
    state = runner.series_state(paths)
    df = runner.summary(paths, state.records, problems=state.problems, broken=state.broken)
    df.to_csv(dest, index=False, encoding="utf-8", lineterminator="\n")
    return dest
