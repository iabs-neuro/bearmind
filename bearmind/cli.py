"""Command line of the batch pipeline: python -m bearmind <command> SERIES ...

Exit codes: 0 ok, 1 partial failure, 2 bad input, 3 series busy, 130 interrupted. Messages are plain ASCII
([OK] / [FAILED] / [WARN]) because students read them in a Windows console. Every write to a record or
config goes through registry.update_record / config.update_yaml, so it is safe next to a running runner.
"""
import argparse
import importlib
import json
import logging
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from bearmind import actions, config, launcher, monitor, registry, runner, scan
from bearmind.paths import SeriesPaths
from bearmind.registry import ALL_STAGES, SIDE_STAGES, STAGES, State
from bearmind.stages import gate

OK, BAD_INPUT, BUSY, INTERRUPTED = 0, 2, 3, 130
HEARTBEAT_MAX_AGE_S = 120
_SHORT = {"crop": "crop", "mc": "mc", "thresholds": "thr", "cnmf": "cnmf", "gate": "gate",
          "autoinspect": "insp", "export": "exp"}
_LETTER = {State.DONE: "D", State.RUNNING: "R", State.WAITING: "W", State.FAILED: "F", State.SKIPPED: "S",
           State.PENDING: "."}
_REASON = {"thresholds": "thresholds", "manual_inspection": "manual inspection", "crop_finalize": "crop box"}
_SIDES = tuple(s for s in ALL_STAGES if s in SIDE_STAGES)  # the letters of the pv column, in this order
_MAX_LISTED = 12
_MAX_UNRECOGNIZED = 10


class CliError(Exception):
    """An expected problem: the message goes to the console, the code is the exit code."""

    def __init__(self, message: str, code: int = BAD_INPUT):
        super().__init__(message)
        self.code = code


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        print(f"[FAILED] {message}", file=sys.stderr)
        raise SystemExit(BAD_INPUT)


def _say(text="") -> None:
    # message text is ASCII; paths keep their letters (Cyrillic folders must stay copyable), and
    # stdout.reconfigure(errors="backslashreplace") covers what the console cannot encode
    print(text)


def _ascii(text) -> str:
    """A stage error for the console: it may carry a localized system message (the logs keep the original)."""
    return str(text).encode("ascii", "backslashreplace").decode("ascii")


def _first_line(text) -> str:
    return (str(text).splitlines() or [""])[0]


def _last_line(text) -> str:
    return (str(text).splitlines() or [""])[-1]


def _stage(value: str) -> str:
    if value not in STAGES:
        raise argparse.ArgumentTypeError(f"unknown stage {value!r}; stages: {', '.join(STAGES)}")
    return value


def _number(value: str, name: str) -> float:
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{name} must be a number, got {value}") from None


def _checked(error_of, name):
    def parse(value):
        x = _number(value, name)
        if name == "--gsig" and math.isfinite(x) and x == int(x):
            x = int(x)
        err = error_of(x)
        if err:
            raise argparse.ArgumentTypeError(err)
        return x
    return parse


# --- helpers ---

def _series(arg: str) -> SeriesPaths:
    root = Path(arg).resolve()
    if not root.is_dir():
        raise CliError(f"series folder not found: {root}")
    return SeriesPaths(root)


def _records(paths) -> dict:
    recs = {r.sid: r for r in registry.all_records(paths)}
    if not recs:
        raise CliError(f"no sessions registered in {paths.root}; run first: scan {paths.root}")
    return recs


def _check_sids(recs: dict, sids) -> None:
    try:
        actions.check_sids(recs, sids)
    except actions.UnknownSession as e:
        known = ", ".join(sorted(recs)) if len(recs) <= _MAX_LISTED else "see the list with: status SERIES"
        raise CliError(f"{e}; known sessions: {known}") from None


def _warn_excluded(rec) -> None:
    if rec.excluded:
        _say(f"[WARN] {rec.sid} is excluded; the runner ignores it until you run: decide SERIES {rec.sid} include")


def _who(args) -> str:
    return args.who or actions.current_user()


def _next_hint(paths) -> str:
    return f"next: run_bearmind run {paths.root}"


# --- commands ---

def cmd_gui(args) -> int:
    try:
        app = importlib.import_module("bearmind.gui.app")  # lazy: the other commands must work without PySide6
    except ImportError as e:
        return _gui_cannot_start(e)
    series = str(Path(args.series).resolve()) if args.series else None
    return app.main([series] if series else [])


def _gui_cannot_start(e: ImportError) -> int:
    """The window started with pythonw has no console: the traceback goes to gui.log (next to machine.yaml, where
    the window logs too), and without a console a message box says where."""
    log_file = config.machine_file().parent / "gui.log"
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}\n{traceback.format_exc()}\n")
    except OSError:
        pass
    text = f"the window cannot start: {type(e).__name__}: {_first_line(e)}; details in {log_file}"
    _say(f"[FAILED] {text}")
    if sys.stdout is None and os.name == "nt":
        import ctypes

        ctypes.windll.user32.MessageBoxW(None, text, "BEARMiND", 0x10)  # MB_ICONERROR
    return 1


def cmd_scan(args) -> int:
    paths = _series(args.series)
    records = scan.scan_series(paths.root)
    excluded = [r for r in records if r.excluded]
    _say(f"[OK] {len(records)} sessions found, {len(excluded)} excluded")
    for r in records:
        for a in r.anomalies:
            if a.code == "registry_recreated":
                _say(f"[WARN] {r.sid}: {a.message}")
    for r in excluded:
        # printed here, not taken from ActionResult.warnings: scan prints these lines in its own order
        why = "; ".join(a.message for a in r.anomalies if a.level == "error") or "excluded by a decision"
        _say(f"[WARN] {r.sid}: excluded - {why}")
    odd = scan.read_unrecognized(paths)
    if odd:
        _say(f"[WARN] {len(odd)} folders not recognized (not scanned as sessions):")
        for path, reason in odd[:_MAX_UNRECOGNIZED]:
            _say(f"         {path} - {reason}")
        if len(odd) > _MAX_UNRECOGNIZED:
            _say(f"         ... and {len(odd) - _MAX_UNRECOGNIZED} more")
        _say(f"         list: {paths.scan_unrecognized_csv}")
    return OK


def cmd_run(args) -> int:
    paths = _series(args.series)
    recs = _records(paths)
    _check_sids(recs, args.sessions or ())
    for sid in args.sessions or ():
        _warn_excluded(recs[sid])
    try:
        res = runner.run(paths.root, until=args.until, sids=args.sessions, once=args.once, console=True)
    except KeyboardInterrupt:
        _say("[WARN] stopped by Ctrl+C; run the same command to resume where it stopped")
        return INTERRUPTED
    except RuntimeError as e:
        if "already being processed" in str(e) or "another batch is running" in str(e):
            _say(f"[FAILED] {e}")
            return BUSY
        raise
    return _report_run(paths, res)


def cmd_stop(args) -> int:
    paths = _series(args.series)
    if not monitor.runner_state(paths).alive:
        _say("[WARN] no runner is running for this series")
        return OK
    launcher.request_stop(paths)
    _say("[OK] stop requested; the runner stops before its next stage")
    return OK


def _report_run(paths, res) -> int:
    """This run's outcome and the series state at exit, labelled so the two cannot be confused; 1 while any
    session of the series has a FAILED stage or cannot be scheduled."""
    st = runner.series_state(paths)
    waiting = sorted({sid for sid, _, _ in st.waiting})
    unrecorded = res.get("not_recorded") or []
    bad = bool(st.failed or st.problems or st.finalize_failed or unrecorded)
    _say(("[FAILED] " if bad else "[OK] ") + ("stopped on request; " if res.get("stopped") else "")
         + f"this run: done {res['done']}, failed {res['failed']}, skipped {res['skipped']} | series now: "
         + f"failed {len(st.failed)}, waiting for a person {len(waiting)}, cannot be scheduled {len(st.problems)}, "
         + f"excluded {len(st.excluded)}")
    for sid, (stage, error) in sorted(st.failed.items())[:_MAX_LISTED]:
        _say(f"         {sid} failed at {stage}: {_clip(_ascii(_last_line(error or '')))}")
    if st.failed:
        _say(f"         after fixing the cause: retry {paths.root} --all-failed")
    for sid, why in sorted(st.problems.items())[:_MAX_LISTED]:
        _say(f"[FAILED] {sid} cannot be scheduled: {_clip(why)}")
    for sid, stage, what, by in st.blocked[:_MAX_LISTED]:
        _say(f"         {sid} {stage} waits for {_REASON.get(what, what)}: blocked by {', '.join(by)}")
    if st.problems:
        _say("         fix the named config file (or rescan), then run again")
    late = sorted(sid for sid in st.finalize_failed if sid not in st.problems)
    if late:
        _say(f"[FAILED] crop box not placed for {', '.join(late)}: {_clip(_ascii(st.finalize_failed[late[0]]))}")
        _say("         the runner tries again on the next run; see the session log for the traceback")
    if unrecorded:
        _say(f"[FAILED] {len(unrecorded)} stages not started or not recorded (registry not writable):")
        for sid, stage, why in unrecorded[:_MAX_LISTED]:
            _say(f"         {sid} {stage}: {_clip(_ascii(why))}")
        _say(f"         close the program that holds files in {paths.registry_file('x').parent}, then run again")
    if len(st.failed) > _MAX_LISTED or len(st.problems) > _MAX_LISTED:
        _say(f"         more in: status {paths.root}")
    if waiting:
        _say(f"[WARN] {len(waiting)} sessions wait for a person; see: status {paths.root}")
    return 1 if bad else OK


def _heartbeat_line(paths) -> str:
    try:
        hb = json.loads(paths.heartbeat.read_text(encoding="utf-8"))
        age = (datetime.now() - datetime.fromisoformat(hb["time"])).total_seconds()
        alive = age <= HEARTBEAT_MAX_AGE_S and hb.get("status") in ("running", "waiting")
    except (OSError, ValueError, KeyError, TypeError):
        return "runner: not running"
    if not alive:
        return "runner: not running"
    cur = hb.get("current")
    where = f", current {cur['sid']}/{cur['stage']}" if cur else (", waiting for a person"
                                                                  if hb.get("status") == "waiting" else "")
    return f"runner: running PID {hb.get('pid')}{where}"


def _side_off(on, stage) -> bool:
    """`on`: series_state.previews of the session ({side stage: on}, None when its config cannot be read)."""
    return on is not None and not on.get(stage, True)


STALE_MARK = "D*"  # done in the registry, but the batch would redo it (runner.StageView.current)


def _pv(rec, on, stale=()) -> str:
    """The pv column: "off" when the session's config turns every preview off, else one mark per side stage
    ("-" for one that is off, D* for one in `stale`: done but the batch would redo it)."""
    if all(_side_off(on, s) for s in _SIDES):
        return "off"
    return "".join("-" if _side_off(on, s) else STALE_MARK if s in stale
                   else _LETTER[rec.stages[s].state] if s in rec.stages else "." for s in _SIDES)


def _row(rec, on=None, stale=()) -> tuple:
    """`stale`: the main stages DONE but out of date (runner.StageView.done_not_current)."""
    cells, reason = [], ""
    for s in STAGES + _SIDES:  # the main stages first: a main failure or wait is the note, not a preview's
        st = rec.stages.get(s)
        if s in SIDE_STAGES:
            if _side_off(on, s):
                continue  # off: an old record holds nothing
        else:
            cells.append(STALE_MARK if s in stale else _LETTER[st.state] if st else ".")
        if st is None:
            continue
        if st.state is State.WAITING and not reason:
            w = runner._what(st)
            reason = _REASON.get(w, w)
        elif st.state is State.FAILED and not reason:
            reason = f"failed {s}: {_ascii(_last_line(st.error or ''))[:60]}"
    return cells, reason


_NOTE_MAX = 160  # characters of a reason in a status row; summary.csv has it whole


def _clip(text) -> str:
    text = str(text)
    return text if len(text) <= _NOTE_MAX else text[:_NOTE_MAX - 3] + "..."


def _excluded_note(rec) -> str:
    codes = list(dict.fromkeys(a.code for a in rec.anomalies if a.level == "error" and not a.data.get("stage")))
    decided = [d for d in rec.decisions if d.what == "exclude"]
    if decided and decided[-1].value == "true" and rec.fps is not None:
        return "excluded by a decision"
    return "excluded: " + ", ".join(codes) if codes else "excluded"


def _note(rec, reason, state) -> str:
    """What the operator must know about the session: why it cannot be scheduled comes first."""
    if rec.sid in state.problems:
        return f"cannot be scheduled: {_clip(state.problems[rec.sid])}"
    if rec.sid in state.finalize_failed and reason:
        return f"{reason}: {_clip(state.finalize_failed[rec.sid])}"
    by = next((b for sid, _, _, b in state.blocked if sid == rec.sid), None)
    if by:
        return f"{reason}: blocked by {', '.join(by)} (cannot be scheduled)"
    if rec.excluded and not reason:
        return _excluded_note(rec)
    return reason


def _waiting_line(state) -> str:
    waits = {}
    for _sid, _stage, what in state.waiting:
        name = _REASON.get(what, what)
        waits[name] = waits.get(name, 0) + 1
    return ("waiting: " + (", ".join(f"{k} {v}" for k, v in sorted(waits.items())) or "nobody")
            + f"; blocked: {len(state.blocked)}; cannot be scheduled: {len(state.problems)}"
            + (f"; crop finalize failed: {len(state.finalize_failed)}" if state.finalize_failed else "")
            + f"; failed: {len(state.failed)}")


def _check_mark(rec) -> str:
    """The worst check level over the done stages: E error, W warning, blank otherwise."""
    levels = {c.level for st in rec.stages.values() if st.state is State.DONE for c in st.checks}
    return "E" if "error" in levels else "W" if "warning" in levels else ""


def cmd_status(args) -> int:
    paths = _series(args.series)
    state = runner.series_state(paths)
    if not state.records and not state.broken:
        raise CliError(f"no sessions registered in {paths.root}; run first: scan {paths.root}")
    runner.summary(paths, state.records, problems=state.problems, broken=state.broken, view=state.view)
    rows = []
    for r in state.records:
        on = state.previews.get(r.sid)
        stale = state.view.done_not_current(r)
        stale_sides = state.view.side_done_not_current(r)
        cells, reason = _row(r, on, stale)
        if (stale or stale_sides) and not reason and r.sid not in state.problems:
            # the stage the batch runs first, in its own order: a preview before the next main stage
            first = state.view.first_ready(r) or state.view.first_not_current(r)
            if first:
                reason = f"out of date: the batch redoes {first} first"
        rows.append((r.sid, "yes " if r.excluded else "no  ", cells + [_pv(r, on, stale_sides)], _check_mark(r),
                     _note(r, reason, state)))
    rows += [(Path(f).stem, "?   ", ["?"] * (len(STAGES) + 1), "?", _clip(state.problems[Path(f).stem]))
             for f, _ in state.broken]
    _say(_heartbeat_line(paths))
    _say(_waiting_line(state))
    odd = scan.read_unrecognized(paths)
    if odd:
        _say(f"unrecognized folders: {len(odd)}; list: {paths.scan_unrecognized_csv}")
    _say("D done, D* done but out of date (the batch redoes it), R running, W waiting, F failed, S skipped, "
         ". not started, ? unknown; "
         "chk: W warning, E error in a check of a done stage; pv: MC and CNMF-E preview videos, off (-) when turned off")
    width = max(len("session"), *(len(sid) for sid, *_ in rows))
    head = ("session".ljust(width) + "  excl  " + " ".join(x.ljust(4) for x in [_SHORT[s] for s in STAGES] + ["pv"])
            + "  chk  note")
    _say(head)
    _say("-" * len(head))
    for sid, excl, cells, mark, note in sorted(rows):
        _say(sid.ljust(width) + "  " + excl + "  " + " ".join(c.ljust(4) for c in cells) + "  "
             + mark.ljust(3) + "  " + note)
    _say(f"summary table: {paths.summary_csv}")
    return OK


_CHOICE_LOG = {
    actions.CHOICE_NOT_LOGGED: "[WARN] {sid}: the choice log cannot be written (open in another program?); the row "
                               "was kept in {pending}",
    actions.CHOICE_ONLY_IN_LOG: "[WARN] {sid}: neither the choice log nor {pending} can be written; the row is in "
                                "the warning printed above",
    actions.CHOICE_PENDING_KEPT: "[WARN] {sid}: the row is in the choice log; earlier rows wait in {pending}, "
                                 "which cannot be read now (open in another program?)",
}


def _say_choice_log(paths, res) -> None:
    pending = paths.threshold_choices_csv.with_name(actions.PENDING_CHOICES)
    for sid, code in res.warnings:
        if code in _CHOICE_LOG:
            _say(_CHOICE_LOG[code].format(sid=sid, pending=pending))


def cmd_thresholds(args) -> int:
    paths = _series(args.series)
    recs = _records(paths)
    if args.action == "set":
        _check_sids(recs, [args.sid])
        _warn_excluded(recs[args.sid])
        res = actions.set_thresholds(paths, args.sid, args.gsig, args.min_corr, args.min_pnr, who=_who(args),
                                     source="cli")
        _say(f"[OK] {args.sid}: gSig={args.gsig} min_corr={args.min_corr} min_pnr={args.min_pnr}")
        _say_choice_log(paths, res)
        _say(_next_hint(paths))
        return OK
    if args.sessions:
        _check_sids(recs, args.sessions)
        sids = list(args.sessions)
    else:
        sids = actions.waiting_threshold_sids(paths)
        if not sids:
            _say("[WARN] no session waits for thresholds")
            return OK
    res = actions.accept_suggested(paths, sids, who=_who(args))
    code = OK
    stale = {sid for sid, w in res.warnings if w == actions.MAPS_OUT_OF_DATE}
    for sid in sids:
        if sid in stale:  # the batch redoes the maps first: a suggestion on them would be chosen unseen
            _say(f"[WARN] {sid}: maps out of date; the batch recomputes them first")
            continue
        if sid in res.failed:
            fcode, detail = res.failed[sid]
            if fcode == "no_suggestion" and not args.sessions:
                # it waits like the others, but only a person can choose for it: not a failure of this command
                _say(f"[WARN] {sid}: no autotune suggestion; set thresholds by hand")
                continue
            if fcode == "no_suggestion":
                _say(f"[FAILED] {sid}: no threshold suggestion (run the thresholds stage first, "
                     f"or set values: thresholds SERIES set {sid} --gsig G --min-corr X --min-pnr Y)")
            else:
                _say(f"[FAILED] {sid}: suggested thresholds are invalid: {detail.replace('--', '')}")
            code = 1
            continue
        _warn_excluded(recs[sid])
        sug = recs[sid].stages["thresholds"].metrics["suggested"]
        _say(f"[OK] {sid}: gSig={sug['gSig']} min_corr={sug['min_corr']} min_pnr={sug['min_pnr']}")
    _say_choice_log(paths, res)
    if code != 1 or len(sids) > 1:
        _say(_next_hint(paths))
    return code


def cmd_gate(args) -> int:
    paths = _series(args.series)
    recs = _records(paths)
    if not args.items:
        raise CliError("give the decision: not-needed or cleaned")
    decision = args.items[-1].replace("-", "_")
    sids = list(args.items[:-1])
    if decision not in gate.GATE_DECISIONS:
        raise CliError(f"unknown decision {args.items[-1]!r}; decisions: not-needed, cleaned")
    if args.all_waiting == bool(sids):
        raise CliError("give session names or --all-waiting, not both and not neither")
    if args.all_waiting:
        sids = actions.waiting_gate_sids(paths)
        if not sids:
            _say("[WARN] no session waits for manual inspection")
            return OK
    _check_sids(recs, sids)
    who, code = _who(args), OK
    # one call for the whole list: each call reads every record of the series
    many = actions.gate_not_needed(paths, sids, who, force=args.force) if decision == "not_needed" else None
    for sid in sids:
        _warn_excluded(recs[sid])
        res = many if many is not None else actions.gate_cleaned(paths, sid, who)
        if sid in res.failed:
            fcode, detail = res.failed[sid]
            if fcode == "already_passed":
                _say(f"[WARN] {sid}: manual inspection already passed; a new decision makes autoinspection and "
                     f"export run again, and autoinspection is not reproducible run to run (driada adds unseeded "
                     f"noise), so the exported neurons can change. Not recorded; to proceed add --force")
            else:
                _say(f"[FAILED] {sid}: manual inspection result not found: {detail}")
                _say("        save the cleaned file at that place, then repeat the command")
            code = 1
        else:
            _say(f"[OK] {sid}: manual inspection {decision.replace('_', '-')}")
    if code == OK:
        _say(_next_hint(paths))
    return code


_DECIDE = ("exclude", "include", "accept-mismatch")


def cmd_decide(args) -> int:
    paths = _series(args.series)
    _check_sids(_records(paths), [args.sid])
    who = _who(args)
    if args.action == "accept-mismatch":
        res = actions.accept_mismatch(paths, [args.sid], who)
    else:
        res = actions.set_excluded(paths, [args.sid], args.action == "exclude", who)
    for sid, wcode in res.warnings:
        if wcode == "no_mismatch_error":
            _say(f"[WARN] {sid} has no frames_ts_mismatch error; the decision is recorded but changes nothing")
        elif wcode == "no_fps_stays_excluded":
            _say(f"[WARN] {sid} has no frame rate (timestamps missing or unusable): it stays excluded, the "
                 f"stages need the frame rate; the decision is recorded and counts once a rescan finds the rate")
        elif wcode == "not_found_stays_excluded":
            _say(f"[WARN] {sid} was not found by the last scan: it stays excluded; the decision is recorded, "
                 f"restore the session folder and rescan")
    rec = registry.load_record(paths, args.sid)
    _say(f"[OK] {args.sid}: {args.action} recorded; session is now "
         + ("excluded" if rec.excluded else "included"))
    return OK


def cmd_rerun(args) -> int:
    paths = _series(args.series)
    _check_sids(_records(paths), args.sids)
    res = actions.rerun_from(paths, args.sids, args.from_stage, _who(args))
    for sid in res.ok:
        _say(f"[OK] {sid}: stages from {args.from_stage} on are reset")
    _say(_next_hint(paths))
    return OK


def cmd_retry(args) -> int:
    paths = _series(args.series)
    recs = _records(paths)
    if bool(args.sids) == args.all_failed:
        raise CliError("give session names or --all-failed, not both and not neither")
    if args.all_failed:
        sids = actions.failed_sids(paths)
        if not sids:
            _say("[OK] no failed sessions")
            return OK
    else:
        _check_sids(recs, args.sids)
        sids = list(args.sids)
    res = actions.retry_failed(paths, sids, _who(args))
    for sid in sids:
        if sid in res.failed:
            _say(f"[WARN] {sid}: no failed stage, skipped")
        else:
            first, *more = actions.retry_stages(recs[sid], paths)  # recs: the states before the reset
            _say(f"[OK] {sid}: reset from {first}" + (f" and {', '.join(more)}" if more else ""))
    if res.ok:
        _say(_next_hint(paths))
    return OK if res.ok else 1


def cmd_inspect(args) -> int:
    paths = _series(args.series)
    recs = _records(paths)
    if args.sid:
        _check_sids(recs, [args.sid])
    # lazy: Bokeh (and, once a session opens, CaImAn) only for this command
    from bearmind import inspect_server

    def stopping(_state):
        _say("[WARN] the inspection server of this series is stopping (it finishes the side files of the last "
             "save); waiting for it to end")

    def running(state) -> int:
        _say(f"[OK] inspection server is running: {state.url}")
        if not args.no_browser:
            import webbrowser
            webbrowser.open(inspect_server.session_url(state.url, args.sid) if args.sid else state.url + "/")
        return OK

    def not_ready(state) -> CliError:
        who = f" (PID {state.pid})" if state.pid is not None else ""
        return CliError(f"the inspection server of this series{who} did not finish starting or stopping in time; "
                        f"its lock file: {paths.inspection_lock}; repeat the command later", BUSY)

    # A server that starts or stops right now is waited for: never a second server, never a dead address. One
    # wait in all, also when another command takes the series in between.
    deadline = time.monotonic() + inspect_server.STOP_WAIT_S
    state = inspect_server.wait_ready(paths, on_stopping=stopping, deadline=deadline)
    if state.available:
        return running(state)
    if state.starting or state.stopping:
        raise not_ready(state)
    # the server draws nothing with matplotlib; a Qt backend would load PySide6 into it (as for the runner)
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        code = inspect_server.serve(paths.root, args.sid, args.port, open_browser=not args.no_browser,
                                    quiet_busy=True)
    except KeyboardInterrupt:
        _say("[OK] inspection server stopped by Ctrl+C")
        return INTERRUPTED
    if code != BUSY:
        return code
    # another inspect command started the series' server between the look above and this start
    state = inspect_server.wait_ready(paths, on_stopping=stopping, deadline=deadline)
    if state.available:
        return running(state)
    raise not_ready(state)


def _port(value: str) -> int:
    try:
        x = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--port must be a whole number 0-65535, got {value}") from None
    if not 0 <= x <= 65535:
        raise argparse.ArgumentTypeError(f"--port must be a whole number 0-65535, got {value}")
    return x


def _preview_setting(value, name):
    if value is None:
        return None
    try:
        x = int(value)
    except ValueError:
        raise CliError(f"{name} must be a whole number >= 1, got {value}") from None
    bad = actions.preview_setting_error(x, name)
    if bad:
        raise CliError(bad)
    return x


def cmd_previews(args) -> int:
    paths = _series(args.series)
    recs = _records(paths)  # a scanned series: a typo in the folder name must not create a config elsewhere
    tbin, ssub = _preview_setting(args.tbin, "--tbin"), _preview_setting(args.ssub, "--ssub")
    on = args.state == "on"
    actions.set_previews(paths, on, tbin=tbin, ssub=ssub)
    if not on:
        _say("[OK] previews off; existing preview files are kept")
    else:
        pv = config.resolve(paths)["preview"]
        _say(f"[OK] previews on (tbin {pv['tbin']}, ssub {pv['ssub']}); the next run makes them for every "
             f"session whose MC / CNMF-E is done")
    own = _own_preview_sessions(paths, recs, on)
    if own:  # a mouse or session config overrides the series setting
        _say(f"{own} sessions follow their own mouse config")
    return OK


def _own_preview_sessions(paths, recs, on: bool) -> int:
    """Sessions whose effective preview.enabled differs from what was just set (unreadable configs are left out)."""
    n = 0
    for rec in recs.values():
        if rec.excluded:
            continue
        try:
            n += bool(config.resolve(paths, rec.sid)["preview"]["enabled"]) != on
        except Exception:
            continue
    return n


# --- parser ---

def build_parser() -> argparse.ArgumentParser:
    p = _Parser(prog="run_bearmind", description="Batch pipeline for calcium-imaging series. "
                "Exit codes: 0 ok, 1 partial failure, 2 bad input, 3 series busy, 130 interrupted.")
    sub = p.add_subparsers(dest="command", required=True, parser_class=_Parser, metavar="command")

    def add(name, func, help_, who=True):
        sp = sub.add_parser(name, help=help_, description=help_)
        sp.add_argument("series", metavar="SERIES", help="series folder")
        if who:
            sp.add_argument("--who", help="your name in the decision log (default: Windows user name)")
        sp.set_defaults(func=func)
        return sp

    add("scan", cmd_scan, "find sessions in the series folder and register them", who=False)
    sp = add("run", cmd_run, "process the series (resumes where it stopped)", who=False)
    sp.add_argument("--until", type=_stage, help=f"last stage to run: {', '.join(STAGES)}")
    sp.add_argument("--sessions", nargs="+", metavar="SID", help="only these sessions")
    sp.add_argument("--once", action="store_true", help="do not wait for people; stop when nothing is ready")
    add("stop", cmd_stop, "ask the runner to stop before its next stage", who=False)
    sp = sub.add_parser("gui", help="open the BEARMiND window", description="open the BEARMiND window")
    sp.add_argument("series", metavar="SERIES", nargs="?", help="series folder (default: choose in the window)")
    sp.set_defaults(func=cmd_gui)
    add("status", cmd_status, "show state of every session and write summary.csv", who=False)

    sp = add("thresholds", cmd_thresholds, "set CNMF-E thresholds")
    ts = sp.add_subparsers(dest="action", required=True, parser_class=_Parser, metavar="action")
    a = ts.add_parser("accept-suggested", help="use the suggested thresholds")
    a.add_argument("--sessions", nargs="+", metavar="SID",
                   help="default: every session waiting for thresholds (maps up to date, none set)")
    s = ts.add_parser("set", help="set thresholds of one session")
    s.add_argument("sid", metavar="SID")
    s.add_argument("--gsig", type=_checked(actions.gsig_error, "--gsig"), required=True)
    s.add_argument("--min-corr", type=_checked(actions.min_corr_error, "--min-corr"), required=True)
    s.add_argument("--min-pnr", type=_checked(actions.min_pnr_error, "--min-pnr"), required=True)

    sp = add("gate", cmd_gate, "manual-inspection decision: gate SERIES SID ... (not-needed | cleaned)")
    sp.add_argument("items", nargs="*", metavar="SID... DECISION", help="session names, then not-needed or cleaned")
    sp.add_argument("--all-waiting", action="store_true", help="all sessions waiting for manual inspection")
    sp.add_argument("--force", action="store_true",
                    help="record not-needed also for a session whose manual inspection already passed")

    sp = add("decide", cmd_decide, "exclude or include a session, or accept a frames/timestamps mismatch")
    sp.add_argument("sid", metavar="SID")
    sp.add_argument("action", choices=sorted(_DECIDE))

    sp = add("rerun", cmd_rerun, "recompute sessions from a stage on")
    sp.add_argument("sids", nargs="+", metavar="SID")
    sp.add_argument("--from", dest="from_stage", type=_stage, required=True, help=f"one of: {', '.join(STAGES)}")

    sp = add("retry", cmd_retry, "recompute sessions from their failed stage (after disk full, locked file)")
    sp.add_argument("sids", nargs="*", metavar="SID")
    sp.add_argument("--all-failed", action="store_true", help="all sessions that have a failed stage")

    sp = add("previews", cmd_previews, "turn the preview videos (MC and CNMF-E QC) on or off for the series",
             who=False)
    sp.add_argument("state", choices=["on", "off"])
    sp.add_argument("--tbin", metavar="N", help="frames averaged into one preview frame (default 2)")
    sp.add_argument("--ssub", metavar="N", help="pixels per side averaged into one preview pixel (default 1: full size)")

    sp = add("inspect", cmd_inspect, "open the manual-inspection window in the browser (one server per series; "
             "a running one is reused)", who=False)
    sp.add_argument("sid", metavar="SID", nargs="?", help="session to open (default: the list of waiting sessions)")
    sp.add_argument("--port", type=_port, help="port of a new server (default: inspection.port; 0 takes a free one)")
    sp.add_argument("--no-browser", action="store_true", help="do not open the browser")
    return p


def _series_log(args):
    try:
        return _series(args.series).series_log
    except CliError:
        return None


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="backslashreplace")
    try:
        args = build_parser().parse_args(sys.argv[1:] if argv is None else argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 0
    if args.command == "gui":  # the series is optional here, so it is not resolved below
        return args.func(args)
    try:
        args.series = str(Path(args.series).resolve())
        return args.func(args)
    except CliError as e:
        _say(f"[FAILED] {e}")
        return e.code
    except KeyboardInterrupt:
        _say("[WARN] stopped by Ctrl+C; run the same command to resume where it stopped")
        return INTERRUPTED
    except Exception as e:
        log_file = SeriesPaths(args.series).series_log
        _log_traceback(log_file)
        _say(f"[FAILED] unexpected error: {type(e).__name__}: {_first_line(e)}; "
             f"details in {log_file}")
        return 1


def _log_traceback(log_file: Path) -> None:
    logger = logging.getLogger("bearmind")
    handler = None
    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(handler)
        logger.error("command failed", exc_info=True)
    except OSError:
        pass
    finally:
        if handler is not None:
            logger.removeHandler(handler)
            handler.close()
