"""Command line of the batch pipeline: python -m bearmind <command> SERIES ...

Exit codes: 0 ok, 1 partial failure, 2 bad input, 3 series busy, 130 interrupted. Messages are plain ASCII
([OK] / [FAILED] / [WARN]) because students read them in a Windows console. Every write to a record or
config goes through registry.update_record / config.update_yaml, so it is safe next to a running runner.
"""
import argparse
import getpass
import json
import logging
import math
import os
import sys
from datetime import datetime
from pathlib import Path

from bearmind import config, provenance, registry, runner, scan
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, Decision, State
from bearmind.stages import WaitingForInput, gate

OK, BAD_INPUT, BUSY, INTERRUPTED = 0, 2, 3, 130
HEARTBEAT_MAX_AGE_S = 120
_SHORT = {"crop": "crop", "mc": "mc", "thresholds": "thr", "cnmf": "cnmf", "gate": "gate",
          "autoinspect": "insp", "export": "exp"}
_LETTER = {State.DONE: "D", State.RUNNING: "R", State.WAITING: "W", State.FAILED: "F", State.SKIPPED: "S",
           State.PENDING: "."}
_REASON = {"thresholds": "thresholds", "manual_inspection": "manual inspection", "crop_finalize": "crop box"}
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


def _now() -> str:
    return datetime.now().isoformat(timespec="milliseconds")  # same format as gate.decide


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


def _gsig_error(x):
    if isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x != int(x) or x < 1:
        return f"--gsig must be a whole number >= 1, got {x}"
    return None


def _min_corr_error(x):
    if not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 < x <= 1:
        return f"--min-corr must be a number above 0 and at most 1, got {x}"
    return None


def _min_pnr_error(x):
    if not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0:
        return f"--min-pnr must be a finite number above 0, got {x}"
    return None


def _threshold_error(gsig, min_corr, min_pnr):
    return _gsig_error(gsig) or _min_corr_error(min_corr) or _min_pnr_error(min_pnr)


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
    unknown = [s for s in sids if s not in recs]
    if unknown:
        known = ", ".join(sorted(recs)) if len(recs) <= _MAX_LISTED else "see the list with: status SERIES"
        raise CliError(f"unknown session {', '.join(unknown)}; known sessions: {known}")


def _waiting_for(rec, stage: str):
    st = rec.stages.get(stage)
    if st is None or st.state is not State.WAITING:
        return None
    return runner._what(st)


def _suggested_unset(paths, rec) -> bool:
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


def _gate_waits(paths, rec) -> bool:
    """The gate waits for a person now: WAITING, or (CNMF done) any state but DONE, since the runner may be
    checking it at this moment. A passed gate, or one holding a decision that counts, is not taken."""
    cnmf, g = rec.stages.get("cnmf"), rec.stages.get("gate")
    if rec.excluded or (g is not None and g.state is State.DONE):
        return False
    if (g is None or g.state is not State.WAITING) and (cnmf is None or cnmf.state is not State.DONE):
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


def _warn_excluded(rec) -> None:
    if rec.excluded:
        _say(f"[WARN] {rec.sid} is excluded; the runner ignores it until you run: decide SERIES {rec.sid} include")


def _who(args) -> str:
    return args.who or getpass.getuser()


def _next_hint(paths) -> str:
    return f"next: run_bearmind run {paths.root}"


# --- commands ---

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
    try:
        res = runner.run(paths.root, until=args.until, sids=args.sessions, once=args.once, console=True)
    except KeyboardInterrupt:
        _say("[WARN] stopped by Ctrl+C; run the same command to resume where it stopped")
        return INTERRUPTED
    except RuntimeError as e:
        if "already being processed" in str(e):
            _say(f"[FAILED] {e}")
            return BUSY
        raise
    return _report_run(paths, res)


def _report_run(paths, res) -> int:
    """This run's outcome and the series state at exit, labelled so the two cannot be confused; 1 while any
    session of the series has a FAILED stage or cannot be scheduled."""
    st = runner.series_state(paths)
    waiting = sorted({sid for sid, _, _ in st.waiting})
    bad = bool(st.failed or st.problems)
    _say(("[FAILED] " if bad else "[OK] ")
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


def _row(rec) -> tuple:
    cells, reason = [], ""
    for s in STAGES:
        st = rec.stages.get(s)
        cells.append(_LETTER[st.state] if st else ".")
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
            + f"; failed: {len(state.failed)}")


def cmd_status(args) -> int:
    paths = _series(args.series)
    state = runner.series_state(paths)
    if not state.records and not state.broken:
        raise CliError(f"no sessions registered in {paths.root}; run first: scan {paths.root}")
    runner.summary(paths, state.records, problems=state.problems, broken=state.broken)
    rows = []
    for r in state.records:
        cells, reason = _row(r)
        rows.append((r.sid, "yes " if r.excluded else "no  ", cells, _note(r, reason, state)))
    rows += [(Path(f).stem, "?   ", ["?"] * len(STAGES), _clip(state.problems[Path(f).stem]))
             for f, _ in state.broken]
    _say(_heartbeat_line(paths))
    _say(_waiting_line(state))
    odd = scan.read_unrecognized(paths)
    if odd:
        _say(f"unrecognized folders: {len(odd)}; list: {paths.scan_unrecognized_csv}")
    _say("D done, R running, W waiting, F failed, S skipped, . not started, ? unknown")
    width = max(len("session"), *(len(sid) for sid, *_ in rows))
    head = "session".ljust(width) + "  excl  " + " ".join(_SHORT[s].ljust(4) for s in STAGES) + "  note"
    _say(head)
    _say("-" * len(head))
    for sid, excl, cells, note in sorted(rows):
        _say(sid.ljust(width) + "  " + excl + "  " + " ".join(c.ljust(4) for c in cells) + "  " + note)
    _say(f"summary table: {paths.summary_csv}")
    return OK


def _write_thresholds(paths, sid: str, gsig, min_corr, min_pnr) -> None:
    config.update_yaml(paths.session_config(sid), {"cnmf": {"gSig": gsig, "min_corr": min_corr,
                                                            "min_pnr": min_pnr}})


def cmd_thresholds(args) -> int:
    paths = _series(args.series)
    recs = _records(paths)
    if args.action == "set":
        _check_sids(recs, [args.sid])
        _warn_excluded(recs[args.sid])
        _write_thresholds(paths, args.sid, args.gsig, args.min_corr, args.min_pnr)
        _say(f"[OK] {args.sid}: gSig={args.gsig} min_corr={args.min_corr} min_pnr={args.min_pnr}")
        _say(_next_hint(paths))
        return OK
    if args.sessions:
        _check_sids(recs, args.sessions)
        sids = list(args.sessions)
    else:
        sids = [s for s, r in sorted(recs.items()) if not r.excluded
                and (_waiting_for(r, "cnmf") == "thresholds" or _suggested_unset(paths, r))]
        if not sids:
            _say("[WARN] no session waits for thresholds")
            return OK
    code = OK
    for sid in sids:
        th = recs[sid].stages["thresholds"]
        sug = th.metrics.get("suggested") if th.state is State.DONE else None
        if not sug:
            _say(f"[FAILED] {sid}: no threshold suggestion (run the thresholds stage first, "
                 f"or set values: thresholds SERIES set {sid} --gsig G --min-corr X --min-pnr Y)")
            code = 1
            continue
        bad = _threshold_error(sug.get("gSig"), sug.get("min_corr"), sug.get("min_pnr"))
        if bad:
            _say(f"[FAILED] {sid}: suggested thresholds are invalid: {bad.replace('--', '')}")
            code = 1
            continue
        _warn_excluded(recs[sid])
        _write_thresholds(paths, sid, sug["gSig"], sug["min_corr"], sug["min_pnr"])
        _say(f"[OK] {sid}: gSig={sug['gSig']} min_corr={sug['min_corr']} min_pnr={sug['min_pnr']}")
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
        sids = [s for s, r in sorted(recs.items()) if _gate_waits(paths, r)]
        if not sids:
            _say("[WARN] no session waits for manual inspection")
            return OK
    _check_sids(recs, sids)
    who, code = _who(args), OK
    for sid in sids:
        _warn_excluded(recs[sid])
        try:
            gate.decide(paths, sid, decision, who)
        except ValueError as e:
            _say(f"[FAILED] {sid}: {e}")
            if decision == "cleaned":
                _say("        save the cleaned file at that place, then repeat the command")
            code = 1
        else:
            _say(f"[OK] {sid}: manual inspection {decision.replace('_', '-')}")
    if code == OK:
        _say(_next_hint(paths))
    return code


_DECIDE = {"exclude": ("exclude", "true"), "include": ("exclude", "false"),
           "accept-mismatch": ("accept_mismatch", "true")}


def cmd_decide(args) -> int:
    paths = _series(args.series)
    _check_sids(_records(paths), [args.sid])
    recs = _records(paths)
    if args.action == "accept-mismatch" and not any(
            a.code == "frames_ts_mismatch" and a.level == "error" for a in recs[args.sid].anomalies):
        _say(f"[WARN] {args.sid} has no frames_ts_mismatch error; the decision is recorded but changes nothing")
    if args.action == "include" and recs[args.sid].fps is None:
        _say(f"[WARN] {args.sid} has no frame rate (timestamps missing or unusable): it stays excluded, the "
             f"stages need the frame rate; the decision is recorded and counts once a rescan finds the rate")
    what, value = _DECIDE[args.action]
    now, who = _now(), _who(args)

    def change(rec):
        rec.decisions.append(Decision(what, value, who, now))
        rec.excluded = scan.evaluate_excluded(rec)  # the same rule as the scan: the latest decision wins

    rec = registry.update_record(paths, args.sid, change)
    _say(f"[OK] {args.sid}: {args.action} recorded; session is now "
         + ("excluded" if rec.excluded else "included"))
    return OK


def cmd_rerun(args) -> int:
    paths = _series(args.series)
    _check_sids(_records(paths), args.sids)
    who = _who(args)
    for sid in args.sids:
        runner.reset_from(paths, sid, args.from_stage, who)
        _say(f"[OK] {sid}: stages from {args.from_stage} on are reset")
    _say(_next_hint(paths))
    return OK


def cmd_retry(args) -> int:
    paths = _series(args.series)
    recs = _records(paths)
    if bool(args.sids) == args.all_failed:
        raise CliError("give session names or --all-failed, not both and not neither")
    if args.all_failed:
        sids = [s for s, r in sorted(recs.items()) if not r.excluded
                and any(r.stages.get(x) and r.stages[x].state is State.FAILED for x in STAGES)]
        if not sids:
            _say("[OK] no failed sessions")
            return OK
    else:
        _check_sids(recs, args.sids)
        sids = list(args.sids)
    who, n = _who(args), 0
    for sid in sids:
        first = next((s for s in STAGES if recs[sid].stages.get(s) and recs[sid].stages[s].state is State.FAILED),
                     None)
        if first is None:
            _say(f"[WARN] {sid}: no failed stage, skipped")
            continue
        runner.reset_from(paths, sid, first, who)
        n += 1
        _say(f"[OK] {sid}: reset from {first}")
    if n:
        _say(_next_hint(paths))
    return OK if n else 1


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
    add("status", cmd_status, "show state of every session and write summary.csv", who=False)

    sp = add("thresholds", cmd_thresholds, "set CNMF-E thresholds", who=False)
    ts = sp.add_subparsers(dest="action", required=True, parser_class=_Parser, metavar="action")
    a = ts.add_parser("accept-suggested", help="use the suggested thresholds")
    a.add_argument("--sessions", nargs="+", metavar="SID",
                   help="default: every session waiting for thresholds, or with a suggestion and none set")
    s = ts.add_parser("set", help="set thresholds of one session")
    s.add_argument("sid", metavar="SID")
    s.add_argument("--gsig", type=_checked(_gsig_error, "--gsig"), required=True)
    s.add_argument("--min-corr", type=_checked(_min_corr_error, "--min-corr"), required=True)
    s.add_argument("--min-pnr", type=_checked(_min_pnr_error, "--min-pnr"), required=True)

    sp = add("gate", cmd_gate, "manual-inspection decision: gate SERIES SID ... (not-needed | cleaned)")
    sp.add_argument("items", nargs="*", metavar="SID... DECISION", help="session names, then not-needed or cleaned")
    sp.add_argument("--all-waiting", action="store_true", help="all sessions waiting for manual inspection")

    sp = add("decide", cmd_decide, "exclude or include a session, or accept a frames/timestamps mismatch")
    sp.add_argument("sid", metavar="SID")
    sp.add_argument("action", choices=sorted(_DECIDE))

    sp = add("rerun", cmd_rerun, "recompute sessions from a stage on")
    sp.add_argument("sids", nargs="+", metavar="SID")
    sp.add_argument("--from", dest="from_stage", type=_stage, required=True, help=f"one of: {', '.join(STAGES)}")

    sp = add("retry", cmd_retry, "recompute sessions from their failed stage (after disk full, locked file)")
    sp.add_argument("sids", nargs="*", metavar="SID")
    sp.add_argument("--all-failed", action="store_true", help="all sessions that have a failed stage")
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
    cwd = os.getcwd()
    try:
        args.series = str(Path(args.series).resolve())  # absolute before the chdir
        os.chdir(provenance.REPO_ROOT)  # the old config.py creates config.json in the current folder
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
    finally:
        os.chdir(cwd)


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
