"""Manual-inspection gate: a session waits here until a person cleans it or marks it not needed.

A decision counts only for the CNMF output it was made on (R28): it must be newer than the CNMF stage's
finish time, and a "cleaned" result file must be newer than the estimates it was cleaned from. Otherwise
the gate waits again, so a CNMF re-run never lets an old manual result through.
"""
from datetime import datetime
from pathlib import Path

from bearmind.registry import Decision, State, StageRecord, update_record
from bearmind.stages import StageResult, WaitingForInput

GATE_DECISIONS = ("cleaned", "not_needed")
_WHAT = "manual_inspection"


def _latest(rec) -> Decision | None:
    found = [d for d in rec.decisions if d.what == _WHAT]
    return found[-1] if found else None


def _cnmf_finished(rec):
    st = rec.stages.get("cnmf")
    return st.finished if st is not None and st.state is State.DONE else None


def _estimates_file(paths, rec) -> Path:
    st = rec.stages.get("cnmf")
    out = (st.outputs if st is not None else {}).get("estimates")
    return paths.root / out if out else paths.estimates(rec.sid)


def decide(paths, sid: str, decision: str, who: str) -> None:
    """Record the decision; the gate goes back to PENDING and only gate.run (through the runner) passes it."""
    if decision not in GATE_DECISIONS:
        raise ValueError(f"decision must be one of {GATE_DECISIONS}, got {decision!r}")
    if decision == "cleaned" and not paths.manual_result(sid).exists():
        raise ValueError(f"manual inspection result not found: {paths.manual_result(sid)}")
    now = datetime.now().isoformat(timespec="milliseconds")  # compared with the CNMF finish time

    def record(rec):
        rec.decisions.append(Decision(_WHAT, decision, who, now))
        st = rec.stages.setdefault("gate", StageRecord())
        st.state, st.params_hash, st.error = State.PENDING, None, None

    update_record(paths, sid, record)


def run(paths, rec, cfg) -> StageResult:
    if not cfg["gate"]["enabled"]:
        return StageResult(metrics={"gate": "disabled"})
    d = _latest(rec)
    if d is None:
        raise WaitingForInput(_WHAT)
    finished = _cnmf_finished(rec)
    if finished and datetime.fromisoformat(d.when) < datetime.fromisoformat(finished):
        raise WaitingForInput(_WHAT, "the decision is older than the current CNMF output")
    if d.value == "cleaned":
        manual, est = paths.manual_result(rec.sid), _estimates_file(paths, rec)
        if not manual.is_file():
            raise WaitingForInput(_WHAT, f"manual result not found: {paths.rel(manual)}")
        if est.is_file() and manual.stat().st_mtime_ns < est.stat().st_mtime_ns:
            raise WaitingForInput(_WHAT, "the manual result is older than the current CNMF output")
    return StageResult(metrics={"gate": d.value, "who": d.who, "when": d.when})


def input_estimates(paths, rec) -> Path:
    d = _latest(rec)
    if d is not None and d.value == "cleaned":
        return paths.manual_result(rec.sid)
    return paths.estimates(rec.sid)
