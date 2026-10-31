"""Manual-inspection gate: a session waits here until a person cleans it or marks it not needed."""
from datetime import datetime
from pathlib import Path

from bearmind.registry import Decision, State, StageRecord, load_record, save_record
from bearmind.stages import StageResult, WaitingForInput

GATE_DECISIONS = ("cleaned", "not_needed")
_WHAT = "manual_inspection"


def _latest(rec) -> Decision | None:
    found = [d for d in rec.decisions if d.what == _WHAT]
    return found[-1] if found else None


def decide(paths, sid: str, decision: str, who: str) -> None:
    if decision not in GATE_DECISIONS:
        raise ValueError(f"decision must be one of {GATE_DECISIONS}, got {decision!r}")
    if decision == "cleaned" and not paths.manual_result(sid).exists():
        raise ValueError(f"manual inspection result not found: {paths.manual_result(sid)}")
    rec = load_record(paths, sid)
    now = datetime.now().isoformat(timespec="seconds")
    rec.decisions.append(Decision(_WHAT, decision, who, now))
    st = rec.stages.setdefault("gate", StageRecord())
    st.state = State.DONE
    st.finished = now
    save_record(paths, rec)


def run(paths, rec, cfg) -> StageResult:
    if not cfg["gate"]["enabled"]:
        return StageResult(metrics={"gate": "disabled"})
    d = _latest(rec)
    if d is None:
        raise WaitingForInput(_WHAT)
    return StageResult(metrics={"gate": d.value, "who": d.who, "when": d.when})


def input_estimates(paths, rec) -> Path:
    d = _latest(rec)
    if d is not None and d.value == "cleaned":
        return paths.manual_result(rec.sid)
    return paths.estimates(rec.sid)
