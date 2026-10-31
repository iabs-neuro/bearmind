import os
import threading
import time
from datetime import datetime, timedelta

import pytest

from bearmind import config
from bearmind.paths import SeriesPaths
from bearmind.registry import SessionRecord, StageRecord, State, load_record, save_record, update_record
from bearmind.stages import WaitingForInput, gate

SID = "NOF_H01_1D"


def _setup(tmp_path):
    paths = SeriesPaths(tmp_path / "series")
    rec = SessionRecord(sid=SID, mouse="NOF_H01", mode="bonsai", parts=[], n_frames=10, fps=20.0,
                        fps_info={}, anomalies=[], decisions=[], excluded=False, stages={})
    save_record(paths, rec)
    return paths, rec, config.load_defaults()


def test_no_decision_waits(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    with pytest.raises(WaitingForInput) as e:
        gate.run(paths, rec, cfg)
    assert e.value.what == "manual_inspection"


def test_not_needed(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    gate.decide(paths, SID, "not_needed", "viktor")
    rec = load_record(paths, SID)
    # R28: decide only records the decision; gate.run (through the runner) sets the gate DONE
    assert rec.stages["gate"].state == State.PENDING and rec.stages["gate"].params_hash is None
    res = gate.run(paths, rec, cfg)
    assert res.metrics["gate"] == "not_needed" and res.metrics["who"] == "viktor"
    assert gate.input_estimates(paths, rec) == paths.estimates(SID)


def test_cleaned_requires_file(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    with pytest.raises(ValueError, match="manual inspection result not found"):
        gate.decide(paths, SID, "cleaned", "viktor")
    assert load_record(paths, SID).decisions == []


def test_cleaned_uses_manual_result(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    f = paths.manual_result(SID)
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    gate.decide(paths, SID, "cleaned", "viktor")
    rec = load_record(paths, SID)
    assert gate.run(paths, rec, cfg).metrics["gate"] == "cleaned"
    assert gate.input_estimates(paths, rec) == f


def test_invalid_decision(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    with pytest.raises(ValueError):
        gate.decide(paths, SID, "maybe", "viktor")


def test_disabled(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    cfg["gate"]["enabled"] = False
    assert gate.run(paths, rec, cfg).metrics == {"gate": "disabled"}


def test_latest_decision_wins(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    f = paths.manual_result(SID)
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    gate.decide(paths, SID, "cleaned", "a")
    gate.decide(paths, SID, "not_needed", "b")
    rec = load_record(paths, SID)
    assert gate.run(paths, rec, cfg).metrics["gate"] == "not_needed"
    assert gate.input_estimates(paths, rec) == paths.estimates(SID)


def test_decide_resets_a_done_gate(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    rec.stages["gate"] = StageRecord(state=State.DONE, params_hash="old", finished="2026-10-31T10:00:00")
    save_record(paths, rec)
    gate.decide(paths, SID, "not_needed", "viktor")
    st = load_record(paths, SID).stages["gate"]
    assert st.state is State.PENDING and st.params_hash is None


def test_decision_older_than_cnmf_output_waits(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    gate.decide(paths, SID, "not_needed", "viktor")
    rec = load_record(paths, SID)
    when = datetime.fromisoformat(rec.decisions[-1].when)
    rec.stages["cnmf"] = StageRecord(state=State.DONE, finished=(when + timedelta(seconds=1)).isoformat())
    with pytest.raises(WaitingForInput) as e:
        gate.run(paths, rec, cfg)
    assert e.value.what == "manual_inspection"
    assert "decision is older than the current CNMF output" in str(e.value)
    rec.stages["cnmf"].finished = (when - timedelta(seconds=1)).isoformat()
    assert gate.run(paths, rec, cfg).metrics["gate"] == "not_needed"


def test_cleaned_result_older_than_estimates_waits(tmp_path):
    paths, rec, cfg = _setup(tmp_path)
    f = paths.manual_result(SID)
    f.parent.mkdir(parents=True)
    f.write_bytes(b"manual")
    est = paths.estimates(SID)
    est.parent.mkdir(parents=True)
    est.write_bytes(b"estimates")
    t = f.stat().st_mtime
    os.utime(est, (t + 10, t + 10))  # CNMF wrote new estimates after the person saved the result
    rec.stages["cnmf"] = StageRecord(state=State.DONE, finished="2000-01-01T00:00:00",
                                     outputs={"estimates": paths.rel(est)})
    save_record(paths, rec)
    gate.decide(paths, SID, "cleaned", "viktor")
    rec = load_record(paths, SID)
    with pytest.raises(WaitingForInput) as e:
        gate.run(paths, rec, cfg)
    assert "manual result is older than the current CNMF output" in str(e.value)
    os.utime(f, (t + 20, t + 20))  # saved again after the new estimates
    assert gate.run(paths, rec, cfg).metrics["gate"] == "cleaned"


def test_decide_keeps_a_concurrent_update(tmp_path):
    """R29: decide rereads under the lock, so a runner write made meanwhile survives."""
    paths, rec, cfg = _setup(tmp_path)
    entered, release = threading.Event(), threading.Event()

    def slow(r):
        entered.set()
        release.wait(10)
        r.stages["cnmf"] = StageRecord(state=State.DONE, params_hash="h")

    t = threading.Thread(target=update_record, args=(paths, SID, slow))
    t.start()
    assert entered.wait(10)
    d = threading.Thread(target=gate.decide, args=(paths, SID, "not_needed", "viktor"))
    d.start()
    time.sleep(0.3)  # decide is now waiting for the lock
    release.set()
    t.join(10)
    d.join(10)
    rec = load_record(paths, SID)
    assert rec.stages["cnmf"].params_hash == "h"
    assert [x.what for x in rec.decisions] == ["manual_inspection"]
