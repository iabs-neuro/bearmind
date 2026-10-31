import pytest

from bearmind import config
from bearmind.paths import SeriesPaths
from bearmind.registry import SessionRecord, State, load_record, save_record
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
    assert rec.stages["gate"].state == State.DONE
    assert rec.stages["gate"].finished == rec.decisions[0].when
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
