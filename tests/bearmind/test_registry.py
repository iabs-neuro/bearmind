import json

import pytest

from bearmind import registry
from bearmind.paths import SeriesPaths
from bearmind.registry import (
    STAGES, Anomaly, Check, Decision, RawPart, SessionRecord, State,
    StageRecord, all_records, load_record, save_record,
)


def make_record(sid="M1_D1"):
    part = RawPart(
        folder="1_Raw/a", chunks=["c0.avi", "c1.avi"], n_frames=100,
        shape=(480, 640), timestamps_file="0_TimeStamps/a.csv",
        ts_format="miniscope", start_ms=12.5, metadata_file=None,
        drop_chunk_tail=True, drop_last_frame=False,
    )
    stages = {s: StageRecord() for s in STAGES}
    stages["mc"] = StageRecord(
        state=State.DONE, params_hash="abc", started="2026-10-31T10:00:00",
        finished="2026-10-31T10:05:00", outputs={"video": "3_MC/x.avi"},
        metrics={"shift": 1.5},
        checks=[Check("mc", "shift", "warning", 1.5, "big shift"),
                Check("mc", "none", "ok", None, "")],
        error=None,
    )
    stages["cnmf"] = StageRecord(state=State.FAILED, error="boom")
    return SessionRecord(
        sid=sid, mouse="M1", mode="series", parts=[part], n_frames=100,
        fps=30.0, fps_info={"source": "ts"},
        anomalies=[Anomaly("A1", "warning", "msg", True, {"k": 1})],
        decisions=[Decision("fps", "30", "user", "2026-10-31T10:00:00")],
        excluded=False, stages=stages,
    )


def test_roundtrip(tmp_path):
    paths = SeriesPaths(tmp_path)
    rec = make_record()
    save_record(paths, rec)
    back = load_record(paths, "M1_D1")
    assert back == rec
    assert isinstance(back.parts[0].shape, tuple)
    assert back.stages["mc"].state is State.DONE
    raw = json.loads(paths.registry_file("M1_D1").read_text(encoding="utf-8"))
    assert raw["stages"]["mc"]["state"] == "done" or raw["stages"]["mc"]["state"] == State.DONE.value


def test_defaults_not_shared():
    a, b = StageRecord(), StageRecord()
    a.outputs["x"] = "y"
    a.checks.append(Check("s", "n", "ok", None, ""))
    assert b.outputs == {} and b.checks == []
    assert Anomaly("c", "ok", "m").data is not Anomaly("c", "ok", "m").data


def test_load_unknown_sid(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_record(SeriesPaths(tmp_path), "nope")


def test_unknown_keys_ignored(tmp_path):
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    f = paths.registry_file("M1_D1")
    d = json.loads(f.read_text(encoding="utf-8"))
    d["future"] = 1
    d["parts"][0]["future"] = 2
    d["stages"]["mc"]["future"] = 3
    d["anomalies"][0]["future"] = 4
    f.write_text(json.dumps(d), encoding="utf-8")
    assert load_record(paths, "M1_D1") == make_record()


def test_save_failure_keeps_old_file(tmp_path, monkeypatch):
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    f = paths.registry_file("M1_D1")
    before = f.read_bytes()

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(registry.json, "dump", boom)
    new = make_record()
    new.excluded = True
    with pytest.raises(OSError):
        save_record(paths, new)
    assert f.read_bytes() == before
    assert not list(f.parent.glob("*.tmp"))


def test_all_records(tmp_path):
    paths = SeriesPaths(tmp_path)
    assert all_records(paths) == []
    save_record(paths, make_record("B"))
    save_record(paths, make_record("A"))
    (paths.registry_file("A").parent / "stale.json.tmp").write_text("{")
    assert [r.sid for r in all_records(paths)] == ["A", "B"]
