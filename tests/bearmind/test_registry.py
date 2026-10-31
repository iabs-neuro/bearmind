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


def test_chunk_frames_roundtrip_and_old_records(tmp_path):
    paths = SeriesPaths(tmp_path)
    rec = make_record()
    rec.parts[0].chunk_frames = [60, 40]
    save_record(paths, rec)
    assert load_record(paths, "M1_D1").parts[0].chunk_frames == [60, 40]
    f = paths.registry_file("M1_D1")
    d = json.loads(f.read_text(encoding="utf-8"))
    del d["parts"][0]["chunk_frames"]  # written before the field existed
    f.write_text(json.dumps(d), encoding="utf-8")
    assert load_record(paths, "M1_D1").parts[0].chunk_frames == []


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


# --- write protocol (R25): reload before save, unique temp files, short lock, PermissionError retry ---

import logging
import os
import threading
import time
from pathlib import Path

import subprocess
import sys

import psutil

from bearmind import fileio
from bearmind.registry import update_record


def _lock_of(paths, sid="M1_D1") -> Path:
    f = paths.registry_file(sid)
    return f.with_name(f.name + ".lock")


def test_params_roundtrip_and_old_records(tmp_path):
    paths = SeriesPaths(tmp_path)
    rec = make_record()
    rec.stages["mc"].params = {"mc": {"max_shifts": [35, 35]}, "fps": 30.0, "box_pos": None}
    save_record(paths, rec)
    assert load_record(paths, "M1_D1").stages["mc"].params == rec.stages["mc"].params
    f = paths.registry_file("M1_D1")
    d = json.loads(f.read_text(encoding="utf-8"))
    del d["stages"]["mc"]["params"]  # written before the field existed
    f.write_text(json.dumps(d), encoding="utf-8")
    assert load_record(paths, "M1_D1").stages["mc"].params == {}


def test_update_record_reloads_before_save(tmp_path):
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    stale = load_record(paths, "M1_D1")
    other = load_record(paths, "M1_D1")  # e.g. a GUI writing a decision meanwhile
    other.decisions.append(Decision("manual_inspection", "not_needed", "viktor", "t"))
    save_record(paths, other)

    def mark(rec):
        rec.stages["cnmf"] = StageRecord(state=State.DONE, params_hash="h")

    out = update_record(paths, "M1_D1", mark)
    back = load_record(paths, "M1_D1")
    assert back == out
    assert back.stages["cnmf"].state is State.DONE
    assert [d.what for d in back.decisions] == ["fps", "manual_inspection"]
    assert stale.stages["cnmf"].state is State.FAILED  # the caller's copy is not the source
    assert not _lock_of(paths).exists()


def test_update_record_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        update_record(SeriesPaths(tmp_path), "nope", lambda r: None)


def test_concurrent_updates_not_lost(tmp_path):
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())

    def worker(k):
        for i in range(15):
            update_record(paths, "M1_D1",
                          lambda r: r.decisions.append(Decision("x", f"{k}-{i}", "w", "t")))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(load_record(paths, "M1_D1").decisions) == 1 + 45
    folder = paths.registry_file("M1_D1").parent
    assert sorted(p.name for p in folder.iterdir()) == ["M1_D1.json"]


def test_stale_lock_is_broken(tmp_path):
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    lock = _lock_of(paths)
    lock.write_text("12345")
    old = time.time() - 120  # holder died long ago
    os.utime(lock, (old, old))
    rec = make_record()
    rec.excluded = True
    save_record(paths, rec)
    assert load_record(paths, "M1_D1").excluded
    assert not lock.exists()


def test_busy_lock_times_out(tmp_path, monkeypatch):
    monkeypatch.setattr(fileio, "LOCK_TIMEOUT_S", 0.3)
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    lock = _lock_of(paths)
    me = psutil.Process()
    lock.write_text(f"{me.pid} {me.create_time()} other-writer")  # fresh, live holder
    t0 = time.monotonic()
    with pytest.raises(TimeoutError, match="lock"):
        save_record(paths, make_record())
    assert time.monotonic() - t0 >= 0.3
    assert lock.exists()  # someone else's lock is left alone


def test_replace_retries_permission_error(tmp_path, monkeypatch):
    monkeypatch.setattr(fileio, "REPLACE_STEP_S", 0.01)
    paths = SeriesPaths(tmp_path)
    real = os.replace
    left = {"n": 3}

    def flaky(src, dst):
        if left["n"]:
            left["n"] -= 1
            raise PermissionError(13, "file is open in another program")
        return real(src, dst)

    monkeypatch.setattr(os, "replace", flaky)
    save_record(paths, make_record())
    assert left["n"] == 0
    assert load_record(paths, "M1_D1") == make_record()


def test_replace_gives_up_and_cleans_temp(tmp_path, monkeypatch):
    monkeypatch.setattr(fileio, "REPLACE_STEP_S", 0.0)
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    calls = {"n": 0}

    def deny(src, dst):
        calls["n"] += 1
        raise PermissionError(13, "denied")

    monkeypatch.setattr(os, "replace", deny)
    with pytest.raises(PermissionError):
        save_record(paths, make_record("M1_D1"))
    assert calls["n"] == fileio.REPLACE_TRIES
    folder = paths.registry_file("M1_D1").parent
    assert sorted(p.name for p in folder.iterdir()) == ["M1_D1.json"]


def test_save_failure_releases_lock(tmp_path, monkeypatch):
    paths = SeriesPaths(tmp_path)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(registry.json, "dump", boom)
    with pytest.raises(OSError):
        save_record(paths, make_record())
    assert list(paths.registry_file("M1_D1").parent.iterdir()) == []


def test_all_records_skips_corrupt(tmp_path, caplog):
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record("A"))
    save_record(paths, make_record("C"))
    paths.registry_file("B").write_text("{", encoding="utf-8")
    d = json.loads(paths.registry_file("C").read_text(encoding="utf-8"))
    d["stages"]["mc"]["state"] = "exploded"
    paths.registry_file("D").write_text(json.dumps(d), encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="bearmind"):
        assert [r.sid for r in all_records(paths)] == ["A", "C"]
    assert "B.json" in caplog.text and "D.json" in caplog.text


def test_numpy_metrics_saved_as_numbers(tmp_path):
    import numpy as np

    paths = SeriesPaths(tmp_path)
    rec = make_record()
    rec.stages["mc"].metrics = {"max_shift_px": np.float32(1.5), "n": np.int64(3), "a": np.arange(2)}
    save_record(paths, rec)
    assert load_record(paths, "M1_D1").stages["mc"].metrics == {"max_shift_px": 1.5, "n": 3, "a": [0, 1]}


def test_corrupt_file_logged_once_per_version(tmp_path, caplog):
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record("A"))
    bad = paths.registry_file("B")
    bad.write_text("{", encoding="utf-8")
    with caplog.at_level(logging.ERROR, logger="bearmind"):
        all_records(paths)
        all_records(paths)
        assert caplog.text.count("B.json") == 1
        bad.write_text("{{", encoding="utf-8")  # changed again: worth a new message
        all_records(paths)
        assert caplog.text.count("B.json") == 2



def _dead_pid() -> int:
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    p.wait()
    return p.pid


def test_dead_holder_lock_taken_over_at_once(tmp_path):
    """I3: a lock whose writer is gone is broken at once, not after the 60 s age."""
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    lock = _lock_of(paths)
    lock.write_text(f"{_dead_pid()} 1.0 dead-writer")
    t0 = time.monotonic()
    rec = make_record()
    rec.excluded = True
    save_record(paths, rec)
    assert time.monotonic() - t0 < 2.0
    assert load_record(paths, "M1_D1").excluded and not lock.exists()
    me = psutil.Process()
    lock.write_text(f"{me.pid} {me.create_time() - 1000} reused-pid")  # same PID, other process
    t0 = time.monotonic()
    save_record(paths, make_record())
    assert time.monotonic() - t0 < 2.0 and not lock.exists()


def test_unreadable_lock_content_counts_as_held(tmp_path, monkeypatch):
    monkeypatch.setattr(fileio, "LOCK_TIMEOUT_S", 0.3)
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    lock = _lock_of(paths)
    lock.write_text("")  # a writer between create and write
    with pytest.raises(fileio.LockTimeout):
        save_record(paths, make_record())
    assert lock.exists()


def test_contenders_take_over_a_dead_lock_one_at_a_time(tmp_path):
    paths = SeriesPaths(tmp_path)
    save_record(paths, make_record())
    _lock_of(paths).write_text(f"{_dead_pid()} 1.0 dead-writer")

    def worker(k):
        for i in range(10):
            update_record(paths, "M1_D1",
                          lambda r: r.decisions.append(Decision("x", f"{k}-{i}", "w", "t")))

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(load_record(paths, "M1_D1").decisions) == 1 + 40
    folder = paths.registry_file("M1_D1").parent
    assert sorted(p.name for p in folder.iterdir()) == ["M1_D1.json"]
