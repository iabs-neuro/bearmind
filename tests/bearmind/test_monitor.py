import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from bearmind import config, monitor, registry
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, Anomaly, Check, SessionRecord, StageRecord, State

NOW = datetime(2026, 11, 7, 12, 0, 0)
FAKE = Path(__file__).with_name("fake_runner.py")
REPO = Path(__file__).resolve().parents[2]
PENDING_STAGES = ("crop", "mc", "thresholds", "cnmf", "autoinspect", "export")  # gate has no default number


@pytest.fixture
def tmp_series(tmp_path):
    p = SeriesPaths(tmp_path / "series")
    p.registry_file("x").parent.mkdir(parents=True)
    return p


def iso(dt):
    return dt.isoformat(timespec="milliseconds")


def done(metrics=None, checks=(), seconds=None):
    st = StageRecord(State.DONE, metrics=metrics or {}, checks=list(checks))
    if seconds is not None:
        st.started = iso(NOW - timedelta(hours=5))
        st.finished = iso(NOW - timedelta(hours=5) + timedelta(seconds=seconds))
    return st


def put(paths, sid, stages=None, mouse=None, excluded=False, anomalies=(), n_frames=1000):
    st = {s: StageRecord() for s in STAGES}
    st.update(stages or {})
    rec = SessionRecord(sid=sid, mouse=mouse or sid.split("_")[1], mode="bonsai", parts=[], n_frames=n_frames,
                        fps=20.0, fps_info={}, anomalies=list(anomalies), decisions=[], excluded=excluded,
                        stages=st)
    registry.save_record(paths, rec)
    return rec


def waiting(why):
    return StageRecord(State.WAITING, error=f"waiting for {why}")


def check(stage, name, level, msg="m"):
    return Check(stage, name, level, 1.0, msg)


def view(paths, cache=None):
    return monitor.series_view(paths, NOW, cache)


def defaults():
    return config.load_defaults()["monitor"]["eta_s_per_frame"]


def spawn(paths, sid, stage, status):
    p = subprocess.Popen([sys.executable, str(FAKE), str(paths.root), sid, stage, status],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert p.stdout.readline().strip() == "ready"
    return p


def stop(p):
    p.stdin.close()
    p.wait(timeout=20)


def test_runner_state_stopped_without_heartbeat(tmp_series):
    r = monitor.runner_state(tmp_series, NOW)
    assert r.status == "stopped" and r.alive is False


def test_runner_state_alive(tmp_series):
    started = datetime.now() - timedelta(seconds=90)
    put(tmp_series, "A_M1_1D", {"mc": StageRecord(State.RUNNING, started=iso(started))})
    p = spawn(tmp_series, "A_M1_1D", "mc", "running")
    try:
        r = monitor.runner_state(tmp_series)
        assert r.alive and r.status == "running" and r.pid == p.pid
        assert r.current == ("A_M1_1D", "mc")
        assert 85 < r.stage_elapsed_s < 150
    finally:
        stop(p)


def test_runner_state_dead_when_pid_gone(tmp_series):
    p = spawn(tmp_series, "A_M1_1D", "mc", "running")
    p.kill()
    p.wait(timeout=20)
    assert tmp_series.lock.exists()  # a killed runner leaves its lock file
    r = monitor.runner_state(tmp_series)
    assert r.status == "dead" and r.alive is False


def test_runner_state_stale_heartbeat(tmp_series):
    p = spawn(tmp_series, "A_M1_1D", "mc", "running")
    try:
        assert monitor.runner_state(tmp_series).alive
        r = monitor.runner_state(tmp_series, datetime.now() + timedelta(seconds=300))
        assert r.alive is False and r.status == "dead"
    finally:
        stop(p)


def test_runner_state_after_normal_exit(tmp_series):
    p = spawn(tmp_series, "-", "", "idle")
    stop(p)
    assert not tmp_series.lock.exists()
    r = monitor.runner_state(tmp_series)
    assert r.alive is False and r.status == "stopped"


def test_fake_runner_clean_exit_is_stopped_not_dead(tmp_series):
    """Like the real runner, the stand-in writes an idle pulse and removes its lock when it ends normally."""
    p = spawn(tmp_series, "A_M1_1D", "mc", "running")
    stop(p)
    hb = monitor._read_json(tmp_series.heartbeat)
    assert hb["status"] == "idle" and hb["current"] is None and hb["exit"] == "finished"
    assert not tmp_series.lock.exists()
    r = monitor.runner_state(tmp_series)
    assert r.alive is False and r.status == "stopped"


def test_cell_levels_and_texts(tmp_series):
    put(tmp_series, "A_M1_1D", {
        "mc": done({"max_shift_px": 18.0}, [check("mc", "mc_max_shift_px", "warning")]),
        "cnmf": waiting("thresholds")})
    put(tmp_series, "A_M1_2D", {"cnmf": StageRecord(State.FAILED, error="Traceback\nMemoryError: out")})
    rows = {r.sid: r for r in view(tmp_series).rows}
    mc = rows["A_M1_1D"].cells["mc"]
    assert (mc.level, mc.text) == ("warning", "18.0 px")
    assert rows["A_M1_1D"].cells["cnmf"].text == "waits"
    f = rows["A_M1_2D"].cells["cnmf"]
    assert (f.level, f.text) == ("error", "failed") and "MemoryError: out" in f.tooltip
    assert rows["A_M1_2D"].cells["export"].level == "none"


def test_cell_texts_match_mockup(tmp_series):
    config.update_yaml(tmp_series.series_config, {"cnmf": {"gSig": 4}})
    put(tmp_series, "A_M1_1D", {
        "crop": done(),
        "mc": done({"max_shift_px": 3.2, "crop": [0, 0, 512, 480]}),
        "thresholds": done({"suggested": {"gSig": 4}}),
        "cnmf": done({"n_components": 231}),
        "gate": done({"gate": "not_needed"}),
        "autoinspect": done({"n_kept": 128}),
        "export": done({"n_neurons": 121})})
    put(tmp_series, "A_M1_2D", {"gate": done({"gate": "disabled"})})
    put(tmp_series, "A_M1_3D", {"gate": done({"gate": "cleaned"})})
    rows = {r.sid: r.cells for r in view(tmp_series).rows}
    assert [rows["A_M1_1D"][s].text for s in STAGES] == [
        "512×480", "3.2 px", "gSig 4", "231 cells", "not needed", "128 kept", "121 neurons"]
    assert rows["A_M1_2D"]["gate"].text == "off"
    assert rows["A_M1_3D"]["gate"].text == "cleaned"


def test_tooltip_uses_check_message(tmp_series):
    msg = "max shift 18.0 px (warn > 2, error > 3)"
    put(tmp_series, "A_M1_1D", {"mc": done({"max_shift_px": 18.0},
                                           [check("mc", "mc_max_shift_px", "error", msg)])})
    assert msg in view(tmp_series).rows[0].cells["mc"].tooltip


def test_excluded_row_cells(tmp_series):
    put(tmp_series, "A_M1_1D", {"mc": done()}, excluded=True)
    put(tmp_series, "A_M1_2D")
    v = view(tmp_series)
    row = next(r for r in v.rows if r.sid == "A_M1_1D")
    assert row.excluded and all(c.level == "excluded" for c in row.cells.values())
    assert v.bars["mc"].total == 1 and v.bars["mc"].done == 0 and v.bars["mc"].pending == 1
    # only the other session is estimated: the excluded one has pending stages too
    assert v.eta_s == pytest.approx(sum(defaults()[s] for s in PENDING_STAGES) * 1000)
    registry.update_record(tmp_series, "A_M1_2D", lambda r: setattr(r, "excluded", True))
    assert view(tmp_series).eta_s is None


def test_bars_count_sessions_per_stage(tmp_series):
    put(tmp_series, "A_M1_1D", {"crop": done(), "mc": done()})
    put(tmp_series, "A_M1_2D", {"crop": done(), "mc": done(), "thresholds": done({"suggested": {"gSig": 4}}),
                                "cnmf": waiting("thresholds")})
    put(tmp_series, "A_M1_3D", {"crop": done(), "mc": StageRecord(State.RUNNING)})
    put(tmp_series, "A_M1_4D", {"crop": done(), "cnmf": StageRecord(State.FAILED, error="x"),
                                "gate": waiting("manual_inspection")})
    b = view(tmp_series).bars
    assert (b["crop"].done, b["crop"].total) == (4, 4)
    assert (b["mc"].done, b["mc"].running, b["mc"].pending) == (2, 1, 1)
    assert b["thresholds"].waiting == 1
    assert b["gate"].waiting == 1
    assert b["cnmf"].failed == 1 and b["cnmf"].pending == 3


def test_waiting_counts(tmp_series):
    put(tmp_series, "A_M1_1D", {"cnmf": waiting("thresholds")})
    # run --until thresholds: a suggestion exists, CNMF-E is still PENDING and no thresholds are set
    put(tmp_series, "A_M1_2D", {"thresholds": done({"suggested": {"gSig": 4, "min_corr": .8, "min_pnr": 5}})})
    put(tmp_series, "A_M1_3D", {"cnmf": done(), "gate": waiting("manual_inspection")})
    put(tmp_series, "A_M1_4D", {"mc": waiting("crop_finalize")})
    v = view(tmp_series)
    assert v.waiting == {"thresholds": 2, "manual_inspection": 1}
    row = next(r for r in v.rows if r.sid == "A_M1_4D")
    assert row.cells["mc"].level == "waiting"


def test_eta_from_history_and_defaults(tmp_series):
    d = defaults()
    # two DONE mc runs of 0.1 and 0.3 s/frame: median 0.2 for pending mc; other stages have no history
    put(tmp_series, "A_M1_1D", {"mc": done(seconds=100)})
    put(tmp_series, "A_M1_2D", {"mc": done(seconds=300)})
    # FAILED and SKIPPED stages are not estimated; the stages after them are pending
    put(tmp_series, "A_M1_3D", {"mc": StageRecord(State.FAILED, error="x"), "crop": StageRecord(State.SKIPPED),
                                "thresholds": StageRecord(State.FAILED, error="x")})
    put(tmp_series, "A_M1_4D")
    rest = sum(d[s] for s in PENDING_STAGES if s != "mc") * 1000
    expected = (rest  # session 1: everything but mc
                + rest  # session 2
                + sum(d[s] for s in ("cnmf", "autoinspect", "export")) * 1000  # session 3
                + 0.2 * 1000 + rest)  # session 4: pending mc at the median
    assert view(tmp_series).eta_s == pytest.approx(expected)


def test_eta_excludes_human_waits(tmp_series):
    d = defaults()
    put(tmp_series, "A_M1_1D", {"crop": done(), "mc": done(), "thresholds": done({"suggested": {}}),
                                "cnmf": waiting("thresholds")})
    assert view(tmp_series).eta_s is None  # nothing but the person's wait
    put(tmp_series, "A_M1_2D", {"crop": done(), "mc": done(), "thresholds": done(),
                                "cnmf": StageRecord(State.RUNNING, started=iso(NOW))})
    got = view(tmp_series).eta_s
    assert got == pytest.approx(sum(d[s] for s in ("cnmf", "autoinspect", "export")) * 1000)


def test_eta_running_stage_minus_elapsed(tmp_series):
    d = defaults()
    put(tmp_series, "A_M1_1D", {"crop": done(),
                                "mc": StageRecord(State.RUNNING, started=iso(NOW - timedelta(seconds=10)))})
    rest = sum(d[s] for s in ("thresholds", "cnmf", "autoinspect", "export")) * 1000
    assert view(tmp_series).eta_s == pytest.approx(d["mc"] * 1000 - 10 + rest)


def test_warning_panel_has_only_six_kinds(tmp_series):
    put(tmp_series, "A_M1_1D", {
        "mc": done({}, [check("mc", "mc_template_corr", "warning")]),
        "cnmf": done({}, [check("cnmf", "cnmf_missed_cells", "warning")]),
        "export": done({}, [check("export", "export_median_snr", "warning")])},
        anomalies=[Anomaly("no_metadata", "warning", "x"), Anomaly("broken_last_frame", "warning", "y"),
                   Anomaly("old_merge_grouping", "warning", "z", data={"stage": "export"})])
    v = view(tmp_series)
    assert v.warnings == [] and v.n_warning_sessions == 0
    cells = v.rows[0].cells
    assert cells["mc"].level == cells["cnmf"].level == cells["export"].level == "warning"


def test_warning_panel_six_kinds_present(tmp_series):
    put(tmp_series, "A_M1_1D", {"cnmf": StageRecord(State.FAILED, error="boom")})
    put(tmp_series, "A_M1_2D", anomalies=[Anomaly("fps_outlier", "warning", "fps 30 vs 20")])
    put(tmp_series, "A_M1_3D", {
        "mc": done({}, [check("mc", "mc_max_shift_px", "warning", "max shift 18.0 px (warn > 10, error > 30)")]),
        "cnmf": done({}, [check("cnmf", "cnmf_n_components_vs_median", "error")]),
        "autoinspect": done({}, [check("autoinspect", "auto_deleted_frac", "warning")]),
        "export": done({}, [check("export", "export_n_neurons", "warning")])})
    w = view(tmp_series).warnings
    assert {x.kind for x in w} == {"stage_failed", "scan_data", "mc_shift", "cnmf_count",
                                   "autoinspect_deleted", "export_count"}
    assert {x.kind for x in w if x.retry} == {"stage_failed"}
    mc = next(x for x in w if x.kind == "mc_shift")
    assert "max shift 18.0 px" in mc.text and "Check the MC preview." in mc.text and mc.stage == "mc"
    assert next(x for x in w if x.kind == "scan_data").stage == ""
    assert "Other sessions continue." in next(x for x in w if x.kind == "stage_failed").text


def test_warning_panel_errors_first(tmp_series):
    put(tmp_series, "A_M1_1D", {"mc": done({}, [check("mc", "mc_max_shift_px", "warning")])})
    put(tmp_series, "A_M1_2D", {"cnmf": StageRecord(State.FAILED, error="boom")})
    v = view(tmp_series)
    assert [(w.level, w.sid) for w in v.warnings] == [("error", "A_M1_2D"), ("warning", "A_M1_1D")]
    assert v.n_warning_sessions == 2


def test_warning_panel_stall_and_crash_text(tmp_series):
    def crash(cause):
        return Anomaly("stage_crashed", "error", "x", data={"stage": "cnmf", "cause": cause})

    put(tmp_series, "A_M1_1D", {"cnmf": StageRecord(State.FAILED, error="killed")}, anomalies=[crash("no_progress")])
    put(tmp_series, "A_M1_2D", {"cnmf": StageRecord(State.FAILED, error="killed")},
        anomalies=[crash("pool_worker_died")])
    put(tmp_series, "A_M1_3D", {"cnmf": StageRecord(State.FAILED, error="a\nlast line")}, anomalies=[crash(None)])
    t = {w.sid: w.text for w in view(tmp_series).warnings}
    assert "stalled: no CPU progress" in t["A_M1_1D"]
    assert "a CaImAn worker died" in t["A_M1_2D"]
    assert "last line" in t["A_M1_3D"]


def test_warning_auto_fixed_mismatch_hidden(tmp_series):
    put(tmp_series, "A_M1_1D", anomalies=[Anomaly("frames_ts_mismatch", "warning", "fixed", auto_fixed=True)])
    put(tmp_series, "A_M1_2D", anomalies=[Anomaly("frames_ts_mismatch", "warning", "open")], excluded=True)
    w = view(tmp_series).warnings
    assert [(x.sid, x.kind) for x in w] == [("A_M1_2D", "scan_data")]


def test_failed_session_does_not_hide_others(tmp_series):
    put(tmp_series, "A_M1_1D", {"crop": done(), "mc": done(), "cnmf": StageRecord(State.RUNNING)})
    put(tmp_series, "A_M1_2D", {"crop": done(), "mc": done(), "cnmf": StageRecord(State.FAILED, error="boom")})
    put(tmp_series, "A_M1_3D", {"crop": done(), "mc": done(), "cnmf": done()})
    v = view(tmp_series)
    cells = {r.sid: r.cells["cnmf"].level for r in v.rows}
    assert cells == {"A_M1_1D": "running", "A_M1_2D": "error", "A_M1_3D": "ok"}
    assert len(v.warnings) == 1 and v.warnings[0].sid == "A_M1_2D" and v.n_warning_sessions == 1


def test_unreadable_registry_file_row(tmp_series):
    put(tmp_series, "A_M1_1D", {"mc": done()})
    bad = tmp_series.registry_file("A_M1_2D")
    bad.write_text("{not json", encoding="utf-8")
    v = view(tmp_series)
    assert {r.sid for r in v.rows} == {"A_M1_1D", "A_M1_2D"}
    row = next(r for r in v.rows if r.unreadable)
    assert row.sid == "A_M1_2D" and all(c.level == "error" and c.text == "?" for c in row.cells.values())
    assert f"registry file cannot be read: {bad}" in row.cells["mc"].tooltip
    assert v.unreadable_files == [bad]
    assert v.warnings == []
    assert next(r for r in v.rows if r.sid == "A_M1_1D").cells["mc"].level == "ok"


def test_cache_rereads_only_changed(tmp_series, monkeypatch):
    for i in (1, 2, 3):
        put(tmp_series, f"A_M1_{i}D", {"thresholds": done({"suggested": {"gSig": 4}})})
    reads, resolves = [], []
    real_read, real_resolve = registry.read_file, config.resolve
    monkeypatch.setattr(registry, "read_file", lambda f: (reads.append(f), real_read(f))[1])
    monkeypatch.setattr(config, "resolve", lambda *a, **k: (resolves.append(1), real_resolve(*a, **k))[1])
    cache = monitor.RecordCache()
    view(tmp_series, cache)
    assert len(reads) == 3 and len(resolves) == 3
    reads.clear(), resolves.clear()
    view(tmp_series, cache)
    assert reads == [] and resolves == []
    registry.update_record(tmp_series, "A_M1_2D", lambda r: setattr(r, "n_frames", 5))
    view(tmp_series, cache)
    assert len(reads) == 1


def test_rows_sorted_by_mouse_then_sid(tmp_series):
    put(tmp_series, "A_M2_1D", mouse="M2")
    put(tmp_series, "B_M1_2D", mouse="M1")
    put(tmp_series, "C_M1_1D", mouse="M1")
    assert [r.sid for r in view(tmp_series).rows] == ["B_M1_2D", "C_M1_1D", "A_M2_1D"]


def test_not_scanned_series(tmp_path):
    v = monitor.series_view(SeriesPaths(tmp_path / "nothing"), NOW)
    assert v.scanned is False and v.rows == [] and v.runner.status == "stopped"


def test_thresholds_cache_keyed_on_session_not_object_identity(tmp_series, monkeypatch):
    rec = put(tmp_series, "A_M1_1D", {"thresholds": done()})
    calls = []
    real = config.resolve
    monkeypatch.setattr(config, "resolve", lambda *a, **k: calls.append(1) or real(*a, **k))
    cache = monitor.RecordCache()
    cache.gsig(tmp_series, rec)
    clone = SessionRecord(**{**rec.__dict__})  # another object for the same session, as after a reread
    cache.gsig(tmp_series, clone)
    assert len(calls) == 1  # an id()-keyed cache would resolve again, and could also serve a reused id's value
    tmp_series.session_config("A_M1_1D").parent.mkdir(parents=True, exist_ok=True)
    tmp_series.session_config("A_M1_1D").write_text("cnmf:\n  gSig: 7\n", encoding="utf-8")
    assert cache.gsig(tmp_series, clone) == 7 and len(calls) == 2
