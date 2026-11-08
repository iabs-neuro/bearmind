import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import psutil
import pytest

from bearmind import actions, config, monitor, registry, runner
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, Anomaly, Check, SessionRecord, StageRecord, State
from tests.bearmind.current_stages import make_current, mark_current

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


def put(paths, sid, stages=None, mouse=None, excluded=False, anomalies=(), n_frames=1000, current=True):
    """A record; its DONE main stages are up to date by the runner's rule unless current=False (then the batch
    would redo them, and the window says they are out of date)."""
    st = {s: StageRecord() for s in STAGES}
    st.update(stages or {})
    rec = SessionRecord(sid=sid, mouse=mouse or sid.split("_")[1], mode="bonsai", parts=[], n_frames=n_frames,
                        fps=20.0, fps_info={}, anomalies=list(anomalies), decisions=[], excluded=excluded,
                        stages=st)
    if current:
        mark_current(paths, rec)
    registry.save_record(paths, rec)
    return rec


def waiting(why):
    return StageRecord(State.WAITING, error=f"waiting for {why}")


def check(stage, name, level, msg="m"):
    return Check(stage, name, level, 1.0, msg)


def view(paths, cache=None):
    return monitor.series_view(paths, NOW, cache)


def defaults():
    """Seconds of each stage for a 1000-frame session (the crop's number is per session, not per frame)."""
    m = config.load_defaults()["monitor"]
    return {**m["eta_s_per_frame"], **{k: v / 1000 for k, v in m["eta_s_per_session"].items()}}


def spawn(paths, sid, stage, status):
    p = subprocess.Popen([sys.executable, str(FAKE), str(paths.root), sid, stage, status],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert p.stdout.readline().strip() == "ready"
    return p


def stop(p):
    p.stdin.close()
    p.wait(timeout=20)


@pytest.fixture
def alive(tmp_series, monkeypatch):
    """A live batch for the model: a fresh pulse and a lock holder (a stage RUNNING without one is interrupted)."""
    def pulse(status="running", **fields):
        hb = {"pid": 1, "time": iso(NOW), "current": None, "status": status, **fields}
        tmp_series.heartbeat.parent.mkdir(parents=True, exist_ok=True)
        tmp_series.heartbeat.write_text(json.dumps(hb), encoding="utf-8")

    monkeypatch.setattr(monitor, "_holds_lock", lambda lock: 1)
    pulse()
    return pulse


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
        "crop": done(), "mc": done({"max_shift_px": 18.0}, [check("mc", "mc_max_shift_px", "warning")]),
        "thresholds": done(), "cnmf": waiting("thresholds")})
    put(tmp_series, "A_M1_2D", {"cnmf": StageRecord(State.FAILED, error="Traceback\nMemoryError: out")})
    rows = {r.sid: r for r in view(tmp_series).rows}
    mc = rows["A_M1_1D"].cells["mc"]
    assert (mc.level, mc.text) == ("warning", "18.0 px")
    assert rows["A_M1_1D"].cells["cnmf"].text == "waits"
    f = rows["A_M1_2D"].cells["cnmf"]
    assert (f.level, f.text) == ("error", "failed") and "MemoryError: out" in f.tooltip
    assert rows["A_M1_2D"].cells["export"].level == "none"


def test_cell_texts_match_mockup(tmp_series):
    config.update_yaml(tmp_series.series_config, {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    put(tmp_series, "A_M1_1D", {
        "crop": done(),
        "mc": done({"max_shift_px": 3.2, "crop": [0, 0, 480, 512]}),
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
    assert view(tmp_series).eta_s == 0.0  # nothing left


def test_bars_count_sessions_per_stage(tmp_series, alive):
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
    put(tmp_series, "A_M1_1D", {"crop": done(), "mc": done(), "thresholds": done(), "cnmf": waiting("thresholds")})
    # run --until thresholds: a suggestion exists, CNMF-E is still PENDING and no thresholds are set
    put(tmp_series, "A_M1_2D", {"crop": done(), "mc": done(),
                                "thresholds": done({"suggested": {"gSig": 4, "min_corr": .8, "min_pnr": 5}})})
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
    assert view(tmp_series).eta_s == 0.0  # nothing but the person's wait: nothing left for the batch
    # CNMF-E runs with the thresholds of the config, so nobody is asked for them
    config.update_yaml(tmp_series.session_config("A_M1_2D"), {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}})
    put(tmp_series, "A_M1_2D", {"crop": done(), "mc": done(), "thresholds": done(),
                                "cnmf": StageRecord(State.RUNNING, started=iso(NOW))})
    got = view(tmp_series).eta_s
    assert got == pytest.approx(sum(d[s] for s in ("cnmf", "autoinspect", "export")) * 1000)


def test_eta_running_stage_minus_elapsed(tmp_series, alive):
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
                   Anomaly("old_merge_grouping", "warning", "z", data={"stage": "export"}),
                   Anomaly("cnmf_downsampled", "warning", "w", data={"stage": "cnmf"})])
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


def test_failed_session_does_not_hide_others(tmp_series, alive):
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
    # one per session: the refresh's runner view, the thresholds and the cells share the cached config
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
    assert v.found_nothing is False


def test_scan_found_no_sessions(tmp_path):
    """M7: the scan ran (its list of unrecognized folders exists) but recognized no session."""
    paths = SeriesPaths(tmp_path / "s")
    paths.scan_unrecognized_csv.parent.mkdir(parents=True)
    paths.scan_unrecognized_csv.write_text("path,reason\n1_Raw/a,folder name is not a session id\n",
                                           encoding="utf-8")
    v = monitor.series_view(paths, NOW)
    assert v.scanned is False and v.found_nothing is True and v.rows == []


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


# --- final fix wave ---

def _break_session_config(paths, sid):
    f = paths.session_config(sid)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("cnmf: [unclosed\n", encoding="utf-8")


def test_unschedulable_session_is_a_red_row_when_the_batch_is_stopped(tmp_series, monkeypatch):
    """I1: computed as the runner does when no runner is alive; red row, reason in every tooltip, out of the
    ETA and out of pending."""
    put(tmp_series, "A_M1_1D")
    put(tmp_series, "A_M1_2D", {"crop": done({"box": [0, 0, 480, 512]})})
    _break_session_config(tmp_series, "A_M1_2D")
    calls, resolves = [], []
    real, real_resolve = runner.schedule_problems, config.resolve
    monkeypatch.setattr(runner, "schedule_problems", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    monkeypatch.setattr(config, "resolve", lambda *a, **k: (resolves.append(1), real_resolve(*a, **k))[1])
    cache = monitor.RecordCache()
    v = view(tmp_series, cache)
    row = next(r for r in v.rows if r.sid == "A_M1_2D")
    assert row.problem.startswith("config: ")
    assert all(c.level == "error" and "cannot be scheduled: config: " in c.tooltip for c in row.cells.values())
    assert row.cells["crop"].text == "512×480"  # what is done stays readable
    ok = next(r for r in v.rows if r.sid == "A_M1_1D")
    assert ok.problem is None and ok.cells["crop"].level == "none"
    assert v.eta_s == pytest.approx(sum(defaults()[s] for s in PENDING_STAGES) * 1000)  # the healthy one only
    assert (v.bars["crop"].done, v.bars["crop"].failed, v.bars["crop"].pending) == (1, 0, 1)
    assert (v.bars["mc"].failed, v.bars["mc"].pending) == (1, 1)
    assert v.warnings == []  # no seventh panel kind
    assert len(resolves) == 2  # one per session, the broken one included: its error is cached too
    view(tmp_series, cache)
    # one runner view per refresh, never one per session; the configs come from the cache until a file changes
    assert len(calls) == 2 and len(resolves) == 2
    tmp_series.session_config("A_M1_2D").write_text("cnmf:\n  gSig: 4\n", encoding="utf-8")
    assert next(r for r in view(tmp_series, cache).rows if r.sid == "A_M1_2D").problem is None


def test_session_without_fps_cannot_be_scheduled(tmp_series):
    rec = put(tmp_series, "A_M1_1D")
    registry.update_record(tmp_series, rec.sid, lambda r: setattr(r, "fps", None))
    row = view(tmp_series).rows[0]
    assert "fps_unavailable" in row.problem and row.cells["mc"].level == "error"


def test_unschedulable_from_the_live_runner_heartbeat(tmp_series, alive, monkeypatch):
    put(tmp_series, "A_M1_1D")
    put(tmp_series, "A_M1_2D")
    alive(status="waiting", unschedulable={"A_M1_2D": "stage parameters: KeyError: 'mc'",
                                           "BROKEN_X9_1D": "broken registry file"})
    monkeypatch.setattr(runner, "schedule_problems", lambda *a, **k: pytest.fail("the live runner has the list"))
    v = view(tmp_series)
    assert v.runner.alive and v.runner.unschedulable["A_M1_2D"].startswith("stage parameters")
    assert {r.sid: r.problem for r in v.rows} == {"A_M1_1D": None, "A_M1_2D": "stage parameters: KeyError: 'mc'"}


def test_crop_cell_is_width_by_height_before_and_after_mc(tmp_series):
    """M1: the crop stage and MC both record [top, left, height, width]; the cell is width×height."""
    put(tmp_series, "A_M1_1D", {"crop": done({"box": [10, 20, 480, 512]})})
    put(tmp_series, "A_M1_2D", {"crop": done({"box": [10, 20, 480, 512]}),
                                "mc": done({"crop": [10, 20, 480, 512]})})
    put(tmp_series, "A_M1_3D", {"crop": done({"box": None}), "mc": done({"crop": None})})
    t = {r.sid: r.cells["crop"].text for r in view(tmp_series).rows}
    assert t == {"A_M1_1D": "512×480", "A_M1_2D": "512×480", "A_M1_3D": "uncropped"}


def test_transient_read_error_is_retried_not_cached(tmp_series, monkeypatch):
    """M2: a PermissionError (the batch replacing the file) is not a broken record."""
    put(tmp_series, "A_M1_1D", {"mc": done()})
    real = registry.read_file
    fail = {"n": 1}

    def flaky(f):
        if fail["n"]:
            fail["n"] -= 1
            raise PermissionError(13, "in use")
        return real(f)

    monkeypatch.setattr(registry, "read_file", flaky)
    cache = monitor.RecordCache()
    assert view(tmp_series, cache).rows[0].unreadable  # nothing read before: shown as unreadable this once
    row = view(tmp_series, cache).rows[0]
    assert not row.unreadable and row.cells["mc"].level == "ok"
    registry.update_record(tmp_series, "A_M1_1D", lambda r: r.stages.update({"mc": StageRecord(State.FAILED)}))
    fail["n"] = 1
    assert view(tmp_series, cache).rows[0].cells["mc"].level == "ok"  # the record read last meanwhile
    assert view(tmp_series, cache).rows[0].cells["mc"].level == "error"  # read again on the next refresh


def test_parse_error_stays_cached(tmp_series, monkeypatch):
    tmp_series.registry_file("A_M1_1D").write_text("{not json", encoding="utf-8")
    cache = monitor.RecordCache()
    assert view(tmp_series, cache).rows[0].unreadable
    reads = []
    monkeypatch.setattr(registry, "read_file", lambda f: reads.append(f))
    assert view(tmp_series, cache).rows[0].unreadable and reads == []


def test_running_stage_without_a_live_batch_is_interrupted(tmp_series):
    """M6: a RUNNING stage left by a stopped or dead batch runs again from its start."""
    put(tmp_series, "A_M1_1D", {"crop": done(),
                                "mc": StageRecord(State.RUNNING, started=iso(NOW - timedelta(seconds=10)))})
    v = view(tmp_series)
    c = v.rows[0].cells["mc"]
    assert (c.level, c.text) == ("warning", "interrupted") and "interrupted" in c.tooltip
    assert (v.bars["mc"].running, v.bars["mc"].pending) == (0, 1)
    rest = sum(defaults()[s] for s in ("thresholds", "cnmf", "autoinspect", "export")) * 1000
    assert v.eta_s == pytest.approx(defaults()["mc"] * 1000 + rest)  # the whole stage, not minus elapsed


def test_manual_inspection_count_is_the_action_set(tmp_series):
    """M11: the "Manual inspection: N" count and "Inspection not needed for all waiting" use one definition."""
    put(tmp_series, "A_M1_1D", {"cnmf": done(), "gate": waiting("manual_inspection")})
    put(tmp_series, "A_M1_2D", {"cnmf": done()})  # the gate not checked yet: the action takes it
    put(tmp_series, "A_M1_3D", {"mc": waiting("crop_finalize")})
    put(tmp_series, "A_M1_4D", {"cnmf": done(), "gate": done({"gate": "not_needed"})})
    v = view(tmp_series)
    assert v.waiting["manual_inspection"] == len(actions.waiting_gate_sids(tmp_series)) == 2
    assert v.bars["gate"].waiting == 2
    g = next(r for r in v.rows if r.sid == "A_M1_2D").cells["gate"]
    assert (g.level, g.text) == ("waiting", "waits")


def test_lock_holder_start_time_read_once_per_pid(tmp_series, monkeypatch):
    """R2: psutil's create_time can cost tens of ms; a PID's start time does not change."""
    monkeypatch.setattr(monitor, "_LOCK_PROC", {}, raising=False)
    calls = []
    real = psutil.Process.create_time
    monkeypatch.setattr(psutil.Process, "create_time", lambda self: (calls.append(self.pid), real(self))[1])
    p = spawn(tmp_series, "-", "", "running")
    try:
        assert monitor.runner_state(tmp_series).alive
        assert monitor.runner_state(tmp_series).alive
        assert calls == [p.pid]
    finally:
        stop(p)
    assert not monitor.runner_state(tmp_series).alive and p.pid not in monitor._LOCK_PROC
    q = spawn(tmp_series, "-", "", "running")  # a new lock holder replaces the entry
    try:
        assert monitor.runner_state(tmp_series).alive and list(monitor._LOCK_PROC) == [q.pid]
    finally:
        stop(q)


def test_warning_text_does_not_repeat_the_stage(tmp_series):
    """R4: the window labels each warning "<sid> · <stage>"; the text after it starts with what happened."""
    put(tmp_series, "A_M1_1D", {"cnmf": StageRecord(State.FAILED, error="boom")},
        anomalies=[Anomaly("stage_crashed", "error", "x", data={"stage": "cnmf", "cause": "pool_worker_died"})])
    put(tmp_series, "A_M1_2D", {"mc": done({}, [check("mc", "mc_max_shift_px", "warning", "max shift 18 px")])})
    w = {x.sid: x.text for x in view(tmp_series).warnings}
    assert w == {"A_M1_1D": "a CaImAn worker died. Other sessions continue.",
                 "A_M1_2D": "max shift 18 px. Check the MC preview."}


def test_runner_state_exit_reason(tmp_series):
    """I2: the final heartbeat says why the batch stopped; an old one without it is plain stopped."""
    tmp_series.heartbeat.parent.mkdir(parents=True, exist_ok=True)
    for exit_ in ("finished", "stopped", "error: RuntimeError: x"):
        tmp_series.heartbeat.write_text(json.dumps({"time": iso(NOW), "status": "idle", "exit": exit_}),
                                        encoding="utf-8")
        r = monitor.runner_state(tmp_series, NOW)
        assert (r.status, r.exit, r.heartbeat_time) == ("stopped", exit_, NOW)
    tmp_series.heartbeat.write_text(json.dumps({"time": iso(NOW), "status": "idle"}), encoding="utf-8")
    assert monitor.runner_state(tmp_series, NOW).exit is None


def test_runner_state_stopping(tmp_series):
    """M5: a stop request waits for the running stage to end."""
    p = spawn(tmp_series, "A_M1_1D", "mc", "running")
    try:
        assert not monitor.runner_state(tmp_series).stopping
        tmp_series.stop_request.write_text("stop\n", encoding="utf-8")
        assert monitor.runner_state(tmp_series).stopping
    finally:
        stop(p)
    assert not monitor.runner_state(tmp_series).stopping  # nothing stops when nothing runs


def test_stale_lock_is_checked_once(tmp_series, monkeypatch):
    """R2: psutil needs ~60 ms to report a dead PID on Windows; the process a lock names never comes back."""
    monkeypatch.setattr(monitor, "_LOCK_PROC", {})
    monkeypatch.setattr(monitor, "_LOCK_GONE", set(), raising=False)
    p = spawn(tmp_series, "A_M1_1D", "mc", "running")
    p.kill()
    p.wait(timeout=20)
    calls = []
    real = psutil.Process
    monkeypatch.setattr(psutil, "Process", lambda pid: (calls.append(pid), real(pid))[1])
    assert monitor.runner_state(tmp_series).status == "dead"
    assert monitor.runner_state(tmp_series).status == "dead"
    assert calls == [p.pid]
    monkeypatch.setattr(psutil, "Process", real)
    q = spawn(tmp_series, "-", "", "running")  # a new runner writes a new lock: checked again
    try:
        assert monitor.runner_state(tmp_series).alive
    finally:
        stop(q)


# --- preview side stages and the downsampling (wave A2) ---

def _preview_on(paths, on=True):
    config.update_yaml(paths.series_config, {"preview": {"enabled": on}})


def test_side_stage_titles():
    assert monitor.STAGE_TITLE["mc_preview"] == "MC preview"
    assert monitor.STAGE_TITLE["cnmf_preview"] == "CNMF-E preview"


def test_downsampled_anomaly_in_cnmf_cell(tmp_series):
    msg = "background downsampled to fit RAM: ssub_B 3 instead of 2 (predicted 40 GiB at 2, 18 GiB at 3)"
    put(tmp_series, "A_M1_1D", {"cnmf": done({"n_components": 80})},
        anomalies=[Anomaly("cnmf_downsampled", "warning", msg, data={"stage": "cnmf"})])
    put(tmp_series, "A_M1_2D", {"cnmf": done({"n_components": 80})},
        anomalies=[Anomaly("cnmf_downsampled", "error", "predicted 99 GiB even at ssub_B 4",
                           data={"stage": "cnmf"})])
    v = view(tmp_series)
    c1, c2 = (r.cells["cnmf"] for r in v.rows)
    assert c1.level == "warning" and "ssub_B 3 instead of 2" in c1.tooltip
    assert monitor.HINTS["cnmf_downsampled"] == "Background downsampled to fit RAM; a tighter crop avoids it."
    assert monitor.HINTS["cnmf_downsampled"] in c1.tooltip
    assert c2.level == "error" and monitor.HINTS["cnmf_downsampled"] in c2.tooltip
    assert v.warnings == []  # the panel keeps its six kinds


def test_failed_preview_in_panel_with_retry(tmp_series):
    _preview_on(tmp_series)
    put(tmp_series, "A_M1_1D", {"mc": done({"max_shift_px": 2.0}),
                                "mc_preview": StageRecord(State.FAILED, error="Traceback" + chr(10) + "OSError: disk full")})
    v = view(tmp_series)
    [w] = v.warnings
    assert (w.kind, w.stage, w.retry, w.level) == ("stage_failed", "mc_preview", True, "error")
    assert "disk full" in w.text
    mc = v.rows[0].cells["mc"]
    assert mc.level == "ok" and "preview: failed" in mc.tooltip


def test_failed_preview_that_is_off_is_not_in_panel(tmp_series):
    put(tmp_series, "A_M1_1D", {"mc": done(), "mc_preview": StageRecord(State.FAILED, error="x")})
    assert view(tmp_series).warnings == []  # runner.first_failed skips a preview its config turns off


def test_preview_line_in_tooltips(tmp_series):
    put(tmp_series, "A_M1_1D", {"mc": done(), "cnmf": done()})
    assert "preview: off" in view(tmp_series).rows[0].cells["mc"].tooltip
    _preview_on(tmp_series)
    cache = monitor.RecordCache()
    cells = view(tmp_series, cache).rows[0].cells
    assert "preview: pending" in cells["mc"].tooltip and "preview: pending" in cells["cnmf"].tooltip
    registry.update_record(tmp_series, "A_M1_1D", lambda r: r.stages.update(
        mc_preview=StageRecord(State.RUNNING), cnmf_preview=StageRecord(State.DONE)))
    cells = view(tmp_series, cache).rows[0].cells
    assert "preview: running" in cells["mc"].tooltip and "preview: done" in cells["cnmf"].tooltip
    assert len(view(tmp_series).rows[0].cells) == len(STAGES)  # the table keeps its seven columns


def test_side_stages_not_in_bars_or_eta(tmp_series):
    _preview_on(tmp_series)
    put(tmp_series, "A_M1_1D", {s: done() for s in STAGES})
    base = view(tmp_series)
    assert set(base.bars) == set(STAGES) and base.eta_s == 0.0
    registry.update_record(tmp_series, "A_M1_1D", lambda r: r.stages.update(
        mc_preview=StageRecord(State.PENDING), cnmf_preview=StageRecord(State.RUNNING)))
    v = view(tmp_series)
    assert set(v.bars) == set(STAGES) and v.eta_s == 0.0


def test_runner_state_reports_a_current_side_stage(tmp_series, alive):
    put(tmp_series, "A_M1_1D", {"cnmf": done(), "cnmf_preview": StageRecord(State.RUNNING, started=iso(NOW))})
    alive(current={"sid": "A_M1_1D", "stage": "cnmf_preview"})
    r = view(tmp_series).runner
    assert r.current == ("A_M1_1D", "cnmf_preview")


def test_crop_eta_is_per_session_not_per_frame(tmp_series):
    m = config.load_defaults()["monitor"]
    # a short and a long session cost the same default crop time
    put(tmp_series, "A_M1_1D", {s: done() for s in STAGES if s != "crop"}, n_frames=600)
    short = view(tmp_series).eta_s
    put(tmp_series, "A_M1_1D", {s: done() for s in STAGES if s != "crop"}, n_frames=36000)
    assert short == pytest.approx(m["eta_s_per_session"]["crop"]) == pytest.approx(90, rel=0.2)
    assert view(tmp_series).eta_s == pytest.approx(short)


def test_crop_eta_from_history_is_median_duration(tmp_series):
    put(tmp_series, "A_M1_1D", {s: done() for s in STAGES if s != "crop"} | {"crop": done(seconds=40)},
        n_frames=600)
    put(tmp_series, "A_M1_2D", {s: done() for s in STAGES if s != "crop"} | {"crop": done(seconds=60)},
        n_frames=36000)
    put(tmp_series, "A_M1_3D", {s: done() for s in STAGES if s != "crop"}, n_frames=18000)
    assert view(tmp_series).eta_s == pytest.approx(50.0)


# --- phase 4 final fix wave: the runner's view in Monitor ---

THRESH = {"cnmf": {"gSig": 4, "min_corr": 0.8, "min_pnr": 8.0}}


def test_thresholds_wait_ends_as_soon_as_thresholds_are_written(tmp_series):
    """A4: the count is actions.waits_for_thresholds on the refresh's view. It drops as soon as the thresholds are
    in the config, though the registry says WAITING until the batch's next check (up to poll_s later)."""
    put(tmp_series, "A_M1_1D", {"crop": done(), "mc": done(), "thresholds": done(), "cnmf": waiting("thresholds")})
    cache = monitor.RecordCache()
    v = view(tmp_series, cache)
    assert v.waiting["thresholds"] == 1 and v.rows[0].cells["thresholds"].text == "waits"
    assert v.bars["thresholds"].waiting == 1
    actions.set_thresholds(tmp_series, "A_M1_1D", 4, 0.8, 8.0)
    v = view(tmp_series, cache)
    assert v.waiting["thresholds"] == 0 and v.bars["thresholds"].waiting == 0
    row = v.rows[0]
    assert row.cells["thresholds"].text == "gSig 4" and row.cells["thresholds"].level == "ok"
    cnmf = row.cells["cnmf"]
    assert (cnmf.level, cnmf.text) == ("none", "–") and monitor.NO_LONGER_WAITS in cnmf.tooltip


def test_out_of_date_stage_in_cell_bar_and_eta(tmp_series):
    """A4: a DONE stage the batch would redo says so in its cell (text and tooltip, the old number kept); it
    counts as pending in the bar and in the time left; the warnings panel keeps its six kinds."""
    config.update_yaml(tmp_series.session_config("A_M1_1D"), THRESH)
    put(tmp_series, "A_M1_1D", {"crop": done(), "mc": done({"max_shift_px": 3.2}), "thresholds": done()})
    assert view(tmp_series).rows[0].cells["mc"].text == "3.2 px"
    config.update_yaml(tmp_series.mouse_config("M1"), {"mc": {"max_shifts": [20, 20]}})  # an MC input changed
    v = view(tmp_series)
    cells = v.rows[0].cells
    assert cells["crop"].level == "ok" and cells["crop"].text == "done"
    for stage in ("mc", "thresholds"):
        assert (cells[stage].level, cells[stage].text) == ("none", "out of date")
        assert "out of date: the batch redoes mc first" in cells[stage].tooltip
    assert "old result: 3.2 px" in cells["mc"].tooltip
    assert (v.bars["mc"].done, v.bars["mc"].pending) == (0, 1) and v.bars["crop"].done == 1
    d = defaults()
    assert v.eta_s == pytest.approx(sum(d[s] for s in PENDING_STAGES if s != "crop") * 1000)
    assert v.waiting["thresholds"] == 0 and v.warnings == []


def test_out_of_date_maps_do_not_wait_for_thresholds(tmp_series):
    """A4: maps DONE but out of date and no thresholds: the batch redoes the earlier stage first, so nobody is
    asked for thresholds yet (the same answer as the Thresholds tab and actions)."""
    put(tmp_series, "A_M1_1D", {"crop": done(), "mc": done(), "thresholds": done({"suggested": {"gSig": 4}}),
                                "cnmf": waiting("thresholds")})
    config.update_yaml(tmp_series.mouse_config("M1"), {"crop": {"fov_threshold_frac": 0.3}})
    v = view(tmp_series)
    assert v.waiting["thresholds"] == 0
    assert v.rows[0].cells["thresholds"].text == "out of date"
    assert actions.waiting_threshold_sids(tmp_series) == []
