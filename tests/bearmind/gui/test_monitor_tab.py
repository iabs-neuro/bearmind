import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QLabel

from bearmind import monitor, registry
from bearmind.gui import monitor_tab
from bearmind.gui.monitor_tab import MonitorTab
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES, SessionRecord, StageRecord, State

FAKE = Path(__file__).resolve().parents[1] / "fake_runner.py"
REPO = Path(__file__).resolve().parents[3]


def put(paths, sid, stages=None, mouse=None, n_frames=1000):
    st = {s: StageRecord() for s in STAGES}
    st.update(stages or {})
    rec = SessionRecord(sid=sid, mouse=mouse or sid.split("_")[1], mode="bonsai", parts=[], n_frames=n_frames,
                        fps=20.0, fps_info={}, anomalies=[], decisions=[], excluded=False, stages=st)
    registry.save_record(paths, rec)
    return rec


def done(**metrics):
    return StageRecord(State.DONE, metrics=metrics)


def spawn(paths, sid, stage, status):
    p = subprocess.Popen([sys.executable, str(FAKE), str(paths.root), sid, stage, status],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "PYTHONPATH": str(REPO)})
    assert p.stdout.readline().strip() == "ready"
    return p


def stop(p):
    p.stdin.close()
    p.wait(timeout=20)


def never(series):  # a start that produces no heartbeat
    never.calls.append(series)
    return 1


never.calls = []


@pytest.fixture
def series4(tmp_path):
    paths = SeriesPaths(tmp_path / "series")
    paths.registry_file("x").parent.mkdir(parents=True)
    base = {"crop": done(box="512x480@0,0"), "mc": done(max_shift_px=3.2)}
    wait = {**base, "thresholds": done(), "cnmf": StageRecord(State.WAITING, error="waiting for thresholds")}
    put(paths, "LT1_J01_1D", wait)
    put(paths, "LT1_J01_2D", wait)
    put(paths, "LT1_J02_1D", {**base, "mc": StageRecord(State.RUNNING)})
    put(paths, "LT1_J02_2D", {"crop": base["crop"]})
    return paths


def tab_for(paths, **kw):
    return MonitorTab(paths.root, **kw)


@pytest.fixture
def live(series4, monkeypatch):
    """A live batch: a fresh pulse and a lock holder (a stage RUNNING without one shows as interrupted)."""
    hb = {"pid": 1, "time": datetime.now().isoformat(timespec="milliseconds"), "current": None,
          "status": "running"}
    series4.heartbeat.parent.mkdir(parents=True, exist_ok=True)
    series4.heartbeat.write_text(json.dumps(hb), encoding="utf-8")
    monkeypatch.setattr(monitor, "_holds_lock", lambda lock: 1)


def test_batch_header_states(qapp, series4):
    t = tab_for(series4)
    assert t.batch_label.text() == "Stopped"
    p = spawn(series4, "LT1_J02_1D", "mc", "running")
    try:
        t.refresh()
        assert t.batch_label.text() == "Running"
        assert "LT1_J02_1D" in t.batch_detail.text() and "MC" in t.batch_detail.text()
    finally:
        stop(p)
    p = spawn(series4, "LT1_J02_1D", "mc", "running")
    p.kill()
    p.wait(timeout=20)
    t.refresh()
    assert t.batch_label.text() == "Not responding"
    assert "last heartbeat" in t.batch_detail.text()


def test_header_heartbeat_age(qapp, series4):
    t = tab_for(series4)
    p = spawn(series4, "LT1_J02_1D", "mc", "running")
    try:
        hb = datetime.fromisoformat(json.loads(series4.heartbeat.read_text())["time"])
        t.refresh(now=hb + timedelta(seconds=4))
        assert "heartbeat 4 s ago" in t.batch_detail.text()
    finally:
        stop(p)


def test_start_disabled_while_alive(qapp, series4):
    t = tab_for(series4)
    p = spawn(series4, "LT1_J02_1D", "mc", "running")
    try:
        t.refresh()
        assert not t.start_button.isEnabled() and t.stop_button.isEnabled()
    finally:
        stop(p)


def test_start_enabled_when_runner_dead(qapp, series4):
    t = tab_for(series4)
    assert t.start_button.isEnabled() and not t.stop_button.isEnabled()
    p = spawn(series4, "LT1_J02_1D", "mc", "running")
    p.kill()
    p.wait(timeout=20)
    t.refresh()
    assert t.start_button.isEnabled() and not t.stop_button.isEnabled()


def test_start_calls_launcher(qapp, series4):
    never.calls = []
    t = tab_for(series4, start=never)
    t.start_button.click()
    t.start_button.click()  # disabled: no second process
    assert never.calls == [series4.root]


def test_start_refused_shows_message(qapp, series4):
    def refuse(series):
        raise RuntimeError("batch already running: PID 7")

    seen = []
    t = tab_for(series4, start=refuse)
    t.status_message.connect(seen.append)
    t.start_button.click()
    assert "already running" in seen[0]
    assert t.start_button.isEnabled()


def test_start_stays_disabled_until_alive_or_grace(qapp, series4):
    never.calls = []
    seen = []
    t = tab_for(series4, start=never)
    t.status_message.connect(seen.append)
    t.start_button.click()
    t.refresh()
    assert not t.start_button.isEnabled() and not seen
    t._clock = lambda: time.monotonic() + monitor_tab.START_GRACE_S + 1
    t.refresh()
    assert t.start_button.isEnabled()
    assert seen == [monitor_tab.DID_NOT_START] and "runner_console.log" in seen[0]


def test_start_wait_ends_when_runner_alive(qapp, series4):
    never.calls = []
    t = tab_for(series4, start=never)
    t.start_button.click()
    p = spawn(series4, "-", "", "waiting")
    try:
        t.refresh()
        assert t._starting_since is None and not t.start_button.isEnabled() and t.stop_button.isEnabled()
    finally:
        stop(p)
    t.refresh()
    assert t.start_button.isEnabled()


def test_stop_disabled_before_alive_and_ignored(qapp, series4):
    t = tab_for(series4, stop=lambda paths: pytest.fail("stop must not run"))
    assert not t.stop_button.isEnabled()
    t._on_stop()
    assert not series4.stop_request.exists()


def test_stop_writes_request(qapp, series4):
    t = tab_for(series4)
    p = spawn(series4, "LT1_J02_1D", "mc", "running")
    try:
        t.refresh()
        t.stop_button.click()
        assert series4.stop_request.exists()
    finally:
        stop(p)


def test_progress_bars_counts_text(qapp, series4, live):
    t = tab_for(series4)
    assert t.bars["mc"].caption.text() == "2 done · 1 running"
    assert "wait" in t.bars["thresholds"].caption.text()
    assert t.bars["crop"].caption.text() == "4 done"
    assert t.progress_title.text() == "Progress — 4 sessions"  # drawn in capitals by the section font


def test_eta_label_mentions_waiting_not_counted(qapp, series4):
    t = tab_for(series4)
    assert t.eta_label.text().startswith("Estimated time left: ~")
    assert "time waiting for you not counted" in t.eta_label.text()
    assert monitor_tab._eta_text(None) == "Estimated time left: unknown"
    assert "~3 h 10 min" in monitor_tab._eta_text(3 * 3600 + 10 * 60)


def test_waiting_buttons_counts_and_signal(qapp, series4):
    t = tab_for(series4)
    seen = []
    t.open_tab.connect(seen.append)
    assert t.thresholds_button.text() == "Thresholds: 2 sessions →"
    assert t.inspection_button.text() == "Manual inspection: 0 sessions →"
    assert t.thresholds_button.isEnabled() and not t.inspection_button.isEnabled()
    t.thresholds_button.click()
    t.inspection_button.click()  # disabled: no signal
    assert seen == ["Thresholds"]


def test_table_flat_sorted(qapp, series4):
    t = tab_for(series4)
    assert t.model.rowCount() == 4
    assert [t.model.item(i, 0).text() for i in range(4)] == ["LT1_J01_1D", "LT1_J01_2D", "LT1_J02_1D", "LT1_J02_2D"]
    heads = [t.model.headerData(c, Qt.Orientation.Horizontal) for c in range(t.model.columnCount())]
    assert heads == ["Session", "Crop", "MC", "Thresholds", "CNMF-E", "Inspection", "Autoinspect", "Export"]


def test_cell_color_text_tooltip(qapp, series4, live):
    t = tab_for(series4)
    view = monitor.series_view(series4)
    for i, row in enumerate(view.rows):
        for c, stage in enumerate(STAGES, start=1):
            cell, item = row.cells[stage], t.model.item(i, c)
            assert item.text() == cell.text and item.toolTip() == cell.tooltip
            want = monitor_tab.CELL_BG.get(cell.level)
            if want is None:
                assert item.background().style() == Qt.BrushStyle.NoBrush
            else:
                assert item.background().color() == QColor(want)
    assert t.model.item(2, 2).background().color() == QColor(monitor_tab.CELL_BG["running"])


def test_unreadable_row_red_question_mark(qapp, series4):
    series4.registry_file("BROKEN_J09_1D").write_text("{not json", encoding="utf-8")
    t = tab_for(series4)
    rows = {t.model.item(i, 0).text(): i for i in range(t.model.rowCount())}
    item = t.model.item(rows["BROKEN_J09_1D"], 1)
    assert item.text() == "?" and item.background().color() == QColor(monitor_tab.CELL_BG["error"])


def test_refresh_keeps_selection(qapp, series4):
    t = tab_for(series4)
    t.table.selectRow(1)
    assert t.selected_sids() == ["LT1_J01_2D"]
    put(series4, "LT1_J01_1D", {"crop": done(box="512x480@0,0")})
    t.refresh()
    assert t.selected_sids() == ["LT1_J01_2D"]
    assert t.model.item(0, 2).text() == "–"  # the changed record is shown


def test_refresh_keeps_selection_when_rows_change(qapp, series4):
    t = tab_for(series4)
    t.table.selectRow(3)
    put(series4, "LT1_J00_1D")  # a new first row: the table is rebuilt
    t.refresh()
    assert t.model.rowCount() == 5 and t.selected_sids() == ["LT1_J02_2D"]


def test_not_scanned_message(qapp, tmp_path):
    t = MonitorTab(tmp_path)
    t.show()
    assert t.not_scanned_label.isVisible() and not t.table.isVisible()
    assert t.not_scanned_label.text().startswith("Series not scanned. Run: run_bearmind.bat scan")
    assert t.batch_label.text() == "Stopped"
    t.close()


def test_slow_refresh_logged_once(qapp, series4, caplog, monkeypatch):
    monkeypatch.setattr(monitor_tab, "SLOW_REFRESH_S", -1.0)
    with caplog.at_level(logging.WARNING, logger="bearmind"):
        t = tab_for(series4)
        t.refresh()
        t.refresh()
    assert len([r for r in caplog.records if "monitor refresh" in r.message]) == 1


def test_refresh_fast_on_300_sessions(qapp, tmp_path):
    paths = SeriesPaths(tmp_path / "series")
    paths.registry_file("x").parent.mkdir(parents=True)
    full = {s: done() for s in ("crop", "mc", "thresholds")}
    for i in range(300):
        put(paths, f"LT1_J{i // 3:03d}_{i % 3 + 1}D", full)
    t = MonitorTab(paths.root)  # the first refresh fills the caches
    t0 = time.perf_counter()
    t.refresh()
    spent = time.perf_counter() - t0
    print(f"refresh of 300 sessions: {spent * 1000:.0f} ms")
    assert t.model.rowCount() == 300
    assert spent < 0.2


# --- final fix wave ---

def _write_heartbeat(paths, **fields):
    hb = {"pid": 9, "time": datetime.now().isoformat(timespec="milliseconds"), "current": None, **fields}
    paths.heartbeat.parent.mkdir(parents=True, exist_ok=True)
    paths.heartbeat.write_text(json.dumps(hb), encoding="utf-8")


@pytest.mark.parametrize("exit_, text, red", [
    ("finished", "Stopped — finished", False),
    ("stopped", "Stopped — stop requested", False),
    ("error: RuntimeError: disk gone", "Stopped — runner error: RuntimeError: disk gone", True),
    (None, "Stopped", False)])
def test_header_exit_reason(qapp, series4, exit_, text, red):
    """I2: why the batch stopped; an old heartbeat without the reason shows plain Stopped."""
    _write_heartbeat(series4, status="idle", **({} if exit_ is None else {"exit": exit_}))
    t = tab_for(series4)
    assert t.header_text() == text
    assert (monitor_tab.ERROR_INK in t.batch_detail.styleSheet()) == red


def test_header_stopping_after_current_stage(qapp, series4):
    """M5: after Stop the header says so until the running stage ends."""
    t = tab_for(series4)
    p = spawn(series4, "LT1_J02_1D", "mc", "running")
    try:
        t.refresh()
        assert t.header_text().startswith("Running")
        series4.stop_request.write_text("stop\n", encoding="utf-8")
        t.refresh()
        assert t.header_text().startswith("Stopping after current stage") and "LT1_J02_1D" in t.header_text()
    finally:
        stop(p)


def test_header_waiting_for_you_only_when_a_person_can_act(qapp, tmp_path):
    """M11: a runner that polls only for the crop box is not waiting for you."""
    paths = SeriesPaths(tmp_path / "series")
    paths.registry_file("x").parent.mkdir(parents=True)
    put(paths, "LT1_J01_1D", {"crop": done(), "mc": StageRecord(State.WAITING, error="waiting for crop_finalize")})
    t = tab_for(paths)
    p = spawn(paths, "-", "", "waiting")
    try:
        t.refresh()
        assert t.header_text().startswith("Running") and "waiting" in t.header_text()
        assert "waiting for you" not in t.header_text()
        put(paths, "LT1_J01_2D", {"cnmf": done(), "gate": StageRecord(State.WAITING,
                                                                      error="waiting for manual_inspection")})
        t.refresh()
        assert "waiting for you" in t.header_text()
    finally:
        stop(p)


def test_quick_exit_after_start_says_it_ran(qapp, series4):
    """M3: a runner that starts and ends between two refreshes did start; its exit reason is shown."""
    def start(series):
        _write_heartbeat(series4, status="idle", exit="finished")
        return 9

    seen = []
    t = tab_for(series4, start=start)
    t.status_message.connect(seen.append)
    t.start_button.click()
    t.refresh()
    assert seen == ["The batch ran and stopped — finished"] and t.start_button.isEnabled()


def test_runner_that_died_after_start_is_reported_after_grace(qapp, series4):
    def start(series):
        _write_heartbeat(series4, status="running")  # a pulse, then the process vanished without its lock
        return 9

    seen = []
    t = tab_for(series4, start=start)
    t.status_message.connect(seen.append)
    t.start_button.click()
    t.refresh()
    assert not seen and not t.start_button.isEnabled()
    t._clock = lambda: time.monotonic() + monitor_tab.START_GRACE_S + 1
    t.refresh()
    assert seen[-1].startswith("The batch ran and stopped — ") and "runner_console.log" in seen[-1]
    assert monitor_tab.DID_NOT_START not in seen


def test_unscanned_folder_disables_start_and_actions(qapp, tmp_path):
    """M3: no Start and no writing action on a folder that is not a scanned series."""
    t = MonitorTab(tmp_path)
    assert not t.start_button.isEnabled()
    assert not any(a.isEnabled() for a in t.menu_actions)
    t.run_action("export_csv")
    assert not (tmp_path / "_bearmind").exists()


def test_eta_text_wording():
    """M4."""
    assert monitor_tab._eta_text(0.0) == "Estimated time left: nothing left"
    assert monitor_tab._eta_text(None) == "Estimated time left: unknown"
    assert monitor_tab._eta_text(3600, running=False) == (
        "Estimated time left: ~1 h 0 min (time waiting for you not counted) (batch not running)")
    assert "~47 h 59 min" in monitor_tab._eta_text(47 * 3600 + 59 * 60)
    assert "~3 d 4 h" in monitor_tab._eta_text((3 * 24 + 4) * 3600 + 120)


def test_eta_label_says_batch_not_running(qapp, series4, live):
    assert "(batch not running)" not in tab_for(series4).eta_label.text()
    series4.heartbeat.unlink()
    assert tab_for(series4).eta_label.text().endswith(" (batch not running)")


def test_interrupted_stage_without_live_batch(qapp, series4):
    """M6: LT1_J02_1D has MC RUNNING in the registry but no batch is alive."""
    t = tab_for(series4)
    item = t.model.item(2, 2)
    assert item.text() == "interrupted" and item.background().color() == QColor(monitor_tab.CELL_BG["warning"])
    assert t.bars["mc"].caption.text() == "2 done"


def test_unschedulable_row_is_red_with_the_reason(qapp, series4):
    """I1: a session the batch cannot schedule is a red row, the reason in every tooltip."""
    f = series4.session_config("LT1_J02_2D")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("cnmf: [unclosed\n", encoding="utf-8")
    t = tab_for(series4)
    i = t._sids.index("LT1_J02_2D")
    assert "cannot be scheduled: config: " in t.model.item(i, 0).toolTip()
    for c in range(1, t.model.columnCount()):
        item = t.model.item(i, c)
        assert item.background().color() == QColor(monitor_tab.CELL_BG["error"])
        assert "cannot be scheduled: config: " in item.toolTip()
    assert t.warnings_header.text() == "Warnings — 0 sessions"


def test_section_headers_share_one_style(qapp, series4):
    """R5: every section header is drawn the same way, in capitals, also after its text changes."""
    t = tab_for(series4)
    heads = [w for w in t.findChildren(QLabel) if isinstance(w, monitor_tab._SectionLabel)]
    assert len(heads) == 6 and t.warnings_header in heads and t.log_title in heads and t.progress_title in heads
    assert len({h.styleSheet() for h in heads}) == 1
    assert all(h.font().capitalization() == QFont.Capitalization.AllUppercase for h in heads)


def test_running_chip_and_accent_buttons(qapp, series4):
    """R5: the status is a coloured chip as in the mockup; the two "Waiting for you" buttons are accent."""
    t = tab_for(series4)
    assert f"background: {monitor_tab.CHIP['Stopped'][0]}" in t.batch_label.styleSheet()
    p = spawn(series4, "LT1_J02_1D", "mc", "running")
    try:
        t.refresh()
        style = t.batch_label.styleSheet()
        assert f"background: {monitor_tab.CHIP['Running'][0]}" in style and "border-radius" in style
    finally:
        stop(p)
    for b in (t.thresholds_button, t.inspection_button):
        assert monitor_tab.ACCENT in b.styleSheet()


def _fully_visible_rows(table) -> int:
    h = table.viewport().height()
    return sum(1 for r in range(table.model().rowCount())
               if 0 <= table.rowViewportPosition(r) and table.rowViewportPosition(r) + table.rowHeight(r) <= h)


def test_table_shows_eight_rows_at_the_default_size(qapp, tmp_path):
    """R3: at 1100x700 (the window's default) the session table gets the room, not the panels below it."""
    from bearmind.gui.app import MainWindow

    paths = SeriesPaths(tmp_path / "series")
    paths.registry_file("x").parent.mkdir(parents=True)
    for i in range(24):
        sid = f"LT1_J{i // 4 + 1:02d}_{i % 4 + 1}D"
        put(paths, sid, {"crop": done(), "mc": StageRecord(State.FAILED, error="boom")} if i < 5 else {"crop": done()})
    paths.logs_dir.mkdir(parents=True)
    paths.series_log.write_text("".join(f"line {i}\n" for i in range(50)), encoding="utf-8")
    w = MainWindow(paths.root)
    w.resize(1100, 700)
    w.show()
    qapp.processEvents()
    try:
        n = _fully_visible_rows(w.monitor.table)
        print(f"fully visible table rows at 1100x700: {n}")
        assert n >= 8
        assert w.monitor.warnings_scroll.height() >= w.monitor.warnings_scroll.minimumHeight() > 0
        assert w.monitor.log_view.height() >= w.monitor.log_view.minimumHeight() > 0
    finally:
        w.monitor.timer.stop()
        w.close()


def test_previews_checkbox_reflects_and_writes_config(qapp, series4):
    from bearmind import config
    t = tab_for(series4)
    assert t.previews_box.text() == "Preview videos" and not t.previews_box.isChecked()
    t.previews_box.click()
    assert config.resolve(series4)["preview"]["enabled"] is True
    assert "Previews on" in t.last_status
    t.previews_box.click()
    assert config.resolve(series4)["preview"]["enabled"] is False
    assert t.last_status.startswith("Previews off")
    config.update_yaml(series4.series_config, {"preview": {"enabled": True}})  # edited elsewhere
    t.refresh()
    assert t.previews_box.isChecked()


def test_header_names_a_running_preview_stage(qapp, series4, live):
    t = tab_for(series4)
    hb = json.loads(series4.heartbeat.read_text())
    hb["current"] = {"sid": "LT1_J02_1D", "stage": "cnmf_preview"}
    series4.heartbeat.write_text(json.dumps(hb), encoding="utf-8")
    t.refresh()
    assert "CNMF-E preview" in t.header_text()


def test_failed_write_returns_the_box_to_the_config_value(qapp, series4, monkeypatch):
    from bearmind import actions
    t = tab_for(series4)

    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(actions, "set_previews", boom)
    t.previews_box.click()
    assert not t.previews_box.isChecked() and "not changed" in t.last_status


def test_mouse_override_shows_partly_checked(qapp, series4):
    from PySide6.QtCore import Qt
    from bearmind import config
    config.update_yaml(series4.mouse_config("J01"), {"preview": {"enabled": True}})
    t = tab_for(series4)
    assert t.previews_box.checkState() == Qt.CheckState.PartiallyChecked
    assert "mouse config overrides" in t.previews_box.toolTip()
    t.previews_box.click()  # sets the series value: on
    assert t.previews_box.checkState() == Qt.CheckState.Checked
    t.previews_box.click()  # series off; J01 still overrides
    assert t.previews_box.checkState() == Qt.CheckState.PartiallyChecked
    assert "2 sessions follow their own mouse config" in t.last_status
