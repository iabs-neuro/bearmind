import os
import re
import threading
import time

import pytest
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QHeaderView, QLabel, QPushButton

from bearmind import actions, fileio, monitor, registry, scan, scan_view
from bearmind.gui.app import MainWindow
from bearmind.gui.monitor_tab import CELL_BG, EXCLUDED_INK, MonitorTab
from bearmind.gui.scan_tab import ScanTab
from bearmind.paths import SeriesPaths
from bearmind.registry import Anomaly

SIDS = ("NOF_H01_1D", "NOF_H01_2D", "NOF_H02_1D")
ASK_RUNNING = "Batch is running. If session files changed, their stages will be recomputed. Rescan?"
CYRILLIC = "[А-Яа-яЁё]"


def build(make_series, tmp_path, sids=SIDS, **kw):
    root = tmp_path / "s"
    for sid in sids:
        make_series(root, sid=sid, **kw)
    return root


class Dialogs:
    def __init__(self, answer=True):
        self.answer = answer
        self.asked, self.messages, self.opened = [], [], []

    def confirm(self, text):
        self.asked.append(text)
        return self.answer

    def message(self, title, text):
        self.messages.append((title, text))

    def open_folder(self, path):
        self.opened.append(path)


def make_tab(series, dlg=None, **kw):
    dlg = dlg or Dialogs()
    tab = ScanTab(series, confirm=dlg.confirm, message=dlg.message, open_folder=dlg.open_folder, **kw)
    tab.dlg = dlg
    return tab


def wait_scan(tab, start=True, timeout_ms=20000):
    """Run the scan and wait for scan_finished; returns the signal's argument."""
    got = []
    loop = QEventLoop()
    tab.scan_finished.connect(lambda ok: (got.append(ok), loop.quit()))
    QTimer.singleShot(timeout_ms, loop.quit)
    if start:
        tab.start_scan()
    if tab.is_scanning() or not got:
        loop.exec()
    assert got, "scan did not finish"
    return got[0]


def scanned(qapp, make_series, tmp_path, **kw):
    tab = make_tab(build(make_series, tmp_path, **kw))
    assert wait_scan(tab)
    return tab


def select(tab, *sids):
    sm = tab.table.selectionModel()
    sm.clearSelection()
    for i in range(tab.model.rowCount()):
        if tab.model.item(i, 0).text() in sids:
            sm.select(tab.model.index(i, 0), sm.SelectionFlag.Select | sm.SelectionFlag.Rows)


def row_of(tab, sid):
    return next(i for i in range(tab.model.rowCount()) if tab.model.item(i, 0).text() == sid)


def add_anomalies(root, sid, *anoms):
    paths = SeriesPaths(root)
    registry.update_record(paths, sid, lambda rec: rec.anomalies.extend(anoms))


def test_scan_button_fills_table(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path)
    tab = make_tab(root)
    assert tab.scan_button.text() == "Scan"
    assert tab.summary_label.text() == "Not scanned yet. Press Scan."
    assert wait_scan(tab) is True
    assert tab.table.model().rowCount() == 3
    assert tab.summary_label.text().startswith("Found 3 sessions")
    assert tab.scan_button.text() == "Rescan"


def test_progress_bar_reaches_total(qapp, make_series, tmp_path):
    tab = make_tab(build(make_series, tmp_path))
    texts = []
    orig = tab.progress_label.setText
    tab.progress_label.setText = lambda t: (texts.append(t), orig(t))
    assert wait_scan(tab)
    assert tab.progress_bar.value() == tab.progress_bar.maximum() == 3
    assert any(re.fullmatch(r"Scanning \S+ … \d+ / 3", t) for t in texts)


def test_window_responds_during_scan(qapp, make_series, tmp_path, monkeypatch):
    real = scan.scan_series

    def slow(series_root, progress=None):
        def tick(i, n, sid):
            time.sleep(0.3)
            if progress:
                progress(i, n, sid)
        return real(series_root, tick)

    monkeypatch.setattr(scan, "scan_series", slow)
    tab = make_tab(build(make_series, tmp_path))
    ticks = []
    timer = QTimer()
    timer.setInterval(50)
    timer.timeout.connect(lambda: ticks.append(1))
    timer.start()
    assert wait_scan(tab)
    timer.stop()
    assert len(ticks) >= 5


def test_columns_and_excluded_text(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path)
    make_series(root, sid="NOF_H03_1D", with_ts=False)  # no timestamps: an error that excludes
    tab = make_tab(root)
    wait_scan(tab)
    heads = [tab.model.headerData(c, Qt.Orientation.Horizontal) for c in range(tab.model.columnCount())]
    assert heads == ["Session", "Mode", "Parts", "Chunks", "Video frames", "Frames to process", "FPS", "Min",
                     "Anomalies", "Excluded"]
    bad, good = row_of(tab, "NOF_H03_1D"), row_of(tab, "NOF_H01_1D")
    assert tab.model.item(bad, 9).text() == "by anomaly"
    assert tab.model.item(good, 9).text() == ""
    select(tab, "NOF_H01_1D")
    tab.exclude_button.click()
    assert tab.model.item(good, 9).text() == "by you"
    assert tab.model.item(good, 0).foreground().color().name() == EXCLUDED_INK
    assert tab.model.item(good, 4).foreground().color().name() == EXCLUDED_INK


def test_details_show_all_anomalies(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path, ts_rows=25)  # 30 frames, 25 timestamps: mismatch error
    sid = SIDS[0]
    extra = [Anomaly("dropped_frames", "warning", "3 gaps in the timestamps", data={"n_gaps": 3, "max_gap_ms": 90}),
             Anomaly("broken_last_frame", "warning", "the last frame is broken", auto_fixed=True)]
    tab = make_tab(root)
    wait_scan(tab)
    add_anomalies(root, sid, *extra)
    tab.refresh()
    select(tab, sid)
    text = tab.details.toPlainText()
    rec = registry.load_record(SeriesPaths(root), sid)
    for a in rec.anomalies:
        assert a.message in text
    assert text.startswith(f"Details — {sid}")
    assert re.search(r"ERROR .*", text) and "WARN 3 gaps" in text
    fixed = next(line for line in text.splitlines() if "the last frame is broken" in line)
    assert "fixed automatically" in fixed and fixed.startswith("FIXED")
    assert "Part folders:" in text and "FPS:" in text and "Decisions: none yet" in text
    folder = registry.load_record(SeriesPaths(root), sid).parts[0].folder
    assert f"Part folders: {folder.replace('/', os.sep)}" in text  # M12: as Explorer shows it


def test_exclude_and_include(qapp, make_series, tmp_path):
    tab = scanned(qapp, make_series, tmp_path)
    sid = SIDS[0]
    paths = SeriesPaths(tab.series)
    select(tab, sid)
    assert tab.exclude_button.isEnabled() and not tab.include_button.isEnabled()
    tab.exclude_button.click()
    rec = registry.load_record(paths, sid)
    assert rec.excluded is True
    d = rec.decisions[-1]
    assert (d.what, d.value) == ("exclude", "true") and d.who
    assert tab.dlg.asked and sid in tab.dlg.asked[-1]
    assert not tab.exclude_button.isEnabled() and tab.include_button.isEnabled()
    tab.include_button.click()
    rec = registry.load_record(paths, sid)
    assert rec.excluded is False
    assert (rec.decisions[-1].what, rec.decisions[-1].value) == ("exclude", "false")


def test_confirm_no_writes_nothing(qapp, make_series, tmp_path):
    tab = scanned(qapp, make_series, tmp_path)
    paths = SeriesPaths(tab.series)
    before = registry.load_record(paths, SIDS[0])
    tab.dlg.answer = False
    select(tab, SIDS[0])
    tab.exclude_button.click()
    assert registry.load_record(paths, SIDS[0]) == before


def test_accept_enabled_only_on_mismatch_error(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path)
    make_series(root, sid="NOF_H03_1D", ts_rows=25)
    tab = make_tab(root)
    wait_scan(tab)
    select(tab, "NOF_H03_1D")
    assert tab.accept_button.isEnabled()
    select(tab, "NOF_H01_1D")
    assert not tab.accept_button.isEnabled()
    select(tab, "NOF_H01_1D", "NOF_H03_1D")
    assert not tab.accept_button.isEnabled()
    select(tab)
    assert not tab.accept_button.isEnabled()
    select(tab, "NOF_H03_1D")
    tab.accept_button.click()
    rec = registry.load_record(SeriesPaths(root), "NOF_H03_1D")
    assert rec.decisions[-1].what == "accept_mismatch"


def test_confirm_stitch_marks_reviewed(qapp, make_series, tmp_path):
    root = make_series(tmp_path / "s", "legacy_split")
    tab = make_tab(root)
    wait_scan(tab)
    sid = tab.model.item(0, 0).text()
    select(tab, sid)
    assert tab.stitch_button.isEnabled()
    assert "with problems 1" in tab.summary_label.text()
    tab.stitch_button.click()
    assert "(checked by you)" in tab.details.toPlainText()
    assert "with problems 0" in tab.summary_label.text()
    assert not tab.stitch_button.isEnabled()  # nothing left to confirm


def test_confirm_stitch_twice_shows_warning(qapp, make_series, tmp_path):
    root = make_series(tmp_path / "s", "legacy_split")
    tab = make_tab(root)
    wait_scan(tab)
    sid = tab.model.item(0, 0).text()
    select(tab, sid)
    tab.stitch_button.click()
    # exercises the actions warning through _run, not the button (disabled after the first confirm)
    tab._run("confirm_stitch")
    assert tab.dlg.messages and "already confirmed" in tab.dlg.messages[-1][1]


def test_open_folder_called_with_part_folder(qapp, make_series, tmp_path):
    tab = scanned(qapp, make_series, tmp_path)
    sid = SIDS[1]
    select(tab, sid, SIDS[0])
    assert not tab.open_folder_button.isEnabled()
    select(tab, sid)
    assert tab.open_folder_button.isEnabled()
    tab.open_folder_button.click()
    rec = registry.load_record(SeriesPaths(tab.series), sid)
    assert tab.dlg.opened == [tab.series / rec.parts[0].folder]


def test_unrecognized_double_click_opens(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path)
    (root / "1_Raw" / "NOF_old_junk").mkdir(parents=True)
    (root / "1_Raw" / "NOF_old_junk" / "a.avi").write_bytes(b"x")
    tab = make_tab(root)
    wait_scan(tab)
    unrec = scan.read_unrecognized(SeriesPaths(root))
    assert unrec, "the scan must report the folder"
    assert "unrecognized folders 1" in tab.summary_label.text()
    item = tab.unrecognized_list.item(0)
    assert item.text() == f"{unrec[0][0]} — {unrec[0][1]}"
    tab.unrecognized_list.itemDoubleClicked.emit(item)
    assert tab.dlg.opened == [root / unrec[0][0]]


def test_broken_registry_file_listed(qapp, make_series, tmp_path):
    """M8: under their own header, not among the unrecognized folders, and counted in the summary."""
    root = build(make_series, tmp_path)
    tab = make_tab(root)
    wait_scan(tab)
    f = SeriesPaths(root).registry_file(SIDS[0])
    f.write_text("{not json", encoding="utf-8")
    tab.refresh()
    texts = [tab.broken_list.item(i).text() for i in range(tab.broken_list.count())]
    assert len(texts) == 1 and f.name in texts[0]
    assert tab.broken_header.text() == "Unreadable registry files" and not tab.broken_header.isHidden()
    assert tab.unrecognized_list.count() == 0 and tab.unrecognized_header.isHidden()
    assert "unreadable registry files 1" in tab.summary_label.text()
    tab.broken_list.itemDoubleClicked.emit(tab.broken_list.item(0))
    assert tab.dlg.opened == [f.parent]


def test_info_and_accepted_lines_in_details(qapp, make_series, tmp_path):
    """I7: single-frame drops are information (no colour, not counted); M1: an accepted mismatch says so."""
    root = build(make_series, tmp_path, sids=("NOF_H01_1D",), ts_rows=25)
    tab = make_tab(root)
    wait_scan(tab)
    sid = "NOF_H01_1D"
    add_anomalies(root, sid, Anomaly("dropped_frames", "warning", "1 gap in timestamps (max 72 ms)",
                                     data={"n_gaps": 1, "max_gap_ms": 72.0, "n_dropped": 1}))
    # one lost frame is 4 % of this 25-frame session; a long real session at 1 % has hundreds
    tab.paths.series_config.write_text("scan:\n  dropped_frames_warn: {share: 0.5}\n", encoding="utf-8")
    actions.accept_mismatch(tab.paths, [sid], "me")
    tab.refresh()
    select(tab, sid)
    text = tab.details.toPlainText()
    info = next(line for line in text.splitlines() if "1 gap in timestamps" in line)
    assert info.startswith("INFO ") and "(information: no gap longer than 1 s" in info
    accepted = next(line for line in text.splitlines() if "timestamp rows" in line)
    assert accepted.startswith("FIXED") and accepted.endswith("(accepted by you)")
    assert tab.model.item(0, 0).background().color().name() == CELL_BG["ok"]  # no warning colour
    assert "with problems 0" in tab.summary_label.text()


def test_rescan_while_runner_alive_asks(qapp, make_series, tmp_path, monkeypatch):
    root = build(make_series, tmp_path)
    tab = make_tab(root)
    wait_scan(tab)
    alive = monitor.RunnerState(alive=True, status="running", pid=1, age_s=1.0, current=None,
                                stage_elapsed_s=None)
    monkeypatch.setattr(monitor, "runner_state", lambda paths, now=None: alive)
    calls = []
    real = scan.scan_series
    monkeypatch.setattr(scan, "scan_series", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    tab.dlg.answer = False
    tab.start_scan()
    assert tab.dlg.asked == [ASK_RUNNING]
    assert not calls and not tab.is_scanning()
    tab.dlg.answer = True
    assert wait_scan(tab)
    assert calls


def test_scan_error_keeps_table(qapp, make_series, tmp_path, monkeypatch):
    tab = scanned(qapp, make_series, tmp_path)
    rows = [tab.model.item(i, 0).text() for i in range(tab.model.rowCount())]

    def boom(*a, **k):
        raise PermissionError("access denied")

    monkeypatch.setattr(scan, "scan_series", boom)
    assert wait_scan(tab) is False
    assert tab.dlg.messages == [("Scan failed", "PermissionError: access denied")]
    assert [tab.model.item(i, 0).text() for i in range(tab.model.rowCount())] == rows
    assert not tab.is_scanning() and tab.scan_button.isEnabled()


def test_scan_error_traceback_logged(qapp, make_series, tmp_path, monkeypatch, caplog):
    tab = scanned(qapp, make_series, tmp_path)
    monkeypatch.setattr(scan, "scan_series", lambda *a, **k: (_ for _ in ()).throw(OSError("gone")))
    with caplog.at_level("ERROR", logger="bearmind"):
        wait_scan(tab)
    assert "Traceback" in caplog.text and "gone" in caplog.text


def test_start_scan_ignored_while_scanning(qapp, make_series, tmp_path, monkeypatch):
    real = scan.scan_series
    calls = []

    def slow(series_root, progress=None):
        calls.append(1)
        time.sleep(0.3)
        return real(series_root, progress)

    monkeypatch.setattr(scan, "scan_series", slow)
    tab = make_tab(build(make_series, tmp_path))
    done = []
    loop = QEventLoop()
    tab.scan_finished.connect(lambda ok: (done.append(ok), loop.quit()))
    tab.start_scan()
    assert tab.is_scanning() and not tab.scan_button.isEnabled()
    select(tab)
    for b in (tab.exclude_button, tab.include_button, tab.accept_button, tab.stitch_button):
        assert not b.isEnabled()
    tab.start_scan()
    QTimer.singleShot(20000, loop.quit)
    loop.exec()
    assert done == [True] and len(calls) == 1


def test_rescan_after_fix_reincludes(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path, sids=("NOF_H01_1D",), with_ts=False)
    sid = "NOF_H01_1D"
    tab = make_tab(root)
    wait_scan(tab)
    assert tab.model.item(0, 9).text() == "by anomaly"
    make_series(root, sid=sid)  # writes the timestamps (the chunks are rewritten identically)
    assert wait_scan(tab)
    assert tab.model.item(0, 9).text() == ""
    assert registry.load_record(SeriesPaths(root), sid).excluded is False


def test_manual_exclude_survives_rescan(qapp, make_series, tmp_path):
    tab = scanned(qapp, make_series, tmp_path)
    select(tab, SIDS[0])
    tab.exclude_button.click()
    assert wait_scan(tab)
    assert tab.model.item(row_of(tab, SIDS[0]), 9).text() == "by you"


def _widget_texts(tab):
    return [w.text() for cls in (QLabel, QPushButton) for w in tab.findChildren(cls)]


def test_scan_tab_texts_are_english(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path, ts_rows=25)
    (root / "1_Raw" / "NOF_old_junk").mkdir(parents=True)
    (root / "1_Raw" / "NOF_old_junk" / "a.avi").write_bytes(b"x")
    tab = make_tab(root)
    assert not re.search(CYRILLIC, " ".join(_widget_texts(tab)))
    wait_scan(tab)
    select(tab, SIDS[0])
    texts = _widget_texts(tab)
    texts += [tab.model.item(r, c).text() for r in range(tab.model.rowCount()) for c in range(10)]
    texts += [tab.model.headerData(c, Qt.Orientation.Horizontal) for c in range(10)]
    texts += [tab.details.toPlainText()]
    texts += [tab.unrecognized_list.item(i).text() for i in range(tab.unrecognized_list.count())]
    assert not re.search(CYRILLIC, "\n".join(texts))


def test_window_scan_tab_replaces_placeholder(qapp, tmp_series):
    w = MainWindow(tmp_series)
    assert isinstance(w.tabs.widget(1), ScanTab)
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == ["Monitor", "Scan", "Thresholds", "Inspection"]
    assert w.tabs.currentIndex() == 0


def test_window_reopen_replaces_scan_tab(qapp, tmp_series, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    w = MainWindow(tmp_series)
    first = w.tabs.widget(1)
    w.open_series(other)
    assert w.tabs.widget(1) is not first and w.tabs.widget(1).series == other.resolve()
    assert w.tabs.count() == 4


def test_scan_finished_refreshes_monitor(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path)
    w = MainWindow(root)
    w.monitor.timer.stop()
    assert w.monitor.model.rowCount() == 0
    scan_tab = w.tabs.widget(1)
    assert wait_scan(scan_tab)
    assert w.monitor.model.rowCount() == 3


def test_monitor_not_scanned_opens_scan_tab(qapp, tmp_path):
    t = MonitorTab(tmp_path)
    t.show()
    assert t.not_scanned_label.text() == "Series not scanned. Open the Scan tab and press Scan."
    assert t.scan_tab_button.isVisible()
    seen = []
    t.open_tab.connect(seen.append)
    t.scan_tab_button.click()
    assert seen == ["Scan"]
    t.close()


def test_monitor_scanned_hides_scan_tab_button(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path)
    scan.scan_series(root)
    t = MonitorTab(root)
    t.show()
    assert not t.scan_tab_button.isVisible()
    t.close()


def _slow_scan(monkeypatch, delay=0.3):
    real = scan.scan_series

    def slow(series_root, progress=None):
        time.sleep(delay)
        return real(series_root, progress)

    monkeypatch.setattr(scan, "scan_series", slow)


@pytest.fixture
def thread_errors(monkeypatch):
    errors = []
    monkeypatch.setattr(threading, "excepthook", lambda a: errors.append(a.exc_value))
    return errors


def test_window_refuses_open_and_close_during_scan(qapp, make_series, tmp_path, monkeypatch, thread_errors):
    _slow_scan(monkeypatch)
    root = build(make_series, tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    shown = []
    w = MainWindow(root, message=lambda title, text: shown.append(text))
    tab = w.scan
    loop = QEventLoop()
    tab.scan_finished.connect(lambda ok: loop.quit())
    tab.start_scan()
    assert w.open_series(other) is False and w.scan is tab
    assert "scan is running" in w.statusBar().currentMessage()
    ev = QCloseEvent()
    w.closeEvent(ev)
    assert not ev.isAccepted()
    # R3: a message box besides the status line, which is easy to miss
    assert shown == ["A scan is running. Wait for it to finish, then try again."] * 2
    QTimer.singleShot(20000, loop.quit)
    loop.exec()
    assert not tab.is_scanning() and w.tabs.widget(1) is tab
    assert w.open_series(other) is True
    assert not thread_errors
    assert len(registry.all_records(SeriesPaths(root))) == 3


def test_shutdown_waits_and_deleted_bridge_is_ignored(qapp, make_series, tmp_path, monkeypatch, thread_errors):
    import shiboken6
    _slow_scan(monkeypatch)
    root = build(make_series, tmp_path)
    tab = make_tab(root)
    tab.start_scan()
    shiboken6.delete(tab._bridge)  # what deleting the tab does to its children
    tab.shutdown()
    assert not tab._thread.is_alive()
    assert not thread_errors
    assert len(registry.all_records(SeriesPaths(root))) == 3  # the scan ran to the end and wrote


def test_scan_finished_emitted_even_if_show_fails(qapp, make_series, tmp_path, monkeypatch):
    tab = make_tab(build(make_series, tmp_path))
    got = []
    tab.scan_finished.connect(got.append)
    seen = []
    monkeypatch.setattr(tab, "_show", lambda view: (seen.append(1), (_ for _ in ()).throw(ValueError("boom")))[1])
    monkeypatch.setattr("sys.excepthook", lambda *a: seen.append(a[0]))  # PySide reports slot errors here
    loop = QEventLoop()
    tab.scan_finished.connect(lambda ok: loop.quit())
    QTimer.singleShot(20000, loop.quit)
    tab.start_scan()
    loop.exec()
    assert got == [True]


def test_tab_refreshes_when_shown(qapp, make_series, tmp_path):
    from bearmind import actions
    tab = scanned(qapp, make_series, tmp_path)
    actions.set_excluded(tab.paths, [SIDS[0]], True, "monitor")  # what Monitor's menu does
    assert tab.model.item(row_of(tab, SIDS[0]), 9).text() == ""
    tab.show()
    assert tab.model.item(row_of(tab, SIDS[0]), 9).text() == "by you"
    tab.close()


def test_scan_tab_decision_refreshes_monitor_at_once(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path)
    scan.scan_series(root)
    w = MainWindow(root)
    w.monitor.timer.stop()
    dlg = Dialogs()
    w.scan._confirm = dlg.confirm
    select(w.scan, SIDS[0])
    w.scan.exclude_button.click()
    m = w.monitor.model
    item = next(m.item(i, 0) for i in range(m.rowCount()) if m.item(i, 0).text() == SIDS[0])
    assert item.foreground().color().name() == EXCLUDED_INK


def test_finish_text_uses_the_scan_total(qapp, make_series, tmp_path):
    """I1: a session the scan no longer finds stays in the table; the finish text counts what was scanned."""
    root = build(make_series, tmp_path, sids=("NOF_H01_1D", "NOF_H01_2D"))
    tab = make_tab(root)
    said = []
    tab.status_message.connect(said.append)
    wait_scan(tab)
    next((root / "1_Raw").rglob("NOF_H01_2D")).rename(tmp_path / "away")
    assert wait_scan(tab)
    assert tab.model.rowCount() == 2 and tab.progress_label.text() == "Scan finished — 1 / 1"
    assert said[-1] == "Scanned 1 session"
    assert tab.model.item(row_of(tab, "NOF_H01_2D"), 9).text() == "by anomaly"
    assert "not found by the last scan" in tab.model.item(row_of(tab, "NOF_H01_2D"), 8).text()
    assert "not found by the last scan 1" in tab.summary_label.text()


def test_missing_raw_fails_the_scan(qapp, make_series, tmp_path):
    root = build(make_series, tmp_path)
    tab = make_tab(root)
    wait_scan(tab)
    (root / "1_Raw").rename(tmp_path / "moved")
    assert wait_scan(tab) is False
    assert tab.dlg.messages[-1][0] == "Scan failed" and "1_Raw not found" in tab.dlg.messages[-1][1]


def _excluded_three(make_series, tmp_path):
    """NOF_H01_1D excluded by you, NOF_H01_2D excluded by anomaly (frame mismatch), NOF_H02_1D included."""
    root = build(make_series, tmp_path, sids=("NOF_H01_1D", "NOF_H02_1D"))
    make_series(root, sid="NOF_H01_2D", ts_rows=25)
    tab = make_tab(root)
    wait_scan(tab)
    actions.set_excluded(tab.paths, ["NOF_H01_1D"], True, "me")
    tab.refresh()
    return tab


def _decisions(tab, sid):
    return [(d.what, d.value) for d in registry.load_record(tab.paths, sid).decisions]


def test_include_only_for_sessions_excluded_by_you(qapp, make_series, tmp_path):
    """I4: Include writes only for rows excluded by you and names only them; an anomaly is fixed by Rescan or
    Accept, which the button's tooltip says."""
    tab = _excluded_three(make_series, tmp_path)
    select(tab, "NOF_H01_2D")
    assert not tab.include_button.isEnabled()
    assert "Rescan" in tab.include_button.toolTip() and "Accept" in tab.include_button.toolTip()
    before = {sid: _decisions(tab, sid) for sid in ("NOF_H01_2D", "NOF_H02_1D")}
    select(tab, "NOF_H01_1D", "NOF_H01_2D", "NOF_H02_1D")
    assert tab.include_button.isEnabled()
    tab.include_button.click()
    assert tab.dlg.asked[-1] == "Session NOF_H01_1D will be included in the batch again."
    assert _decisions(tab, "NOF_H01_1D")[-1] == ("exclude", "false")
    assert {sid: _decisions(tab, sid) for sid in ("NOF_H01_2D", "NOF_H02_1D")} == before


def test_exclude_only_for_sessions_not_excluded(qapp, make_series, tmp_path):
    tab = _excluded_three(make_series, tmp_path)
    before = {sid: _decisions(tab, sid) for sid in ("NOF_H01_1D", "NOF_H01_2D")}
    select(tab, "NOF_H01_1D", "NOF_H01_2D", "NOF_H02_1D")
    tab.exclude_button.click()
    assert tab.dlg.asked[-1].startswith("Session NOF_H02_1D will be excluded")
    assert _decisions(tab, "NOF_H02_1D")[-1] == ("exclude", "true")
    assert {sid: _decisions(tab, sid) for sid in ("NOF_H01_1D", "NOF_H01_2D")} == before


def test_included_by_you_over_an_error(qapp, make_series, tmp_path):
    """I4: the CLI `decide include` still includes a session with an error; the column says who did it."""
    tab = _excluded_three(make_series, tmp_path)
    actions.set_excluded(tab.paths, ["NOF_H01_2D"], False, "me")
    tab.refresh()
    assert tab.model.item(row_of(tab, "NOF_H01_2D"), 9).text() == "included by you"


def test_failure_after_the_writes_refreshes_the_table(qapp, make_series, tmp_path, monkeypatch):
    """I5: the scan wrote the records, then the unrecognized list failed: the table shows what is written."""
    root = build(make_series, tmp_path, sids=("NOF_H01_1D",), ts_rows=25)
    tab = make_tab(root)
    wait_scan(tab)
    assert tab.model.item(0, 9).text() == "by anomaly"
    make_series(root, sid="NOF_H01_1D")  # the person fixes the timestamps
    real = fileio.write_atomic

    def locked(path, *a, **k):
        if str(path).endswith("scan_unrecognized.csv"):
            raise PermissionError("scan_unrecognized.csv is open in Excel")
        return real(path, *a, **k)

    monkeypatch.setattr(fileio, "write_atomic", locked)
    assert wait_scan(tab) is False
    assert tab.dlg.messages[-1] == ("Scan failed", "PermissionError: scan_unrecognized.csv is open in Excel")
    assert tab.model.item(0, 9).text() == ""
    assert registry.load_record(tab.paths, "NOF_H01_1D").excluded is False


def test_refresh_failure_reported_once_then_status_line(qapp, make_series, tmp_path, monkeypatch):
    """I6: like Monitor's refresh: the first failure is reported (a dialog), repeats go to the status line."""
    root = build(make_series, tmp_path, sids=("NOF_H01_1D",))
    reported, said = [], []
    tab = make_tab(root, report=lambda *exc: reported.append(exc[1]))
    tab.status_message.connect(said.append)
    real, fail = scan_view.scan_view, ["share gone"]

    def view(*a, **k):
        if fail[0]:
            raise OSError(fail[0])
        return real(*a, **k)

    monkeypatch.setattr(scan_view, "scan_view", view)
    tab.refresh()
    tab.refresh()
    assert [str(e) for e in reported] == ["share gone"]
    assert said == ["Refresh failed: OSError: share gone"]
    fail[0] = None
    tab.refresh()
    fail[0] = "again"
    tab.refresh()
    assert len(reported) == 2  # a refresh that worked in between: the next failure is reported again


def test_odd_record_row_does_not_break_the_tab(qapp, tmp_path):
    """I6: one record the view cannot read is a row that says so; the table still opens."""
    from bearmind.registry import STAGES, RawPart, SessionRecord, StageRecord
    paths = SeriesPaths(tmp_path / "s")
    rec = SessionRecord(sid="NOF_H01_1D", mouse="NOF_H01", mode="bonsai",
                        parts=[RawPart("1_Raw/x", ["x_0.avi"], 10, (16, 16), None, None, None, None)],
                        n_frames=10, fps=20.0, fps_info={"fps": 20.0, "source": "timestamps"},
                        anomalies=[Anomaly("frames_ts_mismatch", "error", "e", data={"diff": "abc"})],
                        decisions=[], excluded=True, stages={s: StageRecord() for s in STAGES})
    registry.save_record(paths, rec)
    tab = make_tab(paths.root)
    assert tab.model.rowCount() == 1
    assert tab.model.item(0, 8).text().startswith("cannot show: ValueError")
    assert tab.model.item(0, 4).text() == "–"


def test_layout_excluded_column_does_not_stretch(qapp, make_series, tmp_path):
    """R2: only the Anomalies column stretches; anomaly text stays on one line, the rest in the tooltip."""
    tab = scanned(qapp, make_series, tmp_path)
    head = tab.table.horizontalHeader()
    assert not head.stretchLastSection()
    assert head.sectionResizeMode(8) == QHeaderView.ResizeMode.Stretch
    assert head.sectionResizeMode(9) != QHeaderView.ResizeMode.Stretch
    assert not tab.table.wordWrap()


def test_anomalies_cell_greys_fixed_lines(qapp, make_series, tmp_path):
    """R2: fixed lines are marked "(fixed)" and drawn grey; the tooltip lists every line with its note."""
    from PySide6.QtGui import QPainter, QPixmap
    from PySide6.QtWidgets import QStyleOptionViewItem
    from bearmind.gui.scan_tab import SUMMARY_ROLE
    root = build(make_series, tmp_path, sids=("NOF_H01_1D",), ts_rows=28)  # extra chunk frames: fixed
    tab = make_tab(root)
    wait_scan(tab)
    add_anomalies(root, "NOF_H01_1D", Anomaly("no_metadata", "warning", "no start time"))
    tab.refresh()
    item = tab.model.item(0, 8)
    assert item.data(SUMMARY_ROLE) == [("no recording start time", False), ("extra chunk frames: 2 (fixed)", True)]
    assert item.text() == "no recording start time; extra chunk frames: 2 (fixed)"
    assert item.toolTip().splitlines() == ["extra chunk frames: 2 (fixed automatically)", "no recording start time"]
    # the delegate paints the parts (a narrow cell elides the last one) without raising
    pix = QPixmap(120, 24)
    painter = QPainter(pix)
    opt = QStyleOptionViewItem()
    opt.rect = pix.rect()
    opt.widget = tab.table
    try:
        tab.table.itemDelegateForColumn(8).paint(painter, opt, tab.model.index(0, 8))
    finally:
        painter.end()


def test_confirmation_texts(qapp, make_series, tmp_path):
    """M2, M3: the Scan tab says what Accept and Confirm stitch do."""
    root = build(make_series, tmp_path, sids=("NOF_H01_1D",), ts_rows=25)  # 5 frames more than timestamps
    make_series(root, sid="NOF_H01_2D", ts_rows=35)  # 5 timestamps more than frames
    tab = make_tab(root)
    wait_scan(tab)
    select(tab, "NOF_H01_1D")
    tab.dlg.answer = False
    tab.accept_button.click()
    assert tab.dlg.asked[-1] == ("Session NOF_H01_1D will be used without the extra frames at the end; "
                                 "the files stay unchanged.")
    select(tab, "NOF_H01_2D")
    tab.accept_button.click()
    assert tab.dlg.asked[-1] == ("Session NOF_H01_2D will be used without the extra timestamp rows at the end; "
                                 "the files stay unchanged.")
    split = make_tab(make_series(tmp_path / "split", "legacy_split"))
    wait_scan(split)
    sid = split.model.item(0, 0).text()
    select(split, sid)
    split.dlg.answer = False
    split.stitch_button.click()
    assert split.dlg.asked[-1] == f"Session {sid} will be marked as checked: the stitch by start time stays as it is."


def test_open_folder_of_unreadable_and_missing(qapp, make_series, tmp_path):
    """M6: a session whose video cannot be read opens its session folder; a folder that is gone is said so."""
    root = build(make_series, tmp_path, sids=("NOF_H01_1D", "NOF_H01_2D"))
    next((root / "1_Raw").rglob("NOF_H01_1D_1.avi")).write_bytes(bytes(range(256)) * 20)
    tab = make_tab(root)
    wait_scan(tab)
    select(tab, "NOF_H01_1D")
    assert tab.open_folder_button.isEnabled()
    tab.open_folder_button.click()
    assert tab.dlg.opened == [root / "1_Raw" / "HM_NOF_1D" / "NOF_H01_1D"]
    next((root / "1_Raw").rglob("NOF_H01_2D")).rename(tmp_path / "away")
    wait_scan(tab)
    select(tab, "NOF_H01_2D")
    tab.open_folder_button.click()
    assert len(tab.dlg.opened) == 1 and "Folder not found" in tab.dlg.messages[-1][1]


def test_unknown_session_refreshes_the_table(qapp, make_series, tmp_path):
    """M9: a session removed meanwhile (registry file gone): the message, then the table without it."""
    tab = scanned(qapp, make_series, tmp_path)
    select(tab, SIDS[0])
    tab.paths.registry_file(SIDS[0]).unlink()
    tab.exclude_button.click()
    assert "Not in the series any more" in tab.dlg.messages[-1][1]
    assert SIDS[0] not in [tab.model.item(i, 0).text() for i in range(tab.model.rowCount())]


def test_timer_refreshes_only_while_visible(qapp, make_series, tmp_path):
    """M10: like Monitor, the visible tab follows registry changes; a hidden one waits for showEvent."""
    from bearmind import config
    tab = scanned(qapp, make_series, tmp_path)
    assert tab.timer.isActive()
    assert tab.timer.interval() == int(config.load_defaults()["monitor"]["refresh_ms"])
    actions.set_excluded(tab.paths, [SIDS[0]], True, "monitor")
    tab.timer.timeout.emit()
    assert tab.model.item(row_of(tab, SIDS[0]), 9).text() == ""  # hidden: not refreshed
    tab.show()
    actions.set_excluded(tab.paths, [SIDS[1]], True, "monitor")
    select(tab, SIDS[2])
    tab.timer.timeout.emit()
    assert tab.model.item(row_of(tab, SIDS[1]), 9).text() == "by you"
    assert tab.selected_sids() == [SIDS[2]]
    tab.shutdown()
    assert not tab.timer.isActive()
    tab.close()


def test_failed_scan_refreshes_monitor(qapp, make_series, tmp_path, monkeypatch):
    """I5: a scan that failed after writing records: Monitor shows them at once, not at its next tick."""
    root = build(make_series, tmp_path, sids=("NOF_H01_1D",))
    w = MainWindow(root)
    w.monitor.timer.stop()
    assert wait_scan(w.scan) and w.monitor.model.rowCount() == 1
    make_series(root, sid="NOF_H02_1D")
    real = fileio.write_atomic
    monkeypatch.setattr(fileio, "write_atomic", lambda path, *a, **k: (
        (_ for _ in ()).throw(PermissionError("locked")) if str(path).endswith("scan_unrecognized.csv")
        else real(path, *a, **k)))
    w.scan._message = lambda *a: None
    assert wait_scan(w.scan) is False
    assert w.monitor.model.rowCount() == 2


def test_include_leaves_out_not_found_sessions(qapp, make_series, tmp_path):
    """A session whose folder the last scan did not find stays excluded whatever is decided: Include does not
    take it, and the tooltip says to restore the folder and Rescan."""
    tab = _excluded_three(make_series, tmp_path)
    registry.update_record(tab.paths, "NOF_H01_1D",
                           lambda r: r.anomalies.append(Anomaly("not_found_in_scan", "error", "gone")))
    tab.refresh()
    select(tab, "NOF_H01_1D")
    assert not tab.include_button.isEnabled()
    assert "The session folder was not found by the last scan: restore it and Rescan" in         tab.include_button.toolTip()
