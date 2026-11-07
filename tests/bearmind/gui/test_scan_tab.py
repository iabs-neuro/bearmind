import re
import time

from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QLabel, QPushButton

from bearmind import monitor, registry, scan
from bearmind.gui.app import MainWindow
from bearmind.gui.monitor_tab import EXCLUDED_INK, MonitorTab
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
    tab.stitch_button.setEnabled(True)  # a second click through a stale state
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
    root = build(make_series, tmp_path)
    tab = make_tab(root)
    wait_scan(tab)
    f = SeriesPaths(root).registry_file(SIDS[0])
    f.write_text("{not json", encoding="utf-8")
    tab.refresh()
    texts = [tab.unrecognized_list.item(i).text() for i in range(tab.unrecognized_list.count())]
    assert any(str(f) in t or f.name in t for t in texts)


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
