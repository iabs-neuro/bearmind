"""Inspection tab: the sessions whose CNMF-E is done, the series' inspection server, Not needed.

No test starts a real server or browser: spawn and open_url are replaced, and a fake server is a lock file
written by the test, held by the test's own process.
"""
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import psutil
import pytest
from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtWidgets import QLabel, QPushButton

from bearmind import actions, config, inspect_server, launcher, monitor, registry
from bearmind.gui import inspection_tab
from bearmind.gui.inspection_tab import InspectionTab
from bearmind.gui.monitor_tab import CELL_BG, MonitorTab
from bearmind.paths import SeriesPaths
from bearmind.registry import StageRecord, State
from tests.bearmind.gui.test_monitor_tab import done, put

S1, S2, S3, S4 = "LT1_J01_1D", "LT1_J01_2D", "LT1_J02_1D", "LT1_J02_2D"
CYRILLIC = re.compile("[Ѐ-ӿ]")
URL = "http://localhost:5006"
ASK_PASSED = ("Manual inspection of {} already passed. Saving a new result makes autoinspection and export run "
              "again, and the exported neurons can change. Open anyway?")
DID_NOT_START = "The inspection server did not start; see _bearmind/logs/inspect_console.log"
STARTING = "Starting the inspection server (about 10 s)…"
STILL_STARTING = "The inspection server is still starting; it opens your session when it is ready."
NO_SUCH_PID = 999_999_999
REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def series4(tmp_path):
    """S1 waits for manual inspection; S2 passed it (not needed); S3's CNMF-E runs; S4's CNMF-E is done but out
    of date (the batch would redo it)."""
    paths = SeriesPaths(tmp_path / "series")
    paths.registry_file("x").parent.mkdir(parents=True)
    base = {"crop": done(box=[0, 0, 480, 512]), "mc": done(max_shift_px=3.2), "thresholds": done()}
    put(paths, S1, {**base, "cnmf": done(n_components=231)})
    put(paths, S2, {**base, "cnmf": done(n_components=219), "gate": done(gate="not_needed")})
    put(paths, S3, {**base, "cnmf": StageRecord(State.RUNNING)})
    put(paths, S4, {**base, "cnmf": done(n_components=150)})
    registry.update_record(paths, S4, lambda r: setattr(r.stages["cnmf"], "params_hash", "an older run"))
    return paths


def write_lock(paths, url=URL, prepared=None, stopping=False, pid=None, created=None):
    """A lock as the series' server writes it, held by this test process (alive) unless `pid` says otherwise."""
    me = psutil.Process()
    paths.inspection_lock.parent.mkdir(parents=True, exist_ok=True)
    paths.inspection_lock.write_text(json.dumps({
        "pid": me.pid if pid is None else pid, "create_time": me.create_time() if created is None else created,
        "url": url, "current": [], "prepared": prepared, "stopping": stopping}), encoding="utf-8")


class Ui:
    """A tab with every dialog, the server start and the browser replaced; records what it asked, showed, said,
    started and opened."""

    def __init__(self, paths, answer=True):
        self.asked, self.shown, self.status, self.spawned, self.opened = [], [], [], [], []
        self.answer = answer
        self.child = os.getpid()  # the PID the fake start returns: this process, alive while the test runs
        self.tab = InspectionTab(paths.root, confirm=lambda t: (self.asked.append(t), self.answer)[1],
                                 message=lambda title, text: self.shown.append(text),
                                 spawn=lambda cmd: (self.spawned.append(cmd), self.child)[1],
                                 open_url=self.opened.append)
        self.tab.status_message.connect(self.status.append)

    def rows(self):
        m = self.tab.table.model()
        return [m.index(r, 0).data() for r in range(m.rowCount())]

    def select(self, *sids):
        sel = self.tab.table.selectionModel()
        sel.clearSelection()
        m = self.tab.table.model()
        for r, sid in enumerate(self.rows()):
            if sid in sids:
                sel.select(m.index(r, 0), QItemSelectionModel.SelectionFlag.Select
                           | QItemSelectionModel.SelectionFlag.Rows)
        assert sorted(self.tab.selected_sids()) == sorted(sids)


@pytest.fixture
def make_ui():
    made = []

    def make(paths, **kw):
        ui = Ui(paths, **kw)
        made.append(ui)
        return ui

    yield make
    for ui in made:  # no timer of a finished test keeps polling a deleted folder
        ui.tab.shutdown()


def _latest_gate(paths, sid):
    found = [d.value for d in registry.load_record(paths, sid).decisions if d.what == "manual_inspection"]
    return found[-1] if found else None


def test_rows_are_sessions_with_cnmf_done(qapp, series4, make_ui):
    ui = make_ui(series4)
    assert ui.rows() == [S1, S2, S4]  # S3's CNMF-E runs: not a row
    # the cells are Monitor's: text, colour and tooltip
    view = {r.sid: r for r in monitor.series_view(series4).rows}
    m = ui.tab.table.model()
    for r, sid in enumerate(ui.rows()):
        for c, stage in enumerate(("cnmf", "gate", "autoinspect", "export"), start=1):
            cell, ix = view[sid].cells[stage], m.index(r, c)
            assert ix.data() == cell.text and ix.data(Qt.ItemDataRole.ToolTipRole) == cell.tooltip
            bg = ix.data(Qt.ItemDataRole.BackgroundRole)
            assert (bg.color().name() if bg is not None and bg.style() != Qt.BrushStyle.NoBrush else None) \
                == CELL_BG.get(cell.level)
    assert m.index(0, 1).data() == "231 cells" and m.index(1, 2).data() == "not needed"
    assert m.index(2, 1).data() == monitor.OUT_OF_DATE
    # an excluded session leaves the list; one whose CNMF-E is done joins it
    registry.update_record(series4, S1, lambda r: setattr(r, "excluded", True))
    registry.update_record(series4, S3, lambda r: r.stages.update({"cnmf": done(n_components=99)}))
    ui.tab.refresh()
    assert ui.rows() == [S2, S3, S4]


def test_columns_match_mockup(qapp, series4, make_ui):
    m = make_ui(series4).tab.table.model()
    assert [m.headerData(c, Qt.Orientation.Horizontal) for c in range(m.columnCount())] == \
        ["Session", "CNMF-E", "Inspection", "Autoinspect", "Export"]


def test_inspect_button_rules(qapp, series4, make_ui):
    ui = make_ui(series4)
    t = ui.tab
    assert t.inspect_button.text() == "Manual inspection" and t.not_needed_button.text() == "Not needed"
    assert not t.inspect_button.isEnabled() and not t.not_needed_button.isEnabled()  # nothing selected
    ui.select(S1)
    assert t.inspect_button.isEnabled() and t.not_needed_button.isEnabled()
    ui.select(S1, S2)
    assert not t.inspect_button.isEnabled() and t.not_needed_button.isEnabled()
    ui.select(S4)  # CNMF-E done but out of date: the batch redoes it before anyone inspects it
    assert not t.inspect_button.isEnabled()
    assert t.inspect_button.toolTip() == f"CNMF-E of {S4} is out of date: the batch recomputes it first."
    ui.select(S1)
    registry.update_record(series4, S1, lambda r: r.stages.update({"cnmf": StageRecord(State.RUNNING)}))
    t.refresh()  # its CNMF-E runs again: the row goes, and the button with it
    assert S1 not in ui.rows() and t.selected_sids() == [] and not t.inspect_button.isEnabled()
    t.open_inspection(S1)  # asked directly: refused, nothing started or opened
    t.open_inspection(S4)
    assert ui.spawned == [] and ui.opened == []


def test_open_starts_server_once(qapp, series4, make_ui):
    ui = make_ui(series4)
    ui.select(S1)
    ui.tab.inspect_button.click()
    assert ui.spawned == [launcher.inspect_command(series4.root)] and ui.opened == []
    assert ui.status[-1] == STARTING and ui.tab.server_label.text() == STARTING
    ui.tab.refresh()  # the tab's own refresh keeps the line: the server has not written its lock yet
    assert ui.tab.server_label.text() == STARTING
    ui.tab.inspect_button.click()  # a second click during the wait starts no second server
    assert len(ui.spawned) == 1
    write_lock(series4)  # the server listens: its address is in the lock
    end = time.monotonic() + 10
    while not ui.opened and time.monotonic() < end:  # the tab's own timer finds it; the window never blocks
        qapp.processEvents()
        time.sleep(0.02)
    assert ui.opened == [inspect_server.session_url(URL, S1)]
    assert ui.tab.server_label.text() == f"Inspection server: running at {URL}"
    ui.tab.open_inspection(S1)  # running now: opened at once, nothing started
    assert len(ui.spawned) == 1 and ui.opened[-1] == inspect_server.session_url(URL, S1)


def test_open_reuses_running_server(qapp, series4, make_ui):
    write_lock(series4, prepared=S4)
    ui = make_ui(series4)
    ui.select(S1)
    ui.tab.inspect_button.click()
    assert ui.spawned == [] and ui.opened == [inspect_server.session_url(URL, S1)]
    assert not ui.tab.start_timer.isActive()


def test_open_waits_for_a_server_another_process_starts(qapp, series4, make_ui):
    """A server that holds the lock but has no address yet is starting: waited for, not started again."""
    write_lock(series4, url=None)
    ui = make_ui(series4)
    ui.tab.open_inspection(S1)
    assert ui.spawned == [] and ui.opened == [] and ui.tab.start_timer.isActive()
    write_lock(series4)
    ui.tab._poll_server()
    assert ui.spawned == [] and ui.opened == [inspect_server.session_url(URL, S1)]


def _stdin_child():
    """A real process that runs until its stdin closes: a started server process that is not a server."""
    return subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)


def test_server_did_not_start_message(qapp, series4, make_ui):
    """The started inspect command ends without a server: said at once, not after the wait."""
    ui = make_ui(series4)
    now = [100.0]
    ui.tab._clock = lambda: now[0]
    child = _stdin_child()
    try:
        ui.child = child.pid
        ui.tab.open_inspection(S1)
        assert len(ui.spawned) == 1 and ui.tab.start_timer.isActive()
        ui.tab._poll_server()
        assert ui.shown == [] and ui.tab.start_timer.isActive()
        child.stdin.close()
        child.wait(timeout=20)
        ui.tab._poll_server()  # the clock has not moved: the child's end is enough
    finally:
        child.kill()
        child.wait(timeout=20)
    assert ui.shown == [DID_NOT_START] and "inspect_console.log" in ui.status[-1]
    assert ui.opened == [] and not ui.tab.start_timer.isActive()
    assert ui.tab.server_label.text() == "Inspection server: not running"
    ui.child = os.getpid()
    ui.tab.open_inspection(S1)  # a new click is a new start
    assert len(ui.spawned) == 2


def test_child_that_died_at_once(qapp, series4, make_ui):
    ui = make_ui(series4)
    ui.child = NO_SUCH_PID  # gone before the tab looked at it
    ui.tab.open_inspection(S1)
    assert ui.shown == [DID_NOT_START] and not ui.tab.start_timer.isActive() and ui.opened == []


def test_still_starting_after_the_wait(qapp, series4, make_ui):
    """30 s without an address while the started command still runs: the tab says so and keeps waiting, up to the
    inspect command's own bounded wait."""
    ui = make_ui(series4)
    now = [100.0]
    ui.tab._clock = lambda: now[0]
    ui.tab.open_inspection(S1)
    now[0] += inspection_tab.START_WAIT_S + 1
    ui.tab._poll_server()
    assert ui.status[-1] == STILL_STARTING and ui.tab.server_label.text() == STILL_STARTING
    assert ui.shown == [] and ui.tab.start_timer.isActive()
    n = len(ui.status)
    now[0] += 60
    ui.tab._poll_server()
    assert len(ui.status) == n  # said once
    write_lock(series4)
    ui.tab._poll_server()
    assert ui.opened == [inspect_server.session_url(URL, S1)] and not ui.tab.start_timer.isActive()
    # the hard cap: the inspect command itself gives up after its bounded wait
    series4.inspection_lock.unlink()
    ui.tab.open_inspection(S1)
    now[0] += inspect_server.STOP_WAIT_S + inspection_tab.START_WAIT_S - 1
    ui.tab._poll_server()
    assert ui.shown == [] and ui.tab.start_timer.isActive()
    now[0] += 2
    ui.tab._poll_server()
    assert ui.shown == [DID_NOT_START] and not ui.tab.start_timer.isActive()


def test_wait_pauses_while_a_server_stops(qapp, series4, make_ui):
    """A stopping server holds the lock until its side files are written: that time is not counted, and the line
    says the server is stopping. The start is not held back: the inspect command waits for the stop itself."""
    write_lock(series4, stopping=True)
    ui = make_ui(series4)
    now = [100.0]
    ui.tab._clock = lambda: now[0]
    ui.tab.open_inspection(S1)
    assert len(ui.spawned) == 1 and ui.tab.server_label.text() == "Inspection server: stopping…"
    now[0] += 100
    ui.tab._poll_server()
    assert ui.shown == [] and ui.status[-1] == STARTING and ui.tab.start_timer.isActive()
    assert ui.tab.server_label.text() == "Inspection server: stopping…"
    series4.inspection_lock.unlink()  # the old server ended
    ui.tab._poll_server()
    assert ui.tab.server_label.text() == STARTING
    now[0] += inspection_tab.START_WAIT_S - 1  # the 30 s count from the end of the stop
    ui.tab._poll_server()
    assert ui.tab.server_label.text() == STARTING and STILL_STARTING not in ui.status
    now[0] += 2
    ui.tab._poll_server()
    assert ui.status[-1] == STILL_STARTING and len(ui.spawned) == 1
    write_lock(series4)
    ui.tab._poll_server()
    assert ui.opened == [inspect_server.session_url(URL, S1)]


def test_spawn_failure_is_shown(qapp, series4, make_ui, caplog):
    """Any error of the start ends the wait (the line follows the server again); the message names the error's
    type and errno, not its localized text, which goes to gui.log."""
    ui = make_ui(series4)

    def odd(cmd):
        raise ValueError("embedded null character")

    ui.tab._spawn = odd
    with caplog.at_level(logging.ERROR, logger="bearmind"):
        ui.tab.open_inspection(S1)
        assert ui.shown == ["The inspection server did not start: ValueError; details in gui.log"]
        assert not ui.tab.start_timer.isActive() and ui.tab._waiting == []
        assert "embedded null character" in caplog.text
        write_lock(series4)
        ui.tab.refresh()
        assert ui.tab.server_label.text() == f"Inspection server: running at {URL}"
        series4.inspection_lock.unlink()

        def localized(cmd):
            raise FileNotFoundError(2, "Не удается найти указанный файл")

        ui.tab._spawn = localized
        ui.tab.open_inspection(S1)
    assert ui.shown[-1] == "The inspection server did not start: FileNotFoundError (errno 2); details in gui.log"
    assert "Не удается" in caplog.text and not ui.tab.start_timer.isActive()


def test_browser_that_does_not_open_is_named(qapp, series4, make_ui, monkeypatch):
    import webbrowser

    monkeypatch.setattr(webbrowser, "open", lambda url, *a, **k: False)
    assert inspection_tab._default_open_url(URL) is False  # webbrowser.open's answer is passed on
    write_lock(series4)
    ui = make_ui(series4)
    ui.tab._open_url = lambda url: (ui.opened.append(url), False)[1]
    ui.tab.open_inspection(S1)
    url = inspect_server.session_url(URL, S1)
    assert ui.opened == [url]
    assert ui.shown == [f"The browser did not open; open {url} yourself."]
    assert not [s for s in ui.status if s.startswith("Opened in your browser")]


def test_unschedulable_session_cannot_be_opened(qapp, series4, make_ui):
    """A session the batch cannot schedule: whether its CNMF-E is current is unknown, so it is not opened."""
    f = series4.session_config(S1)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("cnmf: [unclosed\n", encoding="utf-8")
    ui = make_ui(series4)
    ui.select(S1)
    assert not ui.tab.inspect_button.isEnabled()
    assert ui.tab.inspect_button.toolTip().startswith(f"{S1} cannot be scheduled: config: ")
    ui.tab.open_inspection(S1)
    assert ui.spawned == [] and ui.opened == [] and ui.shown[0].startswith(f"{S1} cannot be scheduled: ")


def test_window_import_loads_no_bokeh():
    """The window reads the server's lock through bearmind.inspect_lock: Bokeh and Tornado (the server's) are
    not loaded at window start."""
    code = ("import sys, bearmind.gui.app; "
            "print(sorted({m.split('.')[0] for m in sys.modules} & {'bokeh', 'tornado', 'caiman'}))")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONPATH": str(REPO)})
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "[]"


def test_monitor_button_preselects_first_waiting(qapp, series4):
    """Monitor's "Manual inspection: N sessions ->" opens the tab on the first waiting session whose CNMF-E is
    current (as the Thresholds tab does); none: the selection stays."""
    from bearmind.gui.app import MainWindow

    w = MainWindow(series4.root, message=lambda title, text: None)
    try:
        w.monitor.inspection_button.click()
        assert w.tabs.currentIndex() == 3 and w.inspection.selected_sids() == [S1]
        actions.gate_not_needed(series4, [S1], "test")  # S4 still waits, but its CNMF-E is out of date
        w.monitor.refresh()
        sel = w.inspection.table.selectionModel()
        sel.clearSelection()
        sel.select(w.inspection.table.model().index(1, 0), QItemSelectionModel.SelectionFlag.Select
                   | QItemSelectionModel.SelectionFlag.Rows)
        w.show_tab("Monitor")
        w.monitor.inspection_button.click()
        assert w.tabs.currentIndex() == 3 and w.inspection.selected_sids() == [S2]
        s5 = "LT1_J03_1D"
        put(series4, s5, {"crop": done(), "mc": done(), "thresholds": done(), "cnmf": done(n_components=7)})
        w.monitor.refresh()
        w.show_tab("Monitor")
        w.monitor.inspection_button.click()
        assert w.inspection.selected_sids() == [s5]
    finally:
        w.inspection.shutdown()
        w.thresholds.shutdown()
        w.close()


def test_open_already_passed_asks(qapp, series4, make_ui):
    write_lock(series4)
    ui = make_ui(series4, answer=False)
    ui.tab.open_inspection(S2)
    assert ui.asked == [ASK_PASSED.format(S2)] and ui.opened == [] and ui.spawned == []
    ui.answer = True
    ui.tab.open_inspection(S2)
    assert ui.opened == [inspect_server.session_url(URL, S2)]
    ui.tab.open_inspection(S1)  # not passed: no question
    assert len(ui.asked) == 2 and ui.opened[-1] == inspect_server.session_url(URL, S1)


def test_not_needed_writes_decision(qapp, series4, make_ui):
    ui = make_ui(series4)
    decided = []
    ui.tab.decided.connect(lambda: decided.append(1))
    ui.select(S1)
    ui.answer = False
    ui.tab.not_needed_button.click()
    assert ui.asked == [MonitorTab._ask_text("inspection_not_needed", [S1], None)] and _latest_gate(series4, S1) is None
    ui.answer = True
    ui.tab.not_needed_button.click()
    assert _latest_gate(series4, S1) == "not_needed" and decided == [1]
    assert ui.status[-1] == f"Inspection not needed for 1 session: {S1}"
    ui.select(S2)  # already passed: nothing recorded, and the window says why
    ui.tab.not_needed_button.click()
    assert ui.shown == [f"{S2}: {actions.failure_text('already_passed', '')}"]
    assert ui.status[-1] == "Nothing was changed." and decided == [1]


def test_server_label_states(qapp, series4, make_ui):
    ui = make_ui(series4)
    label = ui.tab.server_label
    assert label.text() == "Inspection server: not running"
    for kw, text in (({"url": None}, "Inspection server: starting (about 10 s)…"),
                     ({"stopping": True}, "Inspection server: stopping…"),
                     ({}, f"Inspection server: running at {URL}"),
                     ({"prepared": S4}, f"Inspection server: running at {URL} · next prepared: {S4}"),
                     ({"pid": 999_999_999, "created": 1.0}, "Inspection server: not running")):  # holder gone
        write_lock(series4, **kw)
        ui.tab.refresh()
        assert label.text() == text, kw
    series4.inspection_lock.unlink()
    ui.tab.refresh()
    assert label.text() == "Inspection server: not running"


def test_inspection_tab_texts_are_english(qapp, series4, make_ui):
    ui = make_ui(series4)
    t = ui.tab
    texts = [lab.text() for lab in t.findChildren(QLabel)] + [b.text() for b in t.findChildren(QPushButton)]
    assert t.hint_label.text() == "Opens in your browser. This window stays free."
    m = t.table.model()
    texts += [m.headerData(c, Qt.Orientation.Horizontal) for c in range(m.columnCount())]
    texts += [m.index(r, c).data(role) or "" for r in range(m.rowCount()) for c in range(m.columnCount())
              for role in (Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.ToolTipRole)]
    for sids in ((), (S1, S2), (S4,), (S1,)):
        ui.select(*sids)
        texts += [t.inspect_button.toolTip(), t.not_needed_button.toolTip()]
    t.not_needed_button.click()
    ui.select(S2)
    t.not_needed_button.click()
    t.open_inspection(S1)
    texts.append(t.server_label.text())
    t._clock = lambda: 1e12  # the wait is over
    t._poll_server()
    write_lock(series4, url=None)
    t.refresh()
    texts.append(t.server_label.text())
    write_lock(series4, prepared=S4)
    t.refresh()
    t.open_inspection(S2)
    texts += [t.server_label.text()] + ui.asked + ui.shown + ui.status
    # Not needed for S1, which then counts (opening S1 asks), opening S2 (passed) asks
    assert len(ui.asked) == 4 and ui.shown and ui.status
    texts += [v for k, v in vars(inspection_tab).items() if k.isupper() and isinstance(v, str)]  # every text
    for x in texts:
        assert not CYRILLIC.search(x), x


def test_refresh_only_while_visible(qapp, series4, make_ui, monkeypatch):
    ui = make_ui(series4)
    t = ui.tab
    assert t.timer.interval() == int(config.load_defaults()["monitor"]["refresh_ms"]) and t.timer.isActive()
    calls = []
    real = monitor.series_view
    monkeypatch.setattr(monitor, "series_view", lambda *a, **k: (calls.append(1), real(*a, **k))[1])
    t._tick()  # hidden: no work
    assert calls == []
    t.show()
    qapp.processEvents()
    calls.clear()
    t._tick()
    assert calls == [1]
    t.hide()
    t.shutdown()
    assert not t.timer.isActive() and not t.start_timer.isActive()


def test_refresh_failure_reported_once(qapp, series4, make_ui, monkeypatch):
    reported = []
    t = InspectionTab(series4.root, report=lambda *exc: reported.append(exc[1]), spawn=lambda cmd: 1,
                      open_url=lambda url: None)
    said = []
    t.status_message.connect(said.append)

    def boom(*a, **k):
        raise RuntimeError("registry gone\nmore")

    real = monitor.series_view
    monkeypatch.setattr(monitor, "series_view", boom)
    t.refresh()
    t.refresh()
    assert len(reported) == 1 and said == ["Refresh failed: RuntimeError; details in gui.log"]
    monkeypatch.setattr(monitor, "series_view", real)
    t.refresh()
    assert t._refresh_failed is None
    t.shutdown()


def test_refresh_failure_names_type_and_file(qapp, series4, monkeypatch, caplog):
    """M9: the status bar names the error's type and file, never its text (Windows' is localized); gui.log has
    the whole text."""
    t = InspectionTab(series4.root, report=lambda *exc: None, spawn=lambda cmd: 1, open_url=lambda url: None)
    said = []
    t.status_message.connect(said.append)
    target = series4.registry_file(S1)

    def denied(*a, **k):
        raise PermissionError(13, "Отказано в доступе", str(target))

    monkeypatch.setattr(monitor, "series_view", denied)
    with caplog.at_level(logging.WARNING, logger="bearmind"):
        t.refresh()  # the first goes to the window's report
        t.refresh()
    assert said == [f"Refresh failed: PermissionError on {target.name}; details in gui.log"]
    assert not CYRILLIC.search(said[0])
    assert "Отказано в доступе" in caplog.text
    t.shutdown()


def test_open_asks_when_a_decision_counts(qapp, series4, make_ui):
    """I2: a session whose latest decision counts already (the batch takes it on its next look, the gate is not
    passed yet) asks before opening; a saved result is named, with what the page shows."""
    write_lock(series4)
    paths = series4
    result = paths.manual_result(S1)
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_bytes(b"result")  # newer than the estimates, which this fixture does not have
    actions.gate_cleaned(paths, S1, "ann")
    d = registry.load_record(paths, S1).decisions[-1]
    assert actions.counted_gate_decision(paths, registry.load_record(paths, S1)) == d
    ui = make_ui(paths, answer=False)
    ui.tab.open_inspection(S1)
    when = d.when.replace("T", " ")[:16]
    assert ui.asked == [inspection_tab.ALREADY_DECIDED_ASK.format(
        sid=S1, value="cleaned", saved=inspection_tab.SAVED_BEFORE.format(sid=S1, when=when, who="ann"))]
    assert "A result of LT1_J01_1D was saved" in ui.asked[0] and ui.opened == []
    ui.answer = True
    ui.tab.open_inspection(S1)
    assert ui.opened == [inspect_server.session_url(URL, S1)]


def test_window_wires_the_inspection_tab(qapp, series4):
    from bearmind.gui.app import MainWindow

    w = MainWindow(series4.root, message=lambda title, text: None)
    tab = w.tabs.widget(3)
    assert isinstance(tab, InspectionTab) and w.inspection is tab and w.tabs.tabText(3) == "Inspection"
    # S1 and S4 wait (S4's CNMF-E is out of date, but the gate has no decision for it either)
    assert w.monitor.inspection_button.text() == "Manual inspection: 2 sessions →"
    w.monitor.timer.stop()  # the count below changes with the write, not with Monitor's timer
    tab._confirm = lambda text: True
    sel = tab.table.selectionModel()
    sel.select(tab.table.model().index(0, 0), QItemSelectionModel.SelectionFlag.Select
               | QItemSelectionModel.SelectionFlag.Rows)
    tab.not_needed_button.click()
    assert w.monitor.inspection_button.text() == "Manual inspection: 1 session →"
    tab.status_message.emit("hello")
    assert w.statusBar().currentMessage() == "hello"
    w.open_series(series4.root)  # a second open replaces the tab and stops the old one
    assert w.tabs.widget(3) is not tab and isinstance(w.tabs.widget(3), InspectionTab)
    assert not tab.timer.isActive() and not tab.start_timer.isActive()
    w.inspection.shutdown()
    w.close()
