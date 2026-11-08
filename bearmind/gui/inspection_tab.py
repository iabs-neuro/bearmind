"""Inspection tab: the sessions whose CNMF-E is done, with Monitor's cells for CNMF-E and the stages after it.

"Manual inspection" opens the selected session in the browser, on the series' one inspection server
(bearmind.inspect_server): a running server is reused, else one is started detached (launcher.inspect_command)
and its address is awaited from a timer, so the window never blocks. The server's lock is read through
bearmind.inspect_lock, which needs no Bokeh: the window does not load the server's packages. "Not needed" writes
the gate decision through bearmind.actions, as Monitor's menu does. The dialogs, the server start and the browser
are injectable for the tests.
"""
import logging
import sys
import time
import webbrowser
from pathlib import Path

import psutil

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QPushButton,
                               QTableView, QVBoxLayout, QWidget)

from bearmind import actions, config, inspect_lock, launcher, monitor
from bearmind.gui.monitor_tab import (ACCENT_BUTTON, CELL_BG, CELL_INK, EXCLUDED_INK, MUTED, WARNING_TEXT,
                                      MonitorTab, _SectionLabel, plural)
from bearmind.paths import SeriesPaths
from bearmind.registry import State

log = logging.getLogger("bearmind")

STAGES = ("cnmf", "gate", "autoinspect", "export")  # the columns after Session
SECTION = "Sessions — select one, then press the button"
HINT = "Opens in your browser. This window stays free."
INSPECT_LOG = "inspect_console.log"
# the server writes its address into its lock about 10 s after its start (imports); the time an older server
# spends stopping is not counted
START_WAIT_S = 30.0
START_POLL_MS = 200
SERVER = "Inspection server: {}"
SERVER_NOT_RUNNING = SERVER.format("not running")
SERVER_STARTING = SERVER.format("starting (about 10 s)…")
SERVER_STOPPING = SERVER.format("stopping…")
SERVER_RUNNING = SERVER.format("running at {url}")
NEXT_PREPARED = " · next prepared: {sid}"
STARTING = "Starting the inspection server (about 10 s)…"
DID_NOT_START = f"The inspection server did not start; see _bearmind/logs/{INSPECT_LOG}"
STILL_STARTING = "The inspection server is still starting; it opens your session when it is ready."
SPAWN_FAILED = "The inspection server did not start: {what}; details in gui.log"
NO_BROWSER = "The browser did not open; open {url} yourself."
ALREADY_PASSED_ASK = ("Manual inspection of {sid} already passed. Saving a new result makes autoinspection and "
                      "export run again, and the exported neurons can change. Open anyway?")
OUT_OF_DATE_TIP = "CNMF-E of {sid} is out of date: the batch recomputes it first."
UNSCHEDULABLE_TIP = "{sid} cannot be scheduled: {problem}"
NOT_DONE_TIP = "CNMF-E of {sid} is not done."
SELECT_ONE = "Select one session in the table."
ONLY_ONE = "Select only one session to inspect."
INSPECT_TIP = "Open {sid} for manual inspection in your browser"
NOT_NEEDED_TIP = "Record that the selected sessions need no manual inspection"
OPENED = "Opened in your browser: {sids}"
_STYLE_ROLE = Qt.ItemDataRole.UserRole + 1  # the colours a cell was painted with, as in Monitor


def _default_open_url(url: str) -> bool:
    return webbrowser.open(url)  # False: no browser could be started


def _hard_wait_s() -> float:
    """The longest wait for an address: the inspect command waits up to its own bound (STOP_WAIT_S) for a server
    that starts or stops elsewhere, then starts its own. inspect_server imports Bokeh: only asked for after
    START_WAIT_S, never at window start."""
    from bearmind import inspect_server
    return inspect_server.STOP_WAIT_S + START_WAIT_S


def _child_key(pid):
    """(pid, start time) of the process just started, for telling later whether it still runs; start time None:
    it is gone already. None when the start gave no PID."""
    if not isinstance(pid, int) or isinstance(pid, bool):
        return None
    try:
        return pid, psutil.Process(pid).create_time()
    except psutil.Error:
        return pid, None


def _child_runs(key) -> bool:
    pid, created = key
    # pid_exists: a process that ended but whose handle is still open keeps its start time on Windows
    return inspect_lock.same_process(pid, created) and psutil.pid_exists(pid)


def _spawn_failure(e: BaseException) -> str:
    """The error's type and errno only: an OS error's text is in the system's language (the window is English)."""
    errno = getattr(e, "errno", None)
    return SPAWN_FAILED.format(what=type(e).__name__ + (f" (errno {errno})" if errno is not None else ""))


def server_text(st: inspect_lock.ServerState) -> str:
    """The server line for a ServerState: running (and what it prepared), stopping, starting, or none."""
    if st.available:
        return SERVER_RUNNING.format(url=st.url) + (NEXT_PREPARED.format(sid=st.prepared) if st.prepared else "")
    if st.stopping:
        return SERVER_STOPPING
    if st.starting:
        return SERVER_STARTING
    return SERVER_NOT_RUNNING


class InspectionTab(QWidget):
    status_message = Signal(str)
    decided = Signal()  # a gate decision was written: Monitor's "Manual inspection: N sessions" changes with it

    def __init__(self, series: Path, parent=None, confirm=None, message=None, spawn=None, open_url=None,
                 report=None):
        super().__init__(parent)
        self.series = Path(series)
        self.paths = SeriesPaths(self.series)
        self._confirm = confirm or self._ask_confirm
        self._message = message or self._show_message
        self._spawn = spawn or (lambda cmd: launcher.spawn_detached(cmd, self.paths, INSPECT_LOG))
        self._open_url = open_url or _default_open_url
        # the window installs its exception report as sys.excepthook
        self._report = report or (lambda *exc: sys.excepthook(*exc))
        self._refresh_failed = None  # (type, message) of the failure reported last, until a refresh succeeds
        self._clock = time.monotonic
        self._cache = monitor.RecordCache()
        self._rows: dict[str, monitor.Row] = {}
        self._records: dict = {}  # sid -> the record of the last refresh
        self._sids: list[str] = []
        # a wait for the server's address: the sessions to open once it is there; when it began, when its 30 s end
        # (moved on while a server stops) and when the last look was; whether this wait started a server (one
        # start per wait, however many clicks) and that process; whether "still starting" was said
        self._waiting: list[str] = []
        self._began = self._deadline = self._polled = None
        self._spawned, self._child, self._told = False, None, False
        self._build()
        self.start_timer = QTimer(self)
        self.start_timer.setInterval(START_POLL_MS)
        self.start_timer.timeout.connect(lambda: self._guarded(self._poll_server))
        self.refresh()
        # like Monitor: the batch, Monitor's menu and the server's Save results change the registry; refreshed
        # only while shown
        self.timer = QTimer(self)
        self.timer.setInterval(int(config.load_defaults().get("monitor", {}).get("refresh_ms", 2000)))
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # --- construction ---

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        lay.addWidget(_SectionLabel(SECTION))
        self.model = QStandardItemModel(0, 1 + len(STAGES), self)
        self.model.setHorizontalHeaderLabels(["Session"] + [monitor.STAGE_TITLE[s] for s in STAGES])
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self._fit_rows()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._update_buttons())
        lay.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.inspect_button = QPushButton("Manual inspection")
        self.inspect_button.setStyleSheet(ACCENT_BUTTON)
        self.not_needed_button = QPushButton("Not needed")
        self.hint_label = QLabel(HINT)
        self.hint_label.setStyleSheet(f"color: {MUTED};")
        self.inspect_button.clicked.connect(lambda: self._guarded(self._on_inspect))
        self.not_needed_button.clicked.connect(lambda: self._guarded(self.mark_not_needed))
        row.addWidget(self.inspect_button)
        row.addWidget(self.not_needed_button)
        row.addSpacing(8)
        row.addWidget(self.hint_label)
        row.addStretch(1)
        lay.addLayout(row)
        self.server_label = QLabel(SERVER_NOT_RUNNING)
        self.server_label.setStyleSheet(f"color: {MUTED};")
        self.server_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)  # the address
        lay.addWidget(self.server_label)

    def _fit_rows(self) -> None:
        # Monitor's row height: as tall as the text needs
        h = self.table.fontMetrics().height() + 4
        self.table.verticalHeader().setMinimumSectionSize(h)
        self.table.verticalHeader().setDefaultSectionSize(h)

    # --- dialogs (replaced in the tests) ---

    def _ask_confirm(self, text: str) -> bool:
        return QMessageBox.question(self, "BEARMiND", text) == QMessageBox.StandardButton.Yes

    def _show_message(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _say(self, text: str) -> None:
        self.status_message.emit(text)

    def _guarded(self, fn, *args) -> None:
        try:
            fn(*args)
        except Exception:  # a slot must not close the window: the window reports it
            self._report(*sys.exc_info())

    # --- lifetime and refresh ---

    def shutdown(self) -> None:
        """Before the tab is deleted: stop the timers. A server being started goes on starting on its own."""
        self.timer.stop()
        self._end_wait()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_rows()  # before the tab is on a screen its font is measured at another scaling
        self.refresh()  # the batch, Monitor's menu or a Save results in the browser may have changed the registry

    def _tick(self) -> None:
        if self.isVisible():
            self.refresh()

    def refresh(self) -> None:
        # a failure repeats every tick: the first one goes to the window's report (a dialog), the rest only
        # to the log and the status bar, so the timer does not stack modal dialogs
        try:
            self._refresh()
        except Exception as e:
            first = (str(e).splitlines() or [""])[0]
            kind = (type(e), str(e))
            if self._refresh_failed is None:
                self._refresh_failed = kind
                self._report(type(e), e, e.__traceback__)
            else:
                if kind == self._refresh_failed:
                    log.warning("inspection tab refresh failed again: %s", first)
                else:  # a different failure: keep its traceback in the log, still no dialog
                    self._refresh_failed = kind
                    log.error("inspection tab refresh failed differently", exc_info=(type(e), e, e.__traceback__))
                self._say(f"Refresh failed: {type(e).__name__}: {first}")
        else:
            self._refresh_failed = None

    def _refresh(self) -> None:
        view = monitor.series_view(self.paths, cache=self._cache)
        records, _broken = self._cache.records(self.paths)  # the files series_view just read: no new reads
        self._records = {r.sid: r for r in records}
        rows = [r for r in view.rows if not r.excluded and not r.unreadable
                and r.cells["cnmf"].state == State.DONE.value]
        self._rows = {r.sid: r for r in rows}
        sids = [r.sid for r in rows]  # series_view's order: by mouse, then sid
        if sids != self._sids:
            self._rebuild(sids)
            self._sids = sids
        for i, r in enumerate(rows):
            self._fill_row(i, r)
        if not self._waiting:  # during a wait the line says what the wait is for
            self.server_label.setText(server_text(inspect_lock.server_state(self.paths)))
        self._update_buttons()

    def _rebuild(self, sids: list[str]) -> None:
        selected = set(self.selected_sids())
        scroll = self.table.verticalScrollBar().value()
        self.model.setRowCount(0)
        self.model.setRowCount(len(sids))
        for i in range(len(sids)):
            for c in range(self.model.columnCount()):
                item = QStandardItem()
                item.setEditable(False)
                self.model.setItem(i, c, item)
        sel = self.table.selectionModel()
        for i, sid in enumerate(sids):
            if sid in selected:
                sel.select(self.model.index(i, 0), sel.SelectionFlag.Select | sel.SelectionFlag.Rows)
        self.table.verticalScrollBar().setValue(scroll)

    def _fill_row(self, i: int, row: monitor.Row) -> None:
        """Monitor's cells as they are: the same text, colour and tooltip."""
        tip = row.mouse if row.problem is None else f"{row.mouse}\ncannot be scheduled: {row.problem}"
        first = self.model.item(i, 0)
        if first.text() != row.sid:
            first.setText(row.sid)
        if first.toolTip() != tip:
            first.setToolTip(tip)
        for c, stage in enumerate(STAGES, start=1):
            cell, item = row.cells[stage], self.model.item(i, c)
            # write only changed values: repainting every cell on every refresh flickers and costs time
            if item.text() != cell.text:
                item.setText(cell.text)
            if item.toolTip() != cell.tooltip:
                item.setToolTip(cell.tooltip)
            bg = CELL_BG.get(cell.level)
            ink = EXCLUDED_INK if cell.level == "excluded" else (CELL_INK if bg else None)
            style = f"{bg}|{ink}"
            if item.data(_STYLE_ROLE) != style:
                item.setData(style, _STYLE_ROLE)
                item.setBackground(QBrush(QColor(bg)) if bg else QBrush())
                item.setForeground(QBrush(QColor(ink)) if ink else QBrush())

    def selected_sids(self) -> list[str]:
        rows = sorted({ix.row() for ix in self.table.selectionModel().selectedIndexes()})
        return [self.model.item(r, 0).text() for r in rows]

    # --- buttons ---

    def _refusal(self, sids) -> str:
        """Why Manual inspection cannot open `sids` now ("" when it can): one session, not excluded, one the
        batch can schedule, its CNMF-E done and up to date (Monitor's cell: a CNMF-E the batch would redo is shown
        out of date)."""
        if not sids:
            return SELECT_ONE
        if len(sids) > 1:
            return ONLY_ONE
        sid = sids[0]
        row = self._rows.get(sid)
        if row is None:
            rec = self._records.get(sid)
            if rec is None:
                return f"Not in the series any more: {sid}"
            if rec.excluded:
                return WARNING_TEXT["excluded_session"].format(sid=sid)
            return NOT_DONE_TIP.format(sid=sid)
        if row.problem is not None:  # its configs cannot be resolved: whether its CNMF-E is current is unknown
            return UNSCHEDULABLE_TIP.format(sid=sid, problem=row.problem)
        if row.cells["cnmf"].text == monitor.OUT_OF_DATE:
            return OUT_OF_DATE_TIP.format(sid=sid)
        return ""

    def _update_buttons(self) -> None:
        sids = self.selected_sids()
        why = self._refusal(sids)
        self.inspect_button.setEnabled(not why)
        self.inspect_button.setToolTip(why or INSPECT_TIP.format(sid=sids[0]))
        self.not_needed_button.setEnabled(bool(sids))
        self.not_needed_button.setToolTip(NOT_NEEDED_TIP if sids else "Select sessions in the table first.")

    def _on_inspect(self) -> None:
        sids = self.selected_sids()
        why = self._refusal(sids)
        if why:
            self._say(why)
            return
        self.open_inspection(sids[0])

    def open_inspection(self, sid: str) -> None:
        """Open `sid` in the browser on the series' server; start the server first when none runs."""
        self.refresh()  # the batch may have moved on since the last tick
        why = self._refusal([sid])
        if why:
            self._say(why)
            self._message("BEARMiND", why)
            return
        rec = self._records.get(sid)
        if rec is not None and actions.gate_passed(rec) \
                and not self._confirm(ALREADY_PASSED_ASK.format(sid=sid)):
            return
        if sid not in self._waiting:
            self._waiting.append(sid)
        if self._deadline is None:  # the first click of this wait
            now = self._clock()
            self._began, self._deadline, self._polled = now, now + START_WAIT_S, now
        self._poll_server()
        if self._waiting:  # no address yet: the timer looks again, the window stays free
            self._say(STARTING)
            if not self.start_timer.isActive():
                self.start_timer.start()

    def show_first_waiting(self) -> None:
        """Monitor's "Manual inspection: N sessions": select the first session in list order that waits for
        manual inspection and can be opened now (CNMF-E current); none: the selection stays."""
        self.refresh()
        sid = next((s for s in self._sids if self._records.get(s) is not None
                    and self._cache.gate_waits(self.paths, self._records[s]) and not self._refusal([s])), None)
        if sid is None or self.selected_sids() == [sid]:
            return
        ix = self.model.index(self._sids.index(sid), 0)
        sel = self.table.selectionModel()
        sel.select(ix, sel.SelectionFlag.ClearAndSelect | sel.SelectionFlag.Rows)
        sel.setCurrentIndex(ix, sel.SelectionFlag.NoUpdate)
        self.table.scrollTo(ix)

    def _poll_server(self) -> None:
        if not self._waiting:
            return
        now = self._clock()
        st = inspect_lock.server_state(self.paths)
        if st.available:
            self._open_all(st)
            return
        if st.stopping:  # an older server finishes its side files: its time does not count against the start
            self._deadline += now - self._polled
        self._polled = now
        # a server another process is starting is waited for, not started again. A stopping one is not: the
        # inspect command waits for it to end, then starts its own (its console log says so).
        if not self._spawned and not st.starting:
            self._spawned = True
            try:
                self._child = _child_key(self._spawn(launcher.inspect_command(self.series)))
            except Exception as e:  # any failure of the start ends the wait; the window names it
                log.error("the inspection server of %s did not start", self.series, exc_info=True)
                self._fail(_spawn_failure(e))
                return
        # the started command ended without a server (and nobody else starts one): its console log says why
        if self._child is not None and not st.starting and not _child_runs(self._child):
            self._fail(DID_NOT_START)
            return
        if now >= self._deadline:
            if now - self._began >= _hard_wait_s():
                self._fail(DID_NOT_START)
                return
            if not self._told:  # its process still runs (a slow disk, a first start): keep waiting
                self._told = True
                self._say(STILL_STARTING)
        self.server_label.setText(SERVER_STOPPING if st.stopping else STILL_STARTING if self._told else STARTING)

    def _open_all(self, st: inspect_lock.ServerState) -> None:
        sids = self._waiting
        self._end_wait()
        opened, failed = [], []
        for sid in sids:
            url = inspect_lock.session_url(st.url, sid)
            # an injected opener may return None; only False means the browser did not open
            (failed if self._open_url(url) is False else opened).append((sid, url))
        self.server_label.setText(server_text(st))
        if opened:
            self._say(OPENED.format(sids=", ".join(sid for sid, _ in opened)))
        if failed:
            text = "\n".join(NO_BROWSER.format(url=url) for _, url in failed)
            self._say(text.splitlines()[0])
            self._message("BEARMiND", text)

    def _fail(self, text: str) -> None:
        self._end_wait()
        self.server_label.setText(server_text(inspect_lock.server_state(self.paths)))
        self._say(text)
        self._message("BEARMiND", text)

    def _end_wait(self) -> None:
        self.start_timer.stop()
        self._waiting = []
        self._began = self._deadline = self._polled = None
        self._spawned, self._child, self._told = False, None, False

    def mark_not_needed(self) -> None:
        """Monitor's "Inspection not needed (selected)": the same question, the same write, the same report."""
        sids = self.selected_sids()
        if not sids:
            self._say("Select sessions in the table first.")
            return
        if not self._confirm(MonitorTab._ask_text("inspection_not_needed", sids, None)):
            return
        try:
            res = actions.gate_not_needed(self.paths, sids, actions.current_user())
        except actions.UnknownSession as e:
            self._message("BEARMiND", f"Not in the series any more: {', '.join(e.unknown)}")
            self.refresh()
            return
        self._say(f"Inspection not needed for {plural(len(res.ok))}: {', '.join(res.ok)}" if res.ok
                  else "Nothing was changed.")
        lines = [f"{sid}: {actions.failure_text(code, detail)}" for sid, (code, detail) in res.failed.items()]
        lines += [WARNING_TEXT.get(code, "{sid}: " + code).format(sid=sid) for sid, code in res.warnings]
        if lines:
            self._message("BEARMiND", "\n".join(lines))
        if res.ok:
            self.decided.emit()
        self.refresh()
