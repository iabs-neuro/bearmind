"""Monitor tab: batch header, stage progress, "Waiting for you" buttons, session table, warnings, log tail.

All numbers come from bearmind.monitor.series_view; this module only draws them. Every change of the series
goes through bearmind.actions; the dialogs (confirm, stage, file, message) are injectable for the tests.
"""
import logging
import sys
import time
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QAction, QBrush, QColor, QDesktopServices, QPainter, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog, QLabel,
                               QMenu, QMessageBox, QPushButton, QScrollArea, QTableView, QVBoxLayout, QWidget)

from bearmind import actions, config, launcher, monitor
from bearmind.gui.logs import LogView
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES

log = logging.getLogger("bearmind")

START_GRACE_S = 15.0  # after Start the batch has this long to take its lock and write a heartbeat
SLOW_REFRESH_S = 1.0
DID_NOT_START = "The batch did not start; see _bearmind/logs/runner_console.log"
STATUS_COLOR = {"Running": "#2e7d32", "Not responding": "#c62828", "Stopped": "#757575"}
# The cell colours are light and the ink dark in both light and dark system themes, so the table reads the same.
CELL_BG = {"ok": "#c8e6c9", "warning": "#fff3b0", "error": "#ffcdd2", "running": "#bbdefb", "waiting": "#e1bee7"}
CELL_INK = "#1d2330"
EXCLUDED_INK = "#8a8f98"
SEGMENTS = (("done", "#43a047"), ("running", "#1e88e5"), ("waiting", "#8e24aa"), ("failed", "#e53935"))
LEGEND = (("done", "#43a047"), ("running", "#1e88e5"), ("waits for you", "#8e24aa"), ("failed", "#e53935"))
BAR_BG = "#eef0f3"
_STYLE_ROLE = Qt.ItemDataRole.UserRole + 1
LEVEL_STYLE = {"error": ("ERROR", "#ffcdd2", "#8e0000"), "warning": ("WARN", "#fff3b0", "#6d5300")}
MAX_NAMED = 12  # sessions named in a confirmation before "and N more"

# the same items serve the Actions menu and the table's context menu
ACTIONS = ("exclude", "include", "accept_mismatch", "rerun_from", "retry_failed", "inspection_not_needed",
           "inspection_not_needed_all", "export_csv")
ACTION_TEXT = {
    "exclude": "Exclude", "include": "Include", "accept_mismatch": "Accept frame mismatch",
    "rerun_from": "Rerun from stage…", "retry_failed": "Retry failed (selected or all)",
    "inspection_not_needed": "Inspection not needed (selected)",
    "inspection_not_needed_all": "Inspection not needed for all waiting", "export_csv": "Export summary CSV…",
}
_WARNING_TEXT = {
    "excluded_session": "{sid} is excluded: the runner ignores it until you include it.",
    "no_fps_stays_excluded": "{sid} has no frame rate (timestamps missing or unusable): it stays excluded.",
    "no_mismatch_error": "{sid} has no frame/timestamp mismatch error; the decision was recorded anyway.",
}


def _age(seconds: float) -> str:
    return f"{int(seconds)} s" if seconds < 60 else f"{int(seconds // 60)} min"


def _duration(seconds: float) -> str:
    m = int(seconds // 60)
    if seconds < 60:
        return f"{int(seconds)} s"
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60} min"


def _eta_text(eta_s) -> str:
    if eta_s is None:
        return "Estimated time left: unknown"
    m = round(eta_s / 60)
    left = "less than 1 min" if m < 1 else f"~{_duration(m * 60)}"
    return f"Estimated time left: {left} (time waiting for you not counted)"


def _plural(n: int) -> str:
    return f"{n} session" + ("" if n == 1 else "s")


def _named(sids) -> str:
    sids = list(sids)
    shown = ", ".join(sids[:MAX_NAMED]) + (f" and {len(sids) - MAX_NAMED} more" if len(sids) > MAX_NAMED else "")
    return f"{'Session' if len(sids) == 1 else 'Sessions'} {shown}"


class _SegmentBar(QWidget):
    """One stage: coloured segments proportional to the session counts."""

    def __init__(self):
        super().__init__()
        self.counts: dict[str, int] = {}
        self.total = 0
        self.setFixedHeight(14)
        self.setMinimumWidth(160)

    def set_counts(self, counts: dict[str, int], total: int) -> None:
        if (counts, total) != (self.counts, self.total):
            self.counts, self.total = counts, total
            self.update()

    def paintEvent(self, _event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QColor(BAR_BG))
        p.drawRoundedRect(QRectF(0, 0, w, h), h / 2, h / 2)
        if self.total:
            x = 0.0
            clip = QRectF(0, 0, w, h)
            p.setClipRect(clip)
            for key, color in SEGMENTS:
                seg = w * self.counts.get(key, 0) / self.total
                if seg > 0:
                    p.setBrush(QColor(color))
                    p.drawRect(QRectF(x, 0, seg, h))
                    x += seg
        p.end()


class _StageRow(QWidget):
    def __init__(self, title: str):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        name = QLabel(title)
        name.setFixedWidth(96)
        self.segments = _SegmentBar()
        self.caption = QLabel()
        self.caption.setFixedWidth(150)
        self.caption.setStyleSheet("color: #5d6676;")
        lay.addWidget(name)
        lay.addWidget(self.segments, 1)
        lay.addWidget(self.caption)

    def show_bar(self, b: monitor.StageBar) -> None:
        self.segments.set_counts({"done": b.done, "running": b.running, "waiting": b.waiting,
                                  "failed": b.failed}, b.total)
        parts = [f"{b.done} done"] if b.done or not (b.running or b.waiting or b.failed) else []
        if b.running:
            parts.append(f"{b.running} running")
        if b.waiting:
            parts.append(f"{b.waiting} wait")
        if b.failed:
            parts.append(f"{b.failed} failed")
        self.caption.setText(" · ".join(parts))


def _section_label(text: str) -> QLabel:
    lab = QLabel(text.upper())
    lab.setStyleSheet("color: #5d6676; font-size: 11px; font-weight: 600; letter-spacing: 1px;")
    return lab


class MonitorTab(QWidget):
    open_tab = Signal(str)
    status_message = Signal(str)

    def __init__(self, series: Path, parent=None, start=None, stop=None, confirm=None, choose_stage=None,
                 choose_file=None, message=None, report=None):
        super().__init__(parent)
        self.series = Path(series)
        self.paths = SeriesPaths(self.series)
        self._start = start or launcher.start_detached
        self._stop = stop or launcher.request_stop
        self._confirm = confirm or self._ask_confirm
        self._choose_stage = choose_stage or self._ask_stage
        self._choose_file = choose_file or self._ask_file
        self._message = message or self._show_message
        # the window installs its exception report as sys.excepthook
        self._report = report or (lambda *exc: sys.excepthook(*exc))
        self._refresh_failed = None  # (type, message) of the failure reported last, until a refresh succeeds
        self.last_status = ""
        self._clock = time.monotonic
        self._starting_since: float | None = None
        self._alive = False
        self._cache = monitor.RecordCache()
        self._slow_logged = False
        self._sids: list[str] = []

        self.root_layout = QVBoxLayout(self)
        self._build_header()
        self._build_progress()
        self._build_waiting()
        self._build_table()
        self._build_warnings()
        self._build_log()
        self._build_actions()

        self.timer = QTimer(self)
        self.timer.setInterval(int(config.load_defaults().get("monitor", {}).get("refresh_ms", 2000)))
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        self.timer.start()

    # --- construction ---

    def _build_header(self) -> None:
        self.root_layout.addWidget(_section_label("Batch"))
        row = QHBoxLayout()
        self.batch_label = QLabel("Stopped")
        self.batch_detail = QLabel()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop after current stage")
        self.folder_button = QPushButton("Open folder")
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        self.folder_button.clicked.connect(self._on_folder)
        row.addWidget(self.batch_label)
        row.addWidget(self.batch_detail)
        row.addStretch(1)
        for b in (self.start_button, self.stop_button, self.folder_button):
            row.addWidget(b)
        self.root_layout.addLayout(row)

    def _build_progress(self) -> None:
        self.progress_title = _section_label("Progress")
        self.root_layout.addWidget(self.progress_title)
        self.bars: dict[str, _StageRow] = {}
        for stage in STAGES:
            self.bars[stage] = _StageRow(monitor.STAGE_TITLE[stage])
            self.root_layout.addWidget(self.bars[stage])
        legend = " &nbsp; ".join(f'<span style="color:{c}">&#9632;</span> {t}' for t, c in LEGEND)
        self.legend_label = QLabel(legend)
        self.legend_label.setTextFormat(Qt.TextFormat.RichText)
        self.legend_label.setStyleSheet("color: #5d6676; font-size: 12px;")
        self.eta_label = QLabel()
        self.root_layout.addWidget(self.legend_label)
        self.root_layout.addWidget(self.eta_label)

    def _build_waiting(self) -> None:
        self.root_layout.addWidget(_section_label("Waiting for you"))
        row = QHBoxLayout()
        self.thresholds_button = QPushButton()
        self.inspection_button = QPushButton()
        self.thresholds_button.clicked.connect(lambda: self.open_tab.emit("Thresholds"))
        self.inspection_button.clicked.connect(lambda: self.open_tab.emit("Inspection"))
        row.addWidget(self.thresholds_button)
        row.addWidget(self.inspection_button)
        row.addStretch(1)
        self.root_layout.addLayout(row)

    def _build_table(self) -> None:
        self.root_layout.addWidget(_section_label("Sessions"))
        self.model = QStandardItemModel(0, 1 + len(STAGES), self)
        self.model.setHorizontalHeaderLabels(["Session"] + [monitor.STAGE_TITLE[s] for s in STAGES])
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.not_scanned_label = QLabel(f'Series not scanned. Run: run_bearmind.bat scan "{self.series}"')
        self.not_scanned_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.root_layout.addWidget(self.table, 1)
        self.root_layout.addWidget(self.not_scanned_label)

    def _build_warnings(self) -> None:
        self.warnings_header = _section_label("")
        self.warnings_header.setText("Warnings — 0 sessions")
        self.warnings_list = QWidget()
        self._warnings_layout = QVBoxLayout(self.warnings_list)
        self._warnings_layout.setContentsMargins(0, 0, 0, 0)
        self._warnings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._warnings_sig = None
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(140)
        scroll.setWidget(self.warnings_list)
        self.root_layout.addWidget(self.warnings_header)
        self.root_layout.addWidget(scroll)

    def _build_log(self) -> None:
        self.log_title = _section_label("")
        self.log_view = LogView()
        self.log_view.setMinimumHeight(110)
        self.log_view.setMaximumHeight(200)
        self.root_layout.addWidget(self.log_title)
        self.root_layout.addWidget(self.log_view)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._show_log_of_selection())
        self._show_log_of_selection()

    def _build_actions(self) -> None:
        self.menu_actions: list[QAction] = []
        for name in ACTIONS:
            act = QAction(ACTION_TEXT[name], self)
            act.triggered.connect(lambda _=False, n=name: self._run_guarded(n))
            self.menu_actions.append(act)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._context_menu)

    def _context_menu(self, pos) -> None:
        menu = QMenu(self.table)
        menu.addActions(self.menu_actions)
        menu.exec(self.table.viewport().mapToGlobal(pos))

    # --- dialogs (replaced in the tests) ---

    def _ask_confirm(self, text: str) -> bool:
        return QMessageBox.question(self, "BEARMiND", text) == QMessageBox.StandardButton.Yes

    def _ask_stage(self):
        titles = [monitor.STAGE_TITLE[s] for s in STAGES]
        title, ok = QInputDialog.getItem(self, "Rerun from stage", "Recompute from stage:", titles, 0, False)
        return STAGES[titles.index(title)] if ok else None

    def _ask_file(self, default: Path):
        name, _ = QFileDialog.getSaveFileName(self, "Export summary CSV", str(default), "CSV (*.csv)")
        return Path(name) if name else None

    def _show_message(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _say(self, text: str) -> None:
        self.last_status = text
        self.status_message.emit(text)

    # --- actions ---

    def _run_guarded(self, name: str) -> None:
        try:
            self.run_action(name)
        except Exception:  # a slot must not close the window; the window reports it
            self._report(*sys.exc_info())

    def run_action(self, name: str) -> None:
        if name not in ACTIONS:
            raise ValueError(f"unknown action {name!r}")
        if name == "export_csv":
            return self._export_csv()
        who = actions.current_user()
        selected = self.selected_sids()
        if name == "retry_failed":
            sids = selected if selected else actions.failed_sids(self.paths)
            if not sids:
                return self._say("No failed sessions.")
        elif name == "inspection_not_needed_all":
            sids = actions.waiting_gate_sids(self.paths)
            if not sids:
                return self._say("No sessions are waiting for manual inspection.")
        else:
            sids = selected
            if not sids:
                return self._say("Select sessions in the table first.")
        # the sessions named in the confirmation are the ones written, even if the batch moves on meanwhile
        target = sids
        stage = None
        if name == "rerun_from":
            stage = self._choose_stage()
            if stage is None:
                return
        if not self._confirm(self._ask_text(name, target, stage)):
            return
        try:
            res = {
                "exclude": lambda: actions.set_excluded(self.paths, sids, True, who),
                "include": lambda: actions.set_excluded(self.paths, sids, False, who),
                "accept_mismatch": lambda: actions.accept_mismatch(self.paths, sids, who),
                "rerun_from": lambda: actions.rerun_from(self.paths, sids, stage, who),
                "retry_failed": lambda: actions.retry_failed(self.paths, sids, who),
                "inspection_not_needed": lambda: actions.gate_not_needed(self.paths, sids, who),
                "inspection_not_needed_all": lambda: actions.gate_not_needed(self.paths, sids, who),
            }[name]()
        except actions.UnknownSession as e:
            self._message("BEARMiND", f"Not in the series any more: {', '.join(e.unknown)}")
            return
        self._report_result(name, res, stage)
        self.refresh()

    @staticmethod
    def _ask_text(name: str, sids, stage) -> str:
        who = _named(sids)
        one = len(list(sids)) == 1
        it, they, is_are = ("it", "it", "is") if one else ("them", "they", "are")
        title = monitor.STAGE_TITLE.get(stage, stage)
        return {
            "exclude": f"{who} will be excluded: the runner skips {it} until you include {it} again.",
            "include": f"{who} will be included in the batch again.",
            "accept_mismatch": f"The frame/timestamp mismatch of {who} will be accepted: "
                               f"{'the session is' if one else 'the sessions are'} used as {they} {is_are}.",
            "rerun_from": f"{who} will be recomputed from {title}. Old results stay until the new ones are ready.",
            "retry_failed": f"The failed stage and the later ones of {who} will be reset and run again. "
                            "Old results stay until the new ones are ready.",
            "inspection_not_needed": f"Manual inspection will be marked as not needed for {who}.",
            "inspection_not_needed_all": f"Manual inspection will be marked as not needed for {who}.",
        }[name]

    def _report_result(self, name: str, res, stage) -> None:
        done = {
            "exclude": "Excluded", "include": "Included", "accept_mismatch": "Accepted the frame mismatch of",
            "rerun_from": f"Reset from {monitor.STAGE_TITLE.get(stage, stage)}:",
            "retry_failed": "Reset failed stages of", "inspection_not_needed": "Inspection not needed for",
            "inspection_not_needed_all": "Inspection not needed for",
        }[name]
        if res.ok:
            self._say(f"{done} {_plural(len(res.ok))}: {', '.join(res.ok)}")
        else:
            self._say("Nothing was changed.")
        self._show_problems(res)

    def _show_problems(self, res) -> None:
        lines = [f"{sid}: {actions.failure_text(code, detail)}" for sid, (code, detail) in res.failed.items()]
        lines += [_WARNING_TEXT.get(code, "{sid}: " + code).format(sid=sid) for sid, code in res.warnings]
        if lines:
            self._message("BEARMiND", "\n".join(lines))

    def _export_csv(self) -> None:
        default = self.series / f"{self.series.name}_summary.csv"
        dest = self._choose_file(default)
        if dest is None:
            return
        # the save dialog (with its own overwrite prompt) is the confirmation
        try:
            actions.export_summary(self.paths, Path(dest))
        except OSError as e:  # typically the file is open in Excel
            self._message("BEARMiND", f"The summary was not written: {e}")
            return
        self._say(f"Summary saved: {dest}")

    def _retry(self, sid: str, stage: str) -> None:
        try:
            res = actions.retry_failed(self.paths, [sid], actions.current_user())
        except actions.UnknownSession:
            self._say(f"{sid} is not in the series any more.")
            self.refresh()
            return
        if res.ok:
            text = f"{sid}: reset from {monitor.STAGE_TITLE.get(stage, stage)}"
            self._say(text if self._alive else text + ". Start the batch to run it")
        else:
            self._say("Nothing was changed.")
        self._show_problems(res)
        self.refresh()

    def _on_start(self) -> None:
        if self._starting_since is not None or self._alive:
            return
        try:
            self._start(self.series)
        except RuntimeError as e:  # another batch took the lock between the last refresh and the click
            self.status_message.emit(str(e))
            self.refresh()
            return
        self._starting_since = self._clock()
        self._update_buttons()

    def _on_stop(self) -> None:
        if not self._alive:
            return
        self._stop(self.paths)
        self.status_message.emit("Stop requested: the batch stops before its next stage.")

    def _on_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.series)))

    def _update_buttons(self) -> None:
        self.start_button.setEnabled(not self._alive and self._starting_since is None)
        self.stop_button.setEnabled(self._alive)

    # --- refresh ---

    def refresh(self, now=None) -> None:
        # a failure repeats every tick: the first one goes to the window's report (a dialog), the rest only
        # to the log and the status bar, so the timer does not stack modal dialogs
        try:
            self._refresh(now)
        except Exception as e:
            first = (str(e).splitlines() or [""])[0]
            kind = (type(e), str(e))
            if self._refresh_failed is None:
                self._refresh_failed = kind
                self._report(type(e), e, e.__traceback__)
            else:
                if kind == self._refresh_failed:
                    log.warning("monitor refresh failed again: %s", first)
                else:  # a different failure: keep its traceback in the log, still no dialog
                    self._refresh_failed = kind
                    log.error("monitor refresh failed differently", exc_info=(type(e), e, e.__traceback__))
                self._say(f"Refresh failed: {type(e).__name__}: {first}")
        else:
            self._refresh_failed = None

    def _refresh(self, now=None) -> None:
        t0 = time.perf_counter()
        view = monitor.series_view(self.paths, now, self._cache)
        self._show_runner(view.runner)
        self._show_progress(view)
        self._show_waiting(view.waiting)
        self._show_table(view)
        self._show_warnings(view)
        self.log_view.poll()
        spent = time.perf_counter() - t0
        if spent > SLOW_REFRESH_S and not self._slow_logged:
            self._slow_logged = True  # once: a slow disk would otherwise fill the log every 2 s
            log.warning("monitor refresh took %.1f s for %d sessions", spent, len(view.rows))

    def _show_warnings(self, view: monitor.SeriesView) -> None:
        sig = (view.n_warning_sessions, tuple((w.level, w.kind, w.sid, w.stage, w.text, w.retry)
                                              for w in view.warnings))
        if sig == self._warnings_sig:  # rebuilding would flicker and could swallow a click on Retry
            return
        self._warnings_sig = sig
        self.warnings_header.setText(f"Warnings — {_plural(view.n_warning_sessions)}")
        lay = self._warnings_layout
        while lay.count():
            w = lay.takeAt(0).widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()
        if not view.warnings:
            lay.addWidget(QLabel("No warnings"))
        for w in view.warnings:
            lay.addWidget(self._warning_row(w))

    def _warning_row(self, w: monitor.WarningItem) -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 2, 0, 2)
        word, bg, ink = LEVEL_STYLE[w.level]
        level = QLabel(word)
        level.setFixedWidth(52)
        level.setAlignment(Qt.AlignmentFlag.AlignCenter)
        level.setStyleSheet(f"background: {bg}; color: {ink}; font-size: 11px; font-weight: 600; "
                            "border-radius: 3px; padding: 1px 0;")
        where = w.sid + (f" · {monitor.STAGE_TITLE.get(w.stage, w.stage)}" if w.stage else "")
        title = QLabel(where)
        title.setStyleSheet("font-weight: 600;")
        text = QLabel(w.text)
        text.setWordWrap(True)
        text.setTextFormat(Qt.TextFormat.PlainText)
        h.addWidget(level, 0, Qt.AlignmentFlag.AlignTop)
        h.addWidget(title, 0, Qt.AlignmentFlag.AlignTop)
        h.addWidget(text, 1)
        if w.retry:
            btn = QPushButton("Retry")
            btn.clicked.connect(lambda _=False, sid=w.sid, stage=w.stage: self._retry(sid, stage))
            h.addWidget(btn, 0, Qt.AlignmentFlag.AlignTop)
        return row

    def _show_log_of_selection(self) -> None:
        sids = self.selected_sids()
        sid = sids[0] if sids and sids[0] else None
        path = self.paths.session_log(sid) if sid else self.paths.series_log
        self.log_title.setText(f"Log — {sid or path.name}")
        if path != self.log_view.path:
            self.log_view.show_log(path)

    def _show_runner(self, r: monitor.RunnerState) -> None:
        self._alive = r.alive
        if r.alive:
            self._starting_since = None
            name, detail = "Running", ""
            if r.current:
                detail = f"{r.current[0]} · {monitor.STAGE_TITLE.get(r.current[1], r.current[1])}"
                if r.stage_elapsed_s is not None:
                    detail += f" · {_duration(r.stage_elapsed_s)}"
            elif r.status == "waiting":
                detail = "waiting for you"
            if r.age_s is not None:
                detail += f"   heartbeat {_age(max(0.0, r.age_s))} ago"
        elif r.status == "dead":
            name = "Not responding"
            detail = "no heartbeat" if r.age_s is None else f"last heartbeat {_age(max(0.0, r.age_s))} ago"
        else:
            name, detail = "Stopped", ""
        if not r.alive and self._starting_since is not None and self._clock() - self._starting_since > START_GRACE_S:
            self._starting_since = None
            self.status_message.emit(DID_NOT_START)
        self.batch_label.setText(name)
        self.batch_label.setStyleSheet(f"color: {STATUS_COLOR[name]}; font-weight: 600;")
        self.batch_detail.setText(detail.strip())
        self._update_buttons()

    def _show_progress(self, view: monitor.SeriesView) -> None:
        total = next(iter(view.bars.values())).total
        self.progress_title.setText(f"PROGRESS — {total} SESSIONS")
        for stage, row in self.bars.items():
            row.show_bar(view.bars[stage])
        self.eta_label.setText(_eta_text(view.eta_s))

    def _show_waiting(self, waiting: dict[str, int]) -> None:
        for button, label, key in ((self.thresholds_button, "Thresholds", "thresholds"),
                                   (self.inspection_button, "Manual inspection", "manual_inspection")):
            n = waiting.get(key, 0)
            button.setText(f"{label}: {_plural(n)} →")
            button.setEnabled(n > 0)

    def _show_table(self, view: monitor.SeriesView) -> None:
        self.table.setVisible(view.scanned)
        self.not_scanned_label.setVisible(not view.scanned)
        sids = [r.sid for r in view.rows]
        if sids != self._sids:
            self._rebuild(view.rows)
            self._sids = sids
        for i, row in enumerate(view.rows):
            self._fill_row(i, row)
        self._show_log_of_selection()

    def _rebuild(self, rows: list[monitor.Row]) -> None:
        selected = set(self.selected_sids())
        scroll = self.table.verticalScrollBar().value()
        self.model.setRowCount(0)
        self.model.setRowCount(len(rows))
        for i in range(len(rows)):
            for c in range(self.model.columnCount()):
                item = QStandardItem()
                item.setEditable(False)
                self.model.setItem(i, c, item)
        sel = self.table.selectionModel()
        for i, r in enumerate(rows):
            if r.sid in selected:
                sel.select(self.model.index(i, 0), sel.SelectionFlag.Select | sel.SelectionFlag.Rows)
        self.table.verticalScrollBar().setValue(scroll)

    def _fill_row(self, i: int, row: monitor.Row) -> None:
        first = self.model.item(i, 0)
        tip = row.cells[STAGES[0]].tooltip if row.unreadable else row.mouse
        style = f"first|{row.excluded}"
        if first.text() != row.sid:
            first.setText(row.sid)
        if first.toolTip() != tip:
            first.setToolTip(tip)
        if first.data(_STYLE_ROLE) != style:
            first.setData(style, _STYLE_ROLE)
            first.setForeground(QBrush(QColor(EXCLUDED_INK)) if row.excluded else QBrush())
        for c, stage in enumerate(STAGES, start=1):
            cell = row.cells[stage]
            item = self.model.item(i, c)
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
