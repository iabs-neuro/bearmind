"""Monitor tab: batch header, stage progress, "Waiting for you" buttons, session table, warnings, log tail.

All numbers come from bearmind.monitor.series_view; this module only draws them. Every change of the series
goes through bearmind.actions; the dialogs (confirm, stage, file, message) are injectable for the tests.
"""
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (QAction, QBrush, QColor, QDesktopServices, QFont, QPainter, QStandardItem,
                           QStandardItemModel)
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QInputDialog,
                               QLabel, QMenu, QMessageBox, QPushButton, QScrollArea, QSplitter, QTableView,
                               QVBoxLayout, QWidget)

from bearmind import actions, config, launcher, monitor
from bearmind.gui.logs import LogView
from bearmind.paths import SeriesPaths
from bearmind.registry import STAGES

log = logging.getLogger("bearmind")

START_GRACE_S = 15.0  # after Start the batch has this long to take its lock and write a heartbeat
SLOW_REFRESH_S = 1.0
CONSOLE_LOG = "see _bearmind/logs/runner_console.log"
DID_NOT_START = f"The batch did not start; {CONSOLE_LOG}"
RAN_AND_STOPPED = "The batch ran and stopped — {}"
NO_FINAL_PULSE = f"it ended without a final heartbeat; {CONSOLE_LOG}"
EXIT_TEXT = {"finished": "finished", "stopped": "stop requested", "interrupted": "interrupted",
             "nothing_to_do": "nothing to do"}
MUTED = "#5d6676"
ERROR_INK = "#c62828"
ACCENT = "#2f5bd3"  # the mockup's primary buttons
# status chip (background, ink), the mockup's palette: Running has the colour of running cells
CHIP = {"Running": ("#bbdefb", "#0d47a1"), "Stopping after current stage": ("#fff3b0", "#6d5300"),
        "Not responding": ("#ffcdd2", "#8e0000"), "Stopped": ("#eef0f3", MUTED)}
ACCENT_BUTTON = (f"QPushButton {{ background: {ACCENT}; color: white; border: 1px solid {ACCENT}; "
                 "border-radius: 6px; padding: 4px 10px; }"
                 "QPushButton:disabled { background: #aebde9; border-color: #aebde9; color: #f4f6fc; }")
WARNINGS_MIN_H = 62  # two warning rows with a Retry button; the panel and the log can be dragged larger
LOG_MIN_H = 52  # about three log lines
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
NOT_SCANNED_TEXT = "Series not scanned. Open the Scan tab and press Scan."
NO_SESSIONS_TEXT = "No sessions recognized in this series. See the Scan tab."

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
    "not_found_stays_excluded": "{sid} was not found by the last scan: it stays excluded. "
                                "Restore the session folder and Rescan.",
    "no_mismatch_error": "{sid} has no frame/timestamp mismatch error; the decision was recorded anyway.",
    "already_confirmed": "{sid}: the stitch is already confirmed; nothing was recorded.",
    "config_unreadable": "{sid}: its config cannot be read, so it was left alone.",
}


def _age(seconds: float) -> str:
    return f"{int(seconds)} s" if seconds < 60 else f"{int(seconds // 60)} min"


def _duration(seconds: float) -> str:
    m = int(seconds // 60)
    if seconds < 60:
        return f"{int(seconds)} s"
    return f"{m} min" if m < 60 else f"{m // 60} h {m % 60} min"


def _eta_duration(seconds: float) -> str:
    if seconds > 48 * 3600:  # "~999 h 33 min" reads worse than days
        h = int(seconds // 3600)
        return f"{h // 24} d {h % 24} h"
    return _duration(seconds)


def _eta_text(eta_s, running: bool = True) -> str:
    if eta_s is None:
        left = "unknown"
    elif eta_s <= 0:
        left = "nothing left"
    else:
        m = round(eta_s / 60)
        left = ("less than 1 min" if m < 1 else f"~{_eta_duration(m * 60)}") + " (time waiting for you not counted)"
    text = f"Estimated time left: {left}"
    return text if running else text + " (batch not running)"


def _exit_text(exit_) -> str:
    """Why the batch stopped, from its final heartbeat ("" when the heartbeat does not say)."""
    if not exit_:
        return ""
    if exit_.startswith("error:"):
        return "runner error: " + exit_[len("error:"):].strip()
    return EXIT_TEXT.get(exit_, exit_)


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
        self.caption.setStyleSheet(f"color: {MUTED};")
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


PREVIEWS_TIP = "MC previews and the CNMF-E QC video, made by the batch for finished sessions"


class _PreviewsBox(QCheckBox):
    """Partly checked only shows that sessions differ; a click always sets the series value (on or off)."""

    def __init__(self, text: str):
        super().__init__(text)
        self.setTristate(True)

    def nextCheckState(self) -> None:
        self.setCheckState(Qt.CheckState.Unchecked if self.checkState() == Qt.CheckState.Checked
                           else Qt.CheckState.Checked)


class _SectionLabel(QLabel):
    """A section header. The capitals come from the font, so a text set later ("Warnings — 3 sessions") is
    drawn like the others, and text() keeps the words as written."""

    STYLE = f"color: {MUTED};"

    def __init__(self, text: str = ""):
        super().__init__(text)
        f = self.font()
        f.setCapitalization(QFont.Capitalization.AllUppercase)
        f.setPixelSize(11)
        f.setWeight(QFont.Weight.DemiBold)
        f.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 1.0)
        self.setFont(f)
        self.setStyleSheet(self.STYLE)  # colour only: a font in a style sheet would replace the one above


def _section_box(label: QLabel, widget: QWidget) -> QWidget:
    box = QWidget()
    lay = QVBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(2)
    lay.addWidget(label)
    lay.addWidget(widget, 1)
    return box


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
        self._clicked_at: datetime | None = None  # wall time of the Start click, to the millisecond
        self._alive = False
        self._scanned = False
        self._cache = monitor.RecordCache()
        self._slow_logged = False
        self._sids: list[str] = []

        self.root_layout = QVBoxLayout(self)
        self.root_layout.setContentsMargins(8, 6, 8, 6)
        self.root_layout.setSpacing(4)
        self._build_header()
        self._build_progress()
        self._build_waiting()
        # sessions, warnings and log share a splitter: the table gets the room, the two panels keep a few lines
        # each and can be dragged larger
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self._build_table()
        self._build_warnings()
        self._build_log()
        for i, stretch in enumerate((6, 1, 1)):
            self.splitter.setStretchFactor(i, stretch)
        self.root_layout.addWidget(self.splitter, 1)
        self._build_actions()

        self.timer = QTimer(self)
        self.timer.setInterval(int(config.load_defaults().get("monitor", {}).get("refresh_ms", 2000)))
        self.timer.timeout.connect(self.refresh)
        self.refresh()
        self.timer.start()

    # --- construction ---

    def _build_header(self) -> None:
        self.root_layout.addWidget(_SectionLabel("Batch"))
        row = QHBoxLayout()
        self.batch_label = QLabel("Stopped")  # a chip: its colours are set with the status
        self.batch_detail = QLabel()
        self.start_button = QPushButton("Start")
        self.stop_button = QPushButton("Stop after current stage")
        self.folder_button = QPushButton("Open folder")
        self.previews_box = _PreviewsBox("Preview videos")
        self.previews_box.setToolTip(PREVIEWS_TIP)
        self.previews_box.clicked.connect(self._on_previews)
        self.start_button.clicked.connect(self._on_start)
        self.stop_button.clicked.connect(self._on_stop)
        self.folder_button.clicked.connect(self._on_folder)
        row.addWidget(self.batch_label)
        row.addWidget(self.batch_detail)
        row.addStretch(1)
        for b in (self.start_button, self.stop_button, self.folder_button, self.previews_box):
            row.addWidget(b)
        self.root_layout.addLayout(row)

    def _build_progress(self) -> None:
        # the legend and the time left share the header line, and the "Waiting for you" label sits left of its
        # buttons: at the default 1100x700 (125 % screen scaling) the lines saved are table rows
        self.progress_title = _SectionLabel("Progress")
        legend = " &nbsp; ".join(f'<span style="color:{c}">&#9632;</span> {t}' for t, c in LEGEND)
        self.legend_label = QLabel(legend)
        self.legend_label.setTextFormat(Qt.TextFormat.RichText)
        self.legend_label.setStyleSheet(f"color: {MUTED}; font-size: 12px;")
        self.eta_label = QLabel()
        head = QHBoxLayout()
        head.addWidget(self.progress_title)
        head.addSpacing(16)
        head.addWidget(self.legend_label)
        head.addStretch(1)
        head.addWidget(self.eta_label)
        self.root_layout.addLayout(head)
        self.bars: dict[str, _StageRow] = {}
        bars = QVBoxLayout()
        bars.setSpacing(2)
        for stage in STAGES:
            self.bars[stage] = _StageRow(monitor.STAGE_TITLE[stage])
            bars.addWidget(self.bars[stage])
        self.root_layout.addLayout(bars)

    def _build_waiting(self) -> None:
        row = QHBoxLayout()
        row.addWidget(_SectionLabel("Waiting for you"))
        row.addSpacing(8)
        self.thresholds_button = QPushButton()
        self.inspection_button = QPushButton()
        for b in (self.thresholds_button, self.inspection_button):
            b.setStyleSheet(ACCENT_BUTTON)
        self.thresholds_button.clicked.connect(lambda: self.open_tab.emit("Thresholds"))
        self.inspection_button.clicked.connect(lambda: self.open_tab.emit("Inspection"))
        row.addWidget(self.thresholds_button)
        row.addWidget(self.inspection_button)
        row.addStretch(1)
        self.root_layout.addLayout(row)

    def _build_table(self) -> None:
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
        self.not_scanned_label = QLabel(NOT_SCANNED_TEXT)
        self.scan_tab_button = QPushButton("Open Scan tab")
        self.scan_tab_button.clicked.connect(lambda: self.open_tab.emit("Scan"))
        box = _section_box(_SectionLabel("Sessions"), self.table)
        box.layout().addWidget(self.not_scanned_label)
        box.layout().addWidget(self.scan_tab_button, 0, Qt.AlignmentFlag.AlignLeft)
        self.splitter.addWidget(box)

    def _build_warnings(self) -> None:
        self.warnings_header = _SectionLabel("Warnings — 0 sessions")
        self.warnings_list = QWidget()
        self._warnings_layout = QVBoxLayout(self.warnings_list)
        self._warnings_layout.setContentsMargins(0, 0, 0, 0)
        self._warnings_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._warnings_sig = None
        self.warnings_scroll = QScrollArea()
        self.warnings_scroll.setWidgetResizable(True)
        self.warnings_scroll.setMinimumHeight(WARNINGS_MIN_H)
        self.warnings_scroll.setWidget(self.warnings_list)
        self.splitter.addWidget(_section_box(self.warnings_header, self.warnings_scroll))

    def _build_log(self) -> None:
        self.log_title = _SectionLabel("")
        self.log_view = LogView()
        self.log_view.setMinimumHeight(LOG_MIN_H)
        self.splitter.addWidget(_section_box(self.log_title, self.log_view))
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
        if not self._scanned:  # every action writes into the folder (export too: it refreshes summary.csv)
            return self._say("Not a scanned series: scan it first.")
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
        plan = actions.retry_plan(self.paths, target) if name == "retry_failed" else None
        if not self._confirm(self._ask_text(name, target, stage, plan)):
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
    def _retry_text(sids, plan) -> str:
        """What a retry resets (plan: sid -> actions.retry_stages): a failed main stage and the later ones, or a
        failed preview alone."""
        main = [s for s in sids if plan.get(s) and plan[s][0] in STAGES]
        alone = [s for s in sids if plan.get(s) and plan[s][0] not in STAGES]
        parts = []
        if main or not alone:  # none failed: the action itself says so
            with_previews = [s for s in main if len(plan[s]) > 1]
            parts.append(f"The failed stage and the later ones of {_named(main or sids)} will be reset and run again"
                         + (f", together with {'its' if len(main) == 1 else 'their'} failed preview videos"
                            if with_previews else "") + ".")
        if alone:
            parts.append(f"Only the failed preview videos of {_named(alone)} will be made again; nothing else is "
                         "reset.")
        return " ".join(parts + ["Old results stay until the new ones are ready."])

    @staticmethod
    def _ask_text(name: str, sids, stage, plan=None) -> str:
        if name == "retry_failed":
            return MonitorTab._retry_text(list(sids), plan or {})
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
        if self._starting_since is not None or self._alive or not self._scanned:
            return
        # before the start: the runner's heartbeat times are written to the millisecond
        now = datetime.now()
        clicked = now.replace(microsecond=now.microsecond // 1000 * 1000)
        try:
            self._start(self.series)
        except RuntimeError as e:  # another batch took the lock between the last refresh and the click
            self.status_message.emit(str(e))
            self.refresh()
            return
        self._starting_since, self._clicked_at = self._clock(), clicked
        self._update_buttons()

    def _check_start(self, r: monitor.RunnerState) -> None:
        """After Start, while the batch is not alive. A heartbeat written since the click means the runner ran
        (it can start and end between two refreshes); with its final heartbeat that is known at once, else
        after START_GRACE_S without a live batch."""
        ran = r.heartbeat_time is not None and r.heartbeat_time >= self._clicked_at
        if ran and r.status == "stopped":
            msg = RAN_AND_STOPPED.format(_exit_text(r.exit) or CONSOLE_LOG)
        elif self._clock() - self._starting_since > START_GRACE_S:
            msg = RAN_AND_STOPPED.format(NO_FINAL_PULSE) if ran else DID_NOT_START
        else:
            return
        self._starting_since = self._clicked_at = None
        self.status_message.emit(msg)

    def _on_stop(self) -> None:
        if not self._alive:
            return
        self._stop(self.paths)
        self.status_message.emit("Stop requested: the batch stops before its next stage.")

    def _on_folder(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.series)))

    def _on_previews(self, _checked: bool) -> None:
        # no confirmation: switching on or off is reversible and deletes nothing
        checked = self.previews_box.checkState() == Qt.CheckState.Checked
        try:
            actions.set_previews(self.paths, checked)
        except (OSError, ValueError) as e:
            self._say(f"Previews were not changed: {e}")
        else:
            text = ("Previews on: the batch makes them for sessions whose MC / CNMF-E is done" if checked
                    else "Previews off: existing files are kept")
            own = [v for v in self._session_previews() if v != checked]
            if own:
                text += f". {len(own)} sessions follow their own mouse config"
            self._say(text)
        self._sync_previews()

    def _session_previews(self) -> list:
        """The effective preview.enabled of each non-excluded session (RecordCache.preview_on: the resolved
        config, a mouse or session config can override the series one); sessions with an unreadable config
        are left out."""
        records, _ = self._cache.records(self.paths)
        sigs = {}  # one stat per config file for all sessions
        vals = [self._cache.preview_on(self.paths, r, sigs) for r in records if not r.excluded]
        return [v for v in vals if v is not None]

    def _sync_previews(self) -> None:
        """Show what the sessions will do: all agree -> that value; they differ -> partly checked."""
        vals = set(self._session_previews())
        if not vals:  # no sessions yet: the series config
            try:
                vals = {bool(config.resolve(self.paths)["preview"]["enabled"])}
            except (OSError, KeyError, TypeError, ValueError):  # a config being rewritten: next refresh
                return
        box = self.previews_box
        if len(vals) > 1:
            box.setCheckState(Qt.CheckState.PartiallyChecked)
            box.setToolTip("Differs between sessions: a mouse config overrides it")
        else:
            box.setChecked(vals.pop())
            box.setToolTip(PREVIEWS_TIP)

    def _update_buttons(self) -> None:
        self.start_button.setEnabled(self._scanned and not self._alive and self._starting_since is None)
        self.stop_button.setEnabled(self._alive)
        for act in getattr(self, "menu_actions", ()):  # the actions are built after the first refresh's widgets
            act.setEnabled(self._scanned)

    def header_text(self) -> str:
        """The batch line as read: "Stopped — finished", "Running LT1_J05_2D · MC · 6 min   heartbeat 4 s ago"."""
        return f"{self.batch_label.text()} {self.batch_detail.text()}".strip()

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
        self._scanned = view.scanned
        self._sync_previews()
        self._show_runner(view)
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

    def _show_runner(self, view: monitor.SeriesView) -> None:
        r = view.runner
        self._alive = r.alive
        detail, red = "", False
        if r.alive:
            self._starting_since = self._clicked_at = None
            name = "Stopping after current stage" if r.stopping else "Running"
            if r.current:
                detail = f"{r.current[0]} · {monitor.STAGE_TITLE.get(r.current[1], r.current[1])}"
                if r.stage_elapsed_s is not None:
                    detail += f" · {_duration(r.stage_elapsed_s)}"
            elif r.status == "waiting":
                # the runner also polls for the core's own waits (crop finalize): "you" only for a person's
                person = view.waiting.get("thresholds", 0) + view.waiting.get("manual_inspection", 0)
                detail = "waiting for you" if person else "waiting"
            if r.age_s is not None:
                detail += f"   heartbeat {_age(max(0.0, r.age_s))} ago"
        elif r.status == "dead":
            name = "Not responding"
            detail = "no heartbeat" if r.age_s is None else f"last heartbeat {_age(max(0.0, r.age_s))} ago"
        else:
            name, why = "Stopped", _exit_text(r.exit)
            if why:
                detail, red = f"— {why}", r.exit.startswith("error:")
        if not r.alive and self._starting_since is not None:
            self._check_start(r)
        bg, ink = CHIP[name]
        self.batch_label.setText(name)
        self.batch_label.setStyleSheet(f"background: {bg}; color: {ink}; border-radius: 10px; padding: 2px 9px; "
                                       "font-weight: 600;")
        self.batch_detail.setText(detail.strip())
        self.batch_detail.setStyleSheet(f"color: {ERROR_INK}; font-weight: 600;" if red else "")
        self._update_buttons()

    def _show_progress(self, view: monitor.SeriesView) -> None:
        total = next(iter(view.bars.values())).total
        self.progress_title.setText(f"Progress — {_plural(total)}")
        for stage, row in self.bars.items():
            row.show_bar(view.bars[stage])
        self.eta_label.setText(_eta_text(view.eta_s, view.runner.alive))

    def _show_waiting(self, waiting: dict[str, int]) -> None:
        for button, label, key in ((self.thresholds_button, "Thresholds", "thresholds"),
                                   (self.inspection_button, "Manual inspection", "manual_inspection")):
            n = waiting.get(key, 0)
            button.setText(f"{label}: {_plural(n)} →")
            button.setEnabled(n > 0)

    def _show_table(self, view: monitor.SeriesView) -> None:
        self.table.setVisible(view.scanned)
        # a scan that recognized nothing lists the folders it could not read in the Scan tab
        self.not_scanned_label.setText(NO_SESSIONS_TEXT if view.found_nothing else NOT_SCANNED_TEXT)
        self.not_scanned_label.setVisible(not view.scanned)
        self.scan_tab_button.setVisible(not view.scanned)
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
        if row.unreadable:
            tip = row.cells[STAGES[0]].tooltip
        elif row.problem is not None:
            tip = f"{row.mouse}\ncannot be scheduled: {row.problem}"
        else:
            tip = row.mouse
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

    def _fit_rows(self) -> None:
        # rows as tall as the text needs (the style's default is taller): more sessions on the screen
        h = self.table.fontMetrics().height() + 4
        self.table.verticalHeader().setMinimumSectionSize(h)  # the style's minimum is taller on Windows
        self.table.verticalHeader().setDefaultSectionSize(h)

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._fit_rows()  # before the tab is on a screen its font is measured at another scaling

    def selected_sids(self) -> list[str]:
        rows = sorted({ix.row() for ix in self.table.selectionModel().selectedIndexes()})
        return [self.model.item(r, 0).text() for r in rows]


# public names for the Scan tab, which shows the same warnings and names sessions the same way
WARNING_TEXT = _WARNING_TEXT
named = _named
plural = _plural
