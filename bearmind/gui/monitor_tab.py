"""Monitor tab, parts 1-4: batch header, stage progress, "Waiting for you" buttons, session table.

All numbers come from bearmind.monitor.series_view; this module only draws them. Parts 5-6 (warnings panel,
log) and the Actions menu are added below the table by the next task: they go into `root_layout`.
"""
import logging
import time
from pathlib import Path

from PySide6.QtCore import QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QPainter, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton, QTableView,
                               QVBoxLayout, QWidget)

from bearmind import config, launcher, monitor
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

    def __init__(self, series: Path, parent=None, start=None, stop=None):
        super().__init__(parent)
        self.series = Path(series)
        self.paths = SeriesPaths(self.series)
        self._start = start or launcher.start_detached
        self._stop = stop or launcher.request_stop
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

    # --- actions ---

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
        t0 = time.perf_counter()
        view = monitor.series_view(self.paths, now, self._cache)
        self._show_runner(view.runner)
        self._show_progress(view)
        self._show_waiting(view.waiting)
        self._show_table(view)
        spent = time.perf_counter() - t0
        if spent > SLOW_REFRESH_S and not self._slow_logged:
            self._slow_logged = True  # once: a slow disk would otherwise fill the log every 2 s
            log.warning("monitor refresh took %.1f s for %d sessions", spent, len(view.rows))

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
