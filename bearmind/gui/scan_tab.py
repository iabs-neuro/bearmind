"""Scan tab: scan the series in a worker thread, show every session with all its anomalies, record decisions.

The model is bearmind.scan_view (read-only); every change goes through bearmind.actions. The tab never writes
into 1_Raw or 0_TimeStamps and opens folders in Explorer only. The dialogs are injectable for the tests.
"""
import html
import logging
import os
import sys
import threading
import traceback
from pathlib import Path

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QListWidget, QListWidgetItem,
                               QMessageBox, QProgressBar, QPushButton, QSplitter, QTableView, QTextBrowser,
                               QVBoxLayout, QWidget)

from bearmind import actions, monitor, scan_view
from bearmind.gui.monitor_tab import (ACCENT_BUTTON, CELL_BG, CELL_INK, EXCLUDED_INK, MonitorTab, _WARNING_TEXT,
                                      _named, _plural)
from bearmind.paths import SeriesPaths

log = logging.getLogger("bearmind")

COLUMNS = ("Session", "Mode", "Parts", "Chunks", "Video frames", "Frames to process", "FPS", "Min", "Anomalies",
           "Excluded")
NOT_SCANNED = "Not scanned yet. Press Scan."
ASK_RESCAN = "Batch is running. If session files changed, their stages will be recomputed. Rescan?"
EXCLUDED_TEXT = {"anomaly": "by anomaly", "you": "by you", "": ""}
LEVEL_INK = {"error": ("ERROR", "#8e0000"), "warning": ("WARN", "#6d5300")}
_DONE_TEXT = {
    "exclude": "Excluded", "include": "Included", "accept_mismatch": "Accepted the frame mismatch of",
    "confirm_stitch": "Confirmed the stitch of",
}


def _count(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _num(x, fmt: str) -> str:
    return "–" if x is None else format(x, fmt)


class _Bridge(QObject):
    """Carries the worker thread's messages to the window's thread (a signal emitted from another thread
    reaches a slot of this thread through the event queue)."""
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str, str)


class ScanTab(QWidget):
    scan_finished = Signal(bool)  # True: the scan finished and wrote the registry
    status_message = Signal(str)

    def __init__(self, series: Path, parent=None, confirm=None, message=None, open_folder=None):
        super().__init__(parent)
        self.series = Path(series)
        self.paths = SeriesPaths(self.series)
        self._confirm = confirm or self._ask_confirm
        self._message = message or self._show_message
        self._open_folder = open_folder or (lambda p: os.startfile(str(p)))
        self._scanning = False
        self._view: scan_view.ScanView | None = None
        self._rows: dict[str, scan_view.ScanRow] = {}
        self._bridge = _Bridge(self)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.done.connect(self._on_done)
        self._bridge.failed.connect(self._on_failed)
        self._build()
        self.refresh()

    # --- construction ---

    def _build(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(4)
        top = QHBoxLayout()
        self.scan_button = QPushButton("Scan")
        self.scan_button.setStyleSheet(ACCENT_BUTTON)
        self.scan_button.clicked.connect(self.start_scan)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label = QLabel("")
        top.addWidget(self.scan_button)
        top.addWidget(self.progress_bar, 1)
        top.addWidget(self.progress_label)
        lay.addLayout(top)

        self.summary_label = QLabel(NOT_SCANNED)
        lay.addWidget(self.summary_label)

        self.model = QStandardItemModel(0, len(COLUMNS), self)
        self.model.setHorizontalHeaderLabels(list(COLUMNS))
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(8, QHeaderView.ResizeMode.Stretch)
        self.table.selectionModel().selectionChanged.connect(lambda *_: self._on_selection())

        self.details = QTextBrowser()
        split = QSplitter(Qt.Orientation.Vertical)
        split.setChildrenCollapsible(False)
        split.addWidget(self.table)
        split.addWidget(self.details)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        lay.addWidget(split, 1)

        row = QHBoxLayout()
        self.open_folder_button = QPushButton("Open folder")
        self.exclude_button = QPushButton("Exclude")
        self.include_button = QPushButton("Include")
        self.accept_button = QPushButton("Accept with trimming to common length")
        self.stitch_button = QPushButton("Confirm stitch")
        self.open_folder_button.clicked.connect(self._on_open_folder)
        for button, name in ((self.exclude_button, "exclude"), (self.include_button, "include"),
                             (self.accept_button, "accept_mismatch"), (self.stitch_button, "confirm_stitch")):
            button.clicked.connect(lambda _=False, n=name: self._run_guarded(n))
        for b in (self.open_folder_button, self.exclude_button, self.include_button, self.accept_button,
                  self.stitch_button):
            row.addWidget(b)
        row.addStretch(1)
        lay.addLayout(row)

        self.unrecognized_header = QLabel("Unrecognized folders")
        self.unrecognized_list = QListWidget()
        self.unrecognized_list.setMaximumHeight(90)
        self.unrecognized_list.itemDoubleClicked.connect(self._on_unrecognized)
        lay.addWidget(self.unrecognized_header)
        lay.addWidget(self.unrecognized_list)
        self._update_buttons()

    # --- dialogs (replaced in the tests) ---

    def _ask_confirm(self, text: str) -> bool:
        return QMessageBox.question(self, "BEARMiND", text) == QMessageBox.StandardButton.Yes

    def _show_message(self, title: str, text: str) -> None:
        QMessageBox.warning(self, title, text)

    def _say(self, text: str) -> None:
        self.status_message.emit(text)

    # --- scan ---

    def is_scanning(self) -> bool:
        return self._scanning

    def start_scan(self) -> None:
        if self._scanning:
            return
        if monitor.runner_state(self.paths).alive and not self._confirm(ASK_RESCAN):
            return
        self._scanning = True
        self.scan_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # busy until the first total arrives
        self.progress_label.setText("Scanning …")
        self._update_buttons()
        threading.Thread(target=self._work, name="bearmind-scan", daemon=True).start()

    def _work(self) -> None:
        bridge = self._bridge
        try:
            view = actions.rescan(self.paths, lambda i, n, sid: bridge.progress.emit(i, n, sid))
            bridge.done.emit(view)
        except Exception as e:  # shown by the tab; the traceback goes to the log
            log.error("scan failed:\n%s", traceback.format_exc())
            first = (str(e).splitlines() or [""])[0]
            bridge.failed.emit(type(e).__name__, first)

    def _on_progress(self, i: int, n: int, sid: str) -> None:
        self.progress_bar.setRange(0, max(n, 1))
        self.progress_bar.setValue(i)
        self.progress_label.setText(f"Scanning {sid} … {i} / {n}" if sid else f"Scanning … {i} / {n}")

    def _end_scan(self) -> None:
        self._scanning = False
        self.scan_button.setEnabled(True)

    def _on_done(self, view) -> None:
        self._end_scan()
        n = len(view.rows)
        self.progress_bar.setRange(0, max(n, 1))
        self.progress_bar.setValue(n)
        self.progress_label.setText(f"Scan finished — {n} / {n}")
        self._show(view)
        self._say(f"Scanned {_plural(n)}")
        self.scan_finished.emit(True)

    def _on_failed(self, kind: str, first: str) -> None:
        self._end_scan()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Scan failed")
        self._update_buttons()
        self._message("Scan failed", f"{kind}: {first}")
        self.scan_finished.emit(False)

    # --- showing the model ---

    def refresh(self) -> None:
        self._show(scan_view.scan_view(self.paths))

    def _show(self, view: scan_view.ScanView) -> None:
        keep = set(self.selected_sids())
        scroll = self.table.verticalScrollBar().value()
        self._view = view
        self._rows = {r.sid: r for r in view.rows}
        self.scan_button.setText("Rescan" if view.scanned else "Scan")
        self.summary_label.setText(view.summary_text() if view.scanned else NOT_SCANNED)
        self.table.selectionModel().blockSignals(True)
        self.model.setRowCount(0)
        self.model.setRowCount(len(view.rows))
        for i, r in enumerate(view.rows):
            for c, item in enumerate(self._items(r)):
                self.model.setItem(i, c, item)
        sel = self.table.selectionModel()
        for i, r in enumerate(view.rows):
            if r.sid in keep:
                sel.select(self.model.index(i, 0), sel.SelectionFlag.Select | sel.SelectionFlag.Rows)
        sel.blockSignals(False)
        self.table.verticalScrollBar().setValue(scroll)
        self._show_unrecognized(view)
        self._on_selection()

    @staticmethod
    def _items(r: scan_view.ScanRow) -> list[QStandardItem]:
        texts = [r.sid, r.mode, str(r.n_parts), str(r.n_chunks), _count(r.n_video_frames), _count(r.n_frames),
                 _num(r.fps, ".1f"), _num(r.duration_min, ".1f"), r.summary, EXCLUDED_TEXT.get(r.excluded_by, "")]
        if r.fps is None:  # no frame rate: no frames to process either
            texts[5] = "–"
        items = []
        for c, text in enumerate(texts):
            item = QStandardItem(text)
            item.setEditable(False)
            if c == 0:
                bg = CELL_BG.get(r.level)
                if bg:
                    item.setBackground(QBrush(QColor(bg)))
                    item.setForeground(QBrush(QColor(CELL_INK)))
            if r.excluded:
                item.setForeground(QBrush(QColor(EXCLUDED_INK)))
            items.append(item)
        return items

    def _show_unrecognized(self, view: scan_view.ScanView) -> None:
        self.unrecognized_list.clear()
        for path, reason in view.unrecognized:
            item = QListWidgetItem(f"{path} — {reason}")
            item.setData(Qt.ItemDataRole.UserRole, str(self.series / path))
            self.unrecognized_list.addItem(item)
        for path, reason in view.broken:
            item = QListWidgetItem(f"{path} — {reason}")
            item.setData(Qt.ItemDataRole.UserRole, str(Path(path).parent))
            self.unrecognized_list.addItem(item)
        shown = self.unrecognized_list.count() > 0
        self.unrecognized_header.setVisible(shown)
        self.unrecognized_list.setVisible(shown)

    def _on_unrecognized(self, item: QListWidgetItem) -> None:
        folder = item.data(Qt.ItemDataRole.UserRole)
        if folder:
            self._open_folder(Path(folder))

    # --- selection and buttons ---

    def selected_sids(self) -> list[str]:
        rows = sorted({ix.row() for ix in self.table.selectionModel().selectedIndexes()})
        return [self.model.item(r, 0).text() for r in rows]

    def _on_selection(self) -> None:
        self._show_details()
        self._update_buttons()

    def _selected_rows(self) -> list[scan_view.ScanRow]:
        return [self._rows[s] for s in self.selected_sids() if s in self._rows]

    def _update_buttons(self) -> None:
        rows = self._selected_rows()
        free = bool(rows) and not self._scanning  # a decision during a scan would race the registry rewrite
        self.open_folder_button.setEnabled(len(rows) == 1 and bool(rows[0].folders))
        self.exclude_button.setEnabled(free and any(not r.excluded for r in rows))
        self.include_button.setEnabled(free and any(r.excluded for r in rows))
        self.accept_button.setEnabled(free and all(
            any(a.code == "frames_ts_mismatch" and a.level == "error" for a in r.anomalies) for r in rows))
        self.stitch_button.setEnabled(free and all(
            any(a.code == "split_recording" and a.level == "warning" and not a.reviewed for a in r.anomalies)
            for r in rows))

    def _show_details(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1:
            self.details.setPlainText("Select one session to see its details." if not rows
                                      else f"{len(rows)} sessions selected.")
            return
        r = rows[0]
        out = [f"<b>Details — {html.escape(r.sid)}</b>"]
        for a in r.anomalies:
            if a.auto_fixed:
                out.append(f'<span style="color:#2e7d32">FIXED</span> {html.escape(a.text)} (fixed automatically)')
                continue
            tag, ink = LEVEL_INK.get(a.level, ("WARN", LEVEL_INK["warning"][1]))
            line = f'<span style="color:{ink}"><b>{tag}</b></span> {html.escape(a.text)}'
            out.append(line + (" (checked by you)" if a.reviewed else ""))
        out.append(f"Part folders: {html.escape(', '.join(r.folders) or '–')}")
        out.append(f"FPS: {html.escape(r.fps_text)}")
        out.append("Decisions: none yet" if not r.decisions
                   else "Decisions:<br>" + "<br>".join(html.escape(d) for d in r.decisions))
        self.details.setHtml("<br>".join(out))

    # --- actions ---

    def _on_open_folder(self) -> None:
        rows = self._selected_rows()
        if len(rows) == 1 and rows[0].folders:
            self._open_folder(self.series / rows[0].folders[0])

    @staticmethod
    def _ask_text(name: str, sids) -> str:
        if name == "confirm_stitch":
            who = _named(sids)
            return f"The stitch of {who} will be marked as checked by you."
        return MonitorTab._ask_text(name, sids, None)

    def _run_guarded(self, name: str) -> None:
        try:
            self._run(name)
        except Exception:  # a slot must not close the window: the window reports it
            sys.excepthook(*sys.exc_info())

    def _run(self, name: str) -> None:
        sids = self.selected_sids()
        if not sids or self._scanning:
            return
        if not self._confirm(self._ask_text(name, sids)):
            return
        who = actions.current_user()
        try:
            res = {
                "exclude": lambda: actions.set_excluded(self.paths, sids, True, who),
                "include": lambda: actions.set_excluded(self.paths, sids, False, who),
                "accept_mismatch": lambda: actions.accept_mismatch(self.paths, sids, who),
                "confirm_stitch": lambda: actions.confirm_stitch(self.paths, sids, who),
            }[name]()
        except actions.UnknownSession as e:
            self._message("BEARMiND", f"Not in the series any more: {', '.join(e.unknown)}")
            return
        if res.ok:
            self._say(f"{_DONE_TEXT[name]} {_plural(len(res.ok))}: {', '.join(res.ok)}")
        else:
            self._say("Nothing was changed.")
        lines = [f"{sid}: {actions.failure_text(code, detail)}" for sid, (code, detail) in res.failed.items()]
        lines += [_WARNING_TEXT.get(code, "{sid}: " + code).format(sid=sid) for sid, code in res.warnings]
        if lines:
            self._message("BEARMiND", "\n".join(lines))
        self.refresh()
