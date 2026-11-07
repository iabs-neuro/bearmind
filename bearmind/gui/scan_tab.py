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

from PySide6.QtCore import QObject, QRect, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFontMetrics, QPalette, QStandardItem, QStandardItemModel
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QProgressBar, QPushButton, QSplitter, QStyle,
                               QStyledItemDelegate, QStyleOptionViewItem, QTableView, QTextBrowser, QVBoxLayout,
                               QWidget)

from bearmind import actions, config, monitor, scan_view
from bearmind.gui.monitor_tab import (ACCENT_BUTTON, CELL_BG, CELL_INK, EXCLUDED_INK, WARNING_TEXT, MonitorTab,
                                      named, plural)
from bearmind.paths import SeriesPaths

log = logging.getLogger("bearmind")

COLUMNS = ("Session", "Mode", "Parts", "Chunks", "Video frames", "Frames to process", "FPS", "Min", "Anomalies",
           "Excluded")
ANOMALIES_COLUMN = COLUMNS.index("Anomalies")
NOT_SCANNED = "Not scanned yet. Press Scan."
ASK_RESCAN = "Batch is running. If session files changed, their stages will be recomputed. Rescan?"
EXCLUDED_TEXT = {"anomaly": "by anomaly", "you": "by you", "included": "included by you", "": ""}
LEVEL_INK = {"error": ("ERROR", "#8e0000"), "warning": ("WARN", "#6d5300")}
FIXED_INK = "#2e7d32"
INCLUDE_TIP = ("Include works for sessions excluded by you. A session excluded by an anomaly: fix its files and "
               "press Rescan, or use Accept for a frame mismatch.")
SUMMARY_ROLE = Qt.ItemDataRole.UserRole + 2  # [(text, greyed)] of the Anomalies cell, drawn by _SummaryDelegate
_DONE_TEXT = {
    "exclude": "Excluded", "include": "Included", "accept_mismatch": "Accepted the frame mismatch of",
    "confirm_stitch": "Confirmed the stitch of",
}


def _count(n) -> str:
    return "–" if n is None else f"{n:,}".replace(",", " ")


def _num(x, fmt: str = "") -> str:
    return "–" if x is None else format(x, fmt)


class _Bridge(QObject):
    """Carries the worker thread's messages to the window's thread (a signal emitted from another thread
    reaches a slot of this thread through the event queue)."""
    progress = Signal(int, int, str)
    done = Signal(object)
    failed = Signal(str, str)


class _SummaryDelegate(QStyledItemDelegate):
    """The Anomalies cell on one line: the lines fixed or checked are greyed, so they do not read as errors; the
    part that does not fit is elided (the cell's tooltip has every line)."""

    def paint(self, painter, option, index) -> None:
        parts = index.data(SUMMARY_ROLE)
        if not parts:
            super().paint(painter, option, index)
            return
        opt = QStyleOptionViewItem(option)
        self.initStyleOption(opt, index)
        opt.text = ""
        style = opt.widget.style() if opt.widget is not None else QApplication.style()
        style.drawControl(QStyle.ControlElement.CE_ItemViewItem, opt, painter, opt.widget)  # background, selection
        selected = bool(opt.state & QStyle.StateFlag.State_Selected)
        ink = opt.palette.color(QPalette.ColorRole.HighlightedText if selected else QPalette.ColorRole.Text)
        fg = index.data(Qt.ItemDataRole.ForegroundRole)
        if fg is not None and not selected:  # an excluded row is grey as a whole
            ink = fg.color()
        rect = style.subElementRect(QStyle.SubElement.SE_ItemViewItemText, opt, opt.widget)
        fm = QFontMetrics(opt.font)
        painter.save()
        painter.setFont(opt.font)
        x = rect.left()
        for i, (text, grey) in enumerate(parts):
            for piece, piece_grey in (("; ", False), (text, grey)) if i else ((text, grey),):
                room = rect.right() - x
                cut = fm.horizontalAdvance(piece) > room
                if cut:
                    piece = fm.elidedText(piece, Qt.TextElideMode.ElideRight, max(room, 0))
                painter.setPen(QColor(EXCLUDED_INK) if piece_grey and not selected else ink)
                w = fm.horizontalAdvance(piece)
                painter.drawText(QRect(x, rect.top(), w + 1, rect.height()),
                                 int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft), piece)
                x += w
                if cut:
                    painter.restore()
                    return
        painter.restore()


class ScanTab(QWidget):
    scan_finished = Signal(bool)  # True: the scan finished and wrote the registry; False: it failed
    status_message = Signal(str)
    decided = Signal()  # a decision of this tab was written: Monitor's table changes with it

    def __init__(self, series: Path, parent=None, confirm=None, message=None, open_folder=None, report=None):
        super().__init__(parent)
        self.series = Path(series)
        self.paths = SeriesPaths(self.series)
        self._confirm = confirm or self._ask_confirm
        self._message = message or self._show_message
        self._open_folder = open_folder or (lambda p: os.startfile(str(p)))
        # the window installs its exception report as sys.excepthook
        self._report = report or (lambda *exc: sys.excepthook(*exc))
        self._refresh_failed = None  # (type, message) of the failure reported last, until a refresh succeeds
        self._scanning = False
        self._scan_total: int | None = None  # the scan's own count of sessions, from its progress
        self._past_walk = False  # the scan reported a session: it may have written records before failing
        self._thread: threading.Thread | None = None
        self._view: scan_view.ScanView | None = None
        self._rows: dict[str, scan_view.ScanRow] = {}
        self._cache = monitor.RecordCache()
        self._bridge = _Bridge(self)
        self._bridge.progress.connect(self._on_progress)
        self._bridge.done.connect(self._on_done)
        self._bridge.failed.connect(self._on_failed)
        self._build()
        self.refresh()
        # like Monitor: the batch and Monitor's menu change the registry; refreshed only while shown
        self.timer = QTimer(self)
        self.timer.setInterval(int(config.load_defaults().get("monitor", {}).get("refresh_ms", 2000)))
        self.timer.timeout.connect(self._tick)
        self.timer.start()

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
        self.table.setWordWrap(False)  # one line per session; the Anomalies tooltip has every line
        head = self.table.horizontalHeader()
        head.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        head.setStretchLastSection(False)  # only Anomalies takes the spare width, not the short Excluded
        head.setSectionResizeMode(ANOMALIES_COLUMN, QHeaderView.ResizeMode.Stretch)
        self._summary_delegate = _SummaryDelegate(self.table)
        self.table.setItemDelegateForColumn(ANOMALIES_COLUMN, self._summary_delegate)
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
        self.include_button.setToolTip(INCLUDE_TIP)
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
        self.broken_header = QLabel("Unreadable registry files")
        self.broken_list = QListWidget()
        for header, lst in ((self.unrecognized_header, self.unrecognized_list),
                            (self.broken_header, self.broken_list)):
            lst.setMaximumHeight(90)
            lst.itemDoubleClicked.connect(self._on_list_item)
            lay.addWidget(header)
            lay.addWidget(lst)
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
        self._scan_total, self._past_walk = None, False
        self.scan_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)  # busy until the first total arrives
        self.progress_label.setText("Scanning …")
        self._update_buttons()
        self._thread = threading.Thread(target=self._work, name="bearmind-scan", daemon=True)
        self._thread.start()

    def shutdown(self, timeout_s: float = 60.0) -> None:
        """Before the tab is deleted: stop the timer and wait (bounded) for a running scan, so it ends its
        registry writes."""
        self.timer.stop()
        t = self._thread
        if t is not None and t.is_alive():
            t.join(timeout_s)

    @staticmethod
    def _emit(signal, *args) -> None:
        try:
            signal.emit(*args)
        except RuntimeError:  # the tab was deleted while the scan ran: nobody is left to show the result
            pass

    def _work(self) -> None:
        bridge = self._bridge
        try:
            view = actions.rescan(self.paths, lambda i, n, sid: self._emit(bridge.progress, i, n, sid))
            self._emit(bridge.done, view)
        except Exception as e:  # shown by the tab; the traceback goes to the log
            log.error("scan failed:\n%s", traceback.format_exc())
            first = (str(e).splitlines() or [""])[0]
            self._emit(bridge.failed, type(e).__name__, first)

    def _on_progress(self, i: int, n: int, sid: str) -> None:
        self._scan_total = n
        self._past_walk = self._past_walk or bool(sid)
        self.progress_bar.setRange(0, max(n, 1))
        self.progress_bar.setValue(i)
        self.progress_label.setText(f"Scanning {sid} … {i} / {n}" if sid else f"Scanning … {i} / {n}")

    def _end_scan(self) -> None:
        self._scanning = False
        self.scan_button.setEnabled(True)

    def _on_done(self, view) -> None:
        self._end_scan()
        # the scan's own count: the table also keeps sessions whose folders the scan did not find
        n = self._scan_total if self._scan_total is not None else len(view.rows)
        self.progress_bar.setRange(0, max(n, 1))
        self.progress_bar.setValue(n)
        self.progress_label.setText(f"Scan finished — {n} / {n}")
        try:
            self._show(view)
            self._say(f"Scanned {plural(n)}")
        finally:  # the registry is written: Monitor must refresh even if showing it failed here
            self.scan_finished.emit(True)

    def _on_failed(self, kind: str, first: str) -> None:
        self._end_scan()
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_label.setText("Scan failed")
        if self._past_walk:  # records may have been written before the failure: show what the registry holds
            self.refresh()
        self._update_buttons()
        self._message("Scan failed", f"{kind}: {first}")
        self.scan_finished.emit(False)

    # --- showing the model ---

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._scanning:  # Monitor's menu or the batch may have changed the registry meanwhile
            self.refresh()

    def _tick(self) -> None:
        if self.isVisible() and not self._scanning:  # the scan shows its result itself when it ends
            self.refresh()

    def refresh(self) -> None:
        # a failure repeats every tick: the first one goes to the window's report (a dialog), the rest only
        # to the log and the status bar, so the timer does not stack modal dialogs
        try:
            view = scan_view.scan_view(self.paths, self._cache)
            if view != self._view:  # an unchanged registry: no repaint, no flicker, the selection stays
                self._show(view)
        except Exception as e:
            first = (str(e).splitlines() or [""])[0]
            kind = (type(e), str(e))
            if self._refresh_failed is None:
                self._refresh_failed = kind
                self._report(type(e), e, e.__traceback__)
            else:
                if kind == self._refresh_failed:
                    log.warning("scan tab refresh failed again: %s", first)
                else:  # a different failure: keep its traceback in the log, still no dialog
                    self._refresh_failed = kind
                    log.error("scan tab refresh failed differently", exc_info=(type(e), e, e.__traceback__))
                self._say(f"Refresh failed: {type(e).__name__}: {first}")
        else:
            self._refresh_failed = None

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
        self._show_lists(view)
        self._on_selection()

    @staticmethod
    def _items(r: scan_view.ScanRow) -> list[QStandardItem]:
        texts = [r.sid, r.mode, _num(r.n_parts), _num(r.n_chunks), _count(r.n_video_frames), _count(r.n_frames),
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
            if c == ANOMALIES_COLUMN:
                item.setData(r.summary_parts, SUMMARY_ROLE)
                item.setToolTip("\n".join(a.short + (f" ({a.note})" if a.note else "") for a in r.anomalies))
            if r.excluded:
                item.setForeground(QBrush(QColor(EXCLUDED_INK)))
            items.append(item)
        return items

    @staticmethod
    def _fill_list(header: QLabel, lst: QListWidget, entries) -> None:
        lst.clear()
        for text, folder in entries:
            item = QListWidgetItem(text)
            item.setData(Qt.ItemDataRole.UserRole, folder)
            lst.addItem(item)
        header.setVisible(bool(entries))
        lst.setVisible(bool(entries))

    def _show_lists(self, view: scan_view.ScanView) -> None:
        self._fill_list(self.unrecognized_header, self.unrecognized_list,
                        [(f"{path} — {reason}", str(self.series / path)) for path, reason in view.unrecognized])
        # a registry file opens its folder: the file itself would open in an editor
        self._fill_list(self.broken_header, self.broken_list,
                        [(f"{path} — {reason}", str(Path(path).parent)) for path, reason in view.broken])

    def _on_list_item(self, item: QListWidgetItem) -> None:
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

    @staticmethod
    def _targets(name: str, rows) -> list[str]:
        """The selected sessions a decision changes: Exclude those not excluded yet, Include those excluded by
        a person (an anomaly's exclusion is lifted by fixing the files and Rescan, or by Accept)."""
        if name == "exclude":
            return [r.sid for r in rows if not r.excluded]
        if name == "include":
            return [r.sid for r in rows if r.excluded_by == "you"]
        return [r.sid for r in rows]

    def _update_buttons(self) -> None:
        rows = self._selected_rows()
        free = bool(rows) and not self._scanning  # a decision during a scan would race the registry rewrite
        self.open_folder_button.setEnabled(len(rows) == 1 and bool(rows[0].folder))
        self.exclude_button.setEnabled(free and bool(self._targets("exclude", rows)))
        self.include_button.setEnabled(free and bool(self._targets("include", rows)))
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
        if not r.anomalies and r.summary.startswith("cannot show"):
            out.append(html.escape(r.summary))
        for a in r.anomalies:
            if a.auto_fixed:
                tag = f'<span style="color:{FIXED_INK}">FIXED</span>'
            elif a.info:  # below the warning limits: information, uncoloured
                tag = "INFO"
            else:
                word, ink = LEVEL_INK.get(a.level, ("WARN", LEVEL_INK["warning"][1]))
                tag = f'<span style="color:{ink}"><b>{word}</b></span>'
            out.append(f"{tag} {html.escape(a.text)}" + (f" ({html.escape(a.note)})" if a.note else ""))
        folders = ", ".join(f.replace("/", os.sep) for f in r.folders)  # as Explorer shows them
        out.append(f"Part folders: {html.escape(folders or '–')}")
        out.append(f"FPS: {html.escape(r.fps_text or '–')}")
        out.append("Decisions: none yet" if not r.decisions
                   else "Decisions:<br>" + "<br>".join(html.escape(d) for d in r.decisions))
        self.details.setHtml("<br>".join(out))

    # --- actions ---

    def _on_open_folder(self) -> None:
        rows = self._selected_rows()
        if len(rows) != 1 or not rows[0].folder:
            return
        folder = self.series / rows[0].folder
        if not folder.is_dir():  # removed or renamed since the scan
            self._message("BEARMiND", f"Folder not found: {folder}")
            return
        self._open_folder(folder)

    def _ask_text(self, name: str, sids) -> str:
        who = named(sids)
        if name == "confirm_stitch":
            return f"{who} will be marked as checked: the stitch by start time stays as it is."
        if name == "accept_mismatch":
            # the sign of the mismatch: more frames than timestamp rows, or the other way round
            signs = {a.data.get("diff", 0) > 0 for s in sids if s in self._rows for a in self._rows[s].anomalies
                     if a.code == "frames_ts_mismatch" and a.level == "error"}
            extra = {frozenset({True}): "frames", frozenset({False}): "timestamp rows"}.get(
                frozenset(signs), "frames or timestamp rows")
            return f"{who} will be used without the extra {extra} at the end; the files stay unchanged."
        return MonitorTab._ask_text(name, sids, None)

    def _run_guarded(self, name: str) -> None:
        try:
            self._run(name)
        except Exception:  # a slot must not close the window: the window reports it
            self._report(*sys.exc_info())

    def _run(self, name: str) -> None:
        if self._scanning:
            return
        sids = self._targets(name, self._selected_rows())
        if not sids:
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
            self.refresh()
            return
        if res.ok:
            self._say(f"{_DONE_TEXT[name]} {plural(len(res.ok))}: {', '.join(res.ok)}")
        else:
            self._say("Nothing was changed.")
        lines = [f"{sid}: {actions.failure_text(code, detail)}" for sid, (code, detail) in res.failed.items()]
        lines += [WARNING_TEXT.get(code, "{sid}: " + code).format(sid=sid) for sid, code in res.warnings]
        if lines:
            self._message("BEARMiND", "\n".join(lines))
        self.refresh()
        self.decided.emit()
