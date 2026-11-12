"""Thresholds tab: choose gSig, min_corr and min_pnr per session from the maps of the thresholds stage, with the
MC video of the session next to them.

The model is bearmind.threshold_model (read-only); every write goes through bearmind.actions. The dialogs are
injectable for the tests.
"""
import logging
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLocale, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QDoubleSpinBox, QGridLayout, QHBoxLayout,
                               QHeaderView, QLabel, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider,
                               QSplitter, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from bearmind import actions, config, monitor, registry
from bearmind import threshold_model as tm
from bearmind.gui.monitor_tab import (ACCENT, ACCENT_BUTTON, CELL_BG, CELL_INK, EXCLUDED_INK, MAX_NAMED, MUTED,
                                      WARNING_TEXT)
from bearmind.gui.video_view import VideoView
from bearmind.paths import SeriesPaths
from bearmind.registry import State

log = logging.getLogger("bearmind")

CNMF_DONE_ASK = ("CNMF-E of {sid} is already done. With the new thresholds it will be recomputed, and manual "
                 "inspection will be needed again. Write?")
CNMF_RUNNING_ASK = ("CNMF-E of {sid} is running with the old thresholds. It will run again with the new ones, and "
                    "manual inspection will be needed again. Write?")
# the gate is disabled in the session's resolved config: a new CNMF-E needs no manual inspection
CNMF_DONE_ASK_NO_GATE = "CNMF-E of {sid} is already done. With the new thresholds it will be recomputed. Write?"
CNMF_RUNNING_ASK_NO_GATE = ("CNMF-E of {sid} is running with the old thresholds. It will run again with the new "
                            "ones. Write?")
CHOICE_LOG_TEXT = {
    actions.CHOICE_NOT_LOGGED: "Thresholds written. The choice log is open in another program; the row was kept in "
                               "threshold_choices.pending.csv.",
    actions.CHOICE_ONLY_IN_LOG: "Thresholds written. The choice log is open in another program, and "
                                "threshold_choices.pending.csv cannot be written either; the row was written to the "
                                "warning log (gui.log).",
    actions.CHOICE_PENDING_KEPT: "Thresholds written to the choice log. Earlier rows wait in "
                                 "threshold_choices.pending.csv, which cannot be read now (open in another "
                                 "program?); they are added once it can be read.",
}
ALL_HAVE = "All sessions have thresholds."
NO_SESSIONS = "No sessions in this series yet. Scan the series in the Scan tab."
CORR_STEPS = 100  # slider units per 1.0: min_corr moves in steps of 0.01
PNR_STEPS = 10  # min_pnr moves in steps of 0.1
MAX_UPSCALE = 4.0  # a small map is enlarged at most this much: more only blurs it
THUMB_PX = 96
VIDEO_MIN_W = 320  # px: below this the movie is too small to judge the cells by
RECORD_RETRY_S = 0.1  # a second read of a busy registry file, after fileio's own retries
RED = (255, 0, 0)
STATE_BG = {"waiting": CELL_BG["waiting"], "chosen": CELL_BG["ok"], "cnmf_done": CELL_BG["ok"],
            "config_error": CELL_BG["error"]}
WARN_BUTTON = ("QPushButton { background: #fff3b0; color: #6d5300; border: 1px solid #e0c35c; border-radius: 6px; "
               "padding: 4px 10px; }"
               "QPushButton:disabled { background: #f6f0d6; border-color: #eee3b8; color: #b8ab7d; }")
THUMB_STYLE = (f"QToolButton {{ border: 2px solid transparent; border-radius: 6px; padding: 2px; }}"
               f"QToolButton:checked {{ border-color: {ACCENT}; background: #dde5f8; }}")
ACCEPT_TIP = "Write these thresholds to the session and show the next session that waits"
ACCEPT_MOUSE_TIP = ("Write these thresholds to every session of this mouse that has no thresholds yet and is not "
                    "excluded")
EXCLUDED_NOTE = tm.EXCLUDED_NOTE


def _section(text: str) -> QLabel:
    label = QLabel(text)
    f = label.font()
    f.setCapitalization(QFont.Capitalization.AllUppercase)
    f.setPixelSize(11)
    label.setFont(f)
    label.setStyleSheet(f"color: {MUTED};")
    return label


def _listed(sids) -> str:
    sids = list(sids)
    more = f" and {len(sids) - MAX_NAMED} more" if len(sids) > MAX_NAMED else ""
    return ", ".join(sids[:MAX_NAMED]) + more


def _rgb_image(rgb: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()  # copy: detach from numpy


def _grey_rgb(img: np.ndarray, valid: np.ndarray) -> np.ndarray:
    return np.repeat(tm.display_u8(img, valid)[..., None], 3, axis=2)


class _NumberBox(QDoubleSpinBox):
    """A threshold's spin box. The C locale: a dot whatever the Windows locale (a Russian one took only a comma and
    turned a typed "12.5" into 12.0), as everywhere else in the window. It shows the slider step's decimals, or
    all of an exact start value that lies between steps (an untouched suggestion), which is what Accept writes."""

    def __init__(self, shown: int):
        super().__init__()
        self._shown = shown
        self.setLocale(QLocale.c())
        self.setDecimals(6)  # the value is rounded to these: an exact start value must survive them

    def textFromValue(self, value: float) -> str:
        whole, _, frac = f"{value:.6f}".rstrip("0").partition(".")
        return f"{whole}.{frac.ljust(self._shown, '0')}"


def _values_text(c: float, p: float) -> str:
    return f"min_corr {tm.value_text(c, tm.CORR_DECIMALS)}, min_pnr {tm.value_text(p, tm.PNR_DECIMALS)}"


def _gate_on(cfg: dict) -> bool:
    """The rule of runner.stage_params for the gate: enabled unless the config says otherwise."""
    return bool((cfg.get("gate") or {}).get("enabled", True))


def _window_text(error: str) -> str:
    """actions' validator speaks in CLI flags; the window names the sliders."""
    for flag, name in (("--gsig", "gSig"), ("--min-corr", "min_corr"), ("--min-pnr", "min_pnr")):
        error = error.replace(flag, name)
    return error


class _FitLabel(QLabel):
    """An image scaled to fit the label whole (edges visible, aspect kept), or the reason it cannot be shown."""

    def __init__(self):
        super().__init__(alignment=Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(QSize(120, 90))
        # Ignored: the pixmap must not grow the label, or the layout could never shrink it again
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.setWordWrap(True)
        self.setStyleSheet("background: #0b0d12; color: #d0d4dc; padding: 0px;")
        self._image: QImage | None = None

    def set_image(self, image: QImage) -> None:
        self._image = image
        self._fit()

    def set_reason(self, text: str) -> None:
        self._image = None
        self.setPixmap(QPixmap())
        self.setText(text)

    def image(self) -> QImage | None:
        return self._image

    def _fit(self) -> None:
        if self._image is None or self._image.isNull():
            return
        room = self.contentsRect()
        w, h = self._image.width(), self._image.height()
        s = min(room.width() / w, room.height() / h, MAX_UPSCALE)
        size = QSize(max(1, int(w * s)), max(1, int(h * s)))  # int() floors: the pixmap never exceeds the room
        self.setPixmap(QPixmap.fromImage(self._image).scaled(size, Qt.AspectRatioMode.KeepAspectRatio,
                                                             Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._fit()


class ThresholdsTab(QWidget):
    status_message = Signal(str)
    decided = Signal()  # thresholds were written: Monitor's waiting count changes with them

    def __init__(self, series: Path, parent=None, confirm=None, message=None, report=None):
        super().__init__(parent)
        self.series = Path(series)
        self.paths = SeriesPaths(self.series)
        self._confirm = confirm or self._ask_confirm
        self._message = message or self._show_message
        # the window installs its exception report as sys.excepthook
        self._report = report or (lambda *exc: sys.excepthook(*exc))
        self._refresh_failed = None  # (type, message) of the failure reported last, until a refresh succeeds
        self._items: list[tm.ThresholdItem] = []
        # Monitor's cache: a registry file the batch is replacing (busy for a moment) keeps its last good record,
        # so the shown session and its slider values stay; configs are resolved again only when a file changes
        self._cache = monitor.RecordCache()
        self._records: dict = {}  # sid -> the record of the last refresh
        self._by_sid: dict[str, tm.ThresholdItem] = {}
        self._tree_items: dict[str, QTreeWidgetItem] = {}
        self._sid: str | None = None
        self._maps: tm.ThresholdMaps | None = None
        self._reason = NO_SESSIONS  # why no maps are shown
        self._loaded_sig = None  # what the shown maps were read from; the maps reload when it changes
        self._source = ""
        self._note = ""
        self._g: int | None = None
        self._exact: dict[str, float] = {}  # "corr" / "pnr" -> start value kept exactly until that control moves
        self._min_blob_px = 2
        self._base: dict[str, np.ndarray] = {}  # "corr" / "pnr" -> grey RGB of the selected gSig
        self._images: dict[str, QImage] = {}  # the same with the red mask, at full resolution
        self.gsig_buttons: dict[int, QToolButton] = {}
        # (path, fps, contrast) of the shown session's MC video and the frame to return to. The file is open only
        # while the tab is visible: a hidden tab holds no decoder thread and leaves the file to the batch.
        self._video_args = None
        self._video_index = 0
        self._video_open = False
        self._build()
        self.refresh()
        # like Monitor: the batch and Monitor's menu change the registry; refreshed only while shown
        self.timer = QTimer(self)
        self.timer.setInterval(int(config.load_defaults().get("monitor", {}).get("refresh_ms", 2000)))
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    # --- construction ---

    def _build(self) -> None:
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)
        split.addWidget(self._build_sessions())
        split.addWidget(self._build_work())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([230, 870])
        lay.addWidget(split)
        self._show_no_maps()

    def _build_sessions(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(_section("Sessions"))
        self.session_list = QTreeWidget()
        self.session_list.setColumnCount(2)
        self.session_list.setHeaderLabels(["Session", "State"])
        self.session_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.session_list.setUniformRowHeights(True)
        head = self.session_list.header()
        head.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        head.setStretchLastSection(True)
        self.session_list.currentItemChanged.connect(lambda cur, _prev: self._guarded(self._on_tree, cur))
        lay.addWidget(self.session_list, 1)
        box.setMinimumWidth(180)
        return box

    def _build_work(self) -> QWidget:
        box = QWidget()
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 0, 0, 0)
        lay.setSpacing(4)
        top = QSplitter(Qt.Orientation.Horizontal)
        top.setChildrenCollapsible(False)
        # part 2: the MC video
        video_box = QWidget()
        vl = QVBoxLayout(video_box)
        vl.setContentsMargins(0, 0, 0, 0)
        self.video_title = _section("MC video")
        self.video = VideoView()
        self.video.setMinimumWidth(VIDEO_MIN_W)
        vl.addWidget(self.video_title)
        vl.addWidget(self.video, 1)
        top.addWidget(video_box)
        # part 4: correlation and PNR with the red mask, each with its slider and spin box
        maps_box = QWidget()
        ml = QVBoxLayout(maps_box)
        ml.setContentsMargins(0, 0, 0, 0)
        self.maps_title = _section("Correlation and PNR")
        ml.addWidget(self.maps_title)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        self.corr_view, self.pnr_view = _FitLabel(), _FitLabel()
        self.min_corr_slider, self.min_corr_spin = self._slider(CORR_STEPS, 2, 1.0)
        self.min_pnr_slider, self.min_pnr_spin = self._slider(PNR_STEPS, 1, 30.0)
        for col, (title, view, name, slider, spin) in enumerate((
                ("Correlation", self.corr_view, "min_corr", self.min_corr_slider, self.min_corr_spin),
                ("PNR", self.pnr_view, "min_pnr", self.min_pnr_slider, self.min_pnr_spin))):
            grid.addWidget(QLabel(title), 0, col)
            grid.addWidget(view, 1, col)
            row = QHBoxLayout()
            row.addWidget(QLabel(name))
            row.addWidget(slider, 1)
            row.addWidget(spin)
            grid.addLayout(row, 2, col)
            grid.setColumnStretch(col, 1)
        grid.setRowStretch(1, 1)
        ml.addLayout(grid, 1)
        top.addWidget(maps_box)
        top.setStretchFactor(0, 1)
        top.setStretchFactor(1, 2)
        # Initial sizes in proportion (the splitter scales them to its width): the video as wide as one of the two
        # maps. Without them the splitter follows the size hints, and the slider rows of the maps took nearly all.
        top.setSizes([1000, 2000])
        lay.addWidget(top, 1)
        self.min_corr_slider.valueChanged.connect(
            lambda v: self._guarded(self._on_slider, "corr", self.min_corr_slider, self.min_corr_spin, CORR_STEPS, v))
        self.min_pnr_slider.valueChanged.connect(
            lambda v: self._guarded(self._on_slider, "pnr", self.min_pnr_slider, self.min_pnr_spin, PNR_STEPS, v))
        self.min_corr_spin.valueChanged.connect(
            lambda x: self._guarded(self._on_spin, "corr", self.min_corr_slider, self.min_corr_spin, CORR_STEPS, x))
        self.min_pnr_spin.valueChanged.connect(
            lambda x: self._guarded(self._on_spin, "pnr", self.min_pnr_slider, self.min_pnr_spin, PNR_STEPS, x))
        for slider in (self.min_corr_slider, self.min_pnr_slider):
            slider.sliderReleased.connect(lambda: self._guarded(self._update_numbers))

        # part 3: one preview per gSig
        lay.addWidget(_section("gSig — click a preview"))
        self.gsig_strip = QWidget()
        self._strip_layout = QHBoxLayout(self.gsig_strip)
        self._strip_layout.setContentsMargins(0, 0, 0, 0)
        self._strip_layout.addStretch(1)
        self._gsig_group = QButtonGroup(self)
        self._gsig_group.setExclusive(True)
        scroll = QScrollArea()
        scroll.setWidget(self.gsig_strip)
        scroll.setWidgetResizable(True)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFixedHeight(THUMB_PX + 48)
        lay.addWidget(scroll)

        # parts 5 and 6: counters and where the start values come from
        self.counters_label = QLabel("")
        self.source_label = QLabel("")
        self.source_label.setStyleSheet(f"color: {MUTED};")
        self.source_label.setWordWrap(True)
        lay.addWidget(self.counters_label)
        lay.addWidget(self.source_label)

        # part 7: buttons
        row = QHBoxLayout()
        self.prev_button = QPushButton("← Previous")
        self.next_button = QPushButton("Next →")
        self.reset_button = QPushButton("Reset to suggestion")
        self.accept_button = QPushButton("Accept → next")
        self.accept_button.setStyleSheet(ACCENT_BUTTON)
        self.accept_mouse_button = QPushButton("Accept for all sessions of this mouse")
        self.accept_mouse_button.setStyleSheet(WARN_BUTTON)
        self.prev_button.clicked.connect(lambda: self._guarded(self._step, -1))
        self.next_button.clicked.connect(lambda: self._guarded(self._step, 1))
        self.reset_button.clicked.connect(lambda: self._guarded(self._reset))
        self.accept_button.clicked.connect(lambda: self._guarded(self.accept_and_next))
        self.accept_mouse_button.clicked.connect(lambda: self._guarded(self.accept_for_mouse))
        for b in (self.prev_button, self.next_button, self.reset_button):
            row.addWidget(b)
        row.addStretch(1)
        row.addWidget(self.accept_button)
        row.addWidget(self.accept_mouse_button)
        lay.addLayout(row)
        return box

    @staticmethod
    def _slider(steps: int, decimals: int, top: float) -> tuple[QSlider, QDoubleSpinBox]:
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, round(top * steps))
        slider.setPageStep(5 if steps == CORR_STEPS else 10)
        spin = _NumberBox(decimals)
        spin.setSingleStep(1 / steps)
        spin.setRange(0.0, top)
        spin.setKeyboardTracking(False)  # typing "0.85" must not jump the slider to 0, then 0.8
        return slider, spin

    # --- dialogs (replaced in the tests) ---

    def _ask_confirm(self, text: str, yes: str = "Write", no: str = "Cancel") -> bool:
        box = QMessageBox(QMessageBox.Icon.Question, "BEARMiND", text, parent=self)
        yes_button = box.addButton(yes, QMessageBox.ButtonRole.AcceptRole)
        box.setDefaultButton(box.addButton(no, QMessageBox.ButtonRole.RejectRole))  # a write must be chosen
        box.exec()
        return box.clickedButton() is yes_button

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
        """Before the tab is deleted: stop the timer and release the MC video (the batch may need the file)."""
        self.timer.stop()
        self._close_video()
        self._video_args = None

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.refresh()  # Monitor's menu or the batch may have changed the registry meanwhile
        self._open_video()

    def hideEvent(self, event) -> None:
        super().hideEvent(event)
        self._close_video()

    def _open_video(self) -> None:
        if self._video_args is None or self._video_open or not self.isVisible():
            return
        self.video.set_video(*self._video_args)
        self.video.seek(self._video_index)  # remembered by the view until the file is open
        self._video_open = True

    def _close_video(self) -> None:
        if self._video_open:
            self._video_index = self.video.current_index()
            self.video.set_video(None, 0)
            self._video_open = False

    def _set_video(self, args, keep_position: bool = False) -> None:
        """A new session's video (None: no session), or the same session's after its maps reloaded (MC may have
        replaced the file): the old file is released at once; `keep_position` returns to the frame shown."""
        index = self.video.current_index() if self._video_open else self._video_index
        self._close_video()
        self._video_args, self._video_index = args, index if keep_position else 0
        self._open_video()

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
                    log.warning("thresholds tab refresh failed again: %s", first)
                else:  # a different failure: keep its traceback in the log, still no dialog
                    self._refresh_failed = kind
                    log.error("thresholds tab refresh failed differently", exc_info=(type(e), e, e.__traceback__))
                self._say(f"Refresh failed: {type(e).__name__}: {first}")
        else:
            self._refresh_failed = None

    def _refresh(self) -> None:
        records, _broken = self._cache.records(self.paths)
        self._records = {r.sid: r for r in records}
        items = tm.threshold_items(self.paths, self._cache.stage_view(self.paths), records)
        if items != self._items:
            self._items = items
            self._by_sid = {i.sid: i for i in items}
            self._fill_tree()
        if self._sid not in self._by_sid:
            self._show_first()
            return
        # the shown maps reload only when their file or the thresholds stage changed (the batch made them, or
        # began to recompute them); the slider values stay. The MC video is set again with them: MC may have
        # replaced the file, and its contrast follows the new mean image.
        if self._maps_sig(self._sid) != self._loaded_sig:
            self._load_maps(keep=True)
            self._set_video(self._video_for(self._sid), keep_position=True)
        self._update_buttons()

    def show_first_waiting(self) -> None:
        """Monitor's "Thresholds: N sessions": the first session in list order that waits (none: stay)."""
        self.refresh()
        sid = next((i.sid for i in self._items if i.state == "waiting"), None)
        if sid is not None and sid != self._sid:
            self.show_session(sid)

    def _show_first(self) -> None:
        sid = next((i.sid for i in self._items if i.state == "waiting"), None)
        if sid is None and self._items:
            sid = self._items[0].sid
        if sid is not None:
            self.show_session(sid)
            return
        self._sid = None
        if self._video_args is not None:
            self._set_video(None)
        self.video_title.setText("MC video")
        self._maps, self._reason = None, NO_SESSIONS
        self._show_no_maps()
        self._update_buttons()

    # --- part 1: the session list ---

    def _fill_tree(self) -> None:
        tree = self.session_list
        scroll = tree.verticalScrollBar().value()
        tree.blockSignals(True)
        tree.clear()
        self._tree_items = {}
        mice: dict[str, QTreeWidgetItem] = {}
        for it in self._items:
            parent = mice.get(it.mouse)
            if parent is None:
                parent = mice[it.mouse] = QTreeWidgetItem(tree, [it.mouse])
                parent.setFlags(Qt.ItemFlag.ItemIsEnabled)  # a mouse row is a heading, not a session
                f = parent.font(0)
                f.setBold(True)
                parent.setFont(0, f)
            day = it.sid[len(it.mouse) + 1:] if it.sid.startswith(it.mouse + "_") else it.sid
            state = tm.state_text(it)
            node = QTreeWidgetItem(parent, [day, state])
            node.setData(0, Qt.ItemDataRole.UserRole, it.sid)
            node.setToolTip(0, it.sid)
            node.setToolTip(1, state)
            bg = STATE_BG.get(it.state)
            if bg:
                node.setBackground(1, QBrush(QColor(bg)))
                node.setForeground(1, QBrush(QColor(CELL_INK)))
            elif it.state in ("excluded", "no_maps", "out_of_date"):
                for c in (0, 1):
                    node.setForeground(c, QBrush(QColor(EXCLUDED_INK)))
            self._tree_items[it.sid] = node
        tree.expandAll()
        if self._sid in self._tree_items:
            tree.setCurrentItem(self._tree_items[self._sid])
        tree.blockSignals(False)
        tree.verticalScrollBar().setValue(scroll)

    def _on_tree(self, node) -> None:
        sid = node.data(0, Qt.ItemDataRole.UserRole) if node is not None else None
        if sid and sid != self._sid:
            self.show_session(sid)

    def _step(self, d: int) -> None:
        order = [i.sid for i in self._items]
        if self._sid in order and 0 <= order.index(self._sid) + d < len(order):
            self.show_session(order[order.index(self._sid) + d])

    # --- showing a session ---

    def _record(self, sid: str):
        """The session's record as it is now. A file the batch is replacing can refuse even fileio's retries for a
        moment: one more try after RECORD_RETRY_S, then the record of the last refresh (none if it was removed)."""
        for attempt in range(2):
            try:
                return registry.load_record(self.paths, sid)
            except OSError:
                if attempt == 0:
                    time.sleep(RECORD_RETRY_S)
            except ValueError:  # not a record: the next refresh shows it as broken
                return None
        return self._records.get(sid) if self.paths.registry_file(sid).exists() else None

    def _maps_sig(self, sid: str):
        """What the shown maps depend on: the item's note (why no maps are shown, "" when they are) and the file.
        A move between waiting, chosen and CNMF-E done reloads nothing; going out of date or back does."""
        item = self._by_sid.get(sid)
        try:
            st = self.paths.threshold_maps(sid).stat()
            fsig = (st.st_mtime_ns, st.st_size)
        except OSError:
            fsig = None
        return (item.note if item is not None else None), fsig

    def _session_cfg(self, item: tm.ThresholdItem) -> dict:
        try:
            return config.resolve(self.paths, sid=item.sid, mouse=item.mouse)
        except (config.ConfigError, OSError):
            # a broken config is shown by Monitor; min_blob_px and the start values come from the defaults
            return config.load_defaults()

    def show_session(self, sid: str) -> None:
        if sid not in self._by_sid:
            return
        self._sid = sid
        node = self._tree_items.get(sid)
        if node is not None and self.session_list.currentItem() is not node:
            self.session_list.blockSignals(True)
            self.session_list.setCurrentItem(node)
            self.session_list.blockSignals(False)
        self._load_maps(keep=False)
        self.video_title.setText(f"MC video — {sid}")
        self._set_video(self._video_for(sid))

    def _video_for(self, sid: str) -> tuple:
        """(path, fps, contrast) of the session's MC video; the contrast comes from the shown maps' mean image, so
        the movie and the maps look alike (None without maps)."""
        rec = self._record(sid)
        contrast = None
        if self._maps is not None:
            mean = np.asarray(self._maps.mean, dtype=float)
            fin = mean[np.isfinite(mean)]
            if fin.size:
                lo, hi = np.percentile(fin, [1.0, 99.5])
                contrast = (float(lo), float(hi)) if hi > lo else None
        return self.paths.mc_video(sid), rec.fps if rec is not None and rec.fps else 0, contrast

    def _read_maps(self, item: tm.ThresholdItem):
        """(maps, "") or (None, the reason shown in their place). The item says whether the maps may be shown (the
        runner's rule: out of date, being recomputed, cannot be scheduled, no file); only the file is read here."""
        if item.note:
            return None, item.note
        path = self.paths.threshold_maps(item.sid)
        try:
            maps = tm.load_maps(path)
        except Exception as e:  # any broken file (zip, pickle, missing key) is shown as the reason
            first = (str(e).splitlines() or [""])[0]
            return None, f"Maps cannot be read: {type(e).__name__}: {first}"
        if not maps.usable_gsigs:
            return None, "Maps cannot be read: the file holds no correlation and PNR map pair."
        return maps, ""

    def _start_values(self, item: tm.ThresholdItem, cfg: dict, use_choice: bool = True):
        """(gSig, min_corr, min_pnr, source, note), one rule for the session start and for Reset: a gSig without
        its maps is swapped for a usable one, and the note says so."""
        return tm.start_values(item if use_choice else replace(item, chosen=None), self._maps, cfg)

    def _load_maps(self, keep: bool) -> None:
        item = self._by_sid[self._sid]
        self._loaded_sig = self._maps_sig(self._sid)
        old = None
        if keep and self._maps is not None and self._g is not None:
            old = self.current()[1:]
        self._maps, self._reason = self._read_maps(item)
        cfg = self._session_cfg(item)
        self._min_blob_px = int(cfg["thresholds"]["min_blob_px"])
        if self._maps is None:
            self._show_no_maps()
            self._update_buttons()
            return
        g, c, p, source, note = self._start_values(item, cfg)
        if old is not None and old[0] in self._maps.usable_gsigs:
            g, c, p = old  # the person's values stay when the batch replaced the maps under them
        else:
            self._source, self._note = source, note
        self._build_strip()
        self._apply(g, c, p)
        self._show_source()
        self._update_buttons()

    def _show_no_maps(self) -> None:
        self._g, self._base, self._images = None, {}, {}
        self._clear_strip()
        for view in (self.corr_view, self.pnr_view):
            view.set_reason(self._reason)
        self.maps_title.setText("Correlation and PNR")
        self.counters_label.setText("")
        self.source_label.setText("")
        for w in (self.min_corr_slider, self.min_pnr_slider, self.min_corr_spin, self.min_pnr_spin):
            w.setEnabled(False)

    def _show_source(self) -> None:
        item = self._by_sid.get(self._sid)
        if item is None or self._maps is None:
            return
        chosen_from = tm.choice_source(self.paths, item.sid, item.mouse) if self._source == "chosen" else ""
        text = tm.source_text(self._source, item, chosen_from)
        self.source_label.setText(f"{text} — {self._note}" if self._note else text)

    # --- part 3: the gSig strip ---

    def _clear_strip(self) -> None:
        for b in self.gsig_buttons.values():
            self._gsig_group.removeButton(b)
            self._strip_layout.removeWidget(b)
            b.deleteLater()
        self.gsig_buttons = {}

    def _build_strip(self) -> None:
        self._clear_strip()
        for g in self._maps.usable_gsigs:
            u8 = np.ascontiguousarray(tm.gsig_thumbnail(self._maps, g))
            h, w = u8.shape
            pix = QPixmap.fromImage(QImage(u8.data, w, h, w, QImage.Format.Format_Grayscale8).copy()).scaled(
                THUMB_PX, THUMB_PX, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            b = QToolButton()
            b.setCheckable(True)
            b.setIcon(QIcon(pix))
            b.setIconSize(QSize(THUMB_PX, THUMB_PX))
            b.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
            b.setStyleSheet(THUMB_STYLE)
            b.setText(f"gSig {g}")
            b.clicked.connect(lambda _=False, g=g: self._guarded(self.set_gsig, g))
            self._gsig_group.addButton(b)
            self._strip_layout.insertWidget(self._strip_layout.count() - 1, b)  # before the stretch
            self.gsig_buttons[g] = b

    # --- parts 4 and 5: values, mask, counters ---

    def _set_pnr_top(self, top: float) -> None:
        units = int(np.ceil(top * PNR_STEPS))
        self.min_pnr_slider.blockSignals(True)
        self.min_pnr_slider.setMaximum(units)
        self.min_pnr_slider.blockSignals(False)
        self.min_pnr_spin.blockSignals(True)
        self.min_pnr_spin.setMaximum(units / PNR_STEPS)
        self.min_pnr_spin.blockSignals(False)

    def _put(self, min_corr: float, min_pnr: float) -> None:
        """Slider and spin box values without the change handlers (the caller updates the pictures). A spin box
        shows the exact start value while it is kept (_exact), else the slider's."""
        for name, slider, spin, steps, x in (
                ("corr", self.min_corr_slider, self.min_corr_spin, CORR_STEPS, min_corr),
                ("pnr", self.min_pnr_slider, self.min_pnr_spin, PNR_STEPS, min_pnr)):
            slider.blockSignals(True)
            slider.setValue(round(x * steps))
            slider.blockSignals(False)
            self._sync_spin(spin, self._exact.get(name, slider.value() / steps))

    @staticmethod
    def _sync_spin(spin: QDoubleSpinBox, x: float) -> None:
        spin.blockSignals(True)
        spin.setValue(x)
        spin.blockSignals(False)

    def _select_gsig(self, g: int, keep_pnr: float) -> None:
        # every lookup before any change: a gSig without its maps raises here and leaves the screen and current()
        # as they were
        valid = tm.valid_area(self._maps, g)
        base = {"corr": _grey_rgb(self._maps.cn[g], valid), "pnr": _grey_rgb(self._maps.pnr[g], valid)}
        # the PNR range depends on gSig; a value above it widens the range rather than being cut to it, so the
        # slider never shows another value than the one that will be written
        top = max(tm.slider_range_pnr(self._maps, g), keep_pnr)
        button = self.gsig_buttons[g]
        self._g, self._base = g, base
        self._set_pnr_top(top)
        button.setChecked(True)
        self.maps_title.setText(f"Correlation and PNR, gSig {g}")

    def _apply(self, g: int, min_corr: float, min_pnr: float) -> None:
        """Start values (the session's start, Reset, maps replaced under kept values): kept exactly until the
        person moves that control, so an untouched Accept writes what the source line says, as CLI
        accept-suggested does, even between two slider steps."""
        for w in (self.min_corr_slider, self.min_pnr_slider, self.min_corr_spin, self.min_pnr_spin):
            w.setEnabled(True)
        self._select_gsig(g, min_pnr)
        self._exact = {"corr": float(min_corr), "pnr": float(min_pnr)}
        self._put(min_corr, min_pnr)
        self._update_overlay()
        self._update_numbers()

    def set_gsig(self, g: int) -> None:
        if self._maps is None or g not in self.gsig_buttons:
            return
        self._select_gsig(g, self.current()[3])
        self._sync_spin(self.min_pnr_spin, self.current()[3])
        self._update_overlay()
        self._update_numbers()

    def set_thresholds(self, min_corr: float, min_pnr: float) -> None:
        if self._maps is None:
            return
        if min_pnr > self.min_pnr_slider.maximum() / PNR_STEPS:
            self._set_pnr_top(min_pnr)
        self._exact = {}  # the person set them: the slider steps count
        self._put(min_corr, min_pnr)
        self._update_overlay()
        self._update_numbers()

    def current(self) -> tuple[str, int, float, float]:
        return (self._sid or "", self._g or 0, self._exact.get("corr", self.min_corr_slider.value() / CORR_STEPS),
                self._exact.get("pnr", self.min_pnr_slider.value() / PNR_STEPS))

    def overlay_image(self, name: str) -> QImage | None:
        """The map ("corr" or "pnr") with the red mask at full resolution (None without maps)."""
        return self._images.get(name)

    def _on_spin(self, name: str, slider: QSlider, spin: QDoubleSpinBox, steps: int, x: float) -> None:
        """A typed value: the slider's step value counts from now on, even when the slider does not move (the person
        typed the rounded value of an exact start value)."""
        self._exact.pop(name, None)
        v = round(x * steps)
        if slider.value() != v:
            slider.setValue(v)  # _on_slider updates the rest
            return
        self._sync_spin(spin, v / steps)
        self._update_overlay()
        self._update_numbers()

    def _on_slider(self, name: str, slider: QSlider, spin: QDoubleSpinBox, steps: int, v: int) -> None:
        self._exact.pop(name, None)  # the person moved it (programmatic moves block this signal)
        self._sync_spin(spin, v / steps)
        self._update_overlay()  # the mask follows a drag at once (a few ms on a 600x600 map)
        if not slider.isSliderDown():  # a key, a click on the track or the spin box: no release follows
            self._update_numbers()

    def _update_overlay(self) -> None:
        if self._maps is None or self._g is None:
            return
        _, g, c, p = self.current()
        mask = tm.passing_mask(self._maps.cn[g], self._maps.pnr[g], c, p)
        for name, view in (("corr", self.corr_view), ("pnr", self.pnr_view)):
            rgb = self._base[name].copy()
            rgb[mask] = RED
            self._images[name] = _rgb_image(rgb)
            view.set_image(self._images[name])

    def _update_numbers(self) -> None:
        """Counters and the strip numbers: labelling every gSig takes longer, so it waits for the release."""
        if self._maps is None or self._g is None:
            return
        _, g, c, p = self.current()
        k = tm.counters(self._maps, g, c, p, self._min_blob_px)
        text = f"blobs {k.n_blobs} · single pixels {round(k.single_frac * 100)} %"
        if k.n_mirror_blobs is not None:
            text += f" · blobs on mirrored maps {k.n_mirror_blobs}"
        self.counters_label.setText(text)
        plain = replace(self._maps, cn_mirror={}, pnr_mirror={})  # the strip shows blobs only: no mirror labels
        for g2, b in self.gsig_buttons.items():
            n = k.n_blobs if g2 == g else tm.counters(plain, g2, c, p, self._min_blob_px).n_blobs
            b.setText(f"gSig {g2} · {n}")
        self._update_buttons()

    # --- part 7: buttons ---

    def _refusal(self, item) -> str:
        """Why the shown values cannot be written now ("" when they can)."""
        if item is None:
            return "No session is shown."
        if item.state == "excluded":
            return "The session is excluded: the batch ignores it. Include it first (Scan tab or Monitor)."
        if self._maps is None:
            return self._reason
        bad = actions.threshold_error(*self.current()[1:])
        return _window_text(bad) if bad else ""

    def _update_buttons(self) -> None:
        item = self._by_sid.get(self._sid) if self._sid else None
        order = [i.sid for i in self._items]
        pos = order.index(self._sid) if self._sid in order else -1
        self.prev_button.setEnabled(pos > 0)
        self.next_button.setEnabled(0 <= pos < len(order) - 1)
        why = self._refusal(item)
        for b, tip in ((self.accept_button, ACCEPT_TIP), (self.accept_mouse_button, ACCEPT_MOUSE_TIP)):
            b.setEnabled(not why)
            b.setToolTip(why or tip)
        can_reset = item is not None and self._maps is not None and bool(item.suggested)
        self.reset_button.setEnabled(can_reset)
        no_suggestion = self._maps is not None and not can_reset
        self.reset_button.setToolTip("No autotune suggestion for this session" if no_suggestion else "")

    def _reset(self) -> None:
        item = self._by_sid.get(self._sid)
        if item is None or self._maps is None or not item.suggested:
            return
        g, c, p, source, note = self._start_values(item, self._session_cfg(item), use_choice=False)
        self._apply(g, c, p)
        self._source, self._note = source, note
        self._show_source()

    def _gone(self, names) -> None:
        self._message("BEARMiND", f"Not in the series any more: {', '.join(names)}")
        self.refresh()

    def _run_values(self, cnmf, item) -> tuple:
        """(gSig, min_corr, min_pnr) the CNMF-E run used or uses: its recorded parameters, else the config."""
        prm = cnmf.params or {}
        if all(prm.get(k) is not None for k in ("gSig", "min_corr", "min_pnr")):
            g = prm["gSig"]
            g = g[0] if isinstance(g, (list, tuple)) else g  # cnmf_params stores gSig as (g, g)
            return int(g), float(prm["min_corr"]), float(prm["min_pnr"])
        cfg = self._session_cfg(item).get("cnmf") or {}  # a stage record without parameters (not from the runner)
        return tuple(cfg.get(k) for k in ("gSig", "min_corr", "min_pnr"))

    def _rerun_question(self, rec, item, g: int, c: float, p: float) -> str:
        """The question before a write that makes CNMF-E run again ("" when it does not)."""
        cnmf = rec.stages.get("cnmf")
        if cnmf is None or cnmf.state not in (State.DONE, State.RUNNING):
            return ""
        # the values the run used change nothing it reads (cnmf_params makes them int / float), so CNMF-E does not
        # run again and the question would be false
        if self._run_values(cnmf, item) == (g, c, p):
            return ""
        if _gate_on(self._session_cfg(item)):
            ask = CNMF_DONE_ASK if cnmf.state is State.DONE else CNMF_RUNNING_ASK
        else:
            ask = CNMF_DONE_ASK_NO_GATE if cnmf.state is State.DONE else CNMF_RUNNING_ASK_NO_GATE
        return ask.format(sid=rec.sid)

    def _gate_on_for(self, sid: str) -> bool:
        try:
            return _gate_on(config.resolve(self.paths, sid=sid))
        except Exception:  # cannot be read: the defaults (gate on), as for _session_cfg
            return True

    def accept_and_next(self) -> None:
        item = self._by_sid.get(self._sid) if self._sid else None
        if self._refusal(item):
            return
        sid, g, c, p = self.current()
        # before any question: its modal loop runs the timer, which may reload or clear the maps meanwhile
        maps = self._maps
        counters = tm.counters(maps, g, c, p, self._min_blob_px)
        rec = self._record(sid)  # fresh: CNMF-E may have started or finished, or the session been excluded
        if rec is None:
            self._gone([sid])
            return
        if rec.excluded:
            self._message("BEARMiND", WARNING_TEXT["excluded_session"].format(sid=sid))
            self.refresh()
            return
        question = self._rerun_question(rec, item, g, c, p)
        if question and not self._confirm(question, yes="Write", no="Cancel"):
            return
        try:
            res = actions.set_thresholds(self.paths, sid, g, c, p, who=actions.current_user(), source="window",
                                         counters=counters)
        except actions.UnknownSession as e:
            self._gone(e.unknown)
            return
        self._after_write(res)
        self._next_waiting(f"Thresholds of {sid} written: gSig {g}, {_values_text(c, p)}.")

    def accept_for_mouse(self) -> None:
        item = self._by_sid.get(self._sid) if self._sid else None
        if self._refusal(item):
            return
        sid, g, c, p = self.current()
        # before the question (its modal loop may reload the maps): the counters the person saw, of this session
        counters = tm.counters(self._maps, g, c, p, self._min_blob_px)
        mouse = item.mouse
        try:
            plan = actions.mouse_plan(self.paths, mouse)  # the rule apply_to_mouse writes by
        except actions.UnknownSession:
            self._gone([mouse])
            return
        unreadable = (f"Config cannot be read, left alone: {_listed(f'{sid} ({why})' for sid, why in plan.unreadable)}"
                      if plan.unreadable else "")
        if not plan.will_get:
            self._message("BEARMiND", "\n".join(
                [f"Every session of {mouse} has its own thresholds, is excluded or has a config that cannot be "
                 "read: nothing to write."] + ([unreadable] if unreadable else [])))
            return
        lines = [f"Thresholds gSig {g}, {_values_text(c, p)} for the sessions of {mouse}.",
                 f"Will get these thresholds: {_listed(plan.will_get)}",
                 f"Keep their own: {_listed(plan.keep_own) if plan.keep_own else 'none'}"]
        if plan.excluded:
            lines.append(f"Excluded, left alone: {_listed(plan.excluded)}")
        if unreadable:
            lines.append(unreadable)
        inspect = [s for s in plan.cnmf_done if self._gate_on_for(s)]  # the gate is per session (resolved config)
        no_gate = [s for s in plan.cnmf_done if s not in inspect]
        if inspect:
            lines.append(f"CNMF-E of {_listed(inspect)} is already done: it will be recomputed, and manual "
                         "inspection will be needed again.")
        if no_gate:
            lines.append(f"CNMF-E of {_listed(no_gate)} is already done: it will be recomputed.")
        if not self._confirm("\n".join(lines), yes="Write", no="Cancel"):
            return
        try:
            res = actions.apply_to_mouse(self.paths, mouse, g, c, p, actions.current_user(), shown=sid,
                                         counters=counters)
        except actions.UnknownSession:
            self._gone([mouse])
            return
        self._after_write(res, warn=False)  # its warnings are the groups the question already named
        if sorted(res.ok + list(res.failed)) != sorted(plan.will_get):  # the registry changed after the question
            self._message("BEARMiND", f"The sessions of {mouse} changed after the question. "
                                      f"Written: {_listed(res.ok) if res.ok else 'none'}.")
        total = len(plan.will_get) + len(plan.keep_own) + len(plan.excluded) + len(plan.unreadable)
        self._next_waiting(f"Thresholds written for {len(res.ok)} of {total} sessions of {mouse}.")

    def _after_write(self, res: actions.ActionResult, warn: bool = True) -> None:
        """Says what the write could not do (a failed session, a choice log another program holds), then goes on
        as after any write: what was written changes Monitor's count. `warn`: also the other warnings."""
        lines = [f"Writing failed for {sid}: {detail}" for sid, (_code, detail) in res.failed.items()]
        logged = [code for _sid, code in res.warnings if code in CHOICE_LOG_TEXT]
        lines += [CHOICE_LOG_TEXT[code] for code in dict.fromkeys(logged)]  # once each, for any number of rows
        if warn:
            lines += [WARNING_TEXT.get(code, "{sid}: " + code).format(sid=sid) for sid, code in res.warnings
                      if code not in CHOICE_LOG_TEXT]
        if lines:
            self._message("BEARMiND", "\n".join(lines))
        self.decided.emit()
        self.refresh()

    def _next_waiting(self, done: str) -> None:
        """The next waiting session in list order after the shown one, wrapping to the top; none: stay."""
        order = [i.sid for i in self._items]
        waiting = {i.sid for i in self._items if i.state == "waiting"}
        start = order.index(self._sid) if self._sid in order else -1
        for k in range(1, len(order) + 1):
            sid = order[(start + k) % len(order)]
            if sid in waiting and sid != self._sid:
                self.show_session(sid)
                self._say(f"{done} Next: {sid}.")
                return
        if self._sid in self._by_sid and self._by_sid[self._sid].chosen is not None:
            self._source, self._note = "chosen", ""  # the shown values are this session's choice now
            self._show_source()
        # "all have thresholds" only when it is true: sessions without maps or not reached yet have none
        without = [i.sid for i in self._items if i.state != "excluded" and i.chosen is None]
        rest = ALL_HAVE if not without else \
            f"No other session with maps waits for thresholds. Still without thresholds: {_listed(without)}."
        self._say(f"{done} {rest}")
