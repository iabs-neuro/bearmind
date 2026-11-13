"""Main window of BEARMiND: series chooser and tabs. Tab contents live in their own modules."""
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QCloseEvent
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QTabWidget,
                               QVBoxLayout, QWidget)

from bearmind import config
from bearmind.gui.inspection_tab import InspectionTab
from bearmind.gui.monitor_tab import SELECTION_STYLE, MonitorTab
from bearmind.gui.scan_tab import ScanTab
from bearmind.gui.thresholds_tab import ThresholdsTab

TAB_NAMES = ("Monitor", "Scan", "Thresholds", "Inspection")
MAX_RECENT = 10
NO_RAW_ASK = ("This folder has no 1_Raw subfolder, so it does not look like a series folder:\n{series}\n\n"
              "Opening it creates a _bearmind folder inside it. Open it as a series anyway?")
SCAN_RUNNING = "A scan is running. Wait for it to finish, then try again."


def _hint() -> QWidget:
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label = QLabel("Open a series: Series > Open series…")  # the menu item's own text
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(label)
    return page


def _default_message(title: str, text: str) -> None:
    QMessageBox.warning(None, title, text)


def _default_confirm(text: str) -> bool:
    return QMessageBox.question(None, "BEARMiND", text) == QMessageBox.StandardButton.Yes


class MainWindow(QMainWindow):
    def __init__(self, series: Path | None = None, message=None, confirm=None):
        super().__init__()
        self._message = message or _default_message
        self._confirm = confirm or _default_confirm
        self.series: Path | None = None
        self.resize(1100, 700)
        self.setWindowTitle("BEARMiND")
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        for name in TAB_NAMES:  # every tab is built when a series is opened
            self.tabs.addTab(_hint(), name)
        self._build_menus()
        if series is not None:
            self.open_series(series)

    def _build_menus(self) -> None:
        self.series_menu = self.menuBar().addMenu("Series")
        open_action = QAction("Open series…", self)
        open_action.triggered.connect(self._choose_series)
        self.series_menu.addAction(open_action)
        self.recent_menu = self.series_menu.addMenu("Recent")
        self.actions_menu = self.menuBar().addMenu("Actions")  # filled when a series is opened
        self._fill_recent()

    def _choose_series(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open series folder")
        if folder:
            self.open_series(Path(folder))

    def _recent(self) -> list[str]:
        gui = config.machine_config().get("gui") or {}
        return [str(p) for p in (gui.get("recent_series") or [])]

    def _fill_recent(self) -> None:
        self.recent_menu.clear()
        for path in self._recent():
            act = self.recent_menu.addAction(path)
            act.triggered.connect(lambda _=False, p=path: self.open_series(Path(p)))
        self.recent_menu.setEnabled(not self.recent_menu.isEmpty())

    def open_series(self, series: Path) -> bool:
        series = Path(series)
        if self._scan_running():  # the scan writes the old series' registry: it is not cut off or orphaned
            self._refuse("A scan is running. Wait for it to finish, then open the series.")
            return False
        if not series.is_dir():
            self._message("BEARMiND", f"Series folder not found: {series}")
            return False
        series = series.resolve()
        if not (series / "1_Raw").is_dir() and not (series / "_bearmind").exists():
            # opening 1_Raw itself (or any other folder of a series) would put _bearmind inside it
            if not self._confirm(NO_RAW_ASK.format(series=series)):
                return False
        # the new tabs first: if building them fails, the window keeps the old series, tabs, title and recent list
        built = []
        try:
            built.append(MonitorTab(series))
            built.append(ScanTab(series))
            built.append(ThresholdsTab(series))
            built.append(InspectionTab(series))
        except Exception:
            for tab in built:
                self._retire(tab)
            self.report_exception(*sys.exc_info())
            return False
        self._set_tabs(*built)
        self.series = series
        self.setWindowTitle(f"BEARMiND — {self.series}")
        shown = str(self.series)
        recent = [shown] + [p for p in self._recent() if p != shown]
        # update_yaml deep-merges, and a list is replaced as a whole, so the new order is stored verbatim
        config.update_yaml(config.machine_file(), {"gui": {"recent_series": recent[:MAX_RECENT]}})
        self._fill_recent()
        return True

    @staticmethod
    def _retire(tab: QWidget) -> None:
        """A tab leaving the window: its timers stop, a running scan ends its writes first and the MC video is
        released."""
        if isinstance(tab, MonitorTab):
            tab.timer.stop()
        if isinstance(tab, (ScanTab, ThresholdsTab, InspectionTab)):
            tab.shutdown()
        tab.deleteLater()

    def _replace_tab(self, name: str, tab: QWidget) -> None:
        index = TAB_NAMES.index(name)
        old = self.tabs.widget(index)
        self.tabs.removeTab(index)
        self._retire(old)
        self.tabs.insertTab(index, tab, name)

    def _set_tabs(self, monitor_tab: MonitorTab, scan_tab: ScanTab, thresholds_tab: ThresholdsTab,
                  inspection_tab: InspectionTab) -> None:
        self._replace_tab("Monitor", monitor_tab)
        self._replace_tab("Scan", scan_tab)
        self._replace_tab("Thresholds", thresholds_tab)
        self._replace_tab("Inspection", inspection_tab)
        self.tabs.setCurrentIndex(0)
        self.monitor, self.scan, self.thresholds = monitor_tab, scan_tab, thresholds_tab
        self.inspection = inspection_tab
        self.actions_menu.clear()
        self.actions_menu.addActions(monitor_tab.menu_actions)
        monitor_tab.open_tab.connect(self._open_from_monitor)
        for tab in (monitor_tab, scan_tab, thresholds_tab, inspection_tab):
            tab.status_message.connect(self.statusBar().showMessage)
        # written thresholds change Monitor's "Thresholds: N sessions" at once, not at its next tick; a "Not
        # needed" changes its "Manual inspection: N sessions"
        thresholds_tab.decided.connect(monitor_tab.refresh)
        inspection_tab.decided.connect(monitor_tab.refresh)
        # a finished scan changes the table and the not-scanned state, which the timer would show only later; a
        # failed one may have written records before it failed
        scan_tab.scan_finished.connect(lambda ok: monitor_tab.refresh())
        scan_tab.decided.connect(monitor_tab.refresh)

    def _scan_running(self) -> bool:
        tab = self.tabs.widget(TAB_NAMES.index("Scan"))
        return isinstance(tab, ScanTab) and tab.is_scanning()

    def _refuse(self, status: str) -> None:
        """The status line alone is easy to miss after a click: a message box says it too."""
        self.statusBar().showMessage(status)
        self._message("BEARMiND", SCAN_RUNNING)

    def closeEvent(self, event: QCloseEvent) -> None:
        # simplest safe choice: keep the window until the scan ends (a scan cannot be cancelled half-way)
        if self._scan_running():
            self._refuse("A scan is running. Wait for it to finish, then close the window.")
            event.ignore()
            return
        super().closeEvent(event)

    def show_tab(self, name: str) -> None:
        self.tabs.setCurrentIndex(TAB_NAMES.index(name))

    def _open_from_monitor(self, name: str) -> None:
        """Monitor's buttons: "Thresholds: N sessions" and "Manual inspection: N sessions" open their tab on the
        first session that waits (for inspection: one whose CNMF-E is current)."""
        if name == "Thresholds":
            self.thresholds.show_first_waiting()
        elif name == "Inspection":
            self.inspection.show_first_waiting()
        self.show_tab(name)

    def report_exception(self, exc_type, exc, tb) -> None:
        """Slot exceptions must not close the window: log the traceback, show the first line."""
        _report(self._message, exc_type, exc, tb)


def _report(message, exc_type, exc, tb) -> None:
    text = "".join(traceback.format_exception(exc_type, exc, tb))
    try:
        log = config.machine_file().parent / "gui.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with open(log, "a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat(timespec='seconds')}\n{text}\n")
    except OSError:
        pass
    first = (str(exc).splitlines() or [""])[0]
    message("BEARMiND error", f"{exc_type.__name__}: {first}")


def _gui_log_handler() -> logging.Handler | None:
    """The bearmind logger's warnings (a slow refresh, a refresh that keeps failing) into gui.log: under pythonw
    there is no console, and they would be lost."""
    f = config.machine_file().parent / "gui.log"
    try:
        f.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return None
    h = logging.FileHandler(f, encoding="utf-8", delay=True)  # opened at the first warning
    h.setLevel(logging.WARNING)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger("bearmind").addHandler(h)
    return h


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    handler = _gui_log_handler()
    try:
        app = QApplication.instance() or QApplication(sys.argv[:1])
        app.setStyleSheet(SELECTION_STYLE)
        # before the window exists: under pythonw an exception in the constructor would otherwise vanish
        sys.excepthook = lambda t, e, tb: _report(_default_message, t, e, tb)
        try:
            window = MainWindow(Path(argv[0]) if argv else None)
        except Exception:
            sys.excepthook(*sys.exc_info())
            return 1
        sys.excepthook = window.report_exception
        window.show()
        return app.exec()
    finally:
        if handler is not None:
            logging.getLogger("bearmind").removeHandler(handler)
            handler.close()
