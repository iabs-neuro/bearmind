"""Main window of BEARMiND: series chooser and tabs. Tab contents live in their own modules."""
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QTabWidget,
                               QVBoxLayout, QWidget)

from bearmind import config
from bearmind.gui.monitor_tab import MonitorTab

TAB_NAMES = ("Monitor", "Scan", "Thresholds", "Inspection")
_PHASE = {"Scan": 3, "Thresholds": 4, "Inspection": 5}
PLACEHOLDER_TEXT = "This tab is coming in a later version of BEARMiND."
MAX_RECENT = 10


def _placeholder(name: str) -> QWidget:
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    for text in (PLACEHOLDER_TEXT, f"Planned for phase {_PHASE[name]}."):
        label = QLabel(text)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(label)
    return page


def _hint() -> QWidget:
    page = QWidget()
    lay = QVBoxLayout(page)
    lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label = QLabel("Open a series: Series > Open series...")
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    lay.addWidget(label)
    return page


def _default_message(title: str, text: str) -> None:
    QMessageBox.warning(None, title, text)


class MainWindow(QMainWindow):
    def __init__(self, series: Path | None = None, message=None):
        super().__init__()
        self._message = message or _default_message
        self.series: Path | None = None
        self.resize(1100, 700)
        self.setWindowTitle("BEARMiND")
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        for name in TAB_NAMES:
            # the Monitor tab is built when a series is opened
            self.tabs.addTab(_placeholder(name) if name in _PHASE else _hint(), name)
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
        if not series.is_dir():
            self._message("BEARMiND", f"Series folder not found: {series}")
            return False
        self.series = series.resolve()
        self.setWindowTitle(f"BEARMiND — {self.series}")
        shown = str(self.series)
        recent = [shown] + [p for p in self._recent() if p != shown]
        # update_yaml deep-merges, and a list is replaced as a whole, so the new order is stored verbatim
        config.update_yaml(config.machine_file(), {"gui": {"recent_series": recent[:MAX_RECENT]}})
        self._fill_recent()
        self._set_monitor(MonitorTab(self.series))
        return True

    def _set_monitor(self, tab: MonitorTab) -> None:
        old = self.tabs.widget(0)
        if isinstance(old, MonitorTab):
            old.timer.stop()
        self.tabs.removeTab(0)
        old.deleteLater()
        self.tabs.insertTab(0, tab, TAB_NAMES[0])
        self.tabs.setCurrentIndex(0)
        self.monitor = tab
        self.actions_menu.clear()
        self.actions_menu.addActions(tab.menu_actions)
        tab.open_tab.connect(self.show_tab)
        tab.status_message.connect(self.statusBar().showMessage)

    def show_tab(self, name: str) -> None:
        self.tabs.setCurrentIndex(TAB_NAMES.index(name))

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
