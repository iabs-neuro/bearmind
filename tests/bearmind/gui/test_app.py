import re
import subprocess
import sys
from pathlib import Path

import yaml
from PySide6.QtWidgets import QLabel

from bearmind import config
from bearmind.gui.app import MainWindow

REPO = Path(__file__).resolve().parents[3]


def test_window_opens_with_series(qapp, tmp_series):
    w = MainWindow(tmp_series)
    assert str(tmp_series.resolve()) in w.windowTitle()
    assert [w.tabs.tabText(i) for i in range(w.tabs.count())] == ["Monitor", "Scan", "Thresholds", "Inspection"]


def test_placeholder_tabs_text(qapp, tmp_series):
    w = MainWindow(tmp_series)
    for i, phase in ((2, 4), (3, 5)):
        text = " ".join(lb.text() for lb in w.tabs.widget(i).findChildren(QLabel))
        assert "coming in a later version" in text and f"phase {phase}" in text
        assert not re.search("[А-Яа-яЁё]", text)


def test_show_tab(qapp, tmp_series):
    w = MainWindow(tmp_series)
    w.show_tab("Thresholds")
    assert w.tabs.currentIndex() == 2


def test_recent_series_saved(qapp, tmp_series, tmp_path):
    other = tmp_path / "other"
    other.mkdir()
    w = MainWindow(tmp_series)
    w.open_series(other)
    w.open_series(tmp_series)
    recent = yaml.safe_load(config.machine_file().read_text(encoding="utf-8"))["gui"]["recent_series"]
    assert recent == [str(tmp_series.resolve()), str(other.resolve())]


def test_missing_folder_message(qapp, tmp_path):
    seen = []
    w = MainWindow(message=lambda title, text: seen.append(text))
    assert w.open_series(tmp_path / "nope") is False
    assert "Series folder not found" in seen[0]


def test_excepthook_logs_and_keeps_window(qapp, tmp_series):
    seen = []
    w = MainWindow(tmp_series, message=lambda title, text: seen.append(text))
    try:
        raise ValueError("boom\nsecond line")
    except ValueError:
        w.report_exception(*sys.exc_info())
    log = (config.machine_file().parent / "gui.log").read_text(encoding="utf-8")
    assert "Traceback" in log and "boom" in log
    assert seen == ["ValueError: boom"]
    assert w.tabs.count() == 4


def test_cli_gui_help():
    r = subprocess.run([sys.executable, "-m", "bearmind", "gui", "--help"], cwd=REPO, capture_output=True, text=True)
    assert r.returncode == 0 and "SERIES" in r.stdout


def test_open_series_builds_monitor_tab_and_routes_signals(qapp, tmp_series):
    from bearmind.gui.monitor_tab import MonitorTab

    w = MainWindow(tmp_series)
    tab = w.tabs.widget(0)
    assert isinstance(tab, MonitorTab) and w.tabs.tabText(0) == "Monitor"
    tab.open_tab.emit("Inspection")
    assert w.tabs.currentIndex() == 3
    tab.status_message.emit("hello")
    assert w.statusBar().currentMessage() == "hello"
    w.open_series(tmp_series)  # a second open replaces the tab and stops the old timer
    assert w.tabs.widget(0) is not tab and w.tabs.count() == 4 and not tab.timer.isActive()


def test_main_reports_constructor_failure(qapp, monkeypatch):
    from bearmind.gui import app

    seen = []

    def boom(*a, **k):
        raise RuntimeError("cannot build\nsecond")

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # restored after the test
    monkeypatch.setattr(app, "MainWindow", boom)
    monkeypatch.setattr(app, "_default_message", lambda title, text: seen.append(text))
    assert app.main([]) == 1
    assert seen == ["RuntimeError: cannot build"]
    assert "cannot build" in (config.machine_file().parent / "gui.log").read_text(encoding="utf-8")


def test_monitor_tab_hint_before_any_series(qapp):
    from PySide6.QtWidgets import QLabel

    w = MainWindow()
    texts = [lab.text() for lab in w.tabs.widget(0).findChildren(QLabel)]
    assert texts == ["Open a series: Series > Open series..."]


def test_main_writes_bearmind_warnings_to_gui_log(qapp, monkeypatch):
    """M7: under pythonw there is no console; the bearmind logger's warnings go to gui.log while the window runs."""
    import logging

    from bearmind.gui import app

    def window(*a, **k):
        logging.getLogger("bearmind.monitor").warning("monitor refresh took 3.0 s for 300 sessions")
        raise RuntimeError("stop here")

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)
    monkeypatch.setattr(app, "MainWindow", window)
    monkeypatch.setattr(app, "_default_message", lambda title, text: None)
    assert app.main([]) == 1
    text = (config.machine_file().parent / "gui.log").read_text(encoding="utf-8")
    assert "WARNING bearmind.monitor: monitor refresh took 3.0 s for 300 sessions" in text
    # the handler belongs to this run of the window only
    assert not [h for h in logging.getLogger("bearmind").handlers if isinstance(h, logging.FileHandler)]


def test_gui_import_error_is_logged_and_printed(monkeypatch, capsys):
    """M7: a broken PySide6 install must not vanish without a trace."""
    from bearmind import cli

    monkeypatch.setitem(sys.modules, "bearmind.gui.app", None)  # the import fails with ImportError
    assert cli.main(["gui"]) == 1
    out = capsys.readouterr().out
    log = config.machine_file().parent / "gui.log"
    assert out.startswith("[FAILED] the window cannot start: ModuleNotFoundError") and str(log) in out
    assert "ModuleNotFoundError" in log.read_text(encoding="utf-8")


def test_open_series_failure_keeps_the_old_series(qapp, tmp_path, monkeypatch):
    """I6: the new tabs are built first; if that fails the window keeps the old series, title, tabs and recent
    list, shows the error, and stops the tab already built."""
    from bearmind.gui.monitor_tab import MonitorTab
    from bearmind.gui.scan_tab import ScanTab

    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    seen = []
    w = MainWindow(a, message=lambda title, text: seen.append(text))
    old_tabs, title = [w.tabs.widget(i) for i in range(w.tabs.count())], w.windowTitle()
    made = []
    real_init = MonitorTab.__init__
    monkeypatch.setattr(MonitorTab, "__init__", lambda self, *x, **k: (made.append(self), real_init(self, *x, **k))[1])
    monkeypatch.setattr(ScanTab, "_build", lambda self: (_ for _ in ()).throw(RuntimeError("cannot build\nmore")))
    assert w.open_series(b) is False
    assert seen == ["RuntimeError: cannot build"]
    assert w.series == a.resolve() and w.windowTitle() == title
    assert [w.tabs.widget(i) for i in range(w.tabs.count())] == old_tabs and w.monitor is old_tabs[0]
    recent = yaml.safe_load(config.machine_file().read_text(encoding="utf-8"))["gui"]["recent_series"]
    assert recent == [str(a.resolve())]
    assert len(made) == 1 and not made[0].timer.isActive()
    assert old_tabs[0].timer.isActive()
