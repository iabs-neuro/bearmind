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
    for i, phase in ((1, 3), (2, 4), (3, 5)):
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
