import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before PySide6 is imported

import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def tmp_series(tmp_path):
    p = tmp_path / "series"
    p.mkdir()
    return p


@pytest.fixture(autouse=True)
def _refresh_errors_fail_tests(monkeypatch):
    """A programming error inside refresh() is reported to the window; in tests it must fail the test.
    A test that passes its own `report` keeps it."""
    from bearmind.gui.monitor_tab import MonitorTab

    real = MonitorTab.__init__

    def reraise(exc_type, exc, tb):
        raise exc

    def init(self, *a, **k):
        k.setdefault("report", reraise)
        real(self, *a, **k)

    monkeypatch.setattr(MonitorTab, "__init__", init)
