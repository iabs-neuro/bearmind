import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # before PySide6 is imported

import pytest
from PySide6.QtCore import QEvent
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
    A test that passes its own `report` keeps it. The Inspection tab of a test never starts a real inspection
    server or browser: one the test does not replace raises, and the test fails at its end even when the call
    came from a Qt slot (an exception there is only printed) or was caught by the tab."""
    from bearmind.gui.inspection_tab import InspectionTab
    from bearmind.gui.monitor_tab import MonitorTab
    from bearmind.gui.scan_tab import ScanTab
    from bearmind.gui.thresholds_tab import ThresholdsTab

    def reraise(exc_type, exc, tb):
        raise exc

    hits = []

    def no_real(*a, **k):
        hits.append(a)
        raise AssertionError("a test started a real inspection server or browser")

    for cls in (MonitorTab, ScanTab, ThresholdsTab, InspectionTab):
        def init(self, *a, real=cls.__init__, inspection=cls is InspectionTab, **k):
            k.setdefault("report", reraise)
            if inspection:
                k.setdefault("spawn", no_real)
                k.setdefault("open_url", no_real)
                # its dialogs are modal: offscreen, one would block the test for ever instead of failing it
                k.setdefault("message", lambda title, text: None)
                k.setdefault("confirm", lambda text: False)
            real(self, *a, **k)

        monkeypatch.setattr(cls, "__init__", init)
    yield
    assert not hits, f"a test reached a real inspection server start or browser: {hits}"


@pytest.fixture(autouse=True)
def _join_video_workers(qapp):
    """The tests never run exec(), so aboutToQuit never joins the MC video workers: a worker left running is
    destroyed at interpreter exit (exit code 127 after "all passed"), and the global registry leaks into the next
    test. Widgets deleted with deleteLater go too, since no event loop runs to delete them."""
    yield
    from bearmind.gui import video_view

    video_view._join_all()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete)
