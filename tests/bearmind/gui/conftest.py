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
