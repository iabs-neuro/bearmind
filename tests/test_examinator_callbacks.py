# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import sys
from pathlib import Path

from bokeh.document.document import Document

sys.path.insert(0, str(Path(__file__).parent.parent))

import bm_examinator
from examinator_helpers import App as _App, make_estimates as _estimates


def _open(tmp_path) -> _App:
    return _App(bm_examinator.make_app(str(_estimates(tmp_path)), 20, {"mode": "legacy", "end_frame": 300}))


def test_partial_list_matches_the_view_after_each_callback(tmp_path):
    """Each callback leaves storage.estimates_partial equal to the rows it shows, without restore_callback."""
    app = _open(tmp_path)
    app.select([2]); app.first("Delete sel.")
    assert app.partial() == app.shown() == [0, 1, 3, 4, 5, 6, 7]
    app.select([0, 1]); app.first("Merge sel.")
    assert app.partial() == app.shown() == [1, 2, 3, 4, 5, 6]   # CaImAn renumbers; the merged one is 6
    app.click("Discard all")
    app.select([2, 3, 4, 5]); app.click("Show sel.")
    app.select([1]); app.click("Delete sel.")
    app.first("Revert change")
    assert app.partial() == app.shown() == [2, 3, 4, 5]   # the zoomed view before the delete


def test_delete_merge_zoom_pick_the_selected_rows(tmp_path):
    app = _open(tmp_path)
    steps = [([2], "Delete sel.", [0, 1, 3, 4, 5, 6, 7]),
             ([2], "Delete sel.", [0, 1, 4, 5, 6, 7]),
             ([0, 1], "Merge sel.", [2, 3, 4, 5, 6]),     # CaImAn renumbers; the merged one is 6
             ([0], "Delete sel.", [3, 4, 5, 6]),
             ([1, 2, 3], "Show sel.", [4, 5, 6]),
             ([1, 2], "Show sel.", [5, 6]),               # rows of the zoomed view, not of the full list
             ([0], "Delete sel.", [3, 4, 6])]
    for rows, label, expected in steps:
        app.select(rows); app.click(label)
        assert app.shown() == expected, (rows, label)


def test_revert_restores_the_last_change(tmp_path):
    app = _open(tmp_path)
    app.select([2, 3, 4, 5]); app.click("Show sel.")
    app.select([1]); app.click("Delete sel.")
    assert app.shown() == [0, 1, 2, 4, 5, 6, 7]
    app.select([3]); app.click("Delete sel.")
    app.click("Revert change")
    assert app.shown() == [0, 1, 2, 4, 5, 6, 7]
    app.select([3]); app.click("Delete sel.")
    app.select([0, 1]); app.click("Merge sel.")
    assert app.shown() == [0, 3, 4, 5, 6]
    app.click("Revert change")
    assert app.shown() == [0, 1, 2, 5, 6, 7]
    app.select([0]); app.click("Delete sel.")
    assert app.shown() == [1, 2, 5, 6, 7]


def test_no_selection_after_a_change(tmp_path):
    """A second click without a new selection changes nothing (all rows used to stay selected)."""
    app = _open(tmp_path)
    app.select([2]); app.click("Delete sel.")
    assert list(app.src.selected.indices) == []
    app.click("Delete sel.")
    assert app.shown() == [0, 1, 3, 4, 5, 6, 7]
    # the empty click must not overwrite the snapshot that Revert change restores
    app.click("Revert change")
    assert app.shown() == [0, 1, 2, 3, 4, 5, 6, 7]
    app.select([2]); app.click("Delete sel.")
    assert app.shown() == [0, 1, 3, 4, 5, 6, 7]
    app.select([0, 1]); app.click("Merge sel.")
    assert list(app.src.selected.indices) == []
    app.click("Merge sel.")
    assert app.shown() == [1, 2, 3, 4, 5, 6]
    app.click("Revert change")
    assert app.shown() == [0, 1, 3, 4, 5, 6, 7]


def test_examine_cells_shows_the_app(tmp_path, monkeypatch):
    shown = []
    monkeypatch.setattr(bm_examinator, "show", shown.append)
    bm_examinator.ExamineCells(str(_estimates(tmp_path)), default_fps=20,
                               bkapp_kwargs={"mode": "legacy", "end_frame": 300})
    assert len(shown) == 1 and callable(shown[0])
    shown[0](Document())  # builds without a server
