# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import pickle
import sys
from pathlib import Path

import numpy as np
from bokeh.document.document import Document
from bokeh.events import ButtonClick
from bokeh.models import Button, ColumnDataSource
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.params import CNMFParams
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

import bm_examinator


def _estimates(folder) -> Path:
    """8 round footprints on 48x48, 300 frames; enough for delete, merge, zoom and revert."""
    rng = np.random.default_rng(0)
    n, dims, T = 8, (48, 48), 300
    yy, xx = np.mgrid[0:dims[0], 0:dims[1]]
    A = np.zeros((dims[0] * dims[1], n))
    C = np.zeros((n, T))
    for k in range(n):
        cy, cx = 8 + 10 * (k % 4), 10 + 20 * (k // 4)
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 2.0 ** 2))
        A[:, k] = g.ravel(order="F")
        spikes = (rng.random(T) < 0.03).astype(float)
        C[k] = np.convolve(spikes, np.exp(-np.arange(60) / 5.0))[:T] + 0.01 * k
    est = Estimates(A=csc_matrix(A), C=C, b=None, f=None, dims=dims)
    est.YrA = 0.01 * rng.normal(size=(n, T))
    est.S = np.zeros((n, T))
    est.bl = np.zeros(n)
    est.c1 = np.zeros(n)
    est.neurons_sn = np.ones(n)
    est.g = np.full((n, 1), 0.9)
    est.idx_components = np.arange(n)
    est.idx_components_bad = np.array([], dtype=int)
    est.SNR_comp = np.arange(n) + 1.0
    est.r_values = np.linspace(0.5, 0.9, n)
    est.imax = np.ones(dims)
    est.cnmf_dict = CNMFParams(params_dict={"p": 1, "fr": 20}).to_dict()
    path = Path(folder) / "NOF_H01_1D_estimates.pickle"  # the window takes the session id from the name
    with open(path, "wb") as f:
        pickle.dump(est, f)
    return path


class _App:
    """make_app(...)(Document()) and handles to what the user clicks."""

    def __init__(self, bkapp):
        doc = Document()
        bkapp(doc)
        self.buttons = {}
        sources = []
        for root in doc.roots:
            for m in root.references():
                if isinstance(m, Button):
                    self.buttons[m.label] = m
                elif isinstance(m, ColumnDataSource) and "idx" in m.data:
                    sources.append(m)
        assert len(sources) == 1, len(sources)
        self.src = sources[0]
        self.storage = self.buttons["Delete sel."]._event_callbacks["button_click"][0].keywords["storage"]

    def select(self, rows):
        self.src.selected.indices = list(rows)

    def click(self, label):  # whole chain
        b = self.buttons[label]
        b._trigger_event(ButtonClick(b))

    def first(self, label):  # own callback only
        b = self.buttons[label]
        b._event_callbacks["button_click"][0](ButtonClick(b))

    def shown(self):
        return [int(i) for i in self.src.data["idx"]]

    def partial(self):
        return [int(i) for i in self.storage.estimates_partial.idx_components]


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
