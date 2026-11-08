# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import copy
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bokeh.document.document import Document
from bokeh.models import ColumnDataSource
from caiman.source_extraction.cnmf.estimates import Estimates

sys.path.insert(0, str(Path(__file__).parent.parent))

import bm_examinator
from examinator_helpers import App as _App, changed_columns, make_estimates as _estimates

SELECT = "Select a neuron."
# columns that hold traces, the time axis or contours: large, and sent only when they change
BULK = {"traces", "traces_recon", "times", "t", "xs", "ys"}


def _open(tmp_path) -> _App:
    return _App(bm_examinator.make_app(str(_estimates(tmp_path)), 20, {"mode": "legacy", "end_frame": 300}))


def _normalized(trace):
    return (trace - trace.min()) / (trace.max() - trace.min())


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


def test_revert_button_chain_keeps_zoom(tmp_path):
    app = _open(tmp_path)
    app.select([2, 3, 4, 5]); app.click("Show sel.")
    app.select([1]); app.click("Delete sel.")
    app.click("Revert change")
    assert app.shown() == [2, 3, 4, 5]


def test_metrics_panel_cleared_after_change(tmp_path):
    """The panel never names a neuron that a change removed or renumbered."""
    app = _open(tmp_path)
    app.select([2])
    assert "Neuron #2<" in app.panel.text
    app.click("Delete sel.")
    assert app.panel.text == SELECT
    app.select([0, 1])
    assert "Neuron #0<" in app.panel.text
    app.click("Merge sel.")
    assert app.panel.text == SELECT
    app.select([1])
    assert "Neuron #2<" in app.panel.text   # after the merge the rows are 1 2 3 4 5 6
    app.click("Revert change")
    assert app.panel.text == SELECT
    app.select([3])
    assert "Neuron #4<" in app.panel.text
    app.click("Discard all")
    assert app.panel.text == SELECT


def test_delete_does_not_copy_estimates(tmp_path, monkeypatch):
    app = _open(tmp_path)
    calls = []
    real = copy.deepcopy

    def counting(x, *args, **kwargs):
        if isinstance(x, Estimates):
            calls.append(type(x).__name__)
        return real(x, *args, **kwargs)

    monkeypatch.setattr(copy, "deepcopy", counting)
    for rows, label in [([2], "Delete sel."), ([0, 1], "Merge sel."), ([1, 2, 3], "Show sel."),
                        (None, "Revert change")]:
        calls.clear()
        if rows is not None:
            app.select(rows)
        app.click(label)
        assert calls == [], label


def test_delete_sends_no_traces(tmp_path):
    app = _open(tmp_path)
    for rows, label in [([2], "Delete sel."), ([1, 2, 3], "Show sel."), ([0], "Delete sel."),
                        (None, "Reset view"), (None, "Revert change")]:
        with app.events() as events:
            if rows is not None:
                app.select(rows)
            app.click(label)
        assert events, label
        sent = {col for _, col in changed_columns(events)}
        assert not sent & BULK, (label, sent)


def _time_arrays(app, time):
    """Arrays anywhere in the document's data sources that are a stretch of the session's time axis."""
    found = []

    def is_time(v):
        v = np.asarray(v)
        return (v.ndim == 1 and v.dtype.kind == "f" and len(v) > 2 and np.all(np.diff(v) > 0)
                and np.all(np.isin(v, time)))
    for m in app.doc.roots[0].references():
        if isinstance(m, ColumnDataSource):
            for k, col in m.data.items():
                items = col if len(col) and isinstance(col[0], (list, np.ndarray)) else [col]
                found += [(k, len(v)) for v in items if len(v) and is_time(v)]
    return found


def test_time_axis_sent_once(tmp_path):
    path = _estimates(tmp_path, n=8, T=6000)
    app = _App(bm_examinator.make_app(str(path), 20, {"mode": "legacy", "end_frame": None}))
    time = app.storage.estimates.time
    found = _time_arrays(app, time)
    assert len(found) == 1 and found[0][1] <= 2000, found
    t = app.time_axis()
    assert t[0] == time[0] and t[-1] == time[-1]   # the whole session
    for tr in app.src.data["traces"]:
        assert len(tr) == len(t)


def test_zoom_sends_full_resolution_window(tmp_path):
    path = _estimates(tmp_path, n=8, T=6000)
    app = _App(bm_examinator.make_app(str(path), 20, {"mode": "legacy", "end_frame": None}))
    est = app.storage.estimates
    app.select([2]); app.click("Delete sel.")
    app.zoom(10.0, 20.0)
    t = app.time_axis()
    i0 = int(np.searchsorted(est.time, t[0]))
    frames = np.arange(i0, i0 + len(t))
    assert np.array_equal(t, est.time[frames])          # one point per frame
    assert t[0] <= 10.0 and t[-1] >= 20.0 and len(t) <= 2000
    for s, comp in zip(app._slots(), app.shown()):
        tr = np.asarray(app.src.data["traces"][s])
        assert np.allclose(tr, _normalized(est.C[comp])[frames], atol=1e-6), comp
    app.zoom(-20.0, 400.0)                              # zoomed out again: the whole session
    t = app.time_axis()
    assert t[0] == est.time[0] and t[-1] == est.time[-1] and len(t) <= 2000


def test_merge_recomputes_one_contour(tmp_path, monkeypatch):
    app = _open(tmp_path)
    columns = []
    real = bm_examinator.cm.utils.visualization.get_contours

    def spy(A, *args, **kwargs):
        columns.append(A.shape[1])
        return real(A, *args, **kwargs)

    monkeypatch.setattr(bm_examinator.cm.utils.visualization, "get_contours", spy)
    app.select([0, 1]); app.click("Merge sel.")
    assert columns == [1]
    assert app.shown() == [0, 1, 2, 3, 4, 5, 6]


def test_merge_params_built_once(tmp_path, monkeypatch):
    """Building CNMFParams takes about 0.25 s; manual_merge only reads it, so one per window is enough."""
    app = _open(tmp_path)
    built = []
    real = bm_examinator.params.CNMFParams

    def counting(*args, **kwargs):
        built.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(bm_examinator.params, "CNMFParams", counting)
    app.select([0, 1]); app.click("Merge sel.")
    app.select([2, 3]); app.click("Merge sel.")
    app.click("Revert change")
    app.select([0, 1]); app.click("Merge sel.")
    assert len(built) == 1
    assert app.shown() == [0, 1, 2, 3, 4, 5]   # 8 components, two merges of two in effect


def _capcan_estimates(folder):
    path = _estimates(folder)
    with open(path, "rb") as f:
        est = pickle.load(f)
    rng = np.random.default_rng(1)
    est.metrics_df = pd.DataFrame({"component_idx": range(8), "snr_x": rng.random(8),
                                   "width": rng.random(8) + 1.0})
    with open(path, "wb") as f:
        pickle.dump(est, f)
    return path


def test_capcan_sort_after_merge(tmp_path):
    app = _App(bm_examinator.make_app(str(_capcan_estimates(tmp_path)), 20,
                                      {"mode": "capcan", "end_frame": 300}))
    app.select([0, 1]); app.click("Merge sel.")
    assert app.shown() == [0, 1, 2, 3, 4, 5, 6]          # CaImAn renumbers; the merged one is 6
    app.sort("width")
    heights = app.heights()
    assert heights[6] == max(heights.values())         # no metric for the merged neuron: last
    assert sorted(heights.values()) == list(range(7))
    app.select([5]); app.click("Delete sel.")
    assert app.shown() == [0, 1, 2, 3, 4, 6]


def test_capcan_reconstructions_follow_the_rows(tmp_path):
    path = _capcan_estimates(tmp_path)
    with open(path, "rb") as f:
        est = pickle.load(f)
    est.reconstructions = {k: 0.5 * est.C[k] for k in range(8)}
    with open(path, "wb") as f:
        pickle.dump(est, f)
    app = _App(bm_examinator.make_app(str(path), 20, {"mode": "capcan", "end_frame": 300}))
    app.select([0, 1]); app.click("Merge sel.")
    app.zoom(5.0, 10.0)
    t = app.time_axis()
    assert len(t) < 300
    for s, comp in zip(app._slots(), app.shown()):
        assert len(app.src.data["traces_recon"][s]) == len(app.src.data["traces"][s]) == len(t), comp
    merged = app._slots()[-1]
    assert not np.any(app.src.data["traces_recon"][merged])   # no reconstruction for a merged neuron


def test_decimate_minmax():
    rng = np.random.default_rng(3)
    y = np.sin(np.linspace(0, 60, 10_000)) + 0.1 * rng.normal(size=10_000)
    y[1234], y[8765] = 9.0, -9.0
    frames, values = bm_examinator.decimate_minmax(y)
    assert len(frames) == len(values) <= 2000
    assert np.array_equal(values, y[frames]) and np.all(np.diff(frames) >= 0)
    assert values.max() == y.max() and values.min() == y.min()
    short = y[:500]
    frames, values = bm_examinator.decimate_minmax(short)
    assert np.array_equal(frames, np.arange(500)) and np.array_equal(values, short)
    # the window: the same values for every row, on one time axis from the first to the last frame
    Y = np.stack([y, -y, np.roll(y, 77)])
    frames, values = bm_examinator._decimate_rows(Y, 2000)
    assert len(frames) <= 2000 and np.all(np.diff(frames) > 0) and frames[0] == 0 and frames[-1] == len(y) - 1
    for row, v in zip(Y, values):
        assert np.array_equal(v, bm_examinator.decimate_minmax(row)[1])
