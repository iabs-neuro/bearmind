"""Hooks of the inspection window for a server: prepared session, save hooks, refusal, full-length traces."""
# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bokeh.document.document import Document
from bokeh.models import Button, Div

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import bm_examinator
from examinator_helpers import App, make_estimates

LEGACY = {"mode": "legacy", "end_frame": None}
REFUSAL = "This file is already autoinspected; open 4_Estimates instead."


def _models(doc):
    return [m for root in doc.roots for m in root.references()]


def _status(doc) -> Div:
    divs = [m for m in _models(doc) if isinstance(m, Div) and m.name == "status"]
    assert len(divs) == 1, divs
    return divs[0]


def _boom(*args, **kwargs):
    raise AssertionError("must not be called")


def test_import_has_no_notebook_side_effect():
    code = ("import torch\n"
            "import bokeh.io, bokeh.io.output, bokeh.plotting\n"
            "def boom(*a, **k):\n"
            "    raise RuntimeError('output_notebook called at import')\n"
            "bokeh.io.output_notebook = bokeh.io.output.output_notebook = boom\n"
            "bokeh.plotting.output_notebook = boom\n"
            "import bm_examinator\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]


def test_examine_cells_still_calls_output_notebook(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(bm_examinator, "output_notebook", lambda *a, **k: calls.append("output_notebook"))
    monkeypatch.setattr(bm_examinator, "show", lambda app: calls.append("show"))
    bm_examinator.ExamineCells(str(make_estimates(tmp_path)), default_fps=20,
                               bkapp_kwargs={"mode": "legacy", "end_frame": 300})
    assert calls == ["output_notebook", "show"]


def test_manual_seeds_calls_output_notebook(tmp_path, monkeypatch):
    """show() of an app function needs notebook output; the import no longer turns it on."""
    calls = []
    monkeypatch.setattr(bm_examinator, "output_notebook", lambda *a, **k: calls.append("output_notebook"))
    monkeypatch.setattr(bm_examinator, "show", lambda app: calls.append("show"))
    bm_examinator.ManualSeeds(str(tmp_path / "NOF_H01_1D_MC.tif"))
    assert calls == ["output_notebook", "show"]


def test_load_estimates_default_unchanged(tmp_path, monkeypatch):
    path = make_estimates(tmp_path)
    calls = []

    def fake(name, n_frames, default_fps=20):
        calls.append((name, n_frames, default_fps))
        return np.arange(n_frames) / 7.0

    monkeypatch.setattr(bm_examinator, "get_timestamps", fake)
    est = bm_examinator.LoadEstimates(str(path), default_fps=17)
    assert calls == [("NOF_H01_1D", 300, 17)]
    assert np.array_equal(est.time, np.arange(300) / 7.0) and est.name == str(path)

    calls.clear()
    t = np.linspace(0, 15, 300)
    est = bm_examinator.LoadEstimates(str(path), default_fps=17, time=t)
    assert calls == [] and est.time is t


def test_load_estimates_time_length(tmp_path, monkeypatch):
    """Longer timestamps are cut to the frames, as from a timestamps file; shorter ones cannot be plotted."""
    path = make_estimates(tmp_path)
    monkeypatch.setattr(bm_examinator, "get_timestamps", _boom)
    est = bm_examinator.LoadEstimates(str(path), time=np.arange(310) / 20.0)
    assert np.array_equal(est.time, np.arange(300) / 20.0)
    with pytest.raises(ValueError, match="299 timestamps for 300 frames"):
        bm_examinator.LoadEstimates(str(path), time=np.arange(299) / 20.0)


def test_prepared_session_used(tmp_path, monkeypatch):
    path = make_estimates(tmp_path)
    ref = App(bm_examinator.make_app(str(path), 20, LEGACY))  # the window loading the file itself
    prep = bm_examinator.prepare_session(str(path), bkapp_kwargs=LEGACY)
    assert prep.fname == str(path) and len(prep.contours) == 8
    assert np.array_equal(prep.estimates.time, ref.storage.estimates.time)

    monkeypatch.setattr(bm_examinator, "LoadEstimates", _boom)
    app = App(bm_examinator.make_app(str(path), 20, LEGACY, prepared=prep))
    assert set(app.src.data) == set(ref.src.data)
    for key in ref.src.data:
        assert len(app.src.data[key]) == len(ref.src.data[key]), key
        for a, b in zip(app.src.data[key], ref.src.data[key]):
            assert np.array_equal(np.asarray(a), np.asarray(b)), key
    app.select([2]); app.click("Delete sel.")
    assert app.shown() == [0, 1, 3, 4, 5, 6, 7]
    assert list(prep.estimates.idx_components) == list(range(8))  # the prepared session stays as loaded


def test_prepared_contours_reused(tmp_path, monkeypatch):
    path = make_estimates(tmp_path)
    t = np.arange(300) / 19.5
    prep = bm_examinator.prepare_session(str(path), time=t, bkapp_kwargs=LEGACY)
    assert prep.estimates.time is t
    monkeypatch.setattr(bm_examinator.cm.utils.visualization, "get_contours", _boom)
    again = bm_examinator.prepare_session(str(path), time=t, bkapp_kwargs=LEGACY, contours=prep.contours)
    assert again.contours is prep.contours
    for key in prep.data:
        for a, b in zip(again.data[key], prep.data[key]):
            assert np.array_equal(np.asarray(a), np.asarray(b)), key


def test_final_save_hook_called_and_text_shown(tmp_path, monkeypatch):
    inp, cwd = tmp_path / "in", tmp_path / "cwd"
    inp.mkdir(), cwd.mkdir()
    path = make_estimates(inp)
    monkeypatch.chdir(cwd)
    got = []

    def final_save(est):
        got.append(est)
        return "Saved NOF_H01_1D: 7 neurons"

    app = App(bm_examinator.make_app(str(path), 20, LEGACY, final_save=final_save))
    assert _status(app.doc).text == ""
    app.select([2]); app.click("Delete sel.")
    app.click("Save results")
    assert len(got) == 1 and list(got[0].idx_components) == [0, 1, 3, 4, 5, 6, 7]
    assert _status(app.doc).text == "Saved NOF_H01_1D: 7 neurons"
    assert sorted(os.listdir(inp)) == [path.name] and os.listdir(cwd) == []


def test_progress_save_hook(tmp_path, monkeypatch):
    inp, cwd = tmp_path / "in", tmp_path / "cwd"
    inp.mkdir(), cwd.mkdir()
    path = make_estimates(inp)
    monkeypatch.chdir(cwd)
    got = []

    def progress_save(est):
        got.append(list(est.idx_components))
        return "Progress saved <12:00>"

    app = App(bm_examinator.make_app(str(path), 20, LEGACY, progress_save=progress_save))
    app.select([0, 1]); app.click("Merge sel.")
    app.click("Save progress")
    assert got == [[0, 1, 2, 3, 4, 5, 6]]  # CaImAn renumbers; the merged one is 6
    assert _status(app.doc).text == "Progress saved &lt;12:00&gt;"  # text, not markup
    assert sorted(os.listdir(inp)) == [path.name] and os.listdir(cwd) == []


def test_no_status_line_without_hooks(tmp_path):
    app = App(bm_examinator.make_app(str(make_estimates(tmp_path)), 20, LEGACY))
    assert not [m for m in _models(app.doc) if isinstance(m, Div) and m.name == "status"]


def _autoinspected(folder) -> Path:
    path = make_estimates(folder)
    with open(path, "rb") as f:
        est = pickle.load(f)
    est.metrics_df = pd.DataFrame({"component_idx": range(8)})
    with open(path, "wb") as f:
        pickle.dump(est, f)
    return path


def test_refuse_autoinspected(tmp_path, monkeypatch):
    path = _autoinspected(tmp_path)
    monkeypatch.setattr(bm_examinator, "_legacy_rows", _boom)  # refused before the first view
    doc = Document()
    bm_examinator.make_app(str(path), 20, LEGACY, refuse_autoinspected=True)(doc)
    models = _models(doc)
    assert not [m for m in models if isinstance(m, Button)]
    assert [m.text for m in models if isinstance(m, Div)] == [REFUSAL]
    monkeypatch.undo()

    prep = bm_examinator.prepare_session(str(path), bkapp_kwargs=LEGACY)
    doc = Document()
    bm_examinator.make_app(str(path), 20, LEGACY, prepared=prep, refuse_autoinspected=True)(doc)
    assert [m.text for m in _models(doc) if isinstance(m, Div)] == [REFUSAL]
    # without the flag the notebook path opens such a file as before
    assert App(bm_examinator.make_app(str(path), 20, LEGACY)).shown() == list(range(8))


def test_full_length_traces(tmp_path):
    """No frame limit: the browser gets the whole session (decimated), and a zoom shows every frame, also at
    the session's end."""
    path = make_estimates(tmp_path, n=3, T=100_000)
    for kwargs in (LEGACY, {"mode": "legacy"}):  # end_frame None, and not given at all
        app = App(bm_examinator.make_app(str(path), 20, kwargs))
        est = app.storage.estimates
        time = est.time
        assert len(time) == 100_000
        assert len(app.src.data["traces"]) == 3
        t = app.time_axis()
        assert t[0] == time[0] and t[-1] == time[-1], kwargs
        app.zoom(time[-200], time[-1])
        t = app.time_axis()
        assert len(t) >= 200 and np.array_equal(t, time[-len(t):]), kwargs
        for comp, tr in zip(app.shown(), app.src.data["traces"]):
            c = est.C[comp]
            assert np.allclose(tr, ((c - c.min()) / (c.max() - c.min()))[-len(t):], atol=1e-6), kwargs


@pytest.mark.parametrize("hook, button", [("final_save", "Save results"), ("progress_save", "Save progress")])
def test_save_hook_error_shown(tmp_path, monkeypatch, caplog, hook, button):
    inp, cwd = tmp_path / "in", tmp_path / "cwd"
    inp.mkdir(), cwd.mkdir()
    path = make_estimates(inp)
    monkeypatch.chdir(cwd)
    calls = []

    def save(est):
        calls.append(est)
        if len(calls) == 1:
            raise OSError("disk full\nsecond line")
        return "Saved"

    app = App(bm_examinator.make_app(str(path), 20, LEGACY, **{hook: save}))
    with caplog.at_level(logging.ERROR):
        app.click(button)
    assert _status(app.doc).text == "Save failed: OSError: disk full"
    assert [r.exc_info[0] for r in caplog.records if r.exc_info] == [OSError]  # the traceback is logged
    app.click(button)  # the button still works
    assert len(calls) == 2 and _status(app.doc).text == "Saved"
    assert sorted(os.listdir(inp)) == [path.name] and os.listdir(cwd) == []


def test_prepared_session_must_match(tmp_path):
    path = make_estimates(tmp_path)
    (tmp_path / "other").mkdir()
    other = make_estimates(tmp_path / "other")
    prep = bm_examinator.prepare_session(str(path), bkapp_kwargs=LEGACY)
    assert prep.mode == "legacy" and prep.options["end_frame"] is None and prep.options["cthr"] == 0.3
    # the same file written another way is the same file
    assert App(bm_examinator.make_app(str(path).replace("\\", "/"), 20, LEGACY, prepared=prep)).shown() == \
        list(range(8))
    for fname, fps, kwargs, what in [
            (str(other), 20, LEGACY, "prepared session is of"),
            (str(path), 20, {"mode": "capcan", "end_frame": None}, "mode"),
            (str(path), 20, {**LEGACY, "cthr": 0.4}, "cthr"),
            (str(path), 20, {**LEGACY, "start_frame": 10}, "start_frame"),
            (str(path), 20, {"mode": "legacy", "end_frame": 300}, "end_frame"),
            (str(path), 20, {**LEGACY, "downsampling": 2}, "downsampling"),
            (str(path), 20, {**LEGACY, "corr_thr": 0.5}, "corr_thr"),
            (str(path), 20, {**LEGACY, "correlation_method": "spearman"}, "correlation_method"),
            (str(path), 25, LEGACY, "default_fps")]:
        with pytest.raises(ValueError, match=what):
            bm_examinator.make_app(fname, fps, kwargs, prepared=prep)
