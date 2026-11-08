"""Hooks of the inspection window for a server: prepared session, save hooks, refusal, full-length traces."""
# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import copy
import logging
import os
import pickle
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
from bokeh.document.document import Document
from bokeh.models import Button, Div
from scipy.io import loadmat

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

import bm_examinator
from examinator_helpers import App, make_estimates

LEGACY = {"mode": "legacy", "end_frame": None}
REFUSAL = "This file is already autoinspected; open 4_Estimates instead."
SID = "NOF_H01_1D"  # the session of make_estimates
SIDE_FILES = [f"{SID}_filters", f"{SID}_filters_session.mat", f"{SID}_traces.csv"]


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


def _loaded(folder):
    """The synthetic session as the window holds it: loaded, with its time axis."""
    return bm_examinator.LoadEstimates(str(make_estimates(folder)), time=np.arange(300) / 20.0)


def _read_side_files(folder, base):
    """(traces table, the .mat images, the filter images by file name) of one save of side files."""
    folder = Path(folder)
    traces = pd.read_csv(folder / f"{base}traces.csv")
    mat = loadmat(folder / f"{base}filters_session.mat")["A"]
    tifs = {p.name: tifffile.imread(p) for p in sorted((folder / f"{base}filters").iterdir())}
    return traces, mat, tifs


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _server_final_save(folder, futures):
    """A final_save as the inspection server's: the result whole and uncompressed, the side files in the
    background (their Futures appended to futures)."""
    def final_save(est):
        path = bm_examinator.write_result(est, Path(folder) / f"{SID}_manual_estimates.pickle")
        futures.append(bm_examinator.write_side_files_in_background(est, folder, f"{SID}_"))
        return f"Saved {path.name}"
    return final_save


def test_side_files_skip_deleted(tmp_path):
    """Only kept components: not those of idx_components_bad, nor the ones autoinspection marked deleted (it
    leaves them in idx_components for the window). For those components, the files SaveResults writes."""
    est = _loaded(tmp_path)
    est.idx_components, est.idx_components_bad = np.array([0, 1, 2, 3, 4, 6, 7]), np.array([5])
    est.metrics_df = pd.DataFrame({"component_idx": range(8), "delete": [0, 0, 0, 1, 0, 0, 0, 0]})
    kept = [0, 1, 2, 4, 6, 7]
    assert list(bm_examinator.kept_for_side_files(est)) == kept

    bm_examinator.write_side_files(est, tmp_path / "out", f"{SID}_")
    traces, mat, tifs = _read_side_files(tmp_path / "out", f"{SID}_")
    assert list(traces.columns) == ["time_s", "0", "1", "2", "3", "4", "5"]
    assert np.array_equal(traces["time_s"].to_numpy(), est.time)
    assert np.array_equal(traces.iloc[:, 1:].to_numpy().T, est.C[kept])
    assert mat.shape[0] == 6 and list(tifs) == [f"filter_{i:03d}.tif" for i in range(1, 7)]

    ref = copy.copy(est)
    del ref.metrics_df
    ref.idx_components = np.array(kept)
    (tmp_path / "ref").mkdir()
    ref.name = str(tmp_path / "ref" / f"{SID}_estimates.pickle")  # SaveResults writes next to it
    bm_examinator.SaveResults(ref)
    assert (tmp_path / "out" / f"{SID}_traces.csv").read_bytes() == \
        (tmp_path / "ref" / f"{SID}_traces.csv").read_bytes()
    _, ref_mat, ref_tifs = _read_side_files(tmp_path / "ref", f"{SID}_")
    assert np.array_equal(mat, ref_mat) and list(ref_tifs) == list(tifs)
    assert all(np.array_equal(tifs[k], ref_tifs[k]) for k in tifs)

    # without a delete column, or without component_idx (rows are components then)
    est.metrics_df = pd.DataFrame({"component_idx": range(8)})
    assert list(bm_examinator.kept_for_side_files(est)) == [0, 1, 2, 3, 4, 6, 7]
    est.metrics_df = pd.DataFrame({"delete": [1, 0, 0, 0, 0, 0, 0, 0]})
    assert list(bm_examinator.kept_for_side_files(est)) == [1, 2, 3, 4, 6, 7]


def test_side_files_names(tmp_path):
    est = _loaded(tmp_path)
    folder = tmp_path / "5_Inspection" / SID  # made by the write
    bm_examinator.write_side_files(est, folder, f"{SID}_")
    assert sorted(os.listdir(folder)) == SIDE_FILES
    assert len(os.listdir(folder / f"{SID}_filters")) == 8
    # the folder holds one save: a later one with fewer components leaves no filter of the earlier one
    est.idx_components, est.idx_components_bad = np.array([0, 4, 7]), np.array([1, 2, 3, 5, 6])
    bm_examinator.write_side_files(est, folder, f"{SID}_")
    assert sorted(os.listdir(folder)) == SIDE_FILES
    assert sorted(os.listdir(folder / f"{SID}_filters")) == ["filter_001.tif", "filter_002.tif", "filter_003.tif"]


def test_write_result_whole(tmp_path, monkeypatch):
    est = _loaded(tmp_path)
    path = tmp_path / "5_Inspection" / SID / f"{SID}_manual_estimates.pickle"
    assert bm_examinator.write_result(est, path) == path
    assert sorted(os.listdir(path.parent)) == [path.name]
    assert np.array_equal(_load(path).C, est.C)

    def broken(obj, f, *args, **kwargs):  # the disk fills up halfway
        f.write(b"half")
        raise OSError("disk full")

    monkeypatch.setattr(bm_examinator.pickle, "dump", broken)
    with pytest.raises(OSError):
        bm_examinator.write_result(est, path)
    monkeypatch.undo()
    assert np.array_equal(_load(path).C, est.C)  # the earlier result stays whole


def test_final_save_uncompressed(tmp_path, monkeypatch):
    """The window hands the hooks its working object uncompressed: compress_estimates is the notebook's."""
    inp, cwd, out = tmp_path / "in", tmp_path / "cwd", tmp_path / "5_Inspection" / SID
    inp.mkdir(), cwd.mkdir()
    path = make_estimates(inp)
    monkeypatch.chdir(cwd)
    futures = []
    app = App(bm_examinator.make_app(str(path), 20, {**LEGACY, "compress_estimates": True},
                                     final_save=_server_final_save(out, futures)))
    app.select([2]); app.click("Delete sel.")
    app.click("Save results")
    futures[0].result(timeout=120)
    assert _status(app.doc).text == f"Saved {SID}_manual_estimates.pickle"
    saved = _load(out / f"{SID}_manual_estimates.pickle")
    assert len(saved.SNR_comp) == saved.A.shape[1] == 8
    assert not hasattr(saved, "idx_components_bad_original")
    assert list(saved.idx_components_bad) == [2] and saved.C.dtype == np.float64 and saved.R is not None
    assert sorted(os.listdir(out)) == sorted(SIDE_FILES + [f"{SID}_manual_estimates.pickle"])
    assert pd.read_csv(out / f"{SID}_traces.csv").shape[1] == 1 + 7
    assert sorted(os.listdir(inp)) == [path.name] and os.listdir(cwd) == []


def test_side_files_in_background(tmp_path, monkeypatch):
    """Save results returns while the side files are still written; they hold the components of that save
    even when the window changes meanwhile, and a later save's files are written after an earlier one's."""
    path = make_estimates(tmp_path)
    out = tmp_path / "out"
    gate, written = threading.Event(), []
    real = bm_examinator.write_side_files

    def waiting(est, folder, base, sigma=3):
        assert gate.wait(60)
        written.append([int(c) for c in bm_examinator.kept_for_side_files(est)])
        real(est, folder, base, sigma)

    monkeypatch.setattr(bm_examinator, "write_side_files", waiting)
    futures = []
    app = App(bm_examinator.make_app(str(path), 20, LEGACY, final_save=_server_final_save(out, futures)))
    app.select([2]); app.click("Delete sel.")
    app.click("Save results")  # returns although the side files wait
    assert _status(app.doc).text == f"Saved {SID}_manual_estimates.pickle" and not futures[0].done()
    app.select([0]); app.click("Delete sel.")  # the window goes on
    app.click("Save results")
    gate.set()
    for f in futures:
        f.result(timeout=120)
    assert written == [[0, 1, 3, 4, 5, 6, 7], [1, 3, 4, 5, 6, 7]]
    assert pd.read_csv(out / f"{SID}_traces.csv").shape[1] == 1 + 6  # the last save's
    assert list(_load(out / f"{SID}_manual_estimates.pickle").idx_components) == [1, 3, 4, 5, 6, 7]


def test_side_files_failure_in_future(tmp_path):
    """A failure in the background is not lost: the Future holds it for the server to log and show."""
    est = _loaded(tmp_path)
    est.time = est.time[:10]  # no time for most frames
    future = bm_examinator.write_side_files_in_background(est, tmp_path / "out", f"{SID}_")
    with pytest.raises(ValueError):
        future.result(timeout=60)


def test_notebook_compressed_save_keeps_the_window(tmp_path, monkeypatch):
    """compress_estimates compresses what the notebook path writes, not the window's working object (a
    compressed working object is renumbered, and the next Delete would remove the wrong neuron)."""
    inp, cwd = tmp_path / "in", tmp_path / "cwd"
    inp.mkdir(), cwd.mkdir()
    path = make_estimates(inp)
    monkeypatch.chdir(cwd)
    app = App(bm_examinator.make_app(str(path), 20, {"mode": "legacy", "end_frame": 300,
                                                     "compress_estimates": True}))
    app.select([2]); app.click("Delete sel.")
    app.click("Save progress")
    progress, = cwd.glob("*_estimates.pickle")
    saved = _load(progress)
    assert saved.A.shape[1] == 7 and list(saved.idx_components_bad_original) == [2]  # compressed, as before
    est = app.storage.estimates
    assert est.A.shape[1] == 8 and est.C.dtype == np.float64 and list(est.idx_components_bad) == [2]
    app.select([0]); app.click("Delete sel.")
    assert app.shown() == [1, 3, 4, 5, 6, 7]
    app.click("Save results")
    final = _load(cwd / f"{SID}final_estimates.pickle")
    assert final.A.shape[1] == 6 and list(final.idx_components_bad_original) == [0, 2]
    # SaveResults from the compressed object, as before: float32 traces of the kept neurons
    traces = pd.read_csv(inp / f"{SID}_traces.csv")
    assert np.array_equal(traces.iloc[:, 1:].to_numpy().T.astype(np.float32),
                          est.C[[1, 3, 4, 5, 6, 7]].astype(np.float32))


def test_hooks_get_a_copy(tmp_path):
    """A hook may rebind fields of what it gets (here: compress it) without changing the window, its undo
    record or a reused prepared session."""
    from estimates_compression import compress_estimates_ultra_lightweight
    path = make_estimates(tmp_path)
    prep = bm_examinator.prepare_session(str(path), bkapp_kwargs=LEGACY)
    got = []

    def final_save(est):
        got.append(copy.copy(est))
        if len(got) == 1:
            compress_estimates_ultra_lightweight(est)
            est.idx_components = []
            del est.r_values
        return "Saved"

    app = App(bm_examinator.make_app(str(path), 20, LEGACY, prepared=prep, final_save=final_save))
    app.select([2]); app.click("Delete sel.")
    app.select([0, 1]); app.click("Merge sel.")
    app.click("Save results")
    est = app.storage.estimates
    # CaImAn renumbers; the merged one is 6
    assert list(est.idx_components) == [1, 2, 3, 4, 5, 6] and est.A.shape[1] == 7 and est.C.dtype == np.float64
    assert len(est.r_values) == 7 and est.R is not None
    app.click("Revert change")
    assert app.shown() == [0, 1, 3, 4, 5, 6, 7]
    assert app.storage.estimates.A.shape[1] == 8 and len(app.storage.estimates.r_values) == 8
    app.click("Discard all")
    assert app.shown() == list(range(8))
    app.select([5]); app.click("Delete sel.")
    app.click("Save results")
    last = got[-1]
    assert list(last.idx_components) == [0, 1, 2, 3, 4, 6, 7] and list(last.idx_components_bad) == [5]
    assert last.A.shape[1] == 8 and last.C.dtype == np.float64 and len(last.r_values) == 8
    assert list(prep.estimates.idx_components) == list(range(8)) and prep.estimates.C.dtype == np.float64
