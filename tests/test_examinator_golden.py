"""Golden record of the inspection window's decisions: what Save progress and Save results write.

Recorded on the window before phase 5 (933c6bd). Phase 5 changes how the callbacks work, not what they
decide, so this test stays as it is. BM_GOLDEN_WRITE=1 records tests/data/examinator_golden.npz again
(and then compares with it); run it only on the code the record must describe.

Random sources of the sequences: the synthetic sessions draw from default_rng(0); CaImAn's deconvolution in
Merge seeds np.random with 45 itself; np.random is seeded here as well for anything that draws before it.

Each sequence runs twice against the same record: as the notebook opens a file (make_app loads it and writes the
saves next to it) and as the inspection server does (prepare_session with timestamps and with contours read back
from the server's inspect cache, make_app with the prepared session and save hooks that write through
write_result). After every click the rows the browser draws are checked against the components the clicks act
on, found from the traces and the decisions themselves, never from the window's own numbering.
"""
# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import hashlib
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
from bokeh.models import GlyphRenderer

sys.path.insert(0, str(Path(__file__).parent.parent))

import bm_examinator
from bearmind import inspect_server
from examinator_helpers import App, make_estimates

GOLDEN = Path(__file__).parent / "data" / "examinator_golden.npz"
EXACT = ("idx_components", "idx_components_bad")
FLOATS = ("SNR_comp", "r_values")
# one entry per component along the first axis (A has one column per component); R only when present
PER_COMPONENT = ("C", "YrA", "S", "bl", "c1", "g", "neurons_sn")
RTOL = 1e-12
PROGRESS, FINAL = "Save progress", "Save results"
INPUT_CHANGED = "golden input changed: the helper no longer builds the recorded input"
# Every recorded save follows a Merge or a Revert to a Merge's state, and CaImAn's manual_merge sets both exact
# fields as int arrays: the old window saved arrays there, and the autoinspection that reads the result gets the
# same types. (A save right after a Delete would hold idx_components as a list, in the old window too.)
EXACT_TYPE = np.ndarray
TRACE_ATOL = 1e-6  # the window keeps the normalized traces as float32

# Rows are selected before a click (None: the click takes no selection); "Sort X" clicks the sorting
# button X. Every Save progress and the final Save results are recorded.
SESSIONS = {
    # Delete, Merge, Show sel., Delete in the zoomed view, Revert change, Reset view, Merge, Save progress,
    # Discard all, Delete, Merge, Save results. Revert change returns the zoomed view it was made in; Reset view
    # then shows the full list the Merge rows refer to (the old window ended Revert change on it by itself).
    "s8": dict(n=8, T=300, end_frame=300, clicks=[
        ([2], "Delete sel."), ([0, 1], "Merge sel."), ([1, 2, 3], "Show sel."), ([1], "Delete sel."),
        (None, "Revert change"), (None, "Reset view"), ([0, 2], "Merge sel."), (None, PROGRESS),
        (None, "Discard all"), ([5], "Delete sel."), ([0, 3], "Merge sel."), (None, FINAL)]),
    "s40": dict(n=40, T=2000, end_frame=None, clicks=[
        ([3, 17], "Delete sel."), ([0, 1, 2], "Merge sel."), ([4, 5, 6, 7, 8, 9, 10, 11, 12], "Show sel."),
        ([2], "Delete sel."), (None, "Revert change"), (None, "Reset view"), ([0, 5], "Merge sel."),
        (None, PROGRESS), (None, "Discard all"), ([7, 20, 33], "Delete sel."), ([10, 11, 25], "Merge sel."),
        (None, FINAL)]),
    # sorting, then Delete and Merge by row; Reset view after a zoom, then Merge; Merge then Revert;
    # Delete then Revert. Not Revert right after Reset view: that deferred bug is not recorded.
    "s40b": dict(n=40, T=2000, end_frame=None, clicks=[
        (None, "Sort SNR"), ([5, 9], "Delete sel."), ([0, 3], "Merge sel."),
        ([10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], "Show sel."), (None, "Reset view"),
        ([2, 4, 6], "Merge sel."), ([1, 7], "Merge sel."), (None, "Revert change"), (None, PROGRESS),
        ([12], "Delete sel."), (None, "Revert change"), (None, PROGRESS), (None, "Sort Area"),
        ([8, 30], "Delete sel."), ([20, 21], "Merge sel."), (None, FINAL)]),
}


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _input_sha(est) -> str:
    h = hashlib.sha256()
    for a in (_dense(est.A), est.C, est.YrA, est.S, np.asarray(est.dims)):
        a = np.ascontiguousarray(a)
        h.update(repr((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _fields(est) -> dict:
    names = PER_COMPONENT + (("R",) if getattr(est, "R", None) is not None else ())
    return {f: np.asarray(getattr(est, f)) for f in names}


def _estimate_files(*folders) -> set:
    return {f for d in folders for f in d.glob("*estimates.pickle")}


def _server_app(folder, path, T, end_frame, target):
    """make_app as the inspection server builds it: prepare_session with the session's timestamps and with the
    contours of an earlier preparation read back from the server's inspect cache, then save hooks that write
    through write_result, to the path in target["path"]."""
    kwargs = {"mode": "legacy", "end_frame": end_frame, "enable_gpu_backend": True, "trace_points": 2000}
    time_s = np.arange(T) * 0.05  # what session_time_s gives for 20 frames per second
    fname = path.as_posix()  # as InspectServer.build passes it
    first = bm_examinator.prepare_session(fname, time=time_s, default_fps=inspect_server.DEFAULT_FPS,
                                          bkapp_kwargs=kwargs)
    srv = inspect_server.InspectServer(folder / "series")
    stamp = inspect_server._Stamp(size=1, mtime_ns=1, cnmf_state="done", cnmf_finished=None)
    srv._write_cache("NOF_H01_1D", stamp, kwargs, first.contours, None, None)
    contours, _ = srv._read_cache("NOF_H01_1D", stamp, kwargs, None)
    assert contours is not None and len(contours) == len(first.contours)
    prep = bm_examinator.prepare_session(fname, time=time_s, default_fps=inspect_server.DEFAULT_FPS,
                                         bkapp_kwargs=kwargs, contours=contours)

    def save(est):
        bm_examinator.write_result(est, target.pop("path"))
        return "saved"
    return bm_examinator.make_app(prep.fname, prep.options["default_fps"], kwargs, prepared=prep, final_save=save,
                                  progress_save=save, refuse_autoinspected=True)


def _norm_rows(C):
    """Each row of C scaled to 0..1, as the window draws a trace."""
    X = np.array(C, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        X -= X.min(axis=1, keepdims=True)
        X /= X.max(axis=1, keepdims=True)
    return X


class _Rows:
    """The rows the browser draws, against the components the clicks act on.

    The test keeps its own account of the list on screen (`shown`: component numbers by row position): the full
    list of the working estimates after Delete, Merge and Reset view; the selected positions after Show sel.; the
    list at the last change after Revert change; the loaded list after Discard all. A drawn row's component is
    found from its trace alone: the one row of the working estimates' C whose normalized trace it draws. Every
    drawn row must draw the trace and the contour of the component at its position, and the components a Delete
    or a Merge takes, seen in the decisions it leaves, must be those whose traces the selected rows draw."""

    def __init__(self, app):
        self.app = app
        self.initial = [int(c) for c in app.storage.estimates.idx_components]
        self.shown = list(self.initial)
        self.at_change = list(self.initial)  # Revert change before any change returns to the loaded list
        self.contours = {}
        self.acted = None

    def _filter(self):
        """The data source rows the browser draws, in the order of the list on screen (every glyph of the rows
        shares one filter)."""
        flt, = {r.view.filter for r in self.app.doc.roots[0].references()
                if isinstance(r, GlyphRenderer) and r.data_source is self.app.src}
        return list(flt.indices)

    def _components(self, slots):
        """The component whose trace each data source row draws."""
        N = _norm_rows(self.app.storage.estimates.C)
        out = []
        for s in slots:
            tr = np.asarray(self.app.src.data["traces"][s], dtype=float)
            assert tr.shape == N.shape[1:], (s, tr.shape, N.shape)
            hits = np.flatnonzero(np.all(np.abs(N - tr) <= TRACE_ATOL, axis=1))
            assert len(hits) == 1, (s, hits)
            out.append(int(hits[0]))
        return out

    def _contour(self, est, c):
        """(xs, ys) of the contour of A[:, c] as the window draws it; computed once per footprint."""
        col = sp.csc_matrix(est.A)[:, [c]]
        key = (col.indices.tobytes(), col.data.tobytes())
        if key not in self.contours:
            coors = bm_examinator._component_contours(est, [c], 0.3)[0]
            poly = coors[~np.isnan(coors).any(axis=1)]
            self.contours[key] = (poly[:, 0], est.imax.shape[0] - poly[:, 1])
        return self.contours[key]

    def before(self, rows, label):
        """Before a click with a selection: the selected rows draw the components at those positions."""
        drawn = sorted(self._components(list(self.app.src.selected.indices)))
        assert drawn == sorted(self.shown[r] for r in rows), (label, drawn, rows, self.shown)
        est = self.app.storage.estimates
        self.acted = (label, drawn, [int(c) for c in est.idx_components], est.C)

    def after(self, label):
        """After any click: the account updated, the decisions of a Delete or a Merge, then every drawn row."""
        est = self.app.storage.estimates
        good = [int(c) for c in est.idx_components]
        if label in ("Delete sel.", "Merge sel.", "Show sel."):
            name, drawn, good_before, C_before = self.acted
            self.acted = None
            assert name == label
            if label == "Delete sel.":
                assert sorted(set(good_before) - set(good)) == drawn, (label, drawn)
            elif label == "Merge sel.":
                # CaImAn renumbers the components: the merged ones are those whose traces are gone
                left = {row.tobytes() for row in np.asarray(est.C)}
                merged = sorted(c for c in good_before if np.asarray(C_before[c]).tobytes() not in left)
                assert merged == drawn, (label, merged, drawn)
            if label == "Show sel.":
                self.shown = [c for c in self.shown if c in drawn]
            else:
                self.at_change, self.shown = self.shown, good
        elif label == "Reset view":
            self.shown = good
        elif label == "Revert change":
            self.shown = list(self.at_change)
        elif label == "Discard all":
            self.shown = list(self.initial)
        slots = self._filter()
        comps = self._components(slots)
        assert comps == self.shown, (label, comps, self.shown)
        assert [int(c) for c in self.app.storage.shown] == self.shown, label  # what App.select's rows refer to
        for s, c in zip(slots, comps):
            xs, ys = self._contour(est, c)
            assert np.allclose(self.app.src.data["xs"][s], xs) and np.allclose(self.app.src.data["ys"][s], ys), \
                (label, s, c)


def _run(folder, monkeypatch, n, T, end_frame, clicks, variant="notebook"):
    """(loaded input, [(record name, saved estimates)]): each Save progress file, then Save results'."""
    inp, cwd, kept = folder / "in", folder / "cwd", folder / "saved"
    for d in (inp, cwd, kept):
        d.mkdir(parents=True)
    path = make_estimates(inp, n=n, T=T)
    monkeypatch.chdir(cwd)  # the notebook path writes the saved estimates into the working folder
    np.random.seed(0)
    target = {}
    if variant == "notebook":
        app = App(bm_examinator.make_app(str(path), 20, {"mode": "legacy", "end_frame": end_frame}))
    else:
        app = App(_server_app(folder, path, T, end_frame, target))
    drawn = _Rows(app)
    drawn.after(None)
    saved, second = [], None
    for rows, label in clicks:
        if rows is not None:
            app.select(rows)
        if label.startswith("Sort "):
            app.sort(label[len("Sort "):])
        elif label not in (PROGRESS, FINAL):
            if rows is not None:
                drawn.before(rows, label)
            app.click(label)
        else:
            name = folder.name if label == FINAL else f"{folder.name}_cp{len(saved) + 1}"
            if variant == "server":
                target["path"] = kept / f"{name}.pickle"
                app.click(label)
                assert not target, (label, "the save hook was not called")
                saved.append((name, _load(kept / f"{name}.pickle")))
            else:
                # Save progress names its file by the second; two saves within one second would share a name
                while second is not None and int(time.time()) == second:
                    time.sleep(0.05)
                second = int(time.time())
                before = _estimate_files(inp, cwd)
                app.click(label)
                new = sorted(_estimate_files(inp, cwd) - before)
                assert len(new) == 1, (label, new)
                moved = Path(shutil.move(str(new[0]), str(kept / f"{name}.pickle")))
                saved.append((name, _load(moved)))
        drawn.after(label)
    if variant == "server":
        assert _estimate_files(inp, cwd) == {path}, "a server window wrote next to the input or into the working folder"
    return _load(path), saved


def _record(name, saved, inp) -> dict:
    """What the test compares, kept small: a saved component that is a loaded one unchanged is stored as
    the index of that component ('origin'); only the components the window computed (merges) in full."""
    A_in, A = _dense(inp.A), _dense(saved.A)
    f_in, f_out = _fields(inp), _fields(saved)
    assert set(f_out) <= set(f_in), (set(f_out), set(f_in))
    by_trace = {f_in["C"][j].tobytes(): j for j in range(len(f_in["C"]))}
    origin = []
    for k in range(A.shape[1]):
        j = by_trace.get(f_out["C"][k].tobytes(), -1)
        same = j >= 0 and np.array_equal(A[:, k], A_in[:, j]) and all(
            np.array_equal(a[k], f_in[f][j]) for f, a in f_out.items())
        origin.append(j if same else -1)
    origin = np.array(origin)
    new = origin < 0
    # R: absent, the same values as YrA (CaImAn's CNMF and manual_merge set R = YrA), or its own
    r_kind = "none" if "R" not in f_out else "YrA" if np.array_equal(f_out["R"], f_out["YrA"]) else "own"
    rec = {f"{name}_origin": origin, f"{name}_A_new": A[:, new], f"{name}_R_kind": np.array(r_kind)}
    rec.update({f"{name}_{f}_new": a[new] for f, a in f_out.items() if f != "R" or r_kind == "own"})
    rec.update({f"{name}_{f}": np.asarray(getattr(saved, f)) for f in EXACT + FLOATS})
    return rec


def _expected(g, name, inp) -> dict:
    origin = g[f"{name}_origin"]
    new = origin < 0
    A = np.zeros((inp.A.shape[0], len(origin)))
    A[:, ~new] = _dense(inp.A)[:, origin[~new]]
    A[:, new] = g[f"{name}_A_new"]
    out = {"A": A}
    r_kind = str(g[f"{name}_R_kind"])
    f_in = _fields(inp)
    for f in PER_COMPONENT + (("R",) if r_kind == "own" else ()):
        rows = np.zeros((len(origin),) + f_in[f].shape[1:])
        rows[~new] = f_in[f][origin[~new]]
        rows[new] = g[f"{name}_{f}_new"]
        out[f] = rows
    if r_kind == "YrA":
        out["R"] = out["YrA"]
    return out


def _compare(g, name, saved, inp):
    for f in EXACT:
        value, want = getattr(saved, f), g[f"{name}_{f}"]
        assert isinstance(value, EXACT_TYPE), (name, f, type(value))
        assert value.dtype.kind == want.dtype.kind, (name, f, value.dtype, want.dtype)
        assert np.array_equal(value, want), (name, f)
    for f in FLOATS:
        np.testing.assert_allclose(np.asarray(getattr(saved, f), dtype=float), g[f"{name}_{f}"],
                                   rtol=RTOL, atol=0, err_msg=f"{name} {f}")
    exp = _expected(g, name, inp)
    if str(g[f"{name}_R_kind"]) == "none":
        assert getattr(saved, "R", None) is None, name
    got = _fields(saved)
    for f, want in exp.items():
        have = _dense(saved.A) if f == "A" else got[f]
        np.testing.assert_allclose(have, want, rtol=RTOL, atol=0, err_msg=f"{name} {f}")


@pytest.mark.parametrize("variant", ["notebook", "server"])
def test_golden_click_sequence(tmp_path, monkeypatch, variant):
    runs = {p: _run(tmp_path / p, monkeypatch, variant=variant, **s) for p, s in SESSIONS.items()}
    # recorded from the notebook path only: the server path must reach the same decisions, not define them
    if os.environ.get("BM_GOLDEN_WRITE") == "1" and variant == "notebook":
        GOLDEN.parent.mkdir(exist_ok=True)
        rec = {}
        for p, (inp, saved) in runs.items():
            rec[f"{p}_input_sha256"] = np.array(_input_sha(inp))
            for name, est in saved:
                rec.update(_record(name, est, inp))
        np.savez_compressed(GOLDEN, **rec)
    with np.load(GOLDEN) as g:
        for p, (inp, _) in runs.items():
            assert _input_sha(inp) == str(g[f"{p}_input_sha256"]), INPUT_CHANGED
        names = sorted(name for _, saved in runs.values() for name, _ in saved)
        assert names == sorted(k[:-len("_origin")] for k in g.files if k.endswith("_origin"))
        for p, (inp, saved) in runs.items():
            for name, est in saved:
                _compare(g, name, est, inp)
