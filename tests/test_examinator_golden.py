"""Golden record of the inspection window's decisions: what Save progress and Save results write.

Recorded on the window before phase 5 (933c6bd). Phase 5 changes how the callbacks work, not what they
decide, so this test stays as it is. BM_GOLDEN_WRITE=1 records tests/data/examinator_golden.npz again
(and then compares with it); run it only on the code the record must describe.

Random sources of the sequences: the synthetic sessions draw from default_rng(0); CaImAn's deconvolution in
Merge seeds np.random with 45 itself; np.random is seeded here as well for anything that draws before it.
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
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

import bm_examinator
from examinator_helpers import App, make_estimates

GOLDEN = Path(__file__).parent / "data" / "examinator_golden.npz"
EXACT = ("idx_components", "idx_components_bad")
FLOATS = ("SNR_comp", "r_values")
# one entry per component along the first axis (A has one column per component); R only when present
PER_COMPONENT = ("C", "YrA", "S", "bl", "c1", "g", "neurons_sn")
RTOL = 1e-12
PROGRESS, FINAL = "Save progress", "Save results"
INPUT_CHANGED = "golden input changed: the helper no longer builds the recorded input"

# Rows are selected before a click (None: the click takes no selection); "Sort X" clicks the sorting
# button X. Every Save progress and the final Save results are recorded.
SESSIONS = {
    # Delete, Merge, Show sel., Delete in the zoomed view, Revert change, Merge, Save progress,
    # Discard all, Delete, Merge, Save results
    "s8": dict(n=8, T=300, end_frame=300, clicks=[
        ([2], "Delete sel."), ([0, 1], "Merge sel."), ([1, 2, 3], "Show sel."), ([1], "Delete sel."),
        (None, "Revert change"), ([0, 2], "Merge sel."), (None, PROGRESS), (None, "Discard all"),
        ([5], "Delete sel."), ([0, 3], "Merge sel."), (None, FINAL)]),
    "s40": dict(n=40, T=2000, end_frame=None, clicks=[
        ([3, 17], "Delete sel."), ([0, 1, 2], "Merge sel."), ([4, 5, 6, 7, 8, 9, 10, 11, 12], "Show sel."),
        ([2], "Delete sel."), (None, "Revert change"), ([0, 5], "Merge sel."), (None, PROGRESS),
        (None, "Discard all"), ([7, 20, 33], "Delete sel."), ([10, 11, 25], "Merge sel."), (None, FINAL)]),
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


def _run(folder, monkeypatch, n, T, end_frame, clicks):
    """(loaded input, [(record name, saved estimates)]): each Save progress file, then Save results'."""
    inp, cwd, kept = folder / "in", folder / "cwd", folder / "saved"
    for d in (inp, cwd, kept):
        d.mkdir(parents=True)
    path = make_estimates(inp, n=n, T=T)
    monkeypatch.chdir(cwd)  # the notebook path writes the saved estimates into the working folder
    np.random.seed(0)
    app = App(bm_examinator.make_app(str(path), 20, {"mode": "legacy", "end_frame": end_frame}))
    saved, second = [], None
    for rows, label in clicks:
        if rows is not None:
            app.select(rows)
        if label.startswith("Sort "):
            app.sort(label[len("Sort "):])
            continue
        if label not in (PROGRESS, FINAL):
            app.click(label)
            continue
        # Save progress names its file by the second; two saves within one second would share a name
        while second is not None and int(time.time()) == second:
            time.sleep(0.05)
        second = int(time.time())
        before = _estimate_files(inp, cwd)
        app.click(label)
        new = sorted(_estimate_files(inp, cwd) - before)
        assert len(new) == 1, (label, new)
        name = folder.name if label == FINAL else f"{folder.name}_cp{len(saved) + 1}"
        moved = Path(shutil.move(str(new[0]), str(kept / f"{name}.pickle")))
        saved.append((name, _load(moved)))
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
        assert np.array_equal(np.asarray(getattr(saved, f)), g[f"{name}_{f}"]), (name, f)
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


def test_golden_click_sequence(tmp_path, monkeypatch):
    runs = {p: _run(tmp_path / p, monkeypatch, **s) for p, s in SESSIONS.items()}
    if os.environ.get("BM_GOLDEN_WRITE") == "1":
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
