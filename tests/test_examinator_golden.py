"""Golden record of the inspection window's decisions: what Save results writes after one click sequence.

Recorded on the window before phase 5 (933c6bd). Phase 5 changes how the callbacks work, not what they
decide, so this test stays as it is. BM_GOLDEN_WRITE=1 records tests/data/examinator_golden.npz again
(and then compares with it); run it only on the code the record must describe.

Random sources of the sequence: the synthetic sessions draw from default_rng(0); CaImAn's deconvolution in
Merge seeds np.random with 45 itself; np.random is seeded here as well for anything that draws before it.
"""
# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).parent.parent))

import bm_examinator
from examinator_helpers import App, make_estimates

GOLDEN = Path(__file__).parent / "data" / "examinator_golden.npz"
EXACT = ("idx_components", "idx_components_bad")
FLOATS = ("SNR_comp", "r_values")
ROWS = ("C", "YrA", "S")  # one row per component; A has one column per component
RTOL = 1e-12

# Rows are selected before a click (None: the click takes no selection). The same buttons in both sessions:
# Delete, Merge, Show sel., Delete in the zoomed view, Revert change, Merge, Discard all, Delete, Merge,
# Save results.
SESSIONS = {
    "s8": dict(n=8, T=300, end_frame=300, clicks=[
        ([2], "Delete sel."), ([0, 1], "Merge sel."), ([1, 2, 3], "Show sel."), ([1], "Delete sel."),
        (None, "Revert change"), ([0, 2], "Merge sel."), (None, "Discard all"), ([5], "Delete sel."),
        ([0, 3], "Merge sel."), (None, "Save results")]),
    "s40": dict(n=40, T=2000, end_frame=None, clicks=[
        ([3, 17], "Delete sel."), ([0, 1, 2], "Merge sel."), ([4, 5, 6, 7, 8, 9, 10, 11, 12], "Show sel."),
        ([2], "Delete sel."), (None, "Revert change"), ([0, 5], "Merge sel."), (None, "Discard all"),
        ([7, 20, 33], "Delete sel."), ([10, 11, 25], "Merge sel."), (None, "Save results")]),
}


def _dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _run(folder, monkeypatch, n, T, end_frame, clicks):
    """(saved estimates, loaded input) after the clicks; the notebook path of Save results."""
    folder.mkdir()
    path = make_estimates(folder, n=n, T=T)
    monkeypatch.chdir(folder)  # the notebook path writes <SID>final_estimates.pickle into the working folder
    np.random.seed(0)
    app = App(bm_examinator.make_app(str(path), 20, {"mode": "legacy", "end_frame": end_frame}))
    for rows, label in clicks:
        if rows is not None:
            app.select(rows)
        app.click(label)
    saved = sorted(folder.glob("*final_estimates.pickle"))
    assert len(saved) == 1, saved
    return _load(saved[0]), _load(path)


def _record(p, saved, inp) -> dict:
    """What the test compares, kept small: a saved component that is a loaded one unchanged is stored as
    the index of that component ('origin'); only the components the window computed (merges) in full."""
    A_in, A = _dense(inp.A), _dense(saved.A)
    by_trace = {inp.C[j].tobytes(): j for j in range(inp.C.shape[0])}
    origin = []
    for k in range(A.shape[1]):
        j = by_trace.get(np.asarray(saved.C[k]).tobytes(), -1)
        same = j >= 0 and np.array_equal(A[:, k], A_in[:, j]) and all(
            np.array_equal(getattr(saved, f)[k], getattr(inp, f)[j]) for f in ROWS)
        origin.append(j if same else -1)
    origin = np.array(origin)
    new = origin < 0
    rec = {f"{p}_origin": origin, f"{p}_A_new": A[:, new]}
    rec.update({f"{p}_{f}_new": np.asarray(getattr(saved, f))[new] for f in ROWS})
    rec.update({f"{p}_{f}": np.asarray(getattr(saved, f)) for f in EXACT + FLOATS})
    return rec


def _expected(g, p, inp) -> dict:
    origin = g[f"{p}_origin"]
    new = origin < 0
    A = np.zeros((inp.A.shape[0], len(origin)))
    A[:, ~new] = _dense(inp.A)[:, origin[~new]]
    A[:, new] = g[f"{p}_A_new"]
    out = {"A": A}
    for f in ROWS:
        rows = np.zeros((len(origin), inp.C.shape[1]))
        rows[~new] = getattr(inp, f)[origin[~new]]
        rows[new] = g[f"{p}_{f}_new"]
        out[f] = rows
    return out


def test_golden_click_sequence(tmp_path, monkeypatch):
    runs = {p: _run(tmp_path / p, monkeypatch, **s) for p, s in SESSIONS.items()}
    if os.environ.get("BM_GOLDEN_WRITE") == "1":
        GOLDEN.parent.mkdir(exist_ok=True)
        rec = {}
        for p, (saved, inp) in runs.items():
            rec.update(_record(p, saved, inp))
        np.savez_compressed(GOLDEN, **rec)
    with np.load(GOLDEN) as g:
        for p, (saved, inp) in runs.items():
            for f in EXACT:
                assert np.array_equal(np.asarray(getattr(saved, f)), g[f"{p}_{f}"]), (p, f)
            for f in FLOATS:
                np.testing.assert_allclose(np.asarray(getattr(saved, f), dtype=float), g[f"{p}_{f}"],
                                           rtol=RTOL, atol=0, err_msg=f"{p} {f}")
            exp = _expected(g, p, inp)
            np.testing.assert_allclose(_dense(saved.A), exp["A"], rtol=RTOL, atol=0, err_msg=f"{p} A")
            for f in ROWS:
                np.testing.assert_allclose(np.asarray(getattr(saved, f)), exp[f], rtol=RTOL, atol=0,
                                           err_msg=f"{p} {f}")
