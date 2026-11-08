r"""Speed "before" again, on today's files: the window code before phase 5 (933c6bd, a git worktree), measured as
the phase 5 page measured it (its timing2.py / merge5.py): the button callbacks called one by one on a document
in memory. The only difference from the page: the tiled copies stay in memory (LoadEstimates of the old module is
replaced by a function that returns them), so file loading is not in "open" (0.03 s at 2000 frames on the page).
The size is the PATCH-DOC message of the action's document events, as for "after" (the page summed column bytes).
Save results writes next to estimates.name, which points into OUT (scratch).

Run with the worktree as the working directory (its config.json). usage: python t7_old_speed.py WORKTREE OUT_DIR
"""
import sys

WT = sys.argv[1]
sys.path.insert(0, WT)
import torch  # noqa: F401,E402

import copy  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import pickle  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from bokeh.document import Document  # noqa: E402
from bokeh.document.events import DocumentPatchedEvent  # noqa: E402
from bokeh.models import Button, ColumnDataSource  # noqa: E402
from bokeh.protocol import Protocol  # noqa: E402

import bm_examinator as be  # noqa: E402  (the old module of the worktree)

assert os.path.normcase(os.path.dirname(os.path.abspath(be.__file__))) == os.path.normcase(os.path.abspath(WT)), be.__file__
assert not hasattr(be, "prepare_session"), "not the pre-phase module"
sys.path.append(r"C:\Users\User\PycharmProjects\bearmind\tests")
from examinator_helpers import message_bytes  # noqa: E402

P = r"H:\Projects\_bearmind_test\series_short\5_Inspection\NOF_H01_1D\NOF_H01_1D_processed.pickle"
P4 = r"H:\Projects\_bearmind_test\series_short\4_Estimates\NOF_H01_1D_estimates.pickle"
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
LENGTHS = [int(x) for x in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["2000", "18000", "107107"])]
KW = {"mode": "legacy", "end_frame": 10 ** 7}  # as the page
R = {"by_length": {}}


def load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def tiled(src, T):
    e = copy.copy(src)
    n0 = src.C.shape[1]
    reps = math.ceil(T / n0)
    for k, v in list(vars(src).items()):
        if sp.issparse(v) and v.ndim == 2 and v.shape[1] == n0:
            setattr(e, k, sp.hstack([v] * reps).tocsr()[:, :T])
        elif isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] == n0:
            setattr(e, k, np.tile(v, (1, reps))[:, :T])
    if getattr(src, "R", None) is not None and src.R is src.YrA:
        e.R = e.YrA
    e.time = np.arange(T) / 30.0
    return e


def window(est):
    be.LoadEstimates = lambda name, default_fps=20, _e=est: copy.copy(_e)
    app = be.make_app(est.name, 20, KW)
    doc = Document()
    t = time.perf_counter()
    app(doc)
    open_s = time.perf_counter() - t
    buttons = {m.label: m for m in doc.models if isinstance(m, Button)}
    src = [m for m in doc.models if isinstance(m, ColumnDataSource) and "dummy_id" in m.data][0]
    return doc, buttons, src, open_s


def chain(buttons, label):
    t = time.perf_counter()
    for cb in buttons[label]._event_callbacks.get("button_click", []):
        cb(None)
    return time.perf_counter() - t


def patch_bytes(doc, fn):
    doc.to_json()
    got = []

    def keep(ev):
        if isinstance(ev, DocumentPatchedEvent):
            got.append(ev)
    doc.on_change(keep)
    try:
        fn()
    finally:
        doc.remove_on_change(keep)
    return message_bytes(Protocol().create("PATCH-DOC", got)) if got else 0


base, base4 = load(P), load(P4)
for T in LENGTHS:
    L = R["by_length"][T] = {"kind": "measured" if T == base.C.shape[1] else "tiled copy"}
    d = OUT / f"t7_old_T{T}"
    d.mkdir(parents=True, exist_ok=True)
    e = tiled(base, T)
    e.name = str(d / "NOF_H01_1D_estimates.pickle")
    doc, buttons, src, L["open_s"] = window(e)
    ts = []
    for i in (3, 7, 11, 15):
        t = time.perf_counter()
        src.selected.indices = [i]
        ts.append(time.perf_counter() - t)
    L["select_ms"] = 1000 * statistics.median(ts)
    ds = []
    for _ in range(2):
        src.selected.indices = [5]
        ds.append(chain(buttons, "Delete sel."))
        chain(buttons, "Revert change")
    L["delete_s"] = statistics.median(ds)
    src.selected.indices = [5]
    L["delete_patch_bytes"] = patch_bytes(doc, lambda: chain(buttons, "Delete sel."))
    L["save_results_s"] = chain(buttons, "Save results")
    print(f"[old] T={T} {json.dumps(L)}", flush=True)
    del doc, buttons, src, e
    shutil.rmtree(d, ignore_errors=True)
    # Merge on the CNMF-E estimates (as merge5.py: rows 3 and 4 of the data source)
    e4 = tiled(base4, T)
    d4 = OUT / f"t7_old_M{T}"
    d4.mkdir(parents=True, exist_ok=True)
    e4.name = str(d4 / "NOF_H01_1D_estimates.pickle")
    doc, buttons, src, _ = window(e4)
    src.selected.indices = [3, 4]
    L["merge_s"] = chain(buttons, "Merge sel.")
    del doc, buttons, src
    doc, buttons, src, _ = window(e4)  # a second window for the size: building the message is not timed
    src.selected.indices = [3, 4]
    L["merge_patch_bytes"] = patch_bytes(doc, lambda: chain(buttons, "Merge sel."))
    print(f"[old] T={T} merge_s={L['merge_s']:.3f} merge_patch_bytes={L['merge_patch_bytes']}", flush=True)
    del doc, buttons, src, e4
    shutil.rmtree(d4, ignore_errors=True)
    (OUT / "t7_old_speed.json").write_text(json.dumps(R, indent=1, default=str), encoding="utf-8")
print("[OK] written", OUT / "t7_old_speed.json")
