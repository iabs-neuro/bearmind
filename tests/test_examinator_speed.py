"""Speed of the inspection window's actions on synthetic sessions (slow).

Measured the way the phase 5 page measured the old window: the real button callbacks on a document in
memory, no browser. Time is the Python time of the whole button chain; size is the PATCH-DOC message built
from the document events, i.e. what the server would send to the browser. The synthetic session has 194
components on 608x608 pixels, as the real file of those measurements; the hour has 107 107 frames.
Each test prints its numbers ("[speed] ...", visible with -s).
"""
# bm_examinator imports torch indirectly; on this environment torch's shm.dll
# fails to load (WinError 127) unless torch is imported first.
import torch  # noqa: F401

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from bokeh.document.events import ModelChangedEvent
from bokeh.models import Div, Selection
from bokeh.protocol import Protocol
from caiman.source_extraction.cnmf.estimates import Estimates
from caiman.source_extraction.cnmf.params import CNMFParams
from scipy.signal import lfilter
from scipy.sparse import csc_matrix

sys.path.insert(0, str(Path(__file__).parent.parent))

import bm_examinator
from examinator_helpers import App, changed_columns, message_bytes, patch_doc_bytes

pytestmark = pytest.mark.slow

N_COMP, DIMS, HOUR, SHORT = 194, (608, 608), 107_107, 2_000
MB = 1_000_000
LEGACY = {"mode": "legacy", "end_frame": None}
BULK = {"traces", "traces_recon", "times", "t", "xs", "ys"}


def _session(T, n=N_COMP, dims=DIMS) -> Estimates:
    """n round footprints (radius 12 px) on a grid over dims, AR(1) traces of sparse spikes, T frames."""
    rng = np.random.default_rng(0)
    side = int(np.ceil(np.sqrt(n)))
    step = dims[0] / (side + 1)
    r, sigma = 12, 4.0
    rows, cols, vals = [], [], []
    for k in range(n):
        cy, cx = int(step * (1 + k // side)), int(step * (1 + k % side))
        yy, xx = np.mgrid[cy - r:cy + r + 1, cx - r:cx + r + 1]
        g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
        rows.append((yy + xx * dims[0]).ravel())  # pixel order of CaImAn's A (Fortran)
        cols.append(np.full(g.size, k))
        vals.append(g.ravel())
    A = csc_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                   shape=(dims[0] * dims[1], n))
    spikes = (rng.random((n, T)) < 0.01).astype(float)
    C = lfilter([1.0], [1.0, -0.95], spikes, axis=1) + 0.01 * np.arange(n)[:, None]
    est = Estimates(A=A, C=C, b=None, f=None, dims=dims)
    est.YrA = 0.01 * rng.normal(size=(n, T))
    est.R = est.YrA
    est.S = spikes
    k = np.arange(n)
    est.bl, est.c1, est.neurons_sn = 0.01 * k, 0.02 * k, 1.0 + 0.1 * k
    est.g = (0.9 - 0.001 * k)[:, None]
    est.idx_components = np.arange(n)
    est.idx_components_bad = np.array([], dtype=int)
    est.SNR_comp = 1.0 + 5.0 * rng.random(n)
    est.r_values = rng.random(n)
    est.imax = np.ones(dims)
    est.cnmf_dict = CNMFParams(params_dict={"p": 1, "fr": 20}).to_dict()
    est.time = np.arange(T) / 20.0
    return est


def _fname(folder) -> str:
    return str(Path(folder) / "NOF_H01_1D_estimates.pickle")


@pytest.fixture(scope="module")
def hour(tmp_path_factory):
    """A prepared hour-long session; each test opens its own window on it (the window does not change it)."""
    fname = _fname(tmp_path_factory.mktemp("hour"))
    est = _session(HOUR)
    est.name = fname
    return bm_examinator._prepare(fname, est, 20, LEGACY, None)


def _open(prep) -> App:
    return App(bm_examinator.make_app(prep.fname, 20, LEGACY, prepared=prep))


def _timed(app, label):
    """(seconds, PATCH-DOC bytes, events) of one click on label, the whole chain; rows selected before."""
    with app.events() as events:
        t0 = time.perf_counter()
        app.click(label)
        dt = time.perf_counter() - t0
    return dt, patch_doc_bytes(events), events


def _deletes(app, name):
    results = []
    for rows in ([3], [10, 11], [50]):
        app.select(rows)
        dt, size, events = _timed(app, "Delete sel.")
        assert not {c for _, c in changed_columns(events)} & BULK
        results.append((dt, size))
        print(f"[speed] {name} delete rows={rows} time_s={dt:.4f} patch_bytes={size}")
    return results


def test_delete_payload_small_on_hour_session(hour):
    app = _open(hour)
    first = message_bytes(Protocol().create("PULL-DOC-REPLY", "0", app.doc))
    print(f"[speed] hour first document bytes={first}")
    for dt, size in _deletes(app, "hour"):
        assert dt <= 0.15 and size <= 1 * MB, (dt, size)


def test_delete_fast_short(tmp_path):
    fname = _fname(tmp_path)
    est = _session(SHORT)
    est.name = fname
    app = _open(bm_examinator._prepare(fname, est, 20, LEGACY, None))
    for dt, size in _deletes(app, "short"):
        assert dt <= 0.05 and size <= 0.1 * MB, (dt, size)


def test_merge_time(hour):
    app = _open(hour)
    rows = [0, 1]
    comps = [app.storage.shown[r] for r in rows]
    ref = copy.copy(app.storage.estimates)  # manual_merge replaces attributes; the window's object stays
    params = CNMFParams(params_dict=ref.cnmf_dict)
    t0 = time.perf_counter()
    ref.manual_merge([comps], params=params)
    t_mm = time.perf_counter() - t0
    del ref
    app.select(rows)
    dt, size, events = _timed(app, "Merge sel.")
    print(f"[speed] hour merge time_s={dt:.3f} manual_merge_s={t_mm:.3f} extra_s={dt - t_mm:.3f} "
          f"patch_bytes={size}")
    assert app.shown()[-1] == N_COMP - 2  # the merged one, last
    assert dt <= t_mm + 0.5 and size <= 1 * MB, (dt, t_mm, size)


def test_select_is_cheap(hour):
    app = _open(hour)
    times = []
    with app.events() as events:
        for r in range(20):
            t0 = time.perf_counter()
            app.select([r])
            times.append(time.perf_counter() - t0)
    med = float(np.median(times))
    print(f"[speed] hour select median_ms={1000 * med:.3f} max_ms={1000 * max(times):.3f}")
    assert med <= 0.005, times
    assert changed_columns(events) == set()
    # what changes: the selection (in a real session it comes from the browser) and the metrics panel
    assert {type(e.model) for e in events if isinstance(e, ModelChangedEvent)} <= {Selection, Div}
    assert all(isinstance(e, ModelChangedEvent) for e in events)


def test_save_results_frees_the_window(hour, tmp_path):
    """Save results until the window is free, with a final_save as the inspection server's: the result
    pickled whole and uncompressed, the side files queued for the background thread. The gate decision the
    server also records is a small JSON write and is not part of this measurement."""
    out = tmp_path / "5_Inspection" / "NOF_H01_1D"
    futures = []

    def final_save(est):
        bm_examinator.write_result(est, out / "NOF_H01_1D_manual_estimates.pickle")
        futures.append(bm_examinator.write_side_files_in_background(est, out, "NOF_H01_1D_"))
        return "Saved."

    app = App(bm_examinator.make_app(hour.fname, 20, LEGACY, prepared=hour, final_save=final_save))
    app.select([3])
    app.click("Delete sel.")
    dt, size, _ = _timed(app, "Save results")
    t0 = time.perf_counter()
    futures[0].result(timeout=600)  # before the next test measures: the writer competes for the GIL
    background = time.perf_counter() - t0
    mb = (out / "NOF_H01_1D_manual_estimates.pickle").stat().st_size / MB
    print(f"[speed] hour save_results time_s={dt:.3f} result_mb={mb:.0f} side_files_after_s={background:.1f} "
          f"patch_bytes={size}")
    assert len(list((out / "NOF_H01_1D_filters").iterdir())) == N_COMP - 1
    assert dt <= 0.5, dt
