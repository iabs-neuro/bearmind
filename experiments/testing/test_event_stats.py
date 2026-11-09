"""Integration tests for per-event wavelet statistics on REAL NOF data.

Per project protocol: integration tests on real CaImAn/DRIADA objects only
(no toy examples, no CaImAn/DRIADA mocks). Requires the bearmind env to import
the pipeline (torch must load) and the NOF estimates pickle on disk.

Run:
  conda run -n bearmind python -m pytest test_event_stats.py -v
"""
import os
import numpy as np

from wavelet_backend import set_wavelet_backend
set_wavelet_backend("cpu")  # before driada import; remote GPU run can use 'auto'

from auto_inspector import get_neuron_with_spikes
from capcan_validation import load_estimates
from event_stats import extract_event_records, RECORD_FIELDS

NOF = "data/raw_compressed/NOF_H01_1D_gsig4_mincorr0.92_minpnr7_estimates.pickle"
FPS = 19.76


def _first_active_neuron():
    """Build the first NOF neuron that yields wavelet ridges (RAW trace)."""
    assert os.path.exists(NOF), "fixture missing: %s" % NOF
    est = load_estimates(NOF)
    for comp in est.idx_components:
        tr = np.asarray(est.C[comp], dtype=float)
        if float(np.max(tr) - np.min(tr)) <= 1e-10:
            continue
        neuron, _ = get_neuron_with_spikes(
            tr, fps=FPS, lightweight=True,
            event_method="wavelet", hybrid_kinetics=False)
        if getattr(neuron, "wvt_ridges", None):
            return neuron
    raise AssertionError("No neuron with wavelet ridges found in NOF_H01_1D")


def test_records_match_ridges_and_have_sane_values():
    neuron = _first_active_neuron()
    n_ridges = len(neuron.wvt_ridges)
    recs = extract_event_records(neuron, FPS)
    assert len(recs) == n_ridges, (len(recs), n_ridges)
    for r in recs:
        assert set(RECORD_FIELDS).issubset(r.keys())
        assert r["end_frame"] > r["start_frame"]
        assert r["duration_s"] > 0
        assert r["start_frame"] <= r["peak_frame"] <= r["end_frame"]
        assert r["t_rise_s"] >= 0
        assert (r["t_off_s"] == -1.0) or (r["t_off_s"] >= 0)
        assert np.isfinite(r["amplitude_abs"])
        # amplitude_rel may be NaN only when F0 <= 0
        assert np.isfinite(r["amplitude_rel"]) or r["f0"] <= 0


def test_empty_when_no_ridges():
    class _Holder:  # attribute holder for the no-ridges branch (not a CaImAn/DRIADA mock)
        wvt_ridges = []

        class ca:
            data = np.zeros(100)

    assert extract_event_records(_Holder(), FPS) == []


def test_collect_session_one_real_session():
    from collect_wavelet_event_stats import collect_session
    rows = collect_session(NOF, "NOF_H01_1D", FPS)
    assert len(rows) > 0
    cols = set(rows[0].keys())
    assert {"session_name", "component_idx", "event_index", "t_rise_s",
            "t_off_s", "amplitude_rel", "event_snr"}.issubset(cols)
    assert all(r["session_name"] == "NOF_H01_1D" for r in rows)
