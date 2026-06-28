"""Collect per-event wavelet statistics over the NOF dataset (BEARMiND).

For each NOF session: load CaImAn estimates, run wavelet event detection per
neuron (reusing auto_inspector.get_neuron_with_spikes on the RAW trace), and
write a long-format CSV of per-event stats keyed by (session_name, component_idx)
so it joins to the EBM dataset's metrics_init rows.

Why RAW traces: amplitudes are RELATIVE (dF/F0), which needs a meaningful F0.
Wavelet detection min-max normalizes internally, so detection on the raw trace
gives identical events to the [0,1]-normalized path used by capcan metrics.

ASCII-only output. Usage:
  conda run -n bearmind python collect_wavelet_event_stats.py \
      --raw-path data/raw_compressed --experiment NOF \
      --out nof_wavelet_events.csv --backend auto
"""
import os
import sys
import glob
import argparse

# Backend MUST be selected BEFORE importing auto_inspector (-> driada -> ssqueezepy),
# because it is read from the SSQ_GPU env var at import time. Pre-parse --backend.
def _preparse_backend(argv):
    backend = "auto"
    for i, a in enumerate(argv):
        if a == "--backend" and i + 1 < len(argv):
            backend = argv[i + 1]
        elif a.startswith("--backend="):
            backend = a.split("=", 1)[1]
    return backend


from wavelet_backend import set_wavelet_backend  # noqa: E402
_ACTUAL_BACKEND = set_wavelet_backend(_preparse_backend(sys.argv))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from auto_inspector import get_neuron_with_spikes  # noqa: E402
from capcan_validation import load_estimates, fps_map  # noqa: E402
from event_stats import extract_event_records  # noqa: E402
from naming import extract_session_id  # noqa: E402


def _session_name(pickle_filename):
    """Extract canonical session id (e.g. NOF_H01_1D) from a pickle filename."""
    base = os.path.basename(pickle_filename)
    name = extract_session_id(base)
    if name:
        return name
    return base.split("_estimates")[0]


def _build_shared_wavelet(fps):
    """Pre-compute wavelet + time resolutions once per fps (speeds up batch runs).

    Mirrors auto_inspector.get_multineuron_metrics (lines 448-460). Returns
    (wavelet, rel_wvt_times) or (None, None) on failure (callers fall back to
    per-neuron construction).
    """
    try:
        from ssqueezepy.wavelets import Wavelet, time_resolution
        from driada.experiment.wavelet_event_detection import (
            WVT_EVENT_DETECTION_PARAMS, get_adaptive_wavelet_scales,
        )
        beta = WVT_EVENT_DETECTION_PARAMS.get("beta", 2)
        gamma = WVT_EVENT_DETECTION_PARAMS.get("gamma", 3)
        wavelet = Wavelet(("gmw", {"gamma": gamma, "beta": beta, "centered_scale": True}), N=8196)
        scales = get_adaptive_wavelet_scales(fps)
        rel_wvt_times = [
            time_resolution(wavelet, scale=sc, nondim=False, min_decay=200) for sc in scales
        ]
        return wavelet, rel_wvt_times
    except Exception as ex:
        print("[WARN] shared wavelet precompute failed (%s); falling back per-neuron" % ex)
        return None, None


def collect_session(pickle_path, session_name, fps):
    """Return list of per-event dicts for one session (wavelet detection on RAW traces)."""
    est = load_estimates(pickle_path)
    comps = list(est.idx_components)
    ef = est.C.shape[1]
    wavelet_shared, rel_wvt_times_shared = _build_shared_wavelet(fps)

    rows = []
    for comp_idx in comps:
        trace = np.asarray(est.C[comp_idx, :ef], dtype=float)
        if float(np.max(trace) - np.min(trace)) <= 1e-10:
            continue  # flat trace -> no events
        try:
            neuron, _ = get_neuron_with_spikes(
                trace, fps=fps, lightweight=True,
                event_method="wavelet", hybrid_kinetics=False,
                wavelet_shared=wavelet_shared, rel_wvt_times_shared=rel_wvt_times_shared,
            )
        except Exception as ex:  # detection can fail on degenerate traces
            print("[WARN] %s comp %s: detection failed: %s" % (session_name, comp_idx, ex))
            continue
        for ev_i, rec in enumerate(extract_event_records(neuron, fps)):
            rec["session_name"] = session_name
            rec["component_idx"] = int(comp_idx)
            rec["event_index"] = ev_i
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-path", default="data/raw_compressed")
    ap.add_argument("--experiment", default="NOF")
    ap.add_argument("--out", default="nof_wavelet_events.csv")
    ap.add_argument("--limit", type=int, default=0, help="process at most N sessions (0=all)")
    ap.add_argument("--backend", default="auto", help="auto|gpu|cpu (pre-parsed before import)")
    args = ap.parse_args()

    print("[INFO] wavelet backend: %s" % _ACTUAL_BACKEND)
    pattern = os.path.join(args.raw_path, args.experiment + "_*_estimates.pickle")
    files = sorted(glob.glob(pattern))
    if args.limit > 0:
        files = files[:args.limit]
    print("[INFO] %d %s sessions found" % (len(files), args.experiment))

    all_rows = []
    n_ok = 0
    n_skipped = 0
    for i, fpath in enumerate(files, 1):
        sname = _session_name(fpath)
        fps = fps_map.get(sname, 20)
        if sname not in fps_map:
            print("[WARN] no FPS for %s, using 20" % sname)
        print("[%d/%d] %s (fps=%s)" % (i, len(files), sname, fps))
        try:
            rows = collect_session(fpath, sname, fps)
        except Exception as ex:
            print("[WARN] session %s failed: %s" % (sname, ex))
            n_skipped += 1
            continue
        print("       %d events" % len(rows))
        all_rows.extend(rows)
        n_ok += 1

    df = pd.DataFrame(all_rows)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("[OK] wrote %d events from %d sessions (%d skipped) -> %s"
          % (len(df), n_ok, n_skipped, args.out))


if __name__ == "__main__":
    main()
