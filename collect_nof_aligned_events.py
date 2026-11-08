"""Collect per-event wavelet statistics over the SYNCHRONIZED NOF dataset.

Variant of collect_wavelet_event_stats.py for the *_aligned.npz files produced
by the synchronizer (BEARMiND 'Synchro/synchronized'). Those files hold an
already-curated calcium matrix under key 'Calcium' (neurons x time) plus the
aligned behaviour features; there is NO CaImAn estimates object, so we cannot
use load_estimates / est.idx_components.

Why this script (and not collect_wavelet_event_stats.py):
  - Data is .npz with a 'Calcium' matrix, not a *_estimates.pickle.
  - The local bearmind env has a broken BLAS/LAPACK (scipy.optimize.nnls and
    numpy.linalg.lstsq hard-crash with Windows 0xc06d007f), so we run under the
    'caiman' env which has working LAPACK + driada + ssqueezepy.
  - We call driada Neuron.reconstruct_spikes directly (wavelet, non-iterative)
    instead of auto_inspector.get_neuron_with_spikes, to avoid importing caiman
    (its module import is flaky on this machine) and to keep dependencies thin.
    For lightweight=True, hybrid_kinetics=False, get_neuron_with_spikes reduces
    to exactly this Neuron + reconstruct_spikes call (kinetics are not used by
    extract_event_records).

Detection runs on the RAW synchronized calcium trace: amplitudes are RELATIVE
(dF/F0) which needs a meaningful F0; wavelet detection min-max normalizes
internally, so detection on the raw trace gives identical events.

FPS: read EXACTLY from fps_data.csv (semicolon-delimited '#;Filename;FPS'), keyed
by session id. NOF sessions are 19.76 Hz; 11848 frames / 19.76 = 600 s = the
standard 10-min session, confirming the synchronized timeline is the calcium one.

ASCII-only output. Usage (use the 'driada' env -- driada 1.1 + working LAPACK):
  C:/Users/1/anaconda3/envs/driada/python.exe collect_nof_aligned_events.py \
      --data-path "D:/_Projects/NOF/Synchro/synchronized" \
      --out nof_wavelet_events.csv --backend cpu
"""
import os
import sys
import csv
import glob
import argparse

import numpy as np
import pandas as pd

from event_stats import extract_event_records, RECORD_FIELDS

_HERE = os.path.dirname(os.path.abspath(__file__))


def load_exact_fps(fps_csv=None):
    """Map session_id -> exact (unrounded) FPS from fps_data.csv.

    The CSV is semicolon-delimited with header '#;Filename;FPS'. We key by the
    full Filename and also by its first 10 chars (the canonical session id),
    so 'NOF_H01_1D' resolves whether the CSV stores it bare or suffixed.
    """
    path = fps_csv or os.path.join(_HERE, "fps_data.csv")
    fps = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            name = (row.get("Filename") or "").strip()
            try:
                val = float(row["FPS"])
            except (KeyError, TypeError, ValueError):
                continue
            if not name:
                continue
            fps[name] = val
            fps.setdefault(name[:10], val)
    return fps


def _session_name(npz_path):
    """NOF_H01_1D from .../NOF_H01_1D_aligned.npz."""
    base = os.path.basename(npz_path)
    for suffix in ("_aligned.npz", ".npz"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    return base


def _cpu_only_wavelet_backend(backend):
    """Block torch and choose the wavelet backend; main() calls this before anything imports driada.

    Both are process-wide, so they are done here and not on import: a module that imports this one keeps its own
    torch and its own backend choice.
    """
    # This collector is CPU-only. The local envs ship a broken torch (DLL load
    # crashes 0xc0000139 / 0xc06d007f). ssqueezepy lazily does `import torch`, and
    # the crash is a FATAL native exception that bypasses ssqueezepy's try/except,
    # killing the process. Block torch in sys.modules BEFORE importing ssqueezepy/
    # driada so `import torch` raises a catchable ImportError -> ssqueezepy falls
    # back to its numpy backend (the CPU path we want anyway).
    sys.modules["torch"] = None
    # Backend MUST be selected BEFORE importing driada (-> ssqueezepy), because it is
    # read from the SSQ_GPU env var at import time.
    from wavelet_backend import set_wavelet_backend
    return set_wavelet_backend(backend)


def _build_shared_wavelet(fps):
    """Pre-compute wavelet + time resolutions once per fps (speeds up batch runs).

    Mirrors collect_wavelet_event_stats._build_shared_wavelet. Returns
    (wavelet, rel_wvt_times) or (None, None) on failure (caller falls back to
    per-neuron construction inside driada).
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


def collect_session(npz_path, session_name, fps):
    """Return list of per-event dicts for one synchronized session.

    Components are rows of the 'Calcium' matrix (already curated -> all rows are
    neurons). component_idx == row index, the join key to downstream tables.
    """
    d = np.load(npz_path, allow_pickle=True)
    if "Calcium" not in d:
        raise KeyError("no 'Calcium' key in %s (keys=%s)" % (npz_path, list(d.keys())))
    ca_mat = np.asarray(d["Calcium"], dtype=float)
    if ca_mat.ndim != 2:
        raise ValueError("Calcium is %dD, expected 2D (neurons x time)" % ca_mat.ndim)

    from driada.experiment.neuron import Neuron  # here: after main() chose the backend

    wavelet_shared, rel_wvt_times_shared = _build_shared_wavelet(fps)

    rows = []
    for comp_idx in range(ca_mat.shape[0]):
        trace = ca_mat[comp_idx]
        if float(np.max(trace) - np.min(trace)) <= 1e-10:
            continue  # flat trace -> no events
        try:
            neuron = Neuron(cell_id="", ca=trace, sp=None, fps=fps)
            neuron.reconstruct_spikes(
                method="wavelet", iterative=False, create_event_regions=True,
                fps=fps, wavelet=wavelet_shared, rel_wvt_times=rel_wvt_times_shared,
            )
        except Exception as ex:  # detection can fail on degenerate traces
            print("[WARN] %s comp %s: detection failed: %s" % (session_name, comp_idx, ex))
            continue
        # Pass the ORIGINAL trace: this dataset's Calcium has negative values and
        # DRIADA rectifies neuron.ca.data to non-negative, which would collapse the
        # baseline noise (-> bogus event_snr) and F0 (-> bogus dF/F0). See raw_trace
        # note in event_stats.extract_event_records.
        for ev_i, rec in enumerate(extract_event_records(neuron, fps, raw_trace=trace)):
            rec["session_name"] = session_name
            rec["component_idx"] = int(comp_idx)
            rec["event_index"] = ev_i
            rows.append(rec)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-path", default=r"D:/_Projects/NOF/Synchro/synchronized")
    ap.add_argument("--pattern", default="NOF_*_aligned.npz")
    ap.add_argument("--out", default="nof_wavelet_events.csv")
    ap.add_argument("--fps-csv", default=None, help="override fps_data.csv path")
    ap.add_argument("--default-fps", type=float, default=19.76)
    ap.add_argument("--limit", type=int, default=0, help="process at most N sessions (0=all)")
    ap.add_argument("--backend", default="cpu", help="auto|gpu|cpu (set before driada is imported)")
    args = ap.parse_args()

    actual_backend = _cpu_only_wavelet_backend(args.backend)
    print("[INFO] wavelet backend: %s" % actual_backend)
    fps_lookup = load_exact_fps(args.fps_csv)
    print("[INFO] loaded exact FPS for %d names" % len(fps_lookup))

    files = sorted(glob.glob(os.path.join(args.data_path, args.pattern)))
    if args.limit > 0:
        files = files[: args.limit]
    print("[INFO] %d sessions found in %s" % (len(files), args.data_path))

    all_rows = []
    n_ok = 0
    n_skipped = 0
    for i, fpath in enumerate(files, 1):
        sname = _session_name(fpath)
        fps = fps_lookup.get(sname, fps_lookup.get(sname[:10], args.default_fps))
        if sname not in fps_lookup and sname[:10] not in fps_lookup:
            print("[WARN] no FPS for %s, using default %.4f" % (sname, fps))
        print("[%d/%d] %s (fps=%.4f)" % (i, len(files), sname, fps), flush=True)
        try:
            rows = collect_session(fpath, sname, fps)
        except Exception as ex:
            print("[WARN] session %s failed: %s" % (sname, ex))
            n_skipped += 1
            continue
        print("       %d events" % len(rows), flush=True)
        all_rows.extend(rows)
        n_ok += 1

    cols = RECORD_FIELDS + ["session_name", "component_idx", "event_index"]
    df = pd.DataFrame(all_rows, columns=cols)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df.to_csv(args.out, index=False)
    print("[OK] wrote %d events from %d sessions (%d skipped) -> %s"
          % (len(df), n_ok, n_skipped, args.out))


if __name__ == "__main__":
    main()
