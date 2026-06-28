"""Per-event statistics for wavelet-detected calcium events (BEARMiND).

For each wavelet ridge of a DRIADA Neuron, collect kinetics (t_rise, t_off),
a RELATIVE amplitude (dF/F0), an absolute amplitude and a per-event proxy SNR.
This is the per-event layer that auto_inspector.py does NOT produce (it only
keeps per-neuron aggregates).

Design (agreed with user):
  - Kinetics are MEASURED from the trace (data-driven), not curve-fitted:
      t_rise = onset -> peak ; t_off = peak -> first frame at/below 1/e decay.
  - Amplitude is RELATIVE: dF/F0 = (peak - F0) / F0, F0 = median of the
    pre-event baseline window. This matches DRIADA's amplitude convention
    (Neuron.extract_event_amplitudes, already_dff=False, neuron.py:502-513).
  - SNR is the DRIADA proxy: (peak - baseline_median) / baseline_noise,
    baseline_noise = MAD of non-event frames (Neuron._calc_wavelet_snr).

The trace used for measurement MUST be the RAW calcium signal (neuron.ca.data),
NOT a [0,1]-normalized one, otherwise F0 ~ 0 and dF/F0 is meaningless. Wavelet
detection itself is scale-invariant (it min-max normalizes internally), so
building the Neuron on the raw trace yields identical events.

ASCII-only output (Windows cp1251).
"""
import numpy as np
from scipy.stats import median_abs_deviation


# Order of columns in each per-event record (single source of truth).
RECORD_FIELDS = [
    "start_frame", "end_frame", "start_s", "end_s", "duration_s",
    "peak_frame", "peak_s", "f0", "amplitude_rel", "amplitude_abs",
    "t_rise_s", "t_off_s", "event_snr",
    "ridge_max_ampl", "ridge_max_scale", "ridge_length",
]


def _baseline_stats(ca, events_mask):
    """Baseline median and MAD noise from non-event frames.

    Mirrors DRIADA Neuron._calc_wavelet_snr (median_abs_deviation, scale='normal').
    Returns (median, noise); (nan, nan) if fewer than 10 baseline frames.
    """
    baseline = ca[~events_mask]
    if len(baseline) < 10:
        return np.nan, np.nan
    return float(np.median(baseline)), float(median_abs_deviation(baseline, scale="normal"))


def _measure_kinetics(ca, start, peak_idx, f0, fps):
    """Data-driven per-event kinetics in seconds.

    t_rise: onset -> peak.
    t_off : peak -> first frame at/below 1/e of (peak - baseline). -1 if never reached.
    """
    t_rise_s = (peak_idx - start) / fps
    peak_val = ca[peak_idx]
    target = f0 + (peak_val - f0) / np.e
    seg = ca[peak_idx:]
    below = np.where(seg <= target)[0]
    t_off_s = (below[0] / fps) if len(below) > 0 else -1.0
    return float(t_rise_s), float(t_off_s)


def extract_event_records(neuron, fps, baseline_window_sec=1.0):
    """Build per-event records from a wavelet-detected DRIADA Neuron.

    Parameters
    ----------
    neuron : DRIADA Neuron
        Must have ``wvt_ridges`` populated via reconstruct_spikes(method='wavelet'),
        and ``ca.data`` holding the RAW calcium trace.
    fps : float
        Sampling rate (Hz).
    baseline_window_sec : float
        Pre-event window (seconds) used to estimate per-event F0.

    Returns
    -------
    list of dict
        One dict per detected wavelet event, with keys == RECORD_FIELDS.
        Empty list if no ridges. ``t_off_s == -1`` if the 1/e level is never
        reached; ``amplitude_rel`` is NaN when F0 <= 0 (undefined dF/F0).
    """
    ridges = getattr(neuron, "wvt_ridges", None)
    if not ridges:
        return []

    ca = np.asarray(neuron.ca.data, dtype=float)
    n = len(ca)
    if n == 0:
        return []
    bw = max(1, int(round(baseline_window_sec * fps)))

    # Neuron-level baseline noise from non-event frames (union of ridge spans).
    mask = np.zeros(n, dtype=bool)
    for r in ridges:
        s, e = int(r.start), int(r.end)
        s, e = min(s, e), max(s, e)
        mask[max(0, s):min(n, e + 1)] = True
    base_median, base_noise = _baseline_stats(ca, mask)
    global_median = float(np.median(ca))

    records = []
    for r in ridges:
        s, e = int(r.start), int(r.end)
        s, e = min(s, e), max(s, e)
        s = max(0, s)
        e = min(n - 1, e)
        if e <= s:
            continue
        peak_idx = s + int(np.argmax(ca[s:e + 1]))

        pre = ca[max(0, s - bw):s]
        if len(pre) > 0:
            f0 = float(np.median(pre))
        elif np.isfinite(base_median):
            f0 = base_median
        else:
            f0 = global_median

        peak_val = float(ca[peak_idx])
        amplitude_abs = peak_val - f0
        amplitude_rel = (amplitude_abs / f0) if f0 > 0 else float("nan")

        t_rise_s, t_off_s = _measure_kinetics(ca, s, peak_idx, f0, fps)

        if np.isfinite(base_noise) and base_noise > 0 and np.isfinite(base_median):
            event_snr = float((peak_val - base_median) / base_noise)
        else:
            event_snr = float("nan")

        records.append(dict(
            start_frame=int(s), end_frame=int(e),
            start_s=s / fps, end_s=e / fps, duration_s=(e - s) / fps,
            peak_frame=int(peak_idx), peak_s=peak_idx / fps,
            f0=f0,
            amplitude_rel=float(amplitude_rel), amplitude_abs=float(amplitude_abs),
            t_rise_s=t_rise_s, t_off_s=t_off_s, event_snr=event_snr,
            ridge_max_ampl=float(r.max_ampl),
            ridge_max_scale=float(r.max_scale),
            ridge_length=int(r.length),
        ))
    return records
