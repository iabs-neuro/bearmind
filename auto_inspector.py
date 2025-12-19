import copy
import os
import pickle
from pathlib import Path

from driada.experiment.wavelet_event_detection import extract_wvt_events, WVT_EVENT_DETECTION_PARAMS
from driada.experiment.neuron import Neuron, DEFAULT_T_RISE, DEFAULT_T_OFF, DEFAULT_FPS

from utils import *
import numpy as np
import pandas as pd
import warnings
import tqdm
import time

import matplotlib.pyplot as plt

from caiman.source_extraction.cnmf import params
from caiman.components_evaluation import (
        evaluate_components_CNN, estimate_components_quality_auto,
        select_components_from_metrics, compute_eccentricity,
        compute_event_exceptionality)


from scipy.stats import median_abs_deviation, skew, kurtosis, spearmanr
from scipy.spatial import distance_matrix
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from joblib import Parallel, delayed
from polygon import (get_contours, get_circularities, convex_polygons_min_distance,
                     calculate_polygon_area, calculate_perimeter, get_max_edges, get_convexities,
                     convex_hull, get_aspect_ratios)
from corner_artifacts import detect_edge_artifacts
from ml.data_utils import FEATURE_COLS as ML_FEATURE_COLS, get_feature_cols, NON_FEATURE_COLS

# NaN SEMANTICS for ML features:
# - t_rise, t_off: -1 sentinel = no events detected (informative signal)
# - caiman_snr: capped at max finite value if Inf
# - Other NaNs: degenerate footprints or processing failures
# - EBM handles NaN natively; no imputation required


def compute_correlation_matrix(data, method='pearson'):
    """
    Compute correlation matrix using specified method.

    Args:
        data: 2D array of shape (n_samples, n_features)
        method: 'pearson' or 'spearman'

    Returns:
        Correlation matrix of shape (n_samples, n_samples)
    """
    if method == 'pearson':
        return np.corrcoef(data)
    elif method == 'spearman':
        # spearmanr returns (correlation, p-value) tuple
        # For matrix input with axis=1, computes pairwise correlations between rows
        if data.shape[0] == 1:
            return np.array([[1.0]])
        corr_matrix, _ = spearmanr(data, axis=1)
        return corr_matrix
    else:
        raise ValueError(f"Unknown correlation method: {method}. Use 'pearson' or 'spearman'")


def get_hvals(traces):
    # DEPRECATED: old attempt to quantify "spike vs baseline timing"
    hvals = []
    for tr in traces:
        med = np.median(tr)
        meddev = median_abs_deviation(tr)
        hval = np.round(1.0 * len(np.where(tr >= med + 4 * meddev)[0]) / len(tr), 4)
        hvals.append(hval)

    return hvals


def get_neuron_with_spikes(trace, fps=DEFAULT_FPS, lightweight=True, event_method='threshold', n_iter=2, hybrid_kinetics=True):
    """
    Create a Neuron object and detect events using specified method.

    Args:
        trace: Calcium trace array
        fps: Frames per second
        lightweight: If True, skip expensive re-detection with optimized kinetics
        event_method: Event detection method ('threshold' or 'wavelet')
        n_iter: Number of iterations for iterative reconstruction (default: 2)
        hybrid_kinetics: If True, use cascading kinetics optimization (default: True):
            1. wavelet standard -> 2. wavelet relaxed -> 3. threshold standard -> 4. threshold relaxed -> 5. defaults
    """
    # Create Neuron object
    neuron = Neuron(
        cell_id="",
        ca=trace,
        sp=None,
        fps=fps
    )

    # Stage 1: Initial single-pass detection with default kinetics (conservative)
    neuron.reconstruct_spikes(
        method=event_method,
        iterative=False,  # Single-pass to get initial events safely
        create_event_regions=True,  # Create event regions for quality metrics
        fps=fps  # Pass fps explicitly so DRIADA uses correct value (not DEFAULT_FPS)
    )

    # Validation: check fps propagation (defense against future regressions)
    if hasattr(neuron, 'fps') and neuron.fps != fps:
        warnings.warn(
            f"FPS mismatch detected: neuron.fps={neuron.fps} != expected {fps}. "
            f"This indicates a parameter propagation bug.",
            RuntimeWarning
        )

    # Stage 2: Measure kinetics from initial events
    if hybrid_kinetics:
        # Cascading kinetics optimization
        kinetics_result = _optimize_kinetics_hybrid(neuron, trace, fps)
    else:
        # Standard kinetics optimization
        kinetics_result = neuron.get_kinetics(
            method='direct',
            use_cached=False,
            update_reconstruction=False
        )

    # Stage 3: optionally re-run detection with optimized kinetics
    if not lightweight:
        neuron.reconstruct_spikes(
            method=event_method,
            n_mad=4.0,  # Balanced threshold for noisy data
            min_duration_frames=2,  # Allow shorter events
            create_event_regions=True,
            iterative=True,
            n_iter=n_iter,
            adaptive_thresholds=True
        )

    return neuron, kinetics_result


def _optimize_kinetics_hybrid(neuron, trace, fps):
    """
    Cascading kinetics optimization:
    1. Wavelet standard
    2. Wavelet relaxed (min_events=3, min_r2=0.6)
    3. Threshold standard
    4. Threshold relaxed
    5. Defaults

    Updates neuron.t_rise and neuron.t_off with best available kinetics.
    Returns kinetics_result dict with 'kinetics_source' field.
    """
    kinetics_result = {'optimized': False, 'kinetics_source': 'defaults'}

    # 1. Wavelet standard
    try:
        result = neuron.get_kinetics(
            method='direct', fps=fps,
            use_cached=False, update_reconstruction=False
        )
        if result.get('optimized'):
            result['kinetics_source'] = 'wavelet_standard'
            return result
    except Exception:
        pass

    # 2. Wavelet relaxed
    try:
        result = neuron.get_kinetics(
            method='direct', fps=fps,
            use_cached=False, update_reconstruction=False,
            min_events=3, min_r2=0.6
        )
        if result.get('optimized'):
            result['kinetics_source'] = 'wavelet_relaxed'
            return result
    except Exception:
        pass

    # 3. Threshold standard - need threshold events
    neuron_thr = None
    try:
        neuron_thr = Neuron(cell_id="", ca=trace, sp=None, fps=fps)
        neuron_thr.reconstruct_spikes(
            method='threshold', n_mad=4.0, min_duration_frames=2,
            iterative=False, create_event_regions=True, fps=fps
        )
        result = neuron_thr.get_kinetics(
            method='direct', fps=fps,
            use_cached=False, update_reconstruction=False
        )
        if result.get('optimized'):
            neuron.t_rise = neuron_thr.t_rise
            neuron.t_off = neuron_thr.t_off
            result['kinetics_source'] = 'threshold_standard'
            return result
    except Exception:
        pass

    # 4. Threshold relaxed
    try:
        if neuron_thr is None:
            neuron_thr = Neuron(cell_id="", ca=trace, sp=None, fps=fps)
            neuron_thr.reconstruct_spikes(
                method='threshold', n_mad=4.0, min_duration_frames=2,
                iterative=False, create_event_regions=True, fps=fps
            )
        result = neuron_thr.get_kinetics(
            method='direct', fps=fps,
            use_cached=False, update_reconstruction=False,
            min_events=3, min_r2=0.6
        )
        if result.get('optimized'):
            neuron.t_rise = neuron_thr.t_rise
            neuron.t_off = neuron_thr.t_off
            result['kinetics_source'] = 'threshold_relaxed'
            return result
    except Exception:
        pass

    # 5. All failed - use defaults
    kinetics_result['kinetics_source'] = 'defaults'
    return kinetics_result


def get_signal_metrics(neuron, kinetics_result=None):
    """
    Extract signal-based metrics from a neuron.

    Args:
        neuron: DRIADA Neuron object
        kinetics_result: dict from get_kinetics() with keys:
            - 'optimized': bool (True only if BOTH params measured)
            - 'partially_optimized': bool (True if exactly one measured)
            - 'used_defaults': {'t_rise': bool, 't_off': bool}
    """
    # metrics that don't require precise reconstruction

    n_events = int(np.sum(neuron.asp.data > 0))
    duration_min = len(neuron.asp.data)//neuron.fps/60.0
    epm = n_events/duration_min
    events_dur = 1.0*np.sum(neuron.sp.data.astype(int))
    events_fraction = events_dur/len(neuron.sp.data)

    t_rise = neuron.t_rise/neuron.fps if not pd.isna(neuron.t_rise) else -1
    t_off = neuron.t_off/neuron.fps if not pd.isna(neuron.t_off) else -1

    try:
        event_snr = neuron.get_wavelet_snr()
        # Log transform to handle extreme outliers (corrupted values can reach millions)
        # log1p handles 0 values gracefully: log1p(x) = log(1+x)
        if event_snr > 0:
            event_snr = np.log1p(event_snr)
        else:
            event_snr = 0.0
    except ValueError:
        event_snr = -1

    # Peak amplitude coefficient of variation (consistency of event amplitudes)
    # Real neurons have consistent amplitudes; artifacts vary wildly
    peak_amplitudes = neuron.asp.data[neuron.asp.data > 0]
    if len(peak_amplitudes) > 1 and np.mean(peak_amplitudes) > 0:
        peak_amplitude_cv = np.std(peak_amplitudes) / np.mean(peak_amplitudes)
    else:
        peak_amplitude_cv = np.nan  # not enough events to compute CV

    # Determine kinetics optimization status
    # Uses DRIADA 0.6.4+ fields: optimized, partially_optimized, used_defaults
    kinetics_source = 'unknown'
    if kinetics_result is not None:
        kinetics_optimized = kinetics_result.get('optimized', False)
        partially_optimized = kinetics_result.get('partially_optimized', False)
        kinetics_source = kinetics_result.get('kinetics_source', 'unknown')

        if kinetics_optimized:
            kinetics_opt = 1  # Full success - both t_rise and t_off measured
        elif partially_optimized:
            kinetics_opt = 0.5  # Partial - one measured, one defaulted
        else:
            kinetics_opt = 0  # Full failure - both defaulted or no events
    elif t_rise == -1 or t_off == -1:
        # Fallback: -1 indicates explicit no-events case
        kinetics_opt = 0
    elif t_rise > 0 and t_off > 0:
        # Fallback: assume success if positive values present
        kinetics_opt = 1
    else:
        # Unknown state
        kinetics_opt = 0

    sig_metrics = {
        'events_per_min': epm,
        'events_fraction': events_fraction,
        't_rise': t_rise,
        't_off': t_off,
        'event_snr': event_snr,
        'peak_amplitude_cv': peak_amplitude_cv,
        'kinetics_opt': kinetics_opt,
        'kinetics_source': kinetics_source
    }

    return sig_metrics


def get_reconstruction_quality_metrics(neuron):
    """
    Get reconstruction quality metrics from a DRIADA neuron.

    Always returns the reconstruction array alongside metrics.
    """
    # Get quality metrics
    r2_score = neuron.get_reconstruction_r2()
    event_r2_score = neuron.get_reconstruction_r2(event_only=True)
    nmae = neuron.get_nmae()
    nrmse = neuron.get_nrmse()
    snr_recon = neuron.get_snr_reconstruction()
    rec = neuron.reconstructed
    # Extract scaled data (0-1) from TimeSeries object
    if hasattr(rec, 'scdata'):
        rec = rec.scdata

    rec_metrics = {
            'r2_score': r2_score,
            'event_r2_score': event_r2_score,
            'nmae': nmae,
            'nrmse': nrmse,
            'snr_recon': snr_recon,
            'reconstruction': rec  # Always include reconstruction (as numpy array, scaled 0-1)
        }

    return rec_metrics


def get_single_neuron_metrics(trace, fps=DEFAULT_FPS, include_heavy=False, event_method='threshold', n_iter=2, hybrid_kinetics=True):
    """
    Extract metrics from a single neuron trace.

    Handles flat/zero traces by returning NaN for all metrics.

    Args:
        trace: Calcium trace array
        fps: Frames per second
        include_heavy: If True, compute reconstruction quality metrics
        event_method: Event detection method ('threshold' or 'wavelet')
        n_iter: Number of iterations for iterative reconstruction (default: 2)
        hybrid_kinetics: If True, use cascading kinetics optimization (default: True)
    """
    try:
        neuron, kinetics_result = get_neuron_with_spikes(trace, fps=fps, lightweight=not include_heavy, event_method=event_method, n_iter=n_iter, hybrid_kinetics=hybrid_kinetics)
        signal_metrics = get_signal_metrics(neuron, kinetics_result=kinetics_result)
        if include_heavy:
            rec_metrics = get_reconstruction_quality_metrics(neuron)
            return {**signal_metrics, **rec_metrics}
        else:
            return signal_metrics

    except (ValueError, ZeroDivisionError, Exception) as e:
        # Handle flat/zero traces or other processing failures
        # Return NaN for all metrics
        nan_signal_metrics = {
            'events_per_min': np.nan,
            'events_fraction': np.nan,
            't_rise': np.nan,
            't_off': np.nan,
            'event_snr': np.nan,
            'peak_amplitude_cv': np.nan,
            'kinetics_opt': np.nan,
            'kinetics_source': 'error'
        }

        if include_heavy:
            nan_rec_metrics = {
                'r2_score': np.nan,
                'event_r2_score': np.nan,
                'nmae': np.nan,
                'nrmse': np.nan,
                'snr_recon': np.nan,
                'reconstruction': None  # Placeholder for failed traces
            }
            return {**nan_signal_metrics, **nan_rec_metrics}
        else:
            return nan_signal_metrics


def get_multineuron_metrics(traces, fps=DEFAULT_FPS, include_heavy=False, event_method='threshold', n_iter=2, hybrid_kinetics=True):
    all_metrics = {}
    reconstructions = {}
    n = traces.shape[0]
    metrics_res = Parallel(n_jobs=-1)(
        delayed(get_single_neuron_metrics)(traces[i], fps=fps, include_heavy=include_heavy, event_method=event_method, n_iter=n_iter, hybrid_kinetics=hybrid_kinetics)
        for i in range(traces.shape[0])
    )

    for metric in metrics_res[0].keys():
        if metric == 'reconstruction':
            # Extract reconstructions into separate dict (index → array)
            for i in range(n):
                rec = metrics_res[i].get('reconstruction')
                if rec is not None:
                    reconstructions[i] = rec
        else:
            all_metrics[metric] = [metrics_res[i][metric] for i in range(n)]

    return all_metrics, reconstructions


def footprint_center_distmat(centers):
    centers = np.array(centers)
    dist_matrix = distance_matrix(centers, centers)
    return dist_matrix


def footprint_boundary_distmat(contours, mask=None, verbose=True):
    n = len(contours)
    if verbose:
        print('Computing boundary distances...')
    if mask is None: # mask[i,j] = True means we want to compute for this pair
        mask = np.ones((n,n), dtype=bool)
    else:
        mask = mask.astype(bool)

    cont_distmat = np.full((n,n), np.inf)  # Initialize with inf so uncomputed pairs aren't merged
    for i, c1 in tqdm.tqdm(enumerate(contours)):
        for j, c2 in enumerate(contours):
            if mask[i,j]:
                dist = convex_polygons_min_distance(c1["coordinates"], c2["coordinates"])
                cont_distmat[i,j] = dist
                cont_distmat[j,i] = dist

    return cont_distmat


def get_edge_distances(centers, fov_shape):
    """
    Compute minimum distance from each center to FOV boundary, normalized by FOV size.

    Edge artifacts tend to have low edge_distance values.

    Args:
        centers: List of (y, x) coordinates
        fov_shape: Tuple of (height, width) of FOV

    Returns:
        np.array of relative edge distances (0 = at edge, 0.5 = at center)
    """
    fov_height, fov_width = fov_shape
    # Normalize by half of min dimension so range is [0, 0.5] for center
    norm_factor = min(fov_height, fov_width) / 2.0
    edge_distances = []
    for center in centers:
        cy, cx = center  # Note: center is (y, x) format
        dist_to_edges = [
            cx,                 # distance to left edge
            fov_width - cx,     # distance to right edge
            cy,                 # distance to top edge
            fov_height - cy     # distance to bottom edge
        ]
        edge_distances.append(min(dist_to_edges) / norm_factor)
    return np.array(edge_distances)


def get_nn_distances(distance_matrix):
    """
    Compute nearest neighbor distance for each element from a distance matrix.

    Low values indicate overlapping/merged neurons.

    Args:
        distance_matrix: Square distance matrix (e.g., FCD or FBD)

    Returns:
        np.array of nearest neighbor distances
    """
    n = distance_matrix.shape[0]
    nn_distances = []
    for i in range(n):
        row = distance_matrix[i].copy()
        row[i] = np.inf  # exclude self
        nn_distances.append(np.min(row))
    return np.array(nn_distances)


def get_tau_decays(est, comps_to_select, fps):
    """
    Compute tau_decay from CaImAn autoregressive parameter g.

    tau = -1/log(g) converted from frames to seconds.
    Represents calcium indicator decay time.

    Args:
        est: CaImAn estimates object
        comps_to_select: List of component indices
        fps: Frames per second for unit conversion

    Returns:
        np.array of tau_decay values in seconds
    """
    n_cells = len(comps_to_select)
    if not hasattr(est, 'g'):
        return np.full(n_cells, np.nan)

    tau_decays = []
    for i in comps_to_select:
        g_val = est.g[i]
        # Handle both list and array cases, extract first AR parameter
        if isinstance(g_val, (list, np.ndarray)):
            g_val = g_val[0] if len(g_val) > 0 else np.nan
        # Convert g to tau: tau = -1/log(g) in frames, then convert to seconds
        if not np.isnan(g_val) and 0 < g_val < 1:
            tau_frames = -1.0 / np.log(g_val)
            tau_decays.append(tau_frames / fps)
        else:
            tau_decays.append(np.nan)
    return np.array(tau_decays)


def get_trace_stats(traces):
    """
    Compute trace statistics (skewness, kurtosis, bimodality) for multiple traces.

    Real neurons have high positive skewness (baseline + rare spikes)
    and high kurtosis (heavy tails from spike events).

    Bimodality coefficient: BC = (skewness^2 + 1) / (kurtosis + 3)
    BC > 0.555 suggests bimodality (uniform distribution = 0.555).
    High bimodality in traces indicates values clustering at two levels
    (e.g., baseline and saturated peaks).

    Args:
        traces: 2D array of shape (n_cells, n_timepoints)

    Returns:
        Tuple of (skewnesses, kurtoses, bimodalities) as np.arrays
    """
    n_cells = traces.shape[0]
    skewnesses = []
    kurtoses = []
    bimodalities = []

    for i in range(n_cells):
        trace = traces[i]
        if len(trace) > 3:  # stats need at least 3 points
            s = skew(trace)
            k = kurtosis(trace)  # excess kurtosis (Fisher)
            skewnesses.append(s)
            kurtoses.append(k)

            # Bimodality coefficient: (skew^2 + 1) / (kurtosis + 3)
            # kurtosis from scipy is excess kurtosis, add 3 for regular kurtosis
            if not np.isnan(s) and not np.isnan(k) and (k + 3) != 0:
                bc = (s**2 + 1) / (k + 3)
                bimodalities.append(bc)
            else:
                bimodalities.append(np.nan)
        else:
            skewnesses.append(np.nan)
            kurtoses.append(np.nan)
            bimodalities.append(np.nan)

    return np.array(skewnesses), np.array(kurtoses), np.array(bimodalities)


def get_hurst_exponents(traces, min_window=10, max_windows=20):
    """
    Compute Hurst exponent for multiple traces using R/S analysis.

    The Hurst exponent (H) measures long-range dependence:
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Persistent (trending, positive autocorrelation)
    - H < 0.5: Anti-persistent (mean-reverting)

    Real neurons typically show H ≈ 0.6-0.8 due to calcium kinetics.
    Drift artifacts show H → 1.0 (very persistent).
    Noise-dominated signals show H → 0.5.

    Args:
        traces: 2D array of shape (n_cells, n_timepoints)
        min_window: Minimum window size for R/S analysis
        max_windows: Maximum number of window sizes to use

    Returns:
        np.array of Hurst exponent values
    """
    n_cells = traces.shape[0]
    hurst_values = np.full(n_cells, np.nan)

    for i in range(n_cells):
        trace = np.asarray(traces[i]).flatten()
        n = len(trace)

        if n < 100:
            continue

        max_k = n // 2
        if max_k < min_window:
            continue

        step = max(1, max_k // max_windows)
        rs_values = []
        ns = []

        for k in range(min_window, max_k, step):
            rs = []
            for start in range(0, n - k, k):
                segment = trace[start:start + k]
                mean = np.mean(segment)
                cumdev = np.cumsum(segment - mean)
                R = np.max(cumdev) - np.min(cumdev)
                S = np.std(segment, ddof=1)
                if S > 0:
                    rs.append(R / S)
            if rs:
                rs_values.append(np.mean(rs))
                ns.append(k)

        if len(ns) < 2:
            continue

        try:
            H = np.polyfit(np.log(ns), np.log(rs_values), 1)[0]
            hurst_values[i] = H
        except Exception:
            pass

    return hurst_values


def get_baseline_drifts(traces):
    """
    Compute normalized baseline drift for multiple traces.

    Measures linear trend magnitude normalized by signal range.
    High values indicate slow drift artifacts.

    drift = |slope * n_frames| / trace_range

    Interpretation:
    - drift < 0.1: Stable baseline
    - drift > 0.5: Significant linear trend (potential artifact)

    Args:
        traces: 2D array of shape (n_cells, n_timepoints)

    Returns:
        np.array of baseline drift values
    """
    n_cells = traces.shape[0]
    drift_values = np.full(n_cells, np.nan)

    for i in range(n_cells):
        trace = np.asarray(traces[i]).flatten()
        n = len(trace)

        if n < 10:
            continue

        trace_range = np.max(trace) - np.min(trace)
        if trace_range == 0 or np.isclose(trace_range, 0, atol=1e-10):
            continue

        x = np.arange(n)
        try:
            slope = np.polyfit(x, trace, 1)[0]
            drift_values[i] = abs(slope * n) / trace_range
        except Exception:
            pass

    return drift_values


def get_half_crossing_rates(traces, fps):
    """
    Compute half-crossing rate for each trace (crossings per minute).

    Counts how many times the normalized trace crosses the 0.5 threshold,
    normalized to rate per minute.

    High HCR indicates noisy/continuous activity (artifacts).
    Low HCR indicates sparse calcium events (real neurons).

    Args:
        traces: 2D array of shape (n_cells, n_timepoints)
        fps: Frames per second (required for time normalization)

    Returns:
        np.array of half-crossing rates (crossings per minute)
    """
    n_cells = traces.shape[0]
    n_timepoints = traces.shape[1]
    hcr_values = np.full(n_cells, np.nan)

    # Calculate recording duration in minutes
    if fps is None or fps <= 0:
        raise ValueError("fps must be provided and positive for rate calculation")

    duration_seconds = n_timepoints / fps
    duration_minutes = duration_seconds / 60.0

    for i in range(n_cells):
        trace = np.asarray(traces[i]).flatten()

        # Skip flat traces
        if np.ptp(trace) < 1e-10:
            hcr_values[i] = 0.0
            continue

        # Ensure normalized [0, 1]
        trace_min = np.min(trace)
        trace_max = np.max(trace)
        trace_norm = (trace - trace_min) / (trace_max - trace_min)

        # Count crossings of 0.5 threshold
        above_half = trace_norm > 0.5
        crossings = np.sum(np.abs(np.diff(above_half.astype(int))))

        # Normalize to rate per minute
        hcr_values[i] = crossings / duration_minutes

    return hcr_values


def get_compactnesses(contours, areas):
    """
    Compute footprint compactness for each contour.

    Compactness = area / convex_hull_area.
    Multi-blob neurons (merge artifacts) have low compactness.

    Args:
        contours: List of contour dicts with 'coordinates' key
        areas: List/array of pre-computed areas

    Returns:
        np.array of compactness values
    """
    compactnesses = []
    for i in range(len(contours)):
        coords = contours[i]["coordinates"]
        try:
            hull = convex_hull(coords)
            if len(hull) >= 3:
                hull_area = calculate_polygon_area(hull)
                compactness = areas[i] / hull_area if hull_area > 0 else np.nan
                compactnesses.append(compactness)
            else:
                compactnesses.append(np.nan)
        except Exception:
            compactnesses.append(np.nan)
    return np.array(compactnesses)


def get_saturation_metrics(traces, fps):
    """
    Compute saturation metrics for multiple traces.

    Saturation occurs when peaks stay at maximum instead of decaying.
    Normal double-exponential peaks: ~0.03-0.17 seconds at peak
    Saturated peaks: >0.5 seconds at peak plateau

    Measures time spent within 80% of each peak's value, averaged across peaks.
    Uses Gaussian smoothing (σ=3) to reduce noise and better capture plateau regions.

    Args:
        traces: 2D array of shape (n_cells, n_timepoints)
        fps: Frames per second (for conversion to seconds)

    Returns:
        mean_time_at_peak: np.array of seconds (FPS-independent)
    """
    n_cells = traces.shape[0]
    mean_times_at_peak = np.full(n_cells, np.nan)

    for i in range(n_cells):
        trace = traces[i]
        n_frames = len(trace)

        if n_frames < 100:
            continue

        # Smoothing for robust peak detection and noise reduction
        trace_smooth = gaussian_filter1d(trace, sigma=3)

        # Normalize to 0-1
        trace_min, trace_max = trace_smooth.min(), trace_smooth.max()
        trace_range = trace_max - trace_min
        if trace_range < 1e-10:
            continue

        trace_norm = (trace_smooth - trace_min) / trace_range

        # Find peaks (at least 30% of range, min distance 0.3s)
        min_distance = max(1, int(fps * 0.3))
        peaks, _ = find_peaks(trace_norm, height=0.3, distance=min_distance)

        if len(peaks) == 0:
            continue

        # Limit to first 100 peaks
        peaks = peaks[:100]
        n_peaks = len(peaks)

        # Vectorized: get all peak values and thresholds at once
        peak_values = trace_norm[peaks]
        thresholds = peak_values * 0.80

        # Compute extent for each peak using numpy (much faster than Python loops)
        extents = np.zeros(n_peaks, dtype=np.int32)

        for j in range(n_peaks):
            peak_idx = peaks[j]
            threshold = thresholds[j]

            # Create boolean mask: True where trace >= threshold
            above = trace_norm >= threshold

            # Left extent: find last index below threshold, left of peak
            if peak_idx > 0:
                below_left = np.where(~above[:peak_idx])[0]
                if len(below_left) > 0:
                    left_count = peak_idx - below_left[-1] - 1
                else:
                    left_count = peak_idx
            else:
                left_count = 0

            # Right extent: find first index below threshold, right of peak
            if peak_idx < n_frames - 1:
                below_right = np.where(~above[peak_idx + 1:])[0]
                if len(below_right) > 0:
                    right_count = below_right[0]
                else:
                    right_count = n_frames - peak_idx - 1
            else:
                right_count = 0

            extents[j] = 1 + left_count + right_count

        # Convert mean frames to seconds (FPS-independent)
        mean_times_at_peak[i] = np.mean(extents) / fps

    return mean_times_at_peak


def multisession_corrmat(neurons, corr_threshold, match_threshold, fps=30, sessions_num=5, correlation_method='pearson'):
    match_threshold /= sessions_num
    corr_num = len(neurons)
    corr_mtx_sessions = []

    session_time = neurons.shape[1]//sessions_num
    for session in range(sessions_num):
        ts_start = session * session_time

        corr_mtx = compute_correlation_matrix(neurons[:, ts_start:ts_start + session_time - 1], method=correlation_method)
        corr_mtx = np.where(corr_mtx >= corr_threshold, 1, 0)
        corr_mtx_sessions.append(corr_mtx)

    corr_mtx_sessions = np.array(corr_mtx_sessions)
    match_mtx = np.sum(corr_mtx_sessions, axis=0) / sessions_num
    match_mtx_crop = np.where(match_mtx >= match_threshold, match_mtx, 0)

    CM = match_mtx_crop
    np.fill_diagonal(CM, 0)
    CM[np.isnan(CM)] = 0

    TCM = CM.copy()
    TCM[np.where(TCM < match_threshold)] = 0

    nontrivial_ccs = [comp for comp in list(get_ccs_from_adj(TCM)) if len(comp) > 1]

    group_corr_scores = np.zeros(len(nontrivial_ccs))
    corr_scores = np.zeros(corr_num)
    corr_groups = np.zeros(corr_num)
    for i, group in enumerate(nontrivial_ccs):
        ordered = np.array(sorted(list(group)))
        subnetwork = CM[ordered, :][:, ordered]  # we take corr values from initial corr matrix
        nc = len(ordered)
        group_density = np.sum(subnetwork) / (nc ** 2 - nc)
        group_corr_scores[i] = group_density
        # group_av_nnz = np.mean(subnetwork[np.where(subnetwork != 0)])

    sorted_nontrivial_ccs = [nontrivial_ccs[i] for i in
                             np.argsort(group_corr_scores)[::-1]]  # sort components from highest to lowest score
    sorted_group_corr_scores = sorted(group_corr_scores)
    for i, group in enumerate(sorted_nontrivial_ccs):
        for neuron in group:
            corr_scores[neuron] = sorted_group_corr_scores[i]
            corr_groups[neuron] = len(sorted_group_corr_scores) - i + 1  # big group number = high corr score

    return corr_groups, match_mtx, match_mtx_crop


def estimates_to_metrics(est, fps, comps_to_select=[], cthr=0.3, contours=None,
                         corr_thr=0.6, num_sessions=1, match_threshold=3,
                         sf=None, ef=None, ds=1, include_event_based=True, include_heavy=False,
                         detect_corner_artifacts_flag=True, corner_artifact_params=None,
                         event_method='threshold', correlation_method='pearson', n_iter=2,
                         hybrid_kinetics=True):

    match_threshold = min(match_threshold, num_sessions)

    if len(comps_to_select) == 0:
        comps_to_select = est.idx_components

    n_cells = len(comps_to_select)
    if n_cells == 0:
        return pd.DataFrame(), None, np.array([]), np.array([]), None, {}

    print(f'[1/4] Preparing traces for {n_cells} neurons...')
    t_phase_start = time.time()

    if sf is None:
        sf = 0
    if ef is None:
        ef = est.C.shape[1]

    # Normalize traces, handling flat traces (where max == min)
    traces = []
    for i, tr in enumerate(est.C[comps_to_select, sf:ef][:, ::ds]):
        tr_min = np.min(tr)
        tr_max = np.max(tr)
        tr_range = tr_max - tr_min

        if tr_range == 0 or np.isclose(tr_range, 0, atol=1e-10):
            # Flat trace: set to zeros (will be handled as NaN in metrics)
            normalized = np.zeros_like(tr)
        else:
            normalized = (tr - tr_min) / tr_range

        traces.append(normalized)

    # times = [est.time[sf:ef][::ds] for _ in range(n_cells)]  # Note: time attribute not always present, variable unused

    print(f'[2/4] Computing correlation matrix ({correlation_method})...')
    if len(traces) > 1:
        corr_groups, match_mtx, match_mtx_crop = multisession_corrmat(np.array(traces),
                                                                      corr_thr,
                                                                      match_threshold,
                                                                      fps=fps,
                                                                      sessions_num=num_sessions,
                                                                      correlation_method=correlation_method)
    else:
        corr_groups = None
        match_mtx = None
        match_mtx_crop = None


    print(f'[3/4] Extracting spatial metrics...')
    if contours is None:
        contours = get_contours(est, comps_to_select, cthr=cthr)

    areas = []
    centers = []
    for i, comp in enumerate(comps_to_select):
        coords = contours[i]["coordinates"]
        area = calculate_polygon_area(coords)
        areas.append(area)
        centers.append(contours[i]["CoM"])

    # distance matrices
    FCD = footprint_center_distmat(centers)
    FBD = footprint_boundary_distmat(contours, mask=match_mtx_crop)

    #footprint metrics
    circularities = get_circularities(contours)
    max_edges = get_max_edges(contours)
    convexities = get_convexities(contours)
    aspect_ratios = get_aspect_ratios(contours)

    # Eccentricity from CaImAn - measures elongation of footprint
    try:
        eccentricities = compute_eccentricity(est.A[:, comps_to_select], est.imax.shape)
    except Exception:
        eccentricities = np.full(n_cells, np.nan)

    # Spatial metrics using helper functions
    edge_distances = get_edge_distances(centers, est.imax.shape)
    nn_distances_center = get_nn_distances(FCD)

    # Local density: count of neurons within 50 pixels (excluding self)
    LOCAL_DENSITY_RADIUS = 50  # pixels
    local_densities = np.sum(FCD < LOCAL_DENSITY_RADIUS, axis=1) - 1

    caiman_snrs = est.SNR_comp[comps_to_select]
    # Cap infinities at max valid (finite) value
    finite_mask = np.isfinite(caiman_snrs)
    if finite_mask.any() and (~finite_mask).any():
        max_finite = caiman_snrs[finite_mask].max()
        caiman_snrs = np.where(np.isinf(caiman_snrs), max_finite, caiman_snrs)
    caiman_r_scores = est.r_values[comps_to_select]

    # CaImAn estimates attributes
    noise_levels = est.neurons_sn[comps_to_select] if hasattr(est, 'neurons_sn') else np.full(n_cells, np.nan)
    baselines = est.bl[comps_to_select] if hasattr(est, 'bl') else np.full(n_cells, np.nan)
    tau_decays = get_tau_decays(est, comps_to_select, fps)

    # Trace statistics (skewness, kurtosis, bimodality)
    raw_traces = est.C[comps_to_select, sf:ef]
    trace_skewnesses, trace_kurtoses, trace_bimodalities = get_trace_stats(raw_traces)

    # Saturation metrics (FPS-independent, in seconds)
    mean_times_at_peak = get_saturation_metrics(raw_traces, fps)

    # Long-range dependence (Hurst exponent via R/S analysis)
    hurst_exponents = get_hurst_exponents(raw_traces)

    # Baseline stability (normalized linear trend)
    baseline_drifts = get_baseline_drifts(raw_traces)

    # Footprint compactness
    compactnesses = get_compactnesses(contours, areas)

    metrics = {
        'component_idx': comps_to_select,
        'area': areas,
        'circularity': circularities,
        'max_edge': max_edges,
        'convexity': convexities,
        'aspect_ratio': aspect_ratios,
        'eccentricity': eccentricities,
        'center': centers,
        'edge_distance': edge_distances,
        'nn_distance_center': nn_distances_center,
        'local_density': local_densities,
        'caiman_snr': caiman_snrs,
        'caiman_r_score': caiman_r_scores,
        'noise_level': noise_levels,
        'baseline': baselines,
        'tau_decay': tau_decays,
        'trace_skewness': trace_skewnesses,
        'trace_kurtosis': trace_kurtoses,
        'bimodality': trace_bimodalities,
        'mean_time_at_peak': mean_times_at_peak,
        'hurst_exponent': hurst_exponents,
        'baseline_drift': baseline_drifts,
        'footprint_compactness': compactnesses,
        'corr_groups': corr_groups
    }

    reconstructions = {}
    if include_event_based:
        print(f'[4/4] Computing {event_method} event-based metrics (this may take a while)...')
        t1 = time.time()
        event_based_metrics, local_reconstructions = get_multineuron_metrics(np.array(traces),
                                                      fps=fps,
                                                      include_heavy=include_heavy,
                                                      event_method=event_method,
                                                      n_iter=n_iter,
                                                      hybrid_kinetics=hybrid_kinetics)
        t2 = time.time()
        etime = np.round(t2-t1, 2)
        print(f'      Event metrics completed in {etime}s ({np.round(etime/n_cells, 3)}s/neuron)')
        metrics = {**metrics, **event_based_metrics}

        # Map local indices to component indices for reconstructions
        if include_heavy and local_reconstructions:
            for local_idx, rec in local_reconstructions.items():
                comp_idx = comps_to_select[local_idx]
                reconstructions[comp_idx] = rec
            print(f'      Cached {len(reconstructions)} reconstructions')
    else:
        print(f'[4/4] Skipping event-based metrics (include_event_based=False)')

    metrics_df = pd.DataFrame(metrics)

    # Detect edge artifacts if enabled (combined corner + ellipse detection)
    edge_info = None
    if detect_corner_artifacts_flag:
        if corner_artifact_params is None:
            corner_artifact_params = {}

        try:
            metrics_df, edge_info, _ = detect_edge_artifacts(metrics_df, **corner_artifact_params)
            n_artifacts = (metrics_df['is_corner_artifact'] == 1).sum()
            if n_artifacts > 0:
                corner_only = edge_info.get('n_corner_only', 0)
                ellipse_only = edge_info.get('n_ellipse_only', 0)
                both = edge_info.get('n_both', 0)
                print(f'Edge artifact detection: {n_artifacts} artifacts ({n_artifacts/len(metrics_df)*100:.1f}%) '
                      f'[corner:{corner_only}, ellipse:{ellipse_only}, both:{both}]')
        except Exception as e:
            print(f'Warning: Edge artifact detection failed: {e}')
            metrics_df['is_corner_artifact'] = 0
            edge_info = None

    t_total = time.time() - t_phase_start
    print(f'Metrics extraction completed: {n_cells} neurons in {t_total:.1f}s')

    return metrics_df, match_mtx, FCD, FBD, edge_info, reconstructions


def area_check(series, pxlthr_area):
    metric = (series.area > pxlthr_area)
    return metric


def circularity_check(series, circ_thr):
    metric = (series.circularity <= circ_thr)
    return metric


def max_edge_check(series, maxedge_thr):
    metric = (series.max_edge <= maxedge_thr)
    return metric


def convexity_check(series, convex_thr):
    metric = (series.convexity <= convex_thr)
    return metric


def t_rise_check(series, t_rise_min):
    """Check if rise time is above minimum threshold."""
    if pd.isna(series.t_rise) or series.t_rise < 0:
        return False
    return series.t_rise >= t_rise_min


def caiman_r_score_check(series, r_score_min):
    """Check if CaImAn spatial correlation is above minimum threshold."""
    if pd.isna(series.caiman_r_score):
        return False
    return series.caiman_r_score >= r_score_min


def caiman_snr_check(series, snr_min):
    """Check if CaImAn SNR is above minimum threshold."""
    if pd.isna(series.caiman_snr):
        return False
    return series.caiman_snr >= snr_min


def t_off_check(series, t_off_min):
    """Check if decay time is above minimum threshold."""
    if pd.isna(series.t_off) or series.t_off < 0:
        return False
    return series.t_off >= t_off_min


# =============================================================================
# RULE-BASED THRESHOLD SYSTEM
# =============================================================================

# Default deletion rules (BREAKING CHANGE: inverted area/circularity logic)
# Rules express DELETION conditions (e.g., "area<1" means "DELETE if area < 1")
DEFAULT_DELETION_RULES = [
    'area<1',          # DELETE if area < 1 pixel (reject tiny footprints)
    'circularity>4',   # DELETE if circularity > 4 (reject non-circular)
]


def parse_rule(rule_str):
    """
    Parse a threshold rule string into components.

    Args:
        rule_str: Rule string like "area>6.9" or "circularity<=4"

    Returns:
        tuple: (metric_name, operator, threshold_value)

    Raises:
        ValueError: If rule format invalid or metric not in ML_FEATURE_COLS

    Examples:
        >>> parse_rule("area>6.9")
        ('area', '>', 6.9)
        >>> parse_rule(" circularity <= 4 ")
        ('circularity', '<=', 4.0)
    """
    if not rule_str or not isinstance(rule_str, str):
        raise ValueError(f"Invalid rule: expected non-empty string, got {type(rule_str)}")

    # Strip whitespace
    rule_str = rule_str.strip()

    if not rule_str:
        raise ValueError("Empty rule string after stripping whitespace")

    # Try operators in order (longer first to avoid greedy matching)
    operators = ['>=', '<=', '>', '<']

    for op in operators:
        if op in rule_str:
            parts = rule_str.split(op, 1)
            if len(parts) != 2:
                raise ValueError(f"Invalid rule format: '{rule_str}'. Expected 'metric{op}threshold'")

            metric_name = parts[0].strip()
            threshold_str = parts[1].strip()

            # Validate metric name
            if metric_name not in ML_FEATURE_COLS:
                raise ValueError(
                    f"Unknown metric '{metric_name}' in rule '{rule_str}'. "
                    f"Must be one of: {', '.join(sorted(ML_FEATURE_COLS))}"
                )

            # Parse threshold as float
            try:
                threshold_value = float(threshold_str)
            except ValueError:
                raise ValueError(
                    f"Invalid threshold '{threshold_str}' in rule '{rule_str}'. "
                    f"Threshold must be numeric."
                )

            return (metric_name, op, threshold_value)

    # No operator found
    raise ValueError(
        f"Invalid rule '{rule_str}': no operator found. "
        f"Supported operators: >, <, >=, <="
    )


def evaluate_rule(series, metric_name, operator, threshold):
    """
    Evaluate a single threshold rule on a neuron's metrics.

    CRITICAL SEMANTIC: Rules express DELETION conditions, but this function
    returns PASS/FAIL status for KEEPING the neuron.

    - Return True if neuron PASSES (should be KEPT)
    - Return False if neuron FAILS (should be DELETED)

    Example: Rule "area<1" means "DELETE if area < 1"
    - If area=0.5: meets deletion condition → return False (FAIL - delete)
    - If area=2.0: doesn't meet deletion condition → return True (PASS - keep)

    Implementation uses INVERTED logic:
    - "area<1" → return value >= 1 (neuron passes if area NOT less than 1)
    - "circularity>4" → return value <= 4 (neuron passes if circularity NOT greater than 4)

    Args:
        series: pandas Series with neuron metrics
        metric_name: Name of metric column (e.g., 'area')
        operator: Comparison operator ('>', '<', '>=', '<=')
        threshold: Numeric threshold value

    Returns:
        bool: True if neuron passes (keep), False if fails (delete)
    """
    # Handle missing column
    if metric_name not in series.index:
        # Missing metric → fail (conservative)
        return False

    value = series[metric_name]

    # NaN/sentinel handling (matching current behavior)
    # Event-based metrics: -1 sentinel means "no events detected" (fail)
    if metric_name in ['t_rise', 't_off']:
        if pd.isna(value) or value < 0:
            return False

    # CaImAn metrics: NaN means computation failed (fail)
    elif metric_name in ['caiman_r_score', 'caiman_snr']:
        if pd.isna(value):
            return False

    # Other metrics: NaN handled by pandas comparison (returns False)
    # Infinity: let comparison handle it (may need future refinement)

    # INVERTED operator logic: rules express deletion conditions,
    # but we return True if neuron PASSES (i.e., doesn't meet deletion condition)
    if operator == '<':
        # Rule: "DELETE if value < threshold"
        # Pass if value >= threshold (NOT less than)
        return value >= threshold
    elif operator == '>':
        # Rule: "DELETE if value > threshold"
        # Pass if value <= threshold (NOT greater than)
        return value <= threshold
    elif operator == '<=':
        # Rule: "DELETE if value <= threshold"
        # Pass if value > threshold (NOT less-or-equal)
        return value > threshold
    elif operator == '>=':
        # Rule: "DELETE if value >= threshold"
        # Pass if value < threshold (NOT greater-or-equal)
        return value < threshold
    else:
        raise ValueError(f"Unsupported operator: '{operator}'")


def validate_rules(rules):
    """
    Validate a list of rule strings and parse them.

    Args:
        rules: List of rule strings

    Returns:
        list: List of parsed rules [(metric, op, threshold), ...]

    Raises:
        ValueError: If any rule is invalid (with all errors listed)
    """
    if not isinstance(rules, (list, tuple)):
        raise ValueError(f"Rules must be a list or tuple, got {type(rules)}")

    if len(rules) == 0:
        raise ValueError("Rules list is empty. Provide at least one rule.")

    parsed_rules = []
    errors = []

    for i, rule_str in enumerate(rules):
        try:
            parsed = parse_rule(rule_str)
            parsed_rules.append(parsed)
        except ValueError as e:
            errors.append(f"  Rule {i}: {e}")

    if errors:
        error_msg = "Invalid rules found:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    return parsed_rules


def get_active_metrics_from_rules(rules):
    """
    Extract unique metric names from a list of rules.

    Args:
        rules: List of rule strings (e.g., ['area>6.9', 'circularity<=4'])

    Returns:
        list: Unique metric names (e.g., ['area', 'circularity'])
    """
    parsed_rules = validate_rules(rules)
    metric_names = [metric for metric, op, threshold in parsed_rules]
    return list(dict.fromkeys(metric_names))  # Preserve order, remove duplicates


def _apply_threshold_brain(metrics_df, rules, track_failures=True):
    """
    Apply rule-based deletion logic to determine which neurons to delete.

    Uses string-based rules (e.g., 'area<1', 'circularity>4') to evaluate
    whether neurons should be kept or deleted. All rules must pass (AND logic)
    for a neuron to be kept.

    Args:
        metrics_df: DataFrame with neuron metrics (subset without corner artifacts)
        rules: List of rule strings (e.g., ['area<1', 'circularity>4'])
               Rules express DELETION conditions (e.g., "area<1" = DELETE if area < 1)
        track_failures: If True, track which rules failed for each neuron

    Returns:
        delete_mask: np.ndarray of bool, True = should delete
        failure_info: dict mapping 'failed_<metric>' to arrays (if track_failures)

    Examples:
        >>> rules = ['area<1', 'circularity>4', 't_rise<0.1']
        >>> delete_mask, failures = _apply_threshold_brain(df, rules)
        >>> # Neurons with area < 1 OR circularity > 4 OR t_rise < 0.1 are deleted
    """
    n = len(metrics_df)
    delete_mask = np.zeros(n, dtype=bool)

    # Parse and validate rules
    parsed_rules = validate_rules(rules)

    # Initialize failure tracking (dynamic based on active metrics)
    failure_info = {}
    if track_failures:
        active_metrics = get_active_metrics_from_rules(rules)
        for metric in active_metrics:
            failure_info[f'failed_{metric}'] = np.zeros(n, dtype=int)

    # Evaluate rules for each neuron
    for i, (idx, row) in enumerate(metrics_df.iterrows()):
        all_pass = True

        # Evaluate each rule
        for metric_name, operator, threshold in parsed_rules:
            rule_passes = evaluate_rule(row, metric_name, operator, threshold)

            if not rule_passes:
                all_pass = False
                # Track which rule failed
                if track_failures:
                    failure_info[f'failed_{metric_name}'][i] = 1

        # Delete if ANY rule fails (AND logic - all must pass to keep)
        delete_mask[i] = not all_pass

    return delete_mask, failure_info


def _apply_ml_brain(metrics_df, model_path, threshold=0.5, feature_cols=None):
    """
    Apply ML model-based deletion logic to determine which neurons to delete.

    Uses a trained classifier (e.g., EBM) to predict P(KEEP) for each neuron.
    Neurons with P(KEEP) < threshold are marked for deletion.

    Feature columns are determined in this priority:
    1. model.feature_names_in_ (from sklearn/EBM - most reliable)
    2. feature_cols parameter if provided
    3. ML_FEATURE_COLS fallback

    Args:
        metrics_df: DataFrame with neuron metrics (subset without corner artifacts)
        model_path: Path to pickled model file (required, raises ValueError if None)
        threshold: P(KEEP) below this value triggers deletion (default 0.5)
        feature_cols: Feature columns for model (default: auto-detect from model)

    Returns:
        delete_mask: np.ndarray of bool, True = should delete
        failure_info: dict with 'ml_keep_probability' -> array of P(KEEP) values

    Raises:
        ValueError: If model_path is None
        FileNotFoundError: If model file doesn't exist
    """
    if model_path is None:
        raise ValueError("brain='ml' requires ml_model_path to be specified")

    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"ML model file not found: {model_path}")

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Determine feature columns (priority: model's own list > parameter > fallback)
    if hasattr(model, 'feature_names_in_'):
        feature_cols = list(model.feature_names_in_)
    elif feature_cols is None:
        feature_cols = ML_FEATURE_COLS

    # Extract features
    available_cols = [c for c in feature_cols if c in metrics_df.columns]
    X = metrics_df[available_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    # Add missing columns as NaN
    for col in feature_cols:
        if col not in X.columns:
            X[col] = np.nan
    X = X[feature_cols]

    # Predict probabilities
    probabilities = model.predict_proba(X)[:, 1]  # P(KEEP) = class 1

    # Determine deletions
    delete_mask = probabilities < threshold

    # Return probabilities for tracking
    failure_info = {
        'ml_keep_probability': probabilities
    }

    return delete_mask, failure_info


def _apply_hybrid_brain(metrics_df, rules, model_path, ml_threshold=0.5,
                        feature_cols=None, track_failures=True):
    """
    Apply hybrid brain: rule-based thresholds first, then ML on survivors.

    Neurons that FAIL threshold rules are deleted immediately (ml_keep_probability=NaN).
    Neurons that PASS threshold rules are evaluated by ML model for final decision.

    Args:
        metrics_df: DataFrame with neuron metrics (subset without corner artifacts)
        rules: List of rule strings (e.g., ['area<1', 'circularity>4'])
        model_path: Path to pickled ML model file
        ml_threshold: P(KEEP) below this value triggers deletion (default 0.5)
        feature_cols: Feature columns for ML model (default: ML_FEATURE_COLS)
        track_failures: If True, track which criteria failed for each neuron

    Returns:
        delete_mask: np.ndarray of bool, True = should delete
        failure_info: dict with threshold failure columns + 'ml_keep_probability'
    """
    n = len(metrics_df)

    # Step 1: Apply threshold rules
    threshold_delete, threshold_failures = _apply_threshold_brain(
        metrics_df, rules, track_failures=track_failures
    )

    # Step 2: Initialize ML probabilities as NaN (threshold failures won't get ML evaluation)
    ml_probabilities = np.full(n, np.nan)

    # Step 3: Apply ML to survivors (neurons that passed all thresholds)
    survivors_mask = ~threshold_delete

    if survivors_mask.any():
        survivor_df = metrics_df.iloc[survivors_mask.nonzero()[0]]
        ml_delete, ml_info = _apply_ml_brain(
            survivor_df, model_path, threshold=ml_threshold, feature_cols=feature_cols
        )
        # Map ML results back to full arrays
        survivor_indices = survivors_mask.nonzero()[0]
        ml_probabilities[survivor_indices] = ml_info['ml_keep_probability']

        # Combine: deleted by threshold OR deleted by ML
        for i, orig_idx in enumerate(survivor_indices):
            if ml_delete[i]:
                threshold_delete[orig_idx] = True

    # Combine failure info
    failure_info = threshold_failures.copy()
    failure_info['ml_keep_probability'] = ml_probabilities

    return threshold_delete, failure_info


def metrics_to_decision(metrics_df, match_mtx, FCD, FBD,
                        deletion_rules=None,
                        pxlthr_distance_boundary=5,
                        d_snr_thr=10,
                        enable_merge=True,
                        track_criteria_failures=True,
                        brain='thresholds',
                        ml_model_path=None,
                        ml_threshold=0.5) -> pd.DataFrame:
    """
    Classify neurons for merge/delete/keep decisions using rule-based or ML brains.

    Supports three 'brain' types for deletion decisions:
    - 'thresholds': Rule-based threshold logic (default)
    - 'ml': Machine learning model-based decisions
    - 'hybrid': Thresholds first, then ML on survivors

    Merge logic (enable_merge) is independent of deletion decisions.

    Args:
        metrics_df: DataFrame with neuron metrics
        match_mtx: Correlation match matrix
        FCD: Footprint Center Distance matrix
        FBD: Footprint Boundary Distance matrix
        deletion_rules: List of rule strings for threshold/hybrid brain
                        (e.g., ['area<1', 'circularity>4'])
                        If None, uses DEFAULT_DELETION_RULES
        pxlthr_distance_boundary: Distance threshold for merge detection (pixels)
        d_snr_thr: SNR difference threshold for merge detection
        enable_merge: Enable correlation-based merge detection (default: True)
        track_criteria_failures: Track which criteria failed for each neuron
        brain: Decision brain type - 'thresholds', 'ml', or 'hybrid'
        ml_model_path: Path to ML model pickle (required if brain='ml' or 'hybrid')
        ml_threshold: P(KEEP) threshold for ML brain (default 0.5)

    Returns:
        metrics_df with 'delete' and 'merge' columns added

    Examples:
        >>> # Use default rules
        >>> df = metrics_to_decision(metrics_df, match_mtx, FCD, FBD)

        >>> # Custom rules
        >>> custom_rules = ['area<2', 'circularity>3', 't_rise<0.15']
        >>> df = metrics_to_decision(metrics_df, match_mtx, FCD, FBD,
        ...                           deletion_rules=custom_rules)

        >>> # ML brain
        >>> df = metrics_to_decision(metrics_df, match_mtx, FCD, FBD,
        ...                           brain='ml', ml_model_path='model.pkl')
    """
    # Use default rules if none provided
    if deletion_rules is None:
        deletion_rules = DEFAULT_DELETION_RULES
    series_num = metrics_df.shape[0]
    metrics_df[['delete', 'merge']] = 0

    # Initialize failure tracking columns (dynamic based on rules)
    if track_criteria_failures:
        # Common column
        metrics_df['failed_corner_artifact'] = 0

        if brain == 'thresholds':
            # Initialize columns for active metrics in rules
            active_metrics = get_active_metrics_from_rules(deletion_rules)
            for metric in active_metrics:
                metrics_df[f'failed_{metric}'] = 0

        elif brain == 'ml':
            metrics_df['ml_keep_probability'] = np.nan

        elif brain == 'hybrid':
            # Hybrid needs both threshold failure columns AND ML probability
            active_metrics = get_active_metrics_from_rules(deletion_rules)
            for metric in active_metrics:
                metrics_df[f'failed_{metric}'] = 0
            metrics_df['ml_keep_probability'] = np.nan

    # Step 1: Handle corner artifacts (always, regardless of brain)
    has_corner_artifacts = 'is_corner_artifact' in metrics_df.columns
    if has_corner_artifacts:
        corner_mask = metrics_df['is_corner_artifact'] == 1
        metrics_df.loc[corner_mask, 'delete'] = 1
        if track_criteria_failures:
            metrics_df.loc[corner_mask, 'failed_corner_artifact'] = 1

    # Step 2: Apply brain for deletion decisions (only on non-corner neurons)
    if has_corner_artifacts:
        non_corner_mask = metrics_df['is_corner_artifact'] == 0
    else:
        non_corner_mask = pd.Series([True] * series_num, index=metrics_df.index)

    non_corner_indices = metrics_df[non_corner_mask].index

    if len(non_corner_indices) > 0:
        if brain == 'thresholds':
            # Apply rule-based threshold brain
            delete_mask, failure_info = _apply_threshold_brain(
                metrics_df.loc[non_corner_indices], deletion_rules, track_criteria_failures)

            # Update delete column
            metrics_df.loc[non_corner_indices, 'delete'] = delete_mask.astype(int)

            # Update failure columns
            if track_criteria_failures:
                for col, values in failure_info.items():
                    metrics_df.loc[non_corner_indices, col] = values

        elif brain == 'ml':
            delete_mask, failure_info = _apply_ml_brain(
                metrics_df.loc[non_corner_indices], ml_model_path, ml_threshold)

            # Update delete column
            metrics_df.loc[non_corner_indices, 'delete'] = delete_mask.astype(int)

            # Update probability column for tracking
            if track_criteria_failures:
                metrics_df.loc[non_corner_indices, 'ml_keep_probability'] = failure_info['ml_keep_probability']

        elif brain == 'hybrid':
            # Apply hybrid brain (rules + ML)
            delete_mask, failure_info = _apply_hybrid_brain(
                metrics_df.loc[non_corner_indices], deletion_rules,
                ml_model_path, ml_threshold, track_failures=track_criteria_failures)

            # Update delete column
            metrics_df.loc[non_corner_indices, 'delete'] = delete_mask.astype(int)

            # Update all failure columns (thresholds + ML probability)
            if track_criteria_failures:
                for col, values in failure_info.items():
                    metrics_df.loc[non_corner_indices, col] = values

        else:
            raise ValueError(f"Unknown brain type: {brain}. Supported: 'thresholds', 'ml', 'hybrid'")

    # Step 3: Correlation-based merge logic (UNCHANGED)
    if enable_merge:
        # unique number of clusters
        unique_clusters = metrics_df.loc[metrics_df['corr_groups'] != 0, 'corr_groups'].unique()
        unique_clusters = sorted(unique_clusters, reverse=True)

        metrics_df.sort_values('corr_groups', ascending=False, inplace=True)
        for cluster_id in unique_clusters:
            # indeces of neurons in cluster
            cluster = metrics_df[metrics_df['corr_groups'] == cluster_id]
            cluster_indices = cluster['component_idx'].tolist()

            for index_1 in cluster_indices:
                for index_2 in cluster_indices:
                    # address of current neuron
                    metrics_df_mask_1 = metrics_df['component_idx'] == index_1
                    metrics_df_mask_2 = metrics_df['component_idx'] == index_2

                    # execute only if not already classified to delete
                    if (index_1 != index_2 and metrics_df.loc[metrics_df_mask_1, 'delete'].item() == 0
                            and metrics_df.loc[metrics_df_mask_2, 'delete'].item() == 0):

                        # pandas df indeces matching with FBD matrix
                        FBD_idx_1 = metrics_df.loc[metrics_df_mask_1].index[0]
                        FBD_idx_2 = metrics_df.loc[metrics_df_mask_2].index[0]

                        # SNRs of current neurons
                        snr = [metrics_df.loc[metrics_df_mask_1, 'caiman_snr'].item(),
                               metrics_df.loc[metrics_df_mask_2, 'caiman_snr'].item()]

                        # check for merge/delete/keep choice depending on rules
                        if FBD[FBD_idx_1][FBD_idx_2] <= pxlthr_distance_boundary:
                            d_snr = max(snr) - min(snr)
                            # merge neurons with close snr value
                            if d_snr <= d_snr_thr:
                                metrics_df.loc[metrics_df_mask_1, 'merge'] = cluster_id

                            # delete all with smaller SNR
                            elif snr[0] != max(snr):
                                metrics_df.loc[metrics_df_mask_1, 'delete'] = 1
                        # else : default value is 0 in 'delete' if it has not been classified already

        # Clean up conflicts: if neuron is marked for deletion, remove from merge groups
        # Deletion is primary - no further actions allowed on deleted neurons
        deleted_neurons = metrics_df['delete'] == 1
        metrics_df.loc[deleted_neurons, 'merge'] = 0

    metrics_df.sort_index(inplace=True)
    return metrics_df


def metrics_to_dummy_decision(df):
    df = df.copy()
    # Add 'decision' column: 90% 'ok', 10% 'delete'
    np.random.seed(42)
    delete_mask = np.random.rand(len(df)) < 0.2
    df['decision'] = np.where(delete_mask, 'delete', 'ok')

    # Add 'merge' column: 90% are 0, 10% distributed into groups 1-10
    merge_mask = np.random.rand(len(df)) < 0.2
    df['merge'] = np.where(merge_mask, np.random.randint(1, 11, len(df)), 0)
    return df


def implement_decision(est, df, return_index_mapping=False):
    """
    Apply deletion and merge decisions to estimates.

    BEHAVIOR:
    - DELETIONS: ALL deleted neurons (ML-rejected, threshold-rejected, corner artifacts)
      remain VISIBLE in idx_components. Their deletion status is tracked in
      metrics_df['delete'] column for visualization purposes.
    - MERGES: Only merges affect idx_components structure. Merged source components
      are removed and replaced by the merged result.

    This design allows ExamineCells GUI to display all neurons with their
    ML probabilities and deletion reasons while still applying the merge logic.

    Args:
        est: CaImAn estimates object
        df: Decision DataFrame with 'component_idx', 'decision', 'merge' columns
        return_index_mapping: If True, return (est, mapping_info) where mapping_info
            contains old->new index mapping and merge group info

    Returns:
        If return_index_mapping=False: Modified estimates object
        If return_index_mapping=True: (estimates, mapping_info) where mapping_info is:
            {
                'old_to_new': {old_idx: new_idx, ...},  # For surviving components
                'merged_groups': [  # For each merge group
                    {
                        'new_idx': int,  # New index of merged component
                        'source_indices': [old_idx1, old_idx2, ...],  # Original sources
                        'primary_source': old_idx  # First source component
                    },
                    ...
                ],
                'deleted': [old_idx, ...]  # Deleted component indices
                'nr_before': int,  # C.shape[0] before merge
                'nr_after': int    # C.shape[0] after merge
            }
    """
    est = copy.deepcopy(est)

    # Track deleted components (using ORIGINAL indices)
    # ALL deleted neurons (ML-rejected, threshold-rejected, corner artifacts) stay VISIBLE
    # in idx_components - their deletion status is tracked in metrics_df['delete']
    all_deleted = df[df['decision'] == 'delete']
    deleted_indices = set(all_deleted['component_idx'].tolist())

    # CRITICAL: Capture original idx_components BEFORE any modifications
    # This tells us which components were accepted in the original estimates
    original_idx_components = set(est.idx_components.tolist())

    # NOTE: We do NOT update idx_components_bad - all deleted neurons stay visible
    # Their deletion status is tracked in metrics_df for visualization purposes

    # Collect ALL merge groups to apply in single batch
    all_merge_groups = []
    merge_group_info = []

    components_to_merge = df['merge'].values
    for group_id in np.unique(components_to_merge):
        if group_id != 0:  # 0 means no need to merge
            sel_comps = df[df['merge'] == group_id]['component_idx'].tolist()

            # Filter to components that were originally accepted and not deleted
            sel_comps_valid = [c for c in sel_comps
                              if c in original_idx_components and c not in deleted_indices]

            # Only perform merge if 2 or more valid components remain
            if len(sel_comps_valid) >= 2:
                all_merge_groups.append(sel_comps_valid)
                merge_group_info.append({
                    'group_id': group_id,
                    'source_indices': sel_comps_valid
                })

    # Capture state BEFORE merge for mapping
    nr_before = est.C.shape[0]

    # Apply all merges in single call (if any)
    if all_merge_groups:
        est.manual_merge(all_merge_groups, params=params.CNMFParams(params_dict=est.cnmf_dict))

    # Build mapping info following CaImAn's index transformation logic:
    # - good_neurons = indices NOT in any merge group
    # - mapping: good_neurons[i] -> i (sequential 0..N-1)
    # - merged components get indices N, N+1, ... at the end

    all_merged_sources = set()
    for group in all_merge_groups:
        all_merged_sources.update(group)

    # Good neurons are those NOT involved in any merge (relative to pre-merge indices)
    good_neurons = np.setdiff1d(list(range(nr_before)), list(all_merged_sources))

    # Build old->new mapping for surviving (non-merged) components
    old_to_new = {}
    for new_idx, old_idx in enumerate(sorted(good_neurons)):
        old_to_new[int(old_idx)] = int(new_idx)

    # Add mapping for merged components (they're at the end)
    for i, group_info in enumerate(merge_group_info):
        new_merged_idx = len(good_neurons) + i

        # Primary source is the first component in the merge group
        primary = group_info['source_indices'][0]

        group_info['new_idx'] = new_merged_idx
        group_info['primary_source'] = primary

    # CRITICAL FIX: Rebuild idx_components using NEW indices
    # After manual_merge, the A/C matrices have new indices (0 to nr_after-1)
    # ALL deleted neurons (ML-rejected, threshold-rejected, corner artifacts) stay VISIBLE
    # Only merge source components are excluded (they're replaced by merged result)
    new_idx_components = []
    for old_idx in range(nr_before):
        # Include ALL originally accepted components EXCEPT merged sources
        # Deleted neurons stay visible (marked in metrics_df['delete'])
        if (old_idx in original_idx_components and
            old_idx not in all_merged_sources and
            old_idx in old_to_new):
            new_idx_components.append(old_to_new[old_idx])

    # Add merged result indices (merged components are accepted)
    for group_info in merge_group_info:
        new_idx_components.append(group_info['new_idx'])

    # Update idx_components with properly remapped indices
    est.idx_components = np.array(sorted(new_idx_components))

    mapping_info = {
        'old_to_new': old_to_new,
        'merged_groups': merge_group_info,
        'deleted': list(deleted_indices),
        'nr_before': nr_before,
        'nr_after': est.C.shape[0]
    }

    if not return_index_mapping:
        return est

    return est, mapping_info


def transform_metrics_df_indices(df, mapping_info, est_processed, fps, cthr=0.3,
                                  include_event_based=True, event_method='threshold'):
    """
    Transform metrics_df indices to match post-merge estimates.

    After manual_merge(), the A/C matrices have NEW indices. This function:
    1. Updates component_idx for ALL components (including deleted) using old_to_new mapping
    2. Removes merged SOURCE components from DataFrame (they no longer exist in A/C)
    3. Adds new rows for merged RESULT components with freshly computed metrics

    CRITICAL: Deleted components (including corner artifacts) are KEPT in the DataFrame
    with their new indices. They still exist in A/C matrices, just moved to idx_components_bad.

    Args:
        df: Original decision DataFrame with 'component_idx', 'decision', 'merge' columns
        mapping_info: Dict from implement_decision with index mapping info
        est_processed: CaImAn estimates after implement_decision (with new indices)
        fps: Frames per second for temporal metrics
        cthr: Contour threshold for spatial metrics
        include_event_based: Whether to compute event-based metrics
        event_method: Method for event detection ('threshold' or 'cascade')

    Returns:
        Transformed DataFrame with updated component_idx values
    """
    old_to_new = mapping_info['old_to_new']
    merged_groups = mapping_info['merged_groups']

    # Collect all merged source indices (these no longer exist in A/C - removed by manual_merge)
    all_merged_sources = set()
    for group in merged_groups:
        all_merged_sources.update(group['source_indices'])

    # Build transformed rows for ALL components (including deleted, except merged sources)
    transformed_rows = []

    for _, row in df.iterrows():
        old_idx = int(row['component_idx'])

        # Skip merged source components - they no longer exist in A/C matrices
        if old_idx in all_merged_sources:
            continue

        # Map component to new index (deleted components are mapped too!)
        if old_idx in old_to_new:
            # Use to_dict() for reliable DataFrame construction (avoids Series dtype issues)
            new_row = row.to_dict()
            new_row['component_idx'] = old_to_new[old_idx]
            transformed_rows.append(new_row)

    # Create DataFrame from transformed components
    if transformed_rows:
        transformed_df = pd.DataFrame(transformed_rows)
        # Ensure ml_keep_probability is float dtype (convert None to NaN)
        if 'ml_keep_probability' in transformed_df.columns:
            transformed_df['ml_keep_probability'] = pd.to_numeric(
                transformed_df['ml_keep_probability'], errors='coerce')
    else:
        transformed_df = pd.DataFrame(columns=df.columns)

    # Add rows for merged result components with freshly computed metrics
    if merged_groups:
        # Get indices of merged components
        merged_indices = [g['new_idx'] for g in merged_groups]

        # Use estimates_to_metrics for proper metric computation
        # Returns tuple: (metrics_df, match_mtx, FCD, FBD, edge_info, reconstructions)
        merged_metrics_df, _, _, _, _, _ = estimates_to_metrics(
            est_processed,
            fps=fps,
            comps_to_select=merged_indices,
            cthr=cthr,
            include_event_based=include_event_based,
            event_method=event_method
        )

        # Set proper columns for merged components
        merged_metrics_df['decision'] = 'from_merge'
        merged_metrics_df['ml_keep_probability'] = np.nan  # ML was not applied to merged
        merged_metrics_df['delete'] = 0
        merged_metrics_df['merge'] = 0
        merged_metrics_df['is_corner_artifact'] = 0

        # Ensure failure tracking columns exist (set to 0 for merged components)
        failure_cols = ['failed_area', 'failed_circularity', 'failed_max_edge', 'failed_convexity',
                        'failed_t_rise', 'failed_r_score', 'failed_snr', 'failed_t_off', 'failed_corner_artifact']
        for col in failure_cols:
            if col in transformed_df.columns and col not in merged_metrics_df.columns:
                merged_metrics_df[col] = 0

        # Concatenate
        transformed_df = pd.concat([transformed_df, merged_metrics_df], ignore_index=True)

    # Reset index and sort by component_idx
    if len(transformed_df) > 0:
        transformed_df = transformed_df.sort_values('component_idx').reset_index(drop=True)

    return transformed_df


def save_processed_estimates(est, output_path, session_name=None, compress=False):
    """
    Save processed estimates to pickle file.

    IMPORTANT: This function preserves all attributes attached to the estimates object,
    including est.metrics_df (if set by run_auto_inspection). The cached metrics enable
    EstimatesToSrcFast and EstimatesToSrcFull to skip expensive recomputation on load.

    Args:
        est: CaImAn estimates object after implement_decision
             Should have est.metrics_df attached (DataFrame with computed metrics + ML probabilities)
        output_path: Directory or full path to save the file
        session_name: Optional session name for filename (if output_path is directory)
        compress: If True, apply lightweight compression before saving (removes bad components,
                  converts to float32, sparse S matrix). Default: False

    Returns:
        Path to saved file
    """
    import pickle
    from pathlib import Path

    # Apply compression if requested (before saving)
    if compress:
        from estimates_compression import compress_estimates_ultra_lightweight
        est, savings, total_saved = compress_estimates_ultra_lightweight(est)
        print(f"[save_processed_estimates] Compressed estimates, saved {total_saved:.1f} MB")

    output_path = Path(output_path)

    # Determine full file path
    if output_path.suffix == '.pickle':
        filepath = output_path
    else:
        if session_name:
            filename = f"{session_name}_processed.pickle"
        else:
            filename = "processed_estimates.pickle"
        output_path.mkdir(parents=True, exist_ok=True)
        filepath = output_path / filename

    # Verify metrics_df is present before saving
    has_metrics = hasattr(est, 'metrics_df') and est.metrics_df is not None
    if has_metrics:
        print(f"[save_processed_estimates] Saving estimates WITH cached metrics ({len(est.metrics_df)} rows)")
    else:
        print(f"[save_processed_estimates] WARNING: Saving estimates WITHOUT metrics_df (will recompute on load)")

    # Save estimates with pickle (preserves all attributes including metrics_df)
    with open(filepath, 'wb') as f:
        pickle.dump(est, f)

    return filepath


def validate_decision(est_init, est_gt, est, fps=20):
    df_init, _, _, _, _, _ = estimates_to_metrics(est_init, fps=fps, include_heavy=False)
    df_gt, _, _, _, _, _ = estimates_to_metrics(est_gt, fps=fps, include_heavy=False)
    df, _, _, _, _, _ = estimates_to_metrics(est, fps=fps, include_heavy=False)


import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def match_coordinates(df1, df2, max_distance=None):
    """
    Match coordinates between two dataframes using Hungarian algorithm.

    Args:
        df1, df2: DataFrames with 'center' column containing 2D coordinates
        max_distance: Maximum distance to consider a match (None = no limit)

    Returns:
        matched_pairs: list of (idx1, idx2) tuples
        unmatched_df1: indices from df1 with no match
        unmatched_df2: indices from df2 with no match
    """
    # Extract coordinates
    coords1 = np.array([c for c in df1['center']])
    coords2 = np.array([c for c in df2['center']])

    # Compute pairwise distances
    distances = cdist(coords1, coords2, metric='euclidean')

    # Hungarian algorithm for optimal matching
    row_ind, col_ind = linear_sum_assignment(distances)

    # Filter by max_distance if specified
    matched_pairs = []
    unmatched_df1 = set(range(len(df1)))
    unmatched_df2 = set(range(len(df2)))

    for i, j in zip(row_ind, col_ind):
        if max_distance is None or distances[i, j] <= max_distance:
            matched_pairs.append((i, j, distances[i, j]))
            unmatched_df1.discard(i)
            unmatched_df2.discard(j)

    return matched_pairs, list(unmatched_df1), list(unmatched_df2)


def compute_metrics(initial, ground_truth, auto, max_match_distance=3):
    """
    Compute comprehensive metrics comparing auto vs ground_truth with initial baseline.

    Args:
        initial: Baseline DataFrame
        ground_truth: Ground truth DataFrame
        auto: Automatically generated DataFrame
        max_match_distance: Max distance (pixels) to consider elements as matched

    Returns:
        dict with all metrics
    """
    # Match auto to ground_truth
    matched_auto_gt, unmatched_auto, unmatched_gt = match_coordinates(
        auto, ground_truth, max_match_distance
    )

    # Match initial to ground_truth (for baseline)
    matched_init_gt, unmatched_init, _ = match_coordinates(
        initial, ground_truth, max_match_distance
    )

    # === Metric 1: Detection Metrics (Precision/Recall) ===
    n_auto = len(auto)
    n_gt = len(ground_truth)
    n_matched = len(matched_auto_gt)

    precision = n_matched / n_auto if n_auto > 0 else 0
    recall = n_matched / n_gt if n_gt > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    false_positives = len(unmatched_auto)  # Elements in auto but not in ground_truth
    false_negatives = len(unmatched_gt)  # Elements in ground_truth but not in auto

    # === Metric 2: Positional Accuracy ===
    if matched_auto_gt:
        # Distance errors for matched pairs
        distances_auto = [dist for _, _, dist in matched_auto_gt]
        mean_error_auto = np.mean(distances_auto)
        median_error_auto = np.median(distances_auto)
        std_error_auto = np.std(distances_auto)
        max_error_auto = np.max(distances_auto)
    else:
        mean_error_auto = median_error_auto = std_error_auto = max_error_auto = float('nan')

    # === Metric 3: Baseline Comparison ===
    if matched_init_gt:
        distances_init = [dist for _, _, dist in matched_init_gt]
        mean_error_baseline = np.mean(distances_init)
        improvement = (mean_error_baseline - mean_error_auto) / mean_error_baseline if mean_error_baseline > 0 else 0
    else:
        mean_error_baseline = float('nan')
        improvement = float('nan')

    # === Metric 4: Stability (how much moved) ===
    # Match auto to initial to see how much each element moved
    matched_auto_init, _, _ = match_coordinates(auto, initial, max_match_distance)
    if matched_auto_init:
        movement_distances = [dist for _, _, dist in matched_auto_init]
        mean_movement = np.mean(movement_distances)
        median_movement = np.median(movement_distances)
    else:
      mean_movement = median_movement = float('nan')

    return {
      # Detection metrics
      'precision': precision,
      'recall': recall,
      'f1_score': f1_score,
      'false_positives': false_positives,
      'false_negatives': false_negatives,

      # Positional accuracy (auto vs ground_truth)
      'mean_error': mean_error_auto,
      'median_error': median_error_auto,
      'std_error': std_error_auto,
      'max_error': max_error_auto,

      # Baseline comparison
      'baseline_mean_error': mean_error_baseline,
      'improvement_vs_baseline': improvement,

      # Stability/movement
      'mean_movement_from_initial': mean_movement,
      'median_movement_from_initial': median_movement,

      # Raw counts
      'n_auto': n_auto,
      'n_ground_truth': n_gt,
      'n_matched': n_matched,
    }


def print_report(metrics):
    """Pretty print the metrics report."""
    print("="*60)
    print("EVALUATION REPORT")
    print("="*60)

    print("\n[DETECTION METRICS]")
    print(f"  Precision:        {metrics['precision']:.2%} ({metrics['n_matched']}/{metrics['n_auto']} detected correctly)")
    print(f"  Recall:           {metrics['recall']:.2%} ({metrics['n_matched']}/{metrics['n_ground_truth']} ground truth found)")
    print(f"  F1 Score:         {metrics['f1_score']:.2%}")
    print(f"  False Positives:  {metrics['false_positives']} (detected but shouldn't exist)")
    print(f"  False Negatives:  {metrics['false_negatives']} (missing from detection)")

    print("\n[POSITIONAL ACCURACY] (Auto vs Ground Truth)")
    print(f"  Mean Error:       {metrics['mean_error']:.2f} pixels")
    print(f"  Median Error:     {metrics['median_error']:.2f} pixels")
    print(f"  Std Deviation:    {metrics['std_error']:.2f} pixels")
    print(f"  Max Error:        {metrics['max_error']:.2f} pixels")

    print("\n[BASELINE COMPARISON] (vs Initial)")
    print(f"  Baseline Error:   {metrics['baseline_mean_error']:.2f} pixels")
    print(f"  Improvement:      {metrics['improvement_vs_baseline']:.2%}")

    print("\n[STABILITY] (Movement from Initial)")
    print(f"  Mean Movement:    {metrics['mean_movement_from_initial']:.2f} pixels")
    print(f"  Median Movement:  {metrics['median_movement_from_initial']:.2f} pixels")

    print("="*60)


  # Example usage:
if __name__ == "__main__":
    initial = pd.DataFrame({
      'center': [(10, 20), (50, 60), (100, 100), (150, 150)]
    })

    ground_truth = pd.DataFrame({
      'center': [(12, 22), (51, 61), (102, 103), (155, 152)]
    })

    auto = pd.DataFrame({
      'center': [(13, 21), (52, 62), (200, 200), (156, 153)]  # One FP, one FN
    })

    # Compute metrics
    metrics = compute_metrics(initial, ground_truth, auto, max_match_distance=50)

    # Print report
    print_report(metrics)