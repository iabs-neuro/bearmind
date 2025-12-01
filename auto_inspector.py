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


from scipy.stats import median_abs_deviation, skew, kurtosis
from scipy.spatial import distance_matrix
from joblib import Parallel, delayed
from polygon import (get_contours, get_circularities, convex_polygons_min_distance,
                     calculate_polygon_area, calculate_perimeter, get_max_edges, get_convexities,
                     convex_hull, get_aspect_ratios)
from corner_artifacts import detect_edge_artifacts


# Feature columns expected by ML models (must match ml/data_utils.py FEATURE_COLS)
ML_FEATURE_COLS = [
    'area', 'circularity', 'max_edge', 'convexity', 'caiman_snr', 'caiman_r_score',
    'events_per_min', 'events_fraction', 't_rise', 't_off', 'wavelet_snr',
    'r2_score', 'event_r2_score', 'nmae', 'nrmse', 'snr_recon', 'noise_level',
    'baseline', 'tau_decay', 'trace_skewness', 'footprint_compactness',
    'trace_kurtosis', 'aspect_ratio', 'eccentricity', 'edge_distance', 'nn_distance_center'
]


def get_hvals(traces):
    # DEPRECATED: old attempt to quantify "spike vs baseline timing"
    hvals = []
    for tr in traces:
        med = np.median(tr)
        meddev = median_abs_deviation(tr)
        hval = np.round(1.0 * len(np.where(tr >= med + 4 * meddev)[0]) / len(tr), 4)
        hvals.append(hval)

    return hvals


def get_neuron_with_spikes(trace, fps=DEFAULT_FPS, lightweight=True):
    # Create Neuron object
    neuron = Neuron(
        cell_id="",
        ca=trace,
        sp=None,
        fps=fps
    )

    # Stage 1: Initial single-pass detection with default kinetics (conservative)
    # Using threshold method - simpler and faster than wavelet
    neuron.reconstruct_spikes(
        method='threshold',
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
    kinetics = neuron.get_kinetics(
        method='direct',  # Direct measurement from detected events
        use_cached=False,  # Force recomputation
        #update_reconstruction=not lightweight  # Re-run detection with optimized kinetics
        update_reconstruction=False  # Re-run detection with optimized kinetics
    )

    # Stage 3: optionally re-run detection with optimized kinetics
    if not lightweight:
        neuron.reconstruct_spikes(
            method='threshold',
            n_mad=4.0,  # Balanced threshold for noisy data
            min_duration_frames=2,  # Allow shorter events
            create_event_regions=True,
            iterative=True,
            n_iter=3,
            adaptive_thresholds=True
        )

    return neuron


def get_signal_metrics(neuron):
    # metrics that don't require precise reconstruction

    n_events = int(np.sum(neuron.asp.data > 0))
    duration_min = len(neuron.asp.data)//neuron.fps/60.0
    epm = n_events/duration_min
    events_dur = 1.0*np.sum(neuron.sp.data.astype(int))
    events_fraction = events_dur/len(neuron.sp.data)

    t_rise = neuron.t_rise/neuron.fps if not pd.isna(neuron.t_rise) else -1
    t_off = neuron.t_off/neuron.fps if not pd.isna(neuron.t_off) else -1

    try:
        wavelet_snr = neuron.get_wavelet_snr()
        # Log transform to handle extreme outliers (corrupted values can reach millions)
        # log1p handles 0 values gracefully: log1p(x) = log(1+x)
        if wavelet_snr > 0:
            wavelet_snr = np.log1p(wavelet_snr)
        else:
            wavelet_snr = 0.0
    except ValueError:
        wavelet_snr = -1

    sig_metrics = {
        'events_per_min': epm,
        'events_fraction': events_fraction,
        't_rise': t_rise,
        't_off': t_off,
        'wavelet_snr': wavelet_snr
    }

    return sig_metrics


def get_reconstruction_quality_metrics(neuron,
                                       return_reconstructed=False):

    # Get quality metrics
    r2_score = neuron.get_reconstruction_r2()
    event_r2_score = neuron.get_reconstruction_r2(event_only=True)
    nmae = neuron.get_nmae()
    nrmse = neuron.get_nrmse()
    snr_recon = neuron.get_snr_reconstruction()
    rec = neuron.reconstructed

    rec_metrics = {
            'r2_score': r2_score,
            'event_r2_score': event_r2_score,
            'nmae': nmae,
            'nrmse': nrmse,
            'snr_recon': snr_recon
        }

    if return_reconstructed:
        rec_metrics['reconstruction'] = rec

    return rec_metrics


def get_single_neuron_metrics(trace, fps=DEFAULT_FPS, include_heavy=False):
    """
    Extract metrics from a single neuron trace.

    Handles flat/zero traces by returning NaN for all metrics.
    """
    try:
        neuron = get_neuron_with_spikes(trace, fps=fps, lightweight=not include_heavy)
        signal_metrics = get_signal_metrics(neuron)
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
            'wavelet_snr': np.nan
        }

        if include_heavy:
            nan_rec_metrics = {
                'r2_score': np.nan,
                'event_r2_score': np.nan,
                'nmae': np.nan,
                'nrmse': np.nan,
                'snr_recon': np.nan
            }
            return {**nan_signal_metrics, **nan_rec_metrics}
        else:
            return nan_signal_metrics


def get_multineuron_metrics(traces, fps=DEFAULT_FPS, include_heavy=False):
    all_metrics = {}
    n = traces.shape[0]
    print('Computing wavelet-based events...')
    metrics_res = Parallel(n_jobs=-1)(
        delayed(get_single_neuron_metrics)(traces[i], fps=fps, include_heavy=include_heavy)
        for i in range(traces.shape[0])
    )

    for metric in metrics_res[0].keys():
        all_metrics[metric] = [metrics_res[i][metric] for i in range(n)]

    return all_metrics


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

    cont_distmat = np.zeros((n,n))
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


def get_tau_decays(est, comps_to_select):
    """
    Compute tau_decay from CaImAn autoregressive parameter g.

    tau = -1/log(g) in frames. Represents calcium indicator decay time.

    Args:
        est: CaImAn estimates object
        comps_to_select: List of component indices

    Returns:
        np.array of tau_decay values
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
        # Convert g to tau: tau = -1/log(g) in frames
        if not np.isnan(g_val) and 0 < g_val < 1:
            tau_decays.append(-1.0 / np.log(g_val))
        else:
            tau_decays.append(np.nan)
    return np.array(tau_decays)


def get_trace_stats(traces):
    """
    Compute trace statistics (skewness and kurtosis) for multiple traces.

    Real neurons have high positive skewness (baseline + rare spikes)
    and high kurtosis (heavy tails from spike events).

    Args:
        traces: 2D array of shape (n_cells, n_timepoints)

    Returns:
        Tuple of (skewnesses, kurtoses) as np.arrays
    """
    n_cells = traces.shape[0]
    skewnesses = []
    kurtoses = []

    for i in range(n_cells):
        trace = traces[i]
        if len(trace) > 3:  # stats need at least 3 points
            skewnesses.append(skew(trace))
            kurtoses.append(kurtosis(trace))
        else:
            skewnesses.append(np.nan)
            kurtoses.append(np.nan)

    return np.array(skewnesses), np.array(kurtoses)


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


def multisession_corrmat(neurons, corr_threshold, match_threshold, fps=30, sessions_num=5):
    match_threshold /= sessions_num
    corr_num = len(neurons)
    corr_mtx_sessions = []

    session_time = neurons.shape[1]//sessions_num
    for session in range(sessions_num):
        ts_start = session * session_time

        corr_mtx = np.corrcoef(neurons[:, ts_start:ts_start + session_time - 1])
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
                         sf=None, ef=None, ds=1, include_wavelet=True, include_heavy=False,
                         detect_corner_artifacts_flag=True, corner_artifact_params=None):

    match_threshold = min(match_threshold, num_sessions)

    if len(comps_to_select) == 0:
        comps_to_select = est.idx_components

    n_cells = len(comps_to_select)
    print(n_cells)
    if n_cells == 0:
        return {}

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

    corr_groups, match_mtx, match_mtx_crop = multisession_corrmat(np.array(traces),
                                                                  corr_thr,
                                                                  match_threshold,
                                                                  fps=fps,
                                                                  sessions_num=num_sessions)

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

    caiman_snrs = est.SNR_comp[comps_to_select]
    caiman_r_scores = est.r_values[comps_to_select]

    # CaImAn estimates attributes
    noise_levels = est.neurons_sn[comps_to_select] if hasattr(est, 'neurons_sn') else np.full(n_cells, np.nan)
    baselines = est.bl[comps_to_select] if hasattr(est, 'bl') else np.full(n_cells, np.nan)
    tau_decays = get_tau_decays(est, comps_to_select)

    # Trace statistics (skewness and kurtosis)
    raw_traces = est.C[comps_to_select, sf:ef]
    trace_skewnesses, trace_kurtoses = get_trace_stats(raw_traces)

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
        'caiman_snr': caiman_snrs,
        'caiman_r_score': caiman_r_scores,
        'noise_level': noise_levels,
        'baseline': baselines,
        'tau_decay': tau_decays,
        'trace_skewness': trace_skewnesses,
        'trace_kurtosis': trace_kurtoses,
        'footprint_compactness': compactnesses,
        'corr_groups': corr_groups
    }

    if include_wavelet:
        print('computing wavelet event reconstruction...')
        t1 = time.time()
        event_based_metrics = get_multineuron_metrics(np.array(traces),
                                                      fps=fps,
                                                      include_heavy=include_heavy)
        t2 = time.time()
        etime = np.round(t2-t1, 2)
        #print(f'Elapsed time for metrics: {etime} s, {np.round(etime/n_cells, 2)} s per neuron')
        metrics = {**metrics, **event_based_metrics}

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

    return metrics_df, match_mtx, FCD, FBD, edge_info


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


def _apply_threshold_brain(metrics_df, thresholds, use_checks, track_failures=True):
    """
    Apply threshold-based deletion logic to determine which neurons to delete.

    This is the original threshold-based decision logic, refactored into a standalone
    function to support pluggable 'brains' for deletion decisions.

    Args:
        metrics_df: DataFrame with neuron metrics (subset without corner artifacts)
        thresholds: dict with threshold values:
            - pxlthr_area, circ_thr, maxedge_thr, convex_thr
            - t_rise_min, caiman_r_score_min, caiman_snr_min, t_off_min
        use_checks: dict with boolean flags:
            - use_area_check, use_circularity_check, use_max_edge_check, use_convexity_check
            - use_t_rise_check, use_caiman_r_score_check, use_caiman_snr_check, use_t_off_check
        track_failures: If True, track which criteria failed for each neuron

    Returns:
        delete_mask: np.ndarray of bool, True = should delete
        failure_info: dict mapping failure column names to arrays (if track_failures)
    """
    n = len(metrics_df)
    delete_mask = np.zeros(n, dtype=bool)

    # Initialize failure tracking arrays
    failure_info = {}
    if track_failures:
        failure_cols = ['failed_area', 'failed_circularity', 'failed_max_edge', 'failed_convexity',
                        'failed_t_rise', 'failed_r_score', 'failed_snr', 'failed_t_off']
        for col in failure_cols:
            failure_info[col] = np.zeros(n, dtype=int)

    # Apply checks to each row
    for i, (idx, row) in enumerate(metrics_df.iterrows()):
        area_ok = area_check(row, thresholds['pxlthr_area']) if use_checks['use_area_check'] else True
        circle_ok = circularity_check(row, thresholds['circ_thr']) if use_checks['use_circularity_check'] else True
        max_edge_ok = max_edge_check(row, thresholds['maxedge_thr']) if use_checks['use_max_edge_check'] else True
        convex_ok = convexity_check(row, thresholds['convex_thr']) if use_checks['use_convexity_check'] else True
        t_rise_ok = t_rise_check(row, thresholds['t_rise_min']) if use_checks['use_t_rise_check'] else True
        r_score_ok = caiman_r_score_check(row, thresholds['caiman_r_score_min']) if use_checks['use_caiman_r_score_check'] else True
        snr_ok = caiman_snr_check(row, thresholds['caiman_snr_min']) if use_checks['use_caiman_snr_check'] else True
        t_off_ok = t_off_check(row, thresholds['t_off_min']) if use_checks['use_t_off_check'] else True

        should_delete = not (area_ok and circle_ok and max_edge_ok and convex_ok and
                             t_rise_ok and r_score_ok and snr_ok and t_off_ok)
        delete_mask[i] = should_delete

        # Track failures
        if track_failures and should_delete:
            if use_checks['use_area_check'] and not area_ok:
                failure_info['failed_area'][i] = 1
            if use_checks['use_circularity_check'] and not circle_ok:
                failure_info['failed_circularity'][i] = 1
            if use_checks['use_max_edge_check'] and not max_edge_ok:
                failure_info['failed_max_edge'][i] = 1
            if use_checks['use_convexity_check'] and not convex_ok:
                failure_info['failed_convexity'][i] = 1
            if use_checks['use_t_rise_check'] and not t_rise_ok:
                failure_info['failed_t_rise'][i] = 1
            if use_checks['use_caiman_r_score_check'] and not r_score_ok:
                failure_info['failed_r_score'][i] = 1
            if use_checks['use_caiman_snr_check'] and not snr_ok:
                failure_info['failed_snr'][i] = 1
            if use_checks['use_t_off_check'] and not t_off_ok:
                failure_info['failed_t_off'][i] = 1

    return delete_mask, failure_info


def _apply_ml_brain(metrics_df, model_path, threshold=0.5, feature_cols=None):
    """
    Apply ML model-based deletion logic to determine which neurons to delete.

    Uses a trained classifier (e.g., EBM) to predict P(KEEP) for each neuron.
    Neurons with P(KEEP) < threshold are marked for deletion.

    Args:
        metrics_df: DataFrame with neuron metrics (subset without corner artifacts)
        model_path: Path to pickled model file (required, raises ValueError if None)
        threshold: P(KEEP) below this value triggers deletion (default 0.5)
        feature_cols: Feature columns for model (default: ML_FEATURE_COLS)

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

    if feature_cols is None:
        feature_cols = ML_FEATURE_COLS

    # Load model
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

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


def metrics_to_decision(metrics_df, match_mtx, FCD, FBD,
                        circ_thr=4, maxedge_thr=42, convex_thr=42, pxlthr_area=6.9,
                        pxlthr_distance_boundary=5,
                        d_snr_thr=42,
                        t_rise_min=0.10, caiman_r_score_min=0.05,
                        caiman_snr_min=2.9, t_off_min=1.5,
                        use_circularity_check=True, use_area_check=True, use_max_edge_check=True,
                        use_convexity_check=True, use_corr_check=True,
                        use_t_rise_check=True, use_caiman_r_score_check=True,
                        use_caiman_snr_check=True, use_t_off_check=True,
                        track_criteria_failures=True,
                        brain='thresholds',
                        ml_model_path=None,
                        ml_threshold=0.5) -> pd.DataFrame:
    """
    Classify neurons for merge/delete/keep annotating.

    Supports two 'brain' types for deletion decisions:
    - 'thresholds': Original threshold-based logic (default)
    - 'ml': Machine learning model-based decisions

    Merge logic remains unchanged regardless of brain type.

    Args:
        metrics_df: DataFrame with neuron metrics
        match_mtx: Correlation match matrix
        FCD: Footprint Center Distance matrix
        FBD: Footprint Boundary Distance matrix
        circ_thr: Circularity threshold (for threshold brain)
        maxedge_thr: Max edge threshold (for threshold brain)
        convex_thr: Convexity threshold (for threshold brain)
        pxlthr_area: Area threshold in pixels (for threshold brain)
        pxlthr_distance_boundary: Distance threshold for merge detection
        d_snr_thr: SNR difference threshold for merge detection
        t_rise_min: Minimum rise time (for threshold brain)
        caiman_r_score_min: Minimum CaImAn r-score (for threshold brain)
        caiman_snr_min: Minimum CaImAn SNR (for threshold brain)
        t_off_min: Minimum decay time (for threshold brain)
        use_*_check: Boolean flags to enable/disable individual checks (threshold brain)
        use_corr_check: Enable correlation-based merge detection
        track_criteria_failures: Track which criteria failed for each neuron
        brain: Decision brain type - 'thresholds' or 'ml'
        ml_model_path: Path to ML model pickle (required if brain='ml')
        ml_threshold: P(KEEP) threshold for ML brain (default 0.5)

    Returns:
        metrics_df with 'delete' and 'merge' columns added
    """
    series_num = metrics_df.shape[0]
    metrics_df[['delete', 'merge']] = 0

    # Initialize failure tracking columns
    if track_criteria_failures:
        # Common column
        metrics_df['failed_corner_artifact'] = 0

        if brain == 'thresholds':
            failure_cols = ['failed_area', 'failed_circularity', 'failed_max_edge', 'failed_convexity',
                           'failed_t_rise', 'failed_r_score', 'failed_snr', 'failed_t_off']
            for col in failure_cols:
                metrics_df[col] = 0
        elif brain == 'ml':
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
            # Build threshold and use_checks dicts for the brain function
            thresholds = {
                'pxlthr_area': pxlthr_area,
                'circ_thr': circ_thr,
                'maxedge_thr': maxedge_thr,
                'convex_thr': convex_thr,
                't_rise_min': t_rise_min,
                'caiman_r_score_min': caiman_r_score_min,
                'caiman_snr_min': caiman_snr_min,
                't_off_min': t_off_min
            }
            use_checks = {
                'use_area_check': use_area_check,
                'use_circularity_check': use_circularity_check,
                'use_max_edge_check': use_max_edge_check,
                'use_convexity_check': use_convexity_check,
                'use_t_rise_check': use_t_rise_check,
                'use_caiman_r_score_check': use_caiman_r_score_check,
                'use_caiman_snr_check': use_caiman_snr_check,
                'use_t_off_check': use_t_off_check
            }

            delete_mask, failure_info = _apply_threshold_brain(
                metrics_df.loc[non_corner_indices], thresholds, use_checks, track_criteria_failures)

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

        else:
            raise ValueError(f"Unknown brain type: {brain}. Supported: 'thresholds', 'ml'")

    # Step 3: Correlation-based merge logic (UNCHANGED)
    if use_corr_check:
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


def implement_decision(est, df):
    # deletion
    est = copy.deepcopy(est)
    components_to_del = df[df['decision'] == 'delete']['component_idx']
    temp = est.idx_components_bad.tolist() + components_to_del.tolist()
    est.idx_components_bad = np.sort(temp)
    # print('all bad comps', len(temp))
    est.idx_components = [_ for _ in est.idx_components if _ not in components_to_del]
    components_to_merge = df['merge'].values
    for group_id in np.unique(components_to_merge):
        if group_id != 0:  # 0 means no need to merge
            sel_comps = df[df['merge'] == group_id]['component_idx'].tolist()

            # Filter out components that were already deleted
            # This handles edge cases where merge groups lost members during deletion
            sel_comps_valid = [c for c in sel_comps if c in est.idx_components]

            # Only perform merge if 2 or more valid components remain
            # Single-component groups don't need merging
            if len(sel_comps_valid) >= 2:
                est.manual_merge([sel_comps_valid], params=params.CNMFParams(params_dict=est.cnmf_dict))

    return est


def save_processed_estimates(est, output_path, session_name=None):
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

    Returns:
        Path to saved file
    """
    import pickle
    from pathlib import Path

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
    df_init = estimates_to_metrics(est_init, fps=fps, include_reconstruction=False)
    df_gt = estimates_to_metrics(est_gt, fps=fps, include_reconstruction=False)
    df = estimates_to_metrics(est, fps=fps, include_reconstruction=False)


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