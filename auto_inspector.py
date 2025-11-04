import copy

from driada.experiment.wavelet_event_detection import extract_wvt_events, WVT_EVENT_DETECTION_PARAMS
from driada.experiment.neuron import Neuron, DEFAULT_T_RISE, DEFAULT_T_OFF, DEFAULT_FPS

from utils import *
import numpy as np
import pandas as pd
import warnings
import tqdm

import matplotlib.pyplot as plt

from caiman.source_extraction.cnmf import params
from caiman.components_evaluation import (
        evaluate_components_CNN, estimate_components_quality_auto,
        select_components_from_metrics, compute_eccentricity,
        compute_event_exceptionality)


from scipy.stats import median_abs_deviation
from scipy.spatial import distance_matrix
from joblib import Parallel, delayed
from polygon import (get_contours, get_circularities, convex_polygons_min_distance,
                     calculate_polygon_area, calculate_perimeter, get_max_edges, get_convexities)


def get_hvals(traces):
    hvals = []
    for tr in traces:
        med = np.median(tr)
        meddev = median_abs_deviation(tr)
        hval = np.round(1.0 * len(np.where(tr >= med + 4 * meddev)[0]) / len(tr), 4)
        hvals.append(hval)

    return hvals


def get_multineuron_reconstruction_quality_metrics(traces,
                                                    fps=DEFAULT_FPS):
    print('Computing reconstruction metrics...')
    rec_results = Parallel(n_jobs=-1)(
        delayed(get_reconstruction_quality_metrics)(traces[i], fps=fps)
                for i in range(traces.shape[0])
    )
    r2_scores, mae_values, rmse_values, snr_values = list(zip(*rec_results))
    return r2_scores, mae_values, rmse_values, snr_values


def get_reconstruction_quality_metrics(trace,
                                       fps=DEFAULT_FPS,
                                       return_reconstructed=False):
    # Get metrics for a single neuron

    # Create Neuron object
    neuron = Neuron(
        cell_id="",
        ca=trace,
        sp=None,
        fps=fps
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Reconstruct spikes (required before getting metrics)
        neuron.reconstruct_spikes(method="wavelet")

        # Get quality metrics
        r2_score = neuron.get_reconstruction_r2()
        mae_value = neuron.get_mae()
        rmse_value = neuron.get_noise_ampl()
        snr_value = neuron.get_snr_reconstruction()
        rec = Neuron.get_restored_calcium(neuron.asp.data, DEFAULT_T_RISE, neuron.t_off)

        if return_reconstructed:
            return r2_score, mae_value, rmse_value, snr_value, rec
        else:
            return r2_score, mae_value, rmse_value, snr_value


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


def estimates_to_metrics(est, fps, comps_to_select=[], cthr=0.3,
                         corr_thr=0.6, num_sessions=1, match_threshold=3,
                         sf=None, ef=None, ds=1, include_reconstruction=False):

    match_threshold = min(match_threshold, num_sessions)

    if len(comps_to_select) == 0:
        comps_to_select = est.idx_components

    n_cells = len(comps_to_select)
    if n_cells == 0:
        return {}

    if sf is None:
        sf = 0
    if ef is None:
        ef = est.C.shape[1]

    traces = [(tr - min(tr)) / (np.max(tr) - np.min(tr)) for i, tr in
              enumerate(est.C[comps_to_select, sf:ef][:, ::ds])]
    times = [est.time[sf:ef][::ds] for _ in range(n_cells)]

    corr_groups, match_mtx, match_mtx_crop = multisession_corrmat(np.array(traces), corr_thr, match_threshold,
                                                        fps=fps, sessions_num=num_sessions)

    contours = get_contours(est, comps_to_select, cthr=cthr)
    areas = []
    centers = []
    for i, comp in enumerate(comps_to_select):
        coors = contours[i]["coordinates"]
        area = calculate_polygon_area(coors)
        areas.append(area)
        centers.append(contours[i]["CoM"])

    # distance matrices
    FCD = footprint_center_distmat(centers)
    FBD = footprint_boundary_distmat(contours, mask=match_mtx_crop)

    #footprint metrics
    circularities = get_circularities(contours)
    max_edges = get_max_edges(contours)
    convexities = get_convexities(contours)

    caiman_snrs = est.SNR_comp[comps_to_select]
    caiman_r_scores = est.r_values[comps_to_select]

    metrics = {
        'component_idx': comps_to_select,
        'area': areas,
        'circularity': circularities,
        'max_edge': max_edges,
        'convexity': convexities,
        'center': centers,
        'caiman_snr': caiman_snrs,
        'caiman_r_score': caiman_r_scores,
        'corr_groups': corr_groups
    }

    if include_reconstruction:
        r2_scores, mae_values, rmse_values, snr_values = \
            get_multineuron_reconstruction_quality_metrics(np.array(traces), fps=DEFAULT_FPS)

        rec_metrics = {
            'r2': r2_scores,
            'mae': mae_values,
            'rmse': rmse_values,
            'snr_rec': snr_values
        }
        metrics = {**metrics, **rec_metrics}

    metrics_df = pd.DataFrame(metrics)
    return metrics_df, match_mtx, FCD, FBD


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


def metrics_to_decision(metrics_df, match_mtx, FCD, FBD,
                        circ_thr=4, maxedge_thr=42, convex_thr=42, pxlthr_area=3, pxlthr_distance_boundary=5,
                        d_snr_thr=42,
                        use_circularity_check=True, use_area_check=True, use_max_edge_check=True,
                        use_convexity_check=True, use_corr_check=True) -> pd.DataFrame:
    """
    Classify neurons for merge/delete/keep annotating
    merge groups by clusters' numbers in 'merge' column,
    to delete as 1 and keep as 0 in delete column
    :param metrics_df:
    :param match_mtx:
    :param FCD:
    :param FBD:
    :param circ_thr:
    :param maxedge_thr:
    :param convex_thr:
    :param pxlthr_area:
    :param pxlthr_distance_boundary:
    :param d_snr_thr:
    :param use_circularity_check:
    :param use_area_check:
    :param use_max_edge_check:
    :param use_convexity_check:
    :param use_corr_check:
    :return:
    """
    series_num = metrics_df.shape[0]
    metrics_df[['delete', 'merge']] = 0

    for string_num in range(series_num):
        string = metrics_df.iloc[string_num]

        ### parameters
        area = area_check(string, pxlthr_area) if use_area_check else True
        circle = circularity_check(string, circ_thr) if use_circularity_check else True
        max_edge = max_edge_check(string, maxedge_thr) if use_max_edge_check else True
        convex = convexity_check(string, convex_thr) if use_convexity_check else True

        delete = not (area and circle and max_edge and convex)
        ###

        metrics_df.iloc[string_num, metrics_df.columns.get_loc('delete')] = int(delete)

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
            sel_comps = df[df['merge'] == group_id]['component_idx']
            est.manual_merge([sel_comps], params=params.CNMFParams(params_dict=est.cnmf_dict))

    return est


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


def compute_metrics(initial, ground_truth, auto, max_match_distance=50):
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

    print("\n📊 DETECTION METRICS")
    print(f"  Precision:        {metrics['precision']:.2%} ({metrics['n_matched']}/{metrics['n_auto']} detected correctly)")
    print(f"  Recall:           {metrics['recall']:.2%} ({metrics['n_matched']}/{metrics['n_ground_truth']} ground truth found)")
    print(f"  F1 Score:         {metrics['f1_score']:.2%}")
    print(f"  False Positives:  {metrics['false_positives']} (detected but shouldn't exist)")
    print(f"  False Negatives:  {metrics['false_negatives']} (missing from detection)")

    print("\n📍 POSITIONAL ACCURACY (Auto vs Ground Truth)")
    print(f"  Mean Error:       {metrics['mean_error']:.2f} pixels")
    print(f"  Median Error:     {metrics['median_error']:.2f} pixels")
    print(f"  Std Deviation:    {metrics['std_error']:.2f} pixels")
    print(f"  Max Error:        {metrics['max_error']:.2f} pixels")

    print("\n📈 BASELINE COMPARISON (vs Initial)")
    print(f"  Baseline Error:   {metrics['baseline_mean_error']:.2f} pixels")
    print(f"  Improvement:      {metrics['improvement_vs_baseline']:.2%}")

    print("\n🔄 STABILITY (Movement from Initial)")
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