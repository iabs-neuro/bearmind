
from driada.experiment.wavelet_event_detection import extract_wvt_events, WVT_EVENT_DETECTION_PARAMS
from driada.experiment.neuron import Neuron, DEFAULT_T_RISE, DEFAULT_T_OFF, DEFAULT_FPS

from utils import *
import caiman as cm
import numpy as np
import pandas as pd
import warnings

import matplotlib.pyplot as plt

from caiman.components_evaluation import (
        evaluate_components_CNN, estimate_components_quality_auto,
        select_components_from_metrics, compute_eccentricity,
        compute_event_exceptionality)


from scipy.stats import median_abs_deviation
from joblib import Parallel, delayed


def get_hvals(traces):
    hvals = []
    for tr in traces:
        med = np.median(tr)
        meddev = median_abs_deviation(tr)
        hval = np.round(1.0 * len(np.where(tr >= med + 4 * meddev)[0]) / len(tr), 4)
        hvals.append(hval)

    return hvals


def get_contours(est, comps_to_select, cthr=0.3):
    estimates_data = est.A[:, comps_to_select]
    contours = cm.utils.visualization.get_contours(estimates_data,
                                                   dims=est.imax.shape,
                                                   thr=cthr)
    return contours


def get_circularities(est, comps_to_select, cthr=0.3):
    contours = get_contours(est, comps_to_select, cthr=cthr)
    circularities = []

    for i in range(len(contours)):
        contour = contours[i]["coordinates"]

        area = calculate_polygon_area(contour)
        perimeter = calculate_perimeter(contour)
        circularity = perimeter**2 / (4 * np.pi * area)

        circularities.append(circularity)

    circularities = np.array(circularities)
    return circularities

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

def estimates_to_metrics(est, fps, comps_to_select=[], cthr=0.3, corr_thr=0.6, sf=None, ef=None, ds=1):
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

    contours = get_contours(est, comps_to_select, cthr=cthr)
    areas = []
    centers = []
    for i, comp in enumerate(comps_to_select):
        coors = contours[i]["coordinates"]
        area = calculate_polygon_area(coors)
        areas.append(area)
        centers.append(contours[i]["CoM"])

    circularities = get_circularities(est, comps_to_select, cthr=cthr)
    caiman_snrs = est.SNR_comp[comps_to_select]
    caiman_r_scores = est.r_values[comps_to_select]

    r2_scores, mae_values, rmse_values, snr_values = \
        get_multineuron_reconstruction_quality_metrics(np.array(traces), fps=DEFAULT_FPS)

    metrics = {
        'component_idx': comps_to_select,
        'area': areas,
        'circularity': circularities,
        'center': centers,
        'caiman_snr': caiman_snrs,
        'caiman_r_score': caiman_r_scores,
        'r2': r2_scores,
        'mae': mae_values,
        'rmse': rmse_values,
        'snr_rec': snr_values
    }

    metrics_df = pd.DataFrame(metrics)
    return metrics_df


def area_check(series, pxlthr_area):
    metric = (series.area > pxlthr_area)
    return metric


def circularity_check(series, circ_thr):
    metric = (series.circularity <= circ_thr)
    return metric


def final_check(df, circ_thr, pxlthr_area=3, pxlthr_distance=10):
    # corrss = df['corr'].values
    series_num = df.shape[0]
    df = df.assign(new_column=df['delete'] + df['merge'])

    for string_num in range(series_num):
        string = df.iloc[string_num]

        ### parameters
        area = area_check(string, pxlthr_area)
        circle = circularity_check(string, circ_thr)

        delete = not (area and circle)
        ###

        df.iloc[string_num].delete = int(delete)

    return df
