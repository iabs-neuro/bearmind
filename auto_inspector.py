from bm_examinator import LoadEstimates
from driada.experiment.wavelet_event_detection import extract_wvt_events, WVT_EVENT_DETECTION_PARAMS
from driada.experiment.neuron import Neuron, DEFAULT_T_RISE, DEFAULT_T_OFF, DEFAULT_FPS

from utils import *
import caiman as cm
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from caiman.components_evaluation import (
        evaluate_components_CNN, estimate_components_quality_auto,
        select_components_from_metrics, compute_eccentricity,
        compute_event_exceptionality)


from scipy.stats import median_abs_deviation


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


def get_spike_based_snr(times, traces, fps=DEFAULT_FPS, t_rise=DEFAULT_T_RISE, t_off=DEFAULT_T_OFF):
    st_ev_inds, end_ev_inds, all_ridges = extract_wvt_events(traces, wvt_kwargs=WVT_EVENT_DETECTION_PARAMS)
    for i, tr in enumerate(traces[:1]):
        event_starts = np.array(st_ev_inds[i]).astype(int)
        sp_ampl = np.zeros(len(tr))
        sp = np.zeros(len(tr))

        sp[event_starts] = 1
        sp_ampl[event_starts] = tr[event_starts]

        neuron = Neuron(i,
                        tr,
                        sp,
                        default_t_rise=t_rise,
                        default_t_off=t_off,
                        fps=fps,
                        fit_individual_t_off=False,
                        seed=None)

        restored = neuron.get_restored_calcium(sp_ampl, t_rise, t_off)
        snr = neuron.get_snr()
        mad = neuron.get_mad()
        err = neuron.ca_mse_error(t_off, tr, sp, t_rise)

        print(snr, mad, err)

    return snr, mad, err, restored, tr


def estimates_to_metrics(fname, fps, comps_to_select=[], cthr=0.3, corr_thr=0.6, sf=None, ef=None, ds=1):
    est = LoadEstimates(fname, default_fps=fps)
    print('loaded estimates')

    if len(comps_to_select) == 0:
        comps_to_select = est.idx_components

    n_cells = len(comps_to_select)
    if n_cells == 0:
        return {}

    if sf is None:
        sf = 0
    if ef is None:
        ef = est.C.shape[1]

    traces = [(tr - min(tr)) / (np.max(tr) - np.min(tr)) + i for i, tr in
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

    caiman_snrs = est.SNR_comp[comps_to_select]
    caiman_r_scores = est.r_values[comps_to_select]

    #snr, mad, err, restored, tr = get_spike_based_snr(times, traces, fps=fps)

    metrics = {
        'component_idx': comps_to_select,
        'area': areas,
        'center': centers,
        'caiman_snr': caiman_snrs,
        'caiman_r_score': caiman_r_scores
    }
    metrics_df = pd.DataFrame(metrics)
    return metrics_df


fname = "D://Projects//estim_data//NOF_H01_3D_CR_MC_4_1_0.85_estimates.pickle"
fps=20
df = estimates_to_metrics(fname, fps)
print(df)