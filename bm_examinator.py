# Stuff needed for plotting and widget callbacks
import copy
import html
import logging
import traceback
from dataclasses import dataclass
from scipy.stats import median_abs_deviation, spearmanr

from functools import partial
import tifffile as tfl
import caiman as cm
import pandas as pd
import numpy as np
import pickle
import ipywidgets as ipw
from IPython.display import display
import os
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from bokeh.plotting import figure, show, output_notebook
from bokeh.document.document import Document
from bokeh.models import (LinearColorMapper, CDSView, ColumnDataSource, Plot, CustomJS, Button,
                          RadioButtonGroup, PointDrawTool, TapTool, LabelSet, Div, PreText, CheckboxGroup, Spacer,
                          CustomJSExpr, IndexFilter, Range1d)
from bokeh.core.properties import expr

from bokeh.layouts import column, row
from bokeh.events import Tap, RangesUpdate
from bokeh.io import push_notebook
from glob import glob
from caiman.source_extraction.cnmf import params
from caiman.components_evaluation import (
        evaluate_components_CNN, estimate_components_quality_auto,
        select_components_from_metrics, compute_eccentricity,
        compute_event_exceptionality)

import time
from scipy.ndimage import gaussian_filter
from scipy.io import savemat
from caiman.utils.visualization import inspect_correlation_pnr

from caiman.utils.visualization import nb_inspect_correlation_pnr, inspect_correlation_pnr
from config import (CONFIG, read_config, get_mouse_config_path_from_fname,
                    update_config, get_session_name_from_path, get_session_config_path)
from table_routines import *
from utils import *
from bm_batch_routines import extract_name_with_pattern
from naming import extract_session_id
from auto_inspector import estimates_to_metrics, save_processed_estimates
from polygon import get_contours
import matplotlib.colors as mcolors

AUTOINSPECTED_TEXT = "This file is already autoinspected; open 4_Estimates instead."


def colornum_Metro(num):
    # Returns color for each number as in Moscow Metro
    return {
        1: "red",
        2: "green",
        3: "mediumblue",
        4: "cyan",
        5: "sienna",
        6: "darkorange",
        7: "mediumvioletred",
        8: "gold",
        9: "magenta",
        0: "lawngreen"}.get(num % 10)


def get_ml_probability_color(prob, threshold=0.5, is_deleted=False):
    """
    Map ML keep probability to color gradient.

    Args:
        prob: P(KEEP) probability (0-1)
        threshold: Decision threshold (neurons below this were deleted)
        is_deleted: Whether neuron was deleted

    Returns:
        str: Hex color code
    """
    # Handle None, NaN
    if prob is None or (isinstance(prob, (float, np.floating)) and np.isnan(prob)):
        return '#808080'  # Grey for deleted or missing probability

    # Map probability to 0-1 range for colormap
    # Red (0.0) = at threshold, Green (1.0) = probability of 1.0
    if prob <= threshold:
        # Below threshold (shouldn't happen for kept neurons, but handle it)
        normalized = 0.0
    else:
        # Map [threshold, 1.0] to [0.0, 1.0]
        normalized = (prob - threshold) / (1.0 - threshold)

    # Use RdYlGn colormap (Red-Yellow-Green)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'RedYellowGreen',
        ['#d73027', '#fee08b', '#1a9850']  # Red, Yellow, Green
    )

    rgba = cmap(normalized)
    # Convert RGBA to hex
    return mcolors.to_hex(rgba)


def _same_session_folder(folder, session_prefix):
    # Prefix glob would match HOS_X_1D_2T for HOS_X_1D; require exact ID
    sid = extract_session_id(session_prefix)
    return not sid or extract_session_id(folder.name) == sid


def LoadEstimates(name, default_fps=20, time=None):
    """The estimates in the pickle `name`, with .name and .time (seconds per frame).

    time: the session's timestamps in seconds (a server takes them from the series registry); then
    get_timestamps is not called. Longer than the traces: cut to them, as get_timestamps cuts a file."""
    with open(name, "rb") as f:
        estimates = pickle.load(f, )
    estimates.name = name
    '''
    if not hasattr(estimates, 'imax'):  #temporal hack; normally, imax should be loaded from image simultaneously with estimates
        estimates.imax = LoadImaxFromResults(estimates.name.partition('estimates')[0] + 'results.pickle')
    '''
    n_frames = estimates.C.shape[1]
    if time is None:
        estimates.time = get_timestamps(extract_name_with_pattern(estimates.name),
                                        n_frames,
                                        default_fps=default_fps)
    else:
        time = np.asarray(time)
        if len(time) < n_frames:
            # the traces would be drawn against a shorter time axis
            raise ValueError(f"{name}: {len(time)} timestamps for {n_frames} frames")
        estimates.time = time[:n_frames] if len(time) > n_frames else time

    return estimates


def get_timestamps(name, n_frames, default_fps=20):
    root = CONFIG['ROOT']
    pathway = CONFIG['DATA_PATHWAY']

    # try to load timestamps, in case of failure use constant fps
    print(name)
    ts_files = glob(name + '*timestamp.csv')

    print(ts_files)
    if len(ts_files) == 0:
        # raise FileNotFoundError(f'No timestamp files found for {name}, default fps has been disabled')
        return np.linspace(0, n_frames // default_fps, n_frames)
    else:
        ts_df = pd.read_csv(ts_files[0])
        time_col = find_time_column(ts_df)

        if pathway == 'legacy':
            timeline = ts_df[time_col].values / 1000
        elif pathway == 'bonsai':
            timeline = (ts_df[time_col].values - ts_df[time_col].values[0]) / 10000000
        else:
            raise ValueError('Wrong pathway!')

        return timeline[:n_frames]


def get_fps_from_timestamps(name, default_fps=20, verbose=True):

    ts_files = glob(name + '*.csv')
    if len(ts_files) == 0:
        if verbose:
            print('no timestamps found, reverting to default fps')
        return default_fps
    else:
        print('timestamp found: ', ts_files[0])
        ts_df = pd.read_csv(ts_files[0])
        fps = get_fps(ts_df, verbose=verbose)
        return fps


def _minmax_positions(Y, n_out):
    """Bins of the rows of Y (k, n) for keeping at most n_out points, two per bin: (bin starts (1, nb), position
    of the earlier and of the later of each bin's min and max (k, nb), bin length).

    The last bin is padded with the row's last value; argmin and argmax return the first occurrence, so a
    position never points into the padding."""
    k, n = Y.shape
    b = -(-n // max(n_out // 2, 1))
    nb = -(-n // b)
    pad = nb * b - n
    Z = np.concatenate([Y, np.repeat(Y[:, -1:], pad, axis=1)], axis=1) if pad else Y
    Z = Z.reshape(k, nb, b)
    amin, amax = Z.argmin(axis=2), Z.argmax(axis=2)
    starts = (np.arange(nb) * b)[None, :]
    return starts, starts + np.minimum(amin, amax), starts + np.maximum(amin, amax), b


def decimate_minmax(y: np.ndarray, n_out: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    """(frame index of every kept point, values): the min and the max of each bin, in time order; at most
    n_out points; y not longer than n_out is returned whole."""
    y = np.asarray(y)
    if len(y) <= n_out:
        return np.arange(len(y)), y
    _, first, second, _ = _minmax_positions(y[None, :], n_out)
    frames = np.stack([first[0], second[0]], axis=1).ravel()
    return frames, y[frames]


def _decimate_rows(Y, n_out):
    """decimate_minmax of every row of Y (k, n) on one shared time axis: (frames, values (k, <= n_out)).

    Each bin's two values are those decimate_minmax keeps, in the same order, but drawn at the bin's first and
    last frame rather than at their own: then all rows share one array of frames and the browser gets one time
    axis. The shift stays within a bin, under a pixel when the whole session is shown; a zoomed-in window is
    shown at full resolution."""
    Y = np.asarray(Y)
    n = Y.shape[1]
    if n <= n_out:
        return np.arange(n), Y
    starts, first, second, b = _minmax_positions(Y, n_out)
    rows = np.arange(Y.shape[0])[:, None]
    values = np.stack([Y[rows, first], Y[rows, second]], axis=2).reshape(Y.shape[0], -1)
    s = starts[0]
    return np.stack([s, np.minimum(s + b - 1, n - 1)], axis=1).ravel(), values


def _hval(tr):
    """Share of a trace's points at or above its median + 4 MAD (the H-val sort)."""
    med = np.median(tr)
    meddev = median_abs_deviation(tr)
    return np.round(1.0 * len(np.where(tr >= med + 4 * meddev)[0]) / len(tr), 4)


def _corr_matrix(trace_data, correlation_method):
    """Correlations between the rows of trace_data with the diagonal and NaN set to 0, as the Corr sort uses."""
    n = len(trace_data)
    if correlation_method == 'pearson':
        if n == 1:
            CM = np.array([[1.0]])
        else:
            CM = np.corrcoef(trace_data)
    elif correlation_method == 'spearman':
        if n == 1:
            CM = np.array([[1.0]])
        elif n == 2:
            # spearmanr returns scalar for 2 variables, wrap in 2x2 matrix
            corr_val, _ = spearmanr(trace_data, axis=1)
            CM = np.array([[1.0, corr_val], [corr_val, 1.0]])
        else:
            CM, _ = spearmanr(trace_data, axis=1)
    else:
        raise ValueError(f"Unknown correlation method: {correlation_method}. Use 'pearson' or 'spearman'")
    np.fill_diagonal(CM, 0)
    CM[np.isnan(CM)] = 0
    return CM


def _corr_groups(CM, corr_thr):
    """(corr_scores, corr_groups) per row of a _corr_matrix: groups of rows correlated above corr_thr."""
    TCM = CM.copy()
    TCM[np.where(TCM < corr_thr)] = 0

    nontrivial_ccs = [comp for comp in list(get_ccs_from_adj(TCM)) if len(comp) > 1]

    group_corr_scores = np.zeros(len(nontrivial_ccs))
    corr_scores = np.zeros(len(CM))
    corr_groups = np.zeros(len(CM))
    for i, group in enumerate(nontrivial_ccs):
        ordered = np.array(sorted(list(group)))
        subnetwork = CM[ordered, :][:, ordered]  # we take corr values from initial corr matrix
        nc = len(ordered)
        group_density = np.sum(subnetwork) / (nc ** 2 - nc)
        group_corr_scores[i] = group_density

    sorted_nontrivial_ccs = [nontrivial_ccs[i] for i in np.argsort(group_corr_scores)[::-1]] # sort components from highest to lowest score
    sorted_group_corr_scores = sorted(group_corr_scores)
    for i, group in enumerate(sorted_nontrivial_ccs):
        for neuron in group:
            corr_scores[neuron] = sorted_group_corr_scores[i]
            corr_groups[neuron] = len(sorted_group_corr_scores) - i + 1 # big group number = high corr score
    return corr_scores, corr_groups


def EstimatesToSrc(estimates, comps_to_select=[], cthr=0.3):
    n_cells = len(estimates.idx_components)
    if n_cells == 0:
        return {}
    traces = [tr / np.max(tr) + i for i, tr in enumerate(estimates.C[estimates.idx_components])]
    times = [estimates.time for _ in range(n_cells)]
    colors = [colornum_Metro(i) for i in range(n_cells)]
    estimates_data = estimates.A
    dims = estimates.imax.shape
    cm_conts = cm.utils.visualization.get_contours(estimates_data,
                                                   dims=estimates.imax.shape,
                                                   thr=cthr)
    if len(comps_to_select) == 0:
        comps_to_select = estimates.idx_components

    contours = []
    for i in comps_to_select:
        coors = cm_conts[i]["coordinates"]
        contours.append(coors[~np.isnan(coors).any(axis=1)])

    xs = [[pt[0] for pt in c] for c in contours]
    ys = [[dims[0] - pt[1] for pt in c] for c in contours]  # flip for y-axis inversion
    return dict(xs=xs, ys=ys, times=times, traces=traces, colors=colors, idx=comps_to_select)


def _component_contours(estimates, comps, cthr=0.3):
    """Contour coordinates of the components `comps`, one (k, 2) array each with CaImAn's NaN rows kept:
    what EstimatesToSrcFast draws, and what a server may cache for the same file and cthr."""
    conts = cm.utils.visualization.get_contours(estimates.A[:, comps], dims=estimates.imax.shape, thr=cthr)
    return [c["coordinates"] for c in conts]


def EstimatesToSrcFast(estimates, comps_to_select=[], cthr=0.3, corr_thr=0.6,
                       sf=None, ef=None, ds=1, fps=20,
                       detect_corner_artifacts=True, correlation_method='pearson', contours=None):
    """contours: _component_contours of comps_to_select, if already computed; None computes them."""

    if len(comps_to_select) == 0:
        comps_to_select = estimates.idx_components

    n_cells = len(comps_to_select)
    if n_cells == 0:
        return {}
    if contours is not None and len(contours) != n_cells:
        raise ValueError(f"{len(contours)} contours for {n_cells} components")

    if sf is None:
        sf = 0
    if ef is None:
        ef = estimates.C.shape[1]

    traces = [(tr-min(tr)) / (np.max(tr)-np.min(tr)) + i for i, tr in enumerate(estimates.C[comps_to_select, sf:ef][:, ::ds])]
    traces_flat = [(tr - min(tr)) / (np.max(tr) - np.min(tr)) for i, tr in
              enumerate(estimates.C[comps_to_select, sf:ef][:, ::ds])]

    times = [estimates.time[sf:ef][::ds] for _ in range(n_cells)]
    colors = [colornum_Metro(i) for i in range(n_cells)]

    hvals = [_hval(tr) for tr in traces]

    dims = estimates.imax.shape
    if contours is None:
        contours = _component_contours(estimates, comps_to_select, cthr)

    polygons = []
    areas = []
    for coors in contours:
        area = calculate_polygon_area(coors)
        polygons.append(coors[~np.isnan(coors).any(axis=1)])
        areas.append(area)

    xs = [[pt[0] for pt in c] for c in polygons]
    ys = [[dims[0] - pt[1] for pt in c] for c in polygons]  # flip for y-axis inversion

    # building correlation matrix and assigning corr scores to neurons
    CM = _corr_matrix(estimates.C[comps_to_select, sf:ef], correlation_method)
    corr_scores, corr_groups = _corr_groups(CM, corr_thr)

    return dict(xs=xs, ys=ys, times=times, traces=traces, areas=areas,
                hvals=hvals, colors=colors, corr_scores=corr_scores,
                corr_groups=corr_groups,
                idx=comps_to_select)


def EstimatesToSrcFull(est, fps, comps_to_select=[], cthr=0.3,
                         corr_thr=0.6, num_sessions=1, match_threshold=3,
                         sf=None, ef=None, ds=1,
                         include_event_based=True, include_heavy=False,
                         color_by_ml_probability=False, ml_threshold=0.5,
                         detect_corner_artifacts=True, corner_artifact_params=None,
                         correlation_method='pearson', n_iter=2):

    if len(comps_to_select) == 0:
        comps_to_select = list(est.idx_components)

    if sf is None:
        sf = 0
    if ef is None:
        ef = est.C.shape[1]

    traces = [(tr - min(tr)) / (np.max(tr) - np.min(tr)) + i for i, tr in
              enumerate(est.C[comps_to_select, sf:ef][:, ::ds])]

    # Build reconstruction traces from cached reconstructions (if available)
    traces_recon = None
    if hasattr(est, 'reconstructions') and est.reconstructions:
        traces_recon = []
        for i, comp_idx in enumerate(comps_to_select):
            if comp_idx in est.reconstructions:
                rec = est.reconstructions[comp_idx][sf:ef:ds]
                rec_min, rec_max = np.min(rec), np.max(rec)
                rec_range = rec_max - rec_min
                if rec_range == 0 or np.isclose(rec_range, 0):
                    rec_norm = np.zeros_like(rec) + i
                else:
                    rec_norm = (rec - rec_min) / rec_range + i
                traces_recon.append(rec_norm)
            else:
                # Fallback: use zeros if no reconstruction (don't copy signal)
                num_points = len(traces[i])
                traces_recon.append(np.zeros(num_points) + i)
        print(f'Loaded cached reconstructions for {len(est.reconstructions)} neurons')

    n_cells = len(comps_to_select)
    times = [est.time[sf:ef][::ds] for _ in range(n_cells)]
    colors = [colornum_Metro(i) for i in range(n_cells)]

    # we have to compute contours here OR pollute metrics dataframe (further) with xs, ys and other garbage
    # here we choose option 1 - a little overhead for the sake of modularity

    dims = est.imax.shape
    contours = get_contours(est, comps_to_select, cthr=cthr)
    coords = [c["coordinates"] for c in contours]
    coords = [crd[~np.isnan(crd).any(axis=1)] for crd in coords]

    xs = [[pt[0] for pt in c] for c in coords]
    ys = [[dims[0] - pt[1] for pt in c] for c in coords]  # flip for y-axis inversion

    t1 = time.time()

    # Check if pre-computed metrics exist on the estimates object
    if hasattr(est, 'metrics_df') and est.metrics_df is not None:
        # Use pre-computed metrics from run_auto_inspection()
        mdf = est.metrics_df.copy()
        # Filter to requested components if needed
        if 'component_idx' in mdf.columns and len(comps_to_select) > 0:
            mdf = mdf[mdf['component_idx'].isin(comps_to_select)].reset_index(drop=True)
        print(f'Using pre-computed metrics from estimates.metrics_df ({len(mdf)} neurons)')
    else:
        # Compute metrics from scratch
        mdf, _, _, _, _, _, _ = estimates_to_metrics(est, fps, comps_to_select=comps_to_select, cthr=cthr, contours=contours,
                                            corr_thr=corr_thr, num_sessions=num_sessions, match_threshold=match_threshold,
                                            sf=sf, ef=ef, ds=ds, include_event_based=include_event_based, include_heavy=include_heavy,
                                            detect_corner_artifacts_flag=detect_corner_artifacts, corner_artifact_params=corner_artifact_params,
                                            correlation_method=correlation_method, n_iter=n_iter)

    t2 = time.time()
    etime = np.round(t2 - t1, 2)
    print(f'Elapsed time for all metrics: {etime} s,'
          f'{np.round(etime / n_cells, 2)} s per neuron')

    # Apply ML probability-based coloring if requested
    if color_by_ml_probability:
        if 'ml_keep_probability' in mdf.columns and 'delete' in mdf.columns:
            colors = []
            for i in range(len(mdf)):
                prob = mdf.iloc[i]['ml_keep_probability']
                is_deleted = mdf.iloc[i]['delete'] == 1
                color = get_ml_probability_color(prob, threshold=ml_threshold, is_deleted=is_deleted)
                colors.append(color)
            print(f'Applied ML probability-based coloring (threshold={ml_threshold})')
        else:
            missing_cols = []
            if 'ml_keep_probability' not in mdf.columns:
                missing_cols.append('ml_keep_probability')
            if 'delete' not in mdf.columns:
                missing_cols.append('delete')
            print(f'WARNING: Cannot apply ML coloring - missing columns: {missing_cols}')
            print(f'         Using default Metro colors instead')

    # Build technical dict - only include traces_recon if it exists
    technical = dict(idx=comps_to_select, xs=xs, ys=ys, times=times, traces=traces, colors=colors)
    if traces_recon is not None:
        technical['traces_recon'] = traces_recon

    metrics = {k: v for k, v in mdf.to_dict(orient='list').items() if k not in ['component_idx', 'center']}
    return {**technical, **metrics}, {i: mname for i, mname in enumerate(list(metrics.keys()))}


# keys of PreparedSession.data that are not columns of the window's data source
_NOT_COLUMNS = ('base', 'base_recon', 'cm')


def _legacy_rows(estimates, comps, cthr=0.3, sf=None, ef=None, ds=1, corr_thr=0.6, correlation_method='pearson',
                 contours=None):
    """The window's first view of comps in legacy mode: the per-row values of EstimatesToSrcFast, but each trace
    kept once, normalized and at full resolution ('base', float32 (rows, frames): the window sends a decimated
    or zoomed part of it), no per-row copy of the time axis, and the correlation matrix the Corr sort comes
    from ('cm'), which a merge extends by one row. contours: _component_contours of comps, or None."""
    comps = [int(c) for c in comps]
    n = len(comps)
    if sf is None:
        sf = 0
    if ef is None:
        ef = estimates.C.shape[1]
    if contours is None:
        contours = _component_contours(estimates, comps, cthr) if n else []
    if len(contours) != n:
        raise ValueError(f"{len(contours)} contours for {n} components")

    X = np.array(estimates.C[comps, sf:ef][:, ::ds], dtype=float)
    with np.errstate(invalid='ignore', divide='ignore'):  # a flat trace is NaN, as before
        X -= X.min(axis=1, keepdims=True)
        X /= X.max(axis=1, keepdims=True)
    # on the trace drawn at height i, as EstimatesToSrcFast computes it
    hvals = [_hval(X[i] + i) for i in range(n)]
    base = X.astype(np.float32)
    del X

    dims = estimates.imax.shape
    polygons, areas = [], []
    for coors in contours:
        areas.append(calculate_polygon_area(coors))
        polygons.append(coors[~np.isnan(coors).any(axis=1)])
    xs = [[pt[0] for pt in c] for c in polygons]
    ys = [[dims[0] - pt[1] for pt in c] for c in polygons]  # flip for y-axis inversion

    CM = _corr_matrix(estimates.C[comps, sf:ef], correlation_method) if n else np.zeros((0, 0))
    corr_scores, corr_groups = _corr_groups(CM, corr_thr)
    return dict(idx=comps, xs=xs, ys=ys, areas=areas, hvals=hvals, colors=[colornum_Metro(i) for i in range(n)],
                corr_scores=list(corr_scores), corr_groups=list(corr_groups), base=base, cm=CM)


def _capcan_rows(data):
    """EstimatesToSrcFull's view in the window's form: its traces (and reconstructions) without the row
    offsets as 'base' ('base_recon'), and no per-row copies of the time axis."""
    def strip(rows):
        if len(rows) == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack([np.asarray(tr, dtype=float) - i for i, tr in enumerate(rows)]).astype(np.float32)
    out = {k: v for k, v in data.items() if k not in ('times', 'traces', 'traces_recon')}
    out['idx'] = [int(c) for c in data['idx']]
    out['base'] = strip(data['traces'])
    out['base_recon'] = strip(data['traces_recon']) if data.get('traces_recon') is not None else None
    out['cm'] = None
    return out


def _pearson_with(C, row, sf, ef):
    """Pearson correlation of C[row, sf:ef] with every row of C[:, sf:ef], without copying C (an hour-long C
    is hundreds of MB): sums of products instead of centred copies."""
    X = C[:, sf:ef]
    n = X.shape[1]
    y = X[row] - X[row].mean()
    mx = X.mean(axis=1)
    sxx = np.einsum('ij,ij->i', X, X) - n * mx ** 2
    with np.errstate(invalid='ignore', divide='ignore'):
        return (X @ y) / np.sqrt(sxx * (y @ y))


def _merged_quality(old, cmapping, sel_comps):
    """A per-component quality value (SNR_comp, r_values) after a merge: an untouched component keeps its value
    under its new number, the merged component gets the mean of the merged ones without infinite values."""
    new = np.zeros(len(cmapping) + 1)
    for oi, ni in cmapping.items():
        new[ni] = old[oi]
    merged = old[sel_comps]
    new[-1] = np.mean(merged[~np.isinf(merged)])
    return new


def SaveResults(estimates, sigma=3):
    # traces timestamping and writing
    stamped_traces = np.concatenate(([estimates.time], estimates.C[estimates.idx_components]), axis=0)
    pd.DataFrame(stamped_traces.T).to_csv(estimates.name.partition('estimates')[0] + 'traces.csv', index=False,
                                          header=['time_s', *np.arange(len(estimates.idx_components))])

    # making directory and tiff writing
    fold = estimates.name.partition('estimates')[0] + 'filters'
    if not os.path.exists(fold):
        os.mkdir(fold)
    ims = []
    #ToDo change transposition of filters
    for i, sp in enumerate(estimates.A.T[estimates.idx_components]):
        im = sp.reshape(estimates.imax.shape[::-1]).todense()
        if sigma:  # gaussian smoothing of neural contours, omitted if sigma=0
            im = gaussian_filter(im, sigma=sigma)
        ims.append((im * 255 / np.max(im)).astype(np.uint8))
        tfl.imwrite(fold + f'\\filter_{i + 1:03d}.tif', ims[-1])
    savemat(fold + '_session.mat', {"A": np.array(ims)})


# bkapp_kwargs the first view is built from, with their defaults
_FIRST_VIEW_DEFAULTS = {
    'mode': 'legacy', 'cthr': 0.3, 'start_frame': 0, 'end_frame': None, 'downsampling': 1, 'corr_thr': 0.6,
    'detect_corner_artifacts': True, 'correlation_method': 'pearson',
    # capcan mode only
    'num_sessions': 1, 'match_threshold': 1, 'include_event_based': True, 'include_heavy': False,
    'color_by_ml_probability': False, 'ml_threshold': 0.5, 'corner_artifact_params': None, 'n_iter': 2,
}


def _first_view_options(bkapp_kwargs, default_fps):
    """Everything the first view depends on besides the file; end_frame None is the full length."""
    kw = bkapp_kwargs or {}
    return {**{k: kw.get(k, v) for k, v in _FIRST_VIEW_DEFAULTS.items()}, 'default_fps': default_fps}


def _view_options(bkapp_kwargs):
    """Options of the traces view used both for the first view and after a merge."""
    o = _first_view_options(bkapp_kwargs, None)
    return dict(cthr=o['cthr'], sf=o['start_frame'], ef=o['end_frame'], ds=o['downsampling'],
                corr_thr=o['corr_thr'])


@dataclass
class PreparedSession:
    """What make_app needs to open a session; a server builds it with prepare_session off its IOLoop thread."""
    fname: str
    estimates: "Estimates"  # .time set
    # the first view: one row per component of estimates.idx_components; the window's columns plus 'base'
    # (normalized full-resolution traces), 'base_recon' and 'cm' (see _legacy_rows)
    data: dict
    contours: list | None  # legacy mode: _component_contours of those components; None in capcan mode
    options: dict  # _first_view_options it was built with; make_app refuses other options
    metric_mapping: dict | None = None  # capcan mode: metric names of data by column number

    @property
    def mode(self):
        return self.options['mode']


def _prepare(fname, estimates, default_fps, bkapp_kwargs, contours):
    o = _first_view_options(bkapp_kwargs, default_fps)
    view = _view_options(bkapp_kwargs)
    if o['mode'] == 'legacy':
        if contours is None:
            idx = estimates.idx_components
            contours = _component_contours(estimates, idx, view['cthr']) if len(idx) else []
        data = _legacy_rows(estimates, estimates.idx_components, correlation_method=o['correlation_method'],
                            contours=contours, **view)
        return PreparedSession(fname, estimates, data, contours, o)
    if o['mode'] == 'capcan':
        if contours is not None:
            raise ValueError('cached contours are used in legacy mode only')
        data, metric_mapping = EstimatesToSrcFull(
            estimates, default_fps, comps_to_select=[], cthr=view['cthr'], corr_thr=view['corr_thr'],
            num_sessions=o['num_sessions'], match_threshold=o['match_threshold'],
            sf=view['sf'], ef=view['ef'], ds=view['ds'],
            include_event_based=o['include_event_based'],
            include_heavy=o['include_heavy'],
            color_by_ml_probability=o['color_by_ml_probability'],
            ml_threshold=o['ml_threshold'],
            detect_corner_artifacts=o['detect_corner_artifacts'],
            corner_artifact_params=o['corner_artifact_params'],
            correlation_method=o['correlation_method'],
            n_iter=o['n_iter'])
        return PreparedSession(fname, estimates, _capcan_rows(data), None, o, metric_mapping)
    raise ValueError('wrong operation mode!')


def _check_prepared(prepared, fname, default_fps, bkapp_kwargs):
    """A prepared session built for another file or with other options would show the wrong view."""
    def norm(f):
        return os.path.normcase(os.path.abspath(str(f)))
    if norm(prepared.fname) != norm(fname):
        raise ValueError(f"the prepared session is of {prepared.fname}, not of {fname}")
    for k, v in _first_view_options(bkapp_kwargs, default_fps).items():
        if prepared.options.get(k) != v:
            raise ValueError(f"the prepared session was built with {k}={prepared.options.get(k)!r}, "
                             f"the window asks for {v!r}")


def prepare_session(fname, time=None, default_fps=20, bkapp_kwargs=None, contours=None) -> PreparedSession:
    """Load the estimates and build the window's first view; the slow part of opening a session.

    time: the session's timestamps in seconds (see LoadEstimates). contours: PreparedSession.contours of an
    earlier preparation of the same file with the same cthr (legacy mode); None computes them."""
    estimates = LoadEstimates(fname, default_fps=default_fps, time=time)
    return _prepare(fname, estimates, default_fps, bkapp_kwargs, contours)


def _autoinspected(estimates):
    # autoinspection attaches metrics_df, empty for a session without components
    return getattr(estimates, 'metrics_df', None) is not None


SELECT_TEXT = "Select a neuron."  # the metrics panel after a change, until a neuron is selected

# Estimates attributes that a Delete, CaImAn's manual_merge or the window's quality values after a merge replace
_CHANGED_FIELDS = ('idx_components', 'idx_components_bad', 'A', 'C', 'YrA', 'R', 'S', 'bl', 'c1',
                   'neurons_sn', 'g', 'nr', 'coordinates', 'SNR_comp', 'r_values')

# The browser draws every trace against one time array and at its row's height: one copy of the time axis is
# sent, and a Delete or a sort sends only the heights (dummy_id), never the traces.
_TIME_JS = "return this.data.traces.map(() => time.data.t)"
_HEIGHT_JS = """
const rows = this.data["{col}"], heights = this.data.dummy_id
const out = new Array(rows.length)
for (let i = 0; i < rows.length; i++) {{
  const y = rows[i], h = Number(heights[i]), r = new Float64Array(y.length)
  for (let j = 0; j < y.length; j++) r[j] = y[j] + h
  out[i] = r
}}
return out
"""


@dataclass
class _Numbering:
    """Which row of the window's data source shows which component. A Merge renumbers the components (CaImAn
    drops the merged ones and appends the result) but the rows stay, so no trace or contour is sent again; a
    row whose component no longer exists stays hidden and comes back with Revert change or Discard all."""
    slot_of: dict              # component number -> row
    idx: list                  # row -> component number, -1: no component now
    corr_scores: list | None   # row -> value (legacy mode); recomputed after a merge, as the old full rebuild did
    corr_groups: list | None
    cm: np.ndarray | None      # row x row correlations (diagonal and NaN 0), legacy mode


@dataclass
class _Change:
    """What Revert change restores: the state before the last Delete or Merge, by reference. A Delete and
    CaImAn's manual_merge assign new objects and never change the old ones in place, so nothing is copied."""
    fields: dict               # _CHANGED_FIELDS -> their objects before the change
    shown: list
    numbering: _Numbering


def make_app(fname, default_fps=20, bkapp_kwargs=None, *, prepared=None, final_save=None,
             progress_save=None, refuse_autoinspected=False):
    """
    Interactive neuron examination GUI.

    Args:
        fname: Path to estimates pickle file
        default_fps: Default frames per second (default: 20)
        bkapp_kwargs: Dict with configuration options:
            - mode: 'legacy' or 'capcan' (default: 'legacy')
            - color_by_ml_probability: Color neurons by ML keep probability (default: False)
                ONLY works in 'capcan' mode. Requires estimates with metrics_df containing
                'ml_keep_probability' and 'delete' columns (from run_auto_inspection).
                - Grey: deleted neurons
                - Red to Green: kept neurons (red=low confidence, green=high confidence)
            - ml_threshold: P(KEEP) threshold used for coloring (default: 0.5)
            - ml_model_path: Path to ML model (EBM) for feature importance sorting and
                per-neuron contribution highlighting. If provided, metrics will be sorted
                by global importance and show local contributions when a neuron is selected.
            - metrics_width: Width of the central metrics widget in pixels (default: 200)
            - compress_estimates: If True, compress estimates before saving (removes bad
                components, converts to float32, sparse S matrix). Default: False
            - end_frame: last frame shown (exclusive); None (default) is the full length
            - trace_points: at most this many points per trace are sent to the browser (default 2000):
                min and max of equal bins for the whole session, every frame of a zoomed-in window
            - Other options: size, cthr, downsampling, corr_thr, etc.
        prepared: PreparedSession of fname; the window does not load the file itself.
        final_save: final_save(estimates) -> str, called by "Save results" instead of writing the files
            next to the input; the returned text is shown in the page's status line. None: the notebook
            behaviour.
        progress_save: progress_save(estimates) -> str, the same for "Save progress".
        refuse_autoinspected: an autoinspected input (with metrics_df) opens a page with only
            AUTOINSPECTED_TEXT.
    """
    if bkapp_kwargs is None:
        bkapp_kwargs = {}
    if prepared is not None:
        _check_prepared(prepared, fname, default_fps, bkapp_kwargs)
    operation_mode = bkapp_kwargs.get('mode', 'legacy')
    ml_model_path = bkapp_kwargs.get('ml_model_path', None) if operation_mode == 'capcan' else None
    compress_estimates = bkapp_kwargs.get('compress_estimates', False)

    # ML model state (lazy-loaded)
    _ml_model = [None]  # Use list to allow modification in nested function
    _feature_importances = [None]

    def _get_ml_model():
        """Load model and extract global feature importances (lazy)."""
        if _ml_model[0] is None and ml_model_path is not None:
            import pickle
            from pathlib import Path
            model_path = Path(ml_model_path)
            if model_path.exists():
                with open(model_path, 'rb') as f:
                    _ml_model[0] = pickle.load(f)
                # Extract global importances (individual features only)
                if hasattr(_ml_model[0], 'term_importances') and hasattr(_ml_model[0], 'term_names_'):
                    importances = _ml_model[0].term_importances()
                    names = _ml_model[0].term_names_
                    _feature_importances[0] = {
                        name: imp for name, imp in zip(names, importances)
                        if ' & ' not in name  # Skip interaction terms
                    }
        return _ml_model[0]

    def _get_neuron_contributions(neuron_data):
        """Compute feature contributions for a single neuron using explain_local().

        Returns:
            tuple: (individual_contributions, total_abs_all, intercept)
                - individual_contributions: dict of {feature: score} for individual features
                - total_abs_all: sum of abs(score) for ALL terms (individual + interactions)
                - intercept: the model's intercept value
        """
        model = _get_ml_model()
        if model is None or not hasattr(model, 'explain_local'):
            return {}, 0.0, 0.0

        try:
            import pandas as pd
            # Get feature columns from model
            feature_cols = list(model.feature_names_in_)

            # Build feature dict from neuron_data
            feature_values = {}
            for col in feature_cols:
                if col in neuron_data:
                    value = neuron_data[col]
                    if isinstance(value, (float, int, np.floating, np.integer)):
                        feature_values[col] = value
                    else:
                        feature_values[col] = np.nan
                else:
                    feature_values[col] = np.nan

            # Create DataFrame for model
            X = pd.DataFrame([feature_values])
            X = X.replace([np.inf, -np.inf], np.nan)

            local_exp = model.explain_local(X)
            exp_data = local_exp.data(0)

            # Get intercept from model (not in explain_local output)
            intercept = model.intercept_[0] if hasattr(model, 'intercept_') else 0.0

            # Build contributions dict and compute totals
            contributions = {}
            total_abs_all = 0.0

            for name, score in zip(exp_data['names'], exp_data['scores']):
                total_abs_all += abs(score)
                if ' & ' not in name:  # Store individual features only for display
                    contributions[name] = score

            return contributions, total_abs_all, intercept
        except Exception:
            return {}, 0.0, 0.0
    # This is the main plotting functions which plots all images and traces and contains all button callbacks
    def bkapp(doc):

        class Storage:
            """The window's state: one working Estimates, the rows on screen and the record of the last change."""

            def __init__(self, estimates, shown, numbering):
                self.estimates = estimates  # the working object; the save buttons write it
                self.shown = shown  # component numbers on screen in row order (ascending): all, or a zoomed set
                self.numbering = numbering
                self.undo = None  # _Change of the last Delete or Merge
                self.feedback_dict = {}  # User feedback: {neuron_idx: {feedback_type, ml_prob, ...}}

            def slot_of(self, comp):
                """The data source row of a component."""
                return self.numbering.slot_of[comp]

        operation_mode = bkapp_kwargs.get('mode', 'legacy')
        size = bkapp_kwargs.get('size', 500)
        view = _view_options(bkapp_kwargs)
        cthr = view['cthr']
        ds = view['ds']
        corr_thr = view['corr_thr']
        correlation_method = bkapp_kwargs.get('correlation_method', 'pearson')
        trace_points = int(bkapp_kwargs.get('trace_points', 2000))
        color_by_ml = bkapp_kwargs.get('color_by_ml_probability', False)

        sort_order = bkapp_kwargs.get('sort_order', 'up')
        verbose = bkapp_kwargs.get('verbose', False)

        fill_alpha = bkapp_kwargs.get('fill_alpha', 0.5)
        nonselection_alpha = bkapp_kwargs.get('ns_alpha', 0.2)
        line_width = bkapp_kwargs.get('line_width', 1)
        line_alpha = bkapp_kwargs.get('line_alpha', 1)
        trace_line_width = bkapp_kwargs.get('trace_line_width', 1)
        trace_alpha = bkapp_kwargs.get('trace_alpha', 1)
        bwidth = bkapp_kwargs.get('button_width', 110)

        start_frame = view['sf']
        end_frame = view['ef']
        emergency = bkapp_kwargs.get('oh_shit', False)

        if 'enable_gpu_backend' in bkapp_kwargs:
            backend = "webgl" if bool(bkapp_kwargs.get('enable_gpu_backend')) else "canvas"
        else:
            backend = "canvas"

        # for future resetting
        if prepared is None:
            estimates0 = LoadEstimates(fname, default_fps=default_fps)
        else:
            estimates0 = prepared.estimates
        if refuse_autoinspected and _autoinspected(estimates0):
            doc.add_root(Div(text=AUTOINSPECTED_TEXT))
            return
        if prepared is None:
            prep = _prepare(fname, estimates0, default_fps, bkapp_kwargs, None)
        else:
            prep = prepared
        data = prep.data
        metric_mapping = prep.metric_mapping

        # Extract active deletion metrics for GUI filtering
        active_deletion_metrics = getattr(estimates0, 'active_deletion_metrics', None)

        if operation_mode not in ('legacy', 'capcan'):
            raise ValueError('wrong operation mode!')
        legacy = operation_mode == 'legacy'

        # Traces stay here at full resolution, one normalized row per data source row (data['base'], then one
        # block per merge); the browser gets at most trace_points points of each for the time window `win`
        # (base frames), all on one time array (time_src).
        sf = start_frame or 0
        ef = estimates0.C.shape[1] if end_frame is None else end_frame
        base_time = np.asarray(estimates0.time)[sf:ef][::ds]
        n_base = len(base_time)
        base_blocks = [data['base']]
        recon_blocks = None if data.get('base_recon') is None else [data['base_recon']]
        win = [0, n_base]

        def on_grid(blocks, i0, i1):
            """(frames, rows): every row of blocks on the base frames i0:i1, at most trace_points points each."""
            frames, rows = np.arange(i0, i1), []
            for block in blocks:
                f, values = _decimate_rows(block[:, i0:i1], trace_points)
                frames = i0 + f
                rows.extend(values)
            return frames, rows

        n0 = len(data['idx'])
        numbering0 = _Numbering(slot_of={c: k for k, c in enumerate(data['idx'])}, idx=list(data['idx']),
                                corr_scores=list(data['corr_scores']) if legacy else None,
                                corr_groups=list(data['corr_groups']) if legacy else None, cm=data['cm'])
        frames, rows = on_grid(base_blocks, 0, n_base)
        time_src = ColumnDataSource(data={'t': base_time[frames]})
        # One data source with a row for every component ever shown; a filter picks the rows on screen, so
        # Delete, Show sel. and the sorts send row numbers and heights, not data. Lists are copied: rows are
        # appended to them, and the prepared session must stay as it was built.
        columns = {k: list(v) for k, v in data.items() if k not in _NOT_COLUMNS}
        columns.update(idx=list(numbering0.idx), traces=rows, dummy_x=[-10] * n0, dummy_id=list(range(n0)),
                       metric=list(range(n0)))
        if recon_blocks is not None:
            columns['traces_recon'] = on_grid(recon_blocks, 0, n_base)[1]
        src = ColumnDataSource(data=columns)
        row_filter = IndexFilter(indices=list(range(n0)))

        # the loaded object is never changed: Discard all starts again from it
        storage = Storage(copy.copy(estimates0), [int(c) for c in data['idx']], numbering0)
        storage.mode = operation_mode

        if operation_mode == 'capcan':
            storage.metric_mapping = metric_mapping
            storage.active_deletion_metrics = active_deletion_metrics

        dims = estimates0.imax.shape
        title = fname.rpartition('/')[-1].partition('_estimates')[0]
        #title_add = f'       active comps: {len(curr_traces)}'
        title_add = ''
        title += title_add

        tools1 = ["pan", "tap", "box_select", "zoom_in", "zoom_out", "box_zoom", "reset"]
        tools2 = ["pan", "tap", "box_select", "zoom_in", "zoom_out", "box_zoom", "reset"]
        color_mapper = LinearColorMapper(palette="Greys256", low=1, high=256)

        imwidth = size
        trwidth = size
        '''
        # TODO: fix resolution
        if 'pathway' in bkapp_kwargs:
            if bkapp_kwargs['pathway'] == 'bonsai':
                imwidth = 608
                trwidth = 608

        try:
            title = get_session_name_from_path(fname)
        except Exception:
            title = ''
        '''

        height = int(imwidth * dims[0] / dims[1])
        imdata = np.flip(estimates0.imax, axis=0)  # flip for reverting y-axis

        # main plots, p1 is for image on the left, p2 is for traces on the right
        p1 = figure(width=imwidth, height=height, tools=tools1, toolbar_location='below', title=title,
                    output_backend=backend, background_fill_color='black', border_fill_color='black')
        p1.xgrid.grid_line_color = None
        p1.ygrid.grid_line_color = None
        p1.image(image=[imdata], color_mapper=color_mapper, dh=dims[0], dw=dims[1], x=0, y=0, syncable=False)

        # A fixed x range over the whole session and the selection points at x = -10, padded as the automatic
        # range was: the traces may hold only a zoomed-in window, and Reset must still return to the whole session.
        lo, hi = min(-10.0, float(base_time[0])), float(base_time[-1])
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        p2 = figure(width=trwidth, height=height, tools=tools2, toolbar_location='below', output_backend=backend,
                    x_range=Range1d(lo - pad, hi + pad))

        recon_renderer = None
        if not emergency:
            p1.patches('xs',
                       'ys',
                       fill_alpha=fill_alpha,
                       nonselection_alpha=nonselection_alpha,
                       color='colors',
                       selection_line_color="yellow",
                       line_width=line_width,
                       line_alpha=line_alpha,
                       source=src,
                       view=CDSView(filter=row_filter))

            times = expr(CustomJSExpr(args=dict(time=time_src), code=_TIME_JS))
            p2.multi_line(xs=times,
                          ys=expr(CustomJSExpr(code=_HEIGHT_JS.format(col='traces'))),
                          line_color='colors',
                          line_alpha=trace_alpha,
                          selection_line_width=trace_line_width,
                          source=src,
                          view=CDSView(filter=row_filter))

            if recon_blocks is not None:
                # Reconstruction overlay (dark grey, hidden by default)
                recon_renderer = p2.multi_line(xs=times,
                                               ys=expr(CustomJSExpr(code=_HEIGHT_JS.format(col='traces_recon'))),
                                               line_color='#404040',  # Dark grey
                                               line_alpha=0.9,
                                               line_width=1.5,
                                               source=src,
                                               view=CDSView(filter=row_filter),
                                               visible=False)

            # dummy height property makes traces selectable
            # (since multi_line does not support box selection, we have to plot additional scatter)
            p2.scatter('dummy_x',
                       'dummy_id',
                       source=src,
                       view=CDSView(filter=row_filter),
                       color='black',
                       size=5)

            p2.text(x='dummy_x', y='dummy_id', text='metric',
                    x_offset=5, y_offset=5, anchor="bottom_left",
                    source=src, view=CDSView(filter=row_filter), text_font_size='8pt')

        # this is for points addition
        pts_src = ColumnDataSource({'x': [], 'y': [], 'color': []})
        pts_renderer = p1.scatter(x='x', y='y', source=pts_src, color='color', size=5)
        draw_tool = PointDrawTool(renderers=[pts_renderer], empty_value='yellow')
        p1.add_tools(draw_tool)

        # --- Metrics display on tap ---
        # Narrower in legacy mode (100px), wider in capcan mode (default 200px)
        if operation_mode == 'legacy':
            metrics_width = 100
        else:
            metrics_width = bkapp_kwargs.get('metrics_width', 200)

        metrics_div = Div(
            text="<b>Neuron Metrics</b><br><i>Tap a neuron to see metrics</i>",
            name="metrics",
            width=metrics_width,
            height=height,
            styles={'overflow-y': 'scroll', 'border': '1px solid #ccc', 'padding': '5px',
                    'font-family': 'monospace', 'font-size': '9px'}
        )

        def on_selection_change(attr, old, new):
            """Handle selection changes to show metrics"""
            if len(new) == 0:
                return

            # Get the first selected neuron index
            selected_idx = new[0]

            # Get neuron data from source
            neuron_idx = src.data['idx'][selected_idx]

            # Check if neuron has feedback
            feedback_badge = ''
            if neuron_idx in storage.feedback_dict:
                feedback_type = storage.feedback_dict[neuron_idx]['feedback_type']
                badge_color = '#ff8c00' if feedback_type == 'FP' else '#1e90ff'
                feedback_badge = f" <span style='background-color: {badge_color}; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px;'>{feedback_type}</span>"

            # Build metrics HTML
            html = f"<b>Neuron #{neuron_idx}</b>{feedback_badge}<br>"
            html += "<hr style='margin: 3px 0;'>"

            # Get all metrics from the source
            exclude_keys = ['idx', 'xs', 'ys', 'times', 'traces', 'traces_recon', 'colors', 'dummy_x', 'dummy_id', 'metric']
            metrics_data = {}

            for key in src.data.keys():
                if key not in exclude_keys:
                    value = src.data[key][selected_idx]
                    metrics_data[key] = value

            # Try to get feature importances and neuron contributions from ML model
            # GUARD: Only compute ML contributions in capcan mode with ML model
            if operation_mode == 'capcan' and ml_model_path is not None:
                _get_ml_model()  # Trigger lazy load
                feature_importances = _feature_importances[0] or {}
                contributions, total_abs_all, intercept = _get_neuron_contributions(metrics_data) if feature_importances else ({}, 0.0, 0.0)
            else:
                # Legacy mode: no ML features
                feature_importances = {}
                contributions = {}
                total_abs_all = 0.0
                intercept = 0.0

            # Sort keys: important features first (if model), then alphabetically
            if feature_importances:
                # Separate ML features (in model) from other metrics
                ml_features = [k for k in metrics_data.keys() if k in feature_importances]
                other_keys = [k for k in metrics_data.keys() if k not in feature_importances]

                # Sort ML features by importance (descending)
                ml_features.sort(key=lambda k: feature_importances.get(k, 0), reverse=True)
                other_keys.sort()

                sorted_keys = ml_features + other_keys
            else:
                sorted_keys = sorted(metrics_data.keys())

            # Helper to format value
            def format_value(value, decimals=2):
                if isinstance(value, (int, np.integer)):
                    return f"{value}"
                elif isinstance(value, (float, np.floating)):
                    if np.isnan(value):
                        return "NaN"
                    else:
                        return f"{value:.{decimals}f}"
                return str(value)

            # Check if we have EBM contributions (only EBM models support this)
            has_ebm_contributions = bool(contributions) and total_abs_all > 0

            if has_ebm_contributions:
                # EBM model with local explanations
                # Use total_abs_all which includes interactions (for proper percentage calculation)

                # Color intensity levels (green for positive, red for negative)
                GREEN_INTENSE = '#1B5E20'   # +++
                GREEN_MEDIUM = '#4CAF50'    # ++
                GREEN_LIGHT = '#81C784'     # +
                RED_INTENSE = '#B71C1C'     # ---
                RED_MEDIUM = '#F44336'      # --
                RED_LIGHT = '#E57373'       # -
                BLUE = '#1976D2'            # for delete/ml_keep_probability

                def get_contrib_info(contrib):
                    if contrib is None or total_abs_all == 0:
                        return "", None, None
                    pct = abs(contrib) / total_abs_all * 100
                    is_positive = contrib > 0
                    if pct > 15:
                        sign = "[+++]" if is_positive else "[---]"
                        color = GREEN_INTENSE if is_positive else RED_INTENSE
                    elif pct > 8:
                        sign = "[++]" if is_positive else "[--]"
                        color = GREEN_MEDIUM if is_positive else RED_MEDIUM
                    elif pct > 3:
                        sign = "[+]" if is_positive else "[-]"
                        color = GREEN_LIGHT if is_positive else RED_LIGHT
                    else:
                        return "", None, None  # No sign for <3%
                    return sign, color, pct

                # Show intercept influence (model's prior)
                from scipy.special import expit
                base_prob = expit(intercept) * 100
                net_feature_effect = sum(contributions.values())
                intercept_color = GREEN_INTENSE if intercept > 0 else RED_INTENSE
                html += f"<b style='color: #666;'>ML Decision Drivers:</b><br>"
                html += f"<span style='color: {intercept_color};'><b>Base prior:</b> {base_prob:.0f}% KEEP</span><br>"
                html += f"<span style='color: #666;'><i>Features net: {net_feature_effect:+.2f}</i></span><br>"
                html += "<hr style='margin: 3px 0;'>"

                for key in sorted_keys:
                    value = metrics_data[key]
                    formatted_value = format_value(value)
                    contrib = contributions.get(key)
                    sign, color, pct = get_contrib_info(contrib)

                    if key in ('delete', 'ml_keep_probability'):
                        html += f"<span style='color: {BLUE};'><b>{key}:</b> {formatted_value}</span><br>"
                    elif sign:
                        html += f"<span style='color: {color};'><b>{key}:</b> {formatted_value} {sign} ({pct:.0f}%)</span><br>"
                    else:
                        html += f"<b>{key}:</b> {formatted_value}<br>"
            else:
                # No EBM model - simple display (alphabetically sorted)
                for key in sorted_keys:
                    value = metrics_data[key]
                    formatted_value = format_value(value, decimals=4)

                    if key == 'delete' and value == 1:
                        html += f"<span style='color: red;'><b>{key}:</b> {formatted_value}</span><br>"
                    elif key == 'ml_keep_probability':
                        if isinstance(value, (float, np.floating)) and not np.isnan(value):
                            if value > 0.75:
                                color = 'green'
                            elif value > 0.5:
                                color = 'orange'
                            else:
                                color = 'red'
                            html += f"<span style='color: {color};'><b>{key}:</b> {formatted_value}</span><br>"
                        else:
                            html += f"<b>{key}:</b> {formatted_value}<br>"
                    else:
                        html += f"<b>{key}:</b> {formatted_value}<br>"

            metrics_div.text = html

        # Attach selection callback to the source
        # Attach selection callback to the source
        src.selected.on_change('indices', on_selection_change)

        def on_ranges(event):
            """A zoom or pan of the traces plot ended: send every trace on the time window in view, every frame
            of it when it holds at most trace_points frames."""
            if event.x0 is None or event.x1 is None:
                return
            # one frame beyond each edge, so the lines reach the plot's borders
            i0 = max(int(np.searchsorted(base_time, event.x0, 'left')) - 1, 0)
            i1 = min(int(np.searchsorted(base_time, event.x1, 'right')) + 1, n_base)
            if i1 - i0 < 2 or [i0, i1] == win:
                return  # nothing of the session in view, or the window already sent
            win[:] = [i0, i1]
            frames, rows = on_grid(base_blocks, i0, i1)
            time_src.data = {'t': base_time[frames]}
            new = {'traces': rows}
            if recon_blocks is not None:
                new['traces_recon'] = on_grid(recon_blocks, i0, i1)[1]
            src.data.update(new)

        p2.on_event(RangesUpdate, on_ranges)

        # Button callbacks

        def selected_positions(storage):
            """Positions in storage.shown (rows on screen) of the selected rows."""
            inds = src.selected.indices
            inds = [inds] if isinstance(inds, int) else list(inds)
            position = {storage.slot_of(c): i for i, c in enumerate(storage.shown)}
            return sorted(position[s] for s in inds if s in position)

        def show_rows(storage):
            row_filter.indices = [storage.slot_of(c) for c in storage.shown]

        def after_change(storage, select=()):
            """The rows of storage.shown on screen, the selection, and a metrics panel that names no neuron:
            the one it named may be deleted or renumbered."""
            show_rows(storage)
            src.selected.indices = list(select)
            metrics_div.text = SELECT_TEXT

        def remember(storage):
            est = storage.estimates
            storage.undo = _Change({f: getattr(est, f) for f in _CHANGED_FIELDS if hasattr(est, f)},
                                   list(storage.shown), storage.numbering)

        def use_numbering(storage, numbering):
            """Send the columns that depend on the numbering, when a Merge, its Revert or Discard all changes it."""
            if storage.numbering is numbering:
                return
            storage.numbering = numbering
            n = len(src.data['idx'])

            def full(values, fill):  # rows added after this numbering was made have no component in it
                return list(values) + [fill] * (n - len(values))
            new = {'idx': full(numbering.idx, -1)}
            if legacy:
                new.update(corr_scores=full(numbering.corr_scores, 0.0),
                           corr_groups=full(numbering.corr_groups, 0.0))
            src.data.update(new)

        def add_row(storage, comp, position):
            """A data source row for comp, a merge's result at position in the list; the only contour computed."""
            est = storage.estimates
            coors = _component_contours(est, [comp], cthr)[0]
            polygon = coors[~np.isnan(coors).any(axis=1)]
            tr = np.array(est.C[comp, sf:ef][::ds], dtype=float)
            with np.errstate(invalid='ignore', divide='ignore'):
                tr -= tr.min()
                tr /= tr.max()
            block = tr.astype(np.float32)[None, :]
            base_blocks.append(block)
            if legacy or not color_by_ml:
                color = colornum_Metro(position)
            else:
                color = get_ml_probability_color(None)  # grey: a merged neuron has no ML probability
            row = {k: [np.nan] for k in src.data}  # the file's metrics: none for a merged neuron
            row.update(idx=[comp], xs=[[pt[0] for pt in polygon]], ys=[[dims[0] - pt[1] for pt in polygon]],
                       traces=on_grid([block], *win)[1], colors=[color], dummy_x=[-10], dummy_id=[position],
                       metric=[position])
            if legacy:
                # hvals on the trace drawn at its height, as EstimatesToSrcFast computes it
                row.update(areas=[calculate_polygon_area(coors)], hvals=[_hval(tr + position)],
                           corr_scores=[0.0], corr_groups=[0.0])
            if recon_blocks is not None:
                # no reconstruction for a merged neuron: a flat line, as for any neuron without one
                recon_blocks.append(np.zeros_like(block))
                row['traces_recon'] = on_grid([recon_blocks[-1]], *win)[1]
            src.stream(row)
            return len(src.data['idx']) - 1

        def merged_numbering(storage, cmapping):
            """The numbering after CaImAn's merge: untouched components renumbered (cmapping), rows added for the
            new ones; in legacy mode the Corr groups recomputed over the components now in the list."""
            est = storage.estimates
            old = storage.numbering
            slot_of = {int(cmapping[c]): s for c, s in old.slot_of.items() if c in cmapping}
            good = [int(c) for c in est.idx_components]
            new_comps = [c for c in good if c not in slot_of]
            for c in new_comps:
                slot_of[c] = add_row(storage, c, good.index(c))
            n = len(src.data['idx'])
            idx = [-1] * n
            for c, s in slot_of.items():
                idx[s] = c
            if not legacy:
                return _Numbering(slot_of, idx, None, None, None)
            good_slots = [slot_of[c] for c in good]
            cm = np.zeros((n, n))
            cm[:len(old.cm), :len(old.cm)] = old.cm
            if correlation_method == 'pearson':
                # the other pairs keep their values: one row of correlations, not the whole matrix again
                for c in new_comps:
                    r = _pearson_with(est.C, c, sf, ef)[good]
                    r[np.isnan(r)] = 0
                    cm[slot_of[c], good_slots] = r
                    cm[good_slots, slot_of[c]] = r
                    cm[slot_of[c], slot_of[c]] = 0
            else:
                cm[np.ix_(good_slots, good_slots)] = _corr_matrix(est.C[good, sf:ef], correlation_method)
            scores, groups = _corr_groups(cm[np.ix_(good_slots, good_slots)], corr_thr)
            corr_scores = list(old.corr_scores) + [0.0] * (n - len(old.corr_scores))
            corr_groups = list(old.corr_groups) + [0.0] * (n - len(old.corr_groups))
            for s, a, g in zip(good_slots, scores, groups):
                corr_scores[s], corr_groups[s] = float(a), float(g)
            return _Numbering(slot_of, idx, corr_scores, corr_groups, cm)

        # as before, Revert change before any change returns to the file as loaded
        remember(storage)

        merge_params = []

        def cnmf_params(estimates):
            """The session's CNMFParams, built at the first merge only: building takes about 0.25 s, and
            manual_merge only reads it (cnmf_dict is the file's and never changes)."""
            if not merge_params:
                merge_params.append(params.CNMFParams(params_dict=estimates.cnmf_dict))
            return merge_params[0]

        def sort_callback(event, storage=None, rb=None):
            regime = storage.mode
            est = storage.estimates
            shown = storage.shown
            slots = [storage.slot_of(c) for c in shown]
            mode = rb.active

            def column(name):
                return np.array([src.data[name][s] for s in slots])

            if regime == 'capcan':
                # automated metric
                if mode == 0:
                    metric = np.arange(len(shown))
                else:
                    # Use GUI metric mapping (filtered/reordered metrics)
                    metric = column(storage.gui_metric_mapping[mode - 1])

            elif regime == 'legacy':
                if mode == 0:
                    metric = np.arange(len(shown))
                elif mode == 1:
                    # trace SNR for each component
                    metric = np.array(est.SNR_comp)[shown]
                elif mode == 2:
                    # space correlation for each component
                    metric = np.array(est.r_values)[shown]
                elif mode == 3:
                    # % of high values (>median + 4*MAD) for each component
                    metric = column('hvals')
                elif mode == 4:
                    # area of each component
                    metric = column('areas')
                elif mode == 5:
                    # correlation with other components from corr matrix (belonging to the same connected component)
                    metric = column('corr_groups')
                elif mode in (6, 7, 8, 9):
                    # reconstruction: r2, mae, rmse, snr (only in capcan mode with reconstructions)
                    name = {6: 'r2_score', 7: 'nmae', 8: 'nrmse', 9: 'snr_recon'}[mode]
                    if name in src.data:
                        metric = column(name)
                    else:
                        print(f"WARNING: {name} not available, sorting by SNR instead")
                        metric = column('snrs')
                else:
                    raise ValueError('wrong RadioButton value')

            else:
                raise NotImplementedError()

            if sort_order == 'up':
                indices = np.argsort(metric)
            elif sort_order == 'down':
                indices = np.argsort(metric)[::-1]
            else:
                raise ValueError('Wrong order! Only "up" and "down" are supported')
            # The rows keep their order (positions of storage.shown, which a click refers to); only the
            # heights the traces are drawn at and the labels change.
            heights = list(src.data['dummy_id'])
            labels = list(src.data['metric'])
            for rank, pos in enumerate(indices):
                heights[slots[pos]] = rank
            for pos, s in enumerate(slots):
                labels[s] = np.round(metric[pos], 2)
            src.data.update(dummy_id=heights, metric=labels)

        def del_callback(event, storage=None):
            sel_inds = selected_positions(storage)
            if len(sel_inds) == 0:
                # Before the record: an empty click must not overwrite what Revert change restores.
                return
            estimates = storage.estimates
            if verbose:
                print('               Delete in progress...')
            sel_comps = np.array([ind for i, ind in enumerate(storage.shown) if i in sel_inds])
            if verbose:
                print('sel_inds:', sel_inds)
                print('num est comp before:', len(estimates.idx_components))
                print('est comp before:', estimates.idx_components)
                print('shown before:', storage.shown)
                print('sel_comps:', sel_comps)
            remember(storage)
            temp = list(estimates.idx_components_bad) + list(sel_comps)
            estimates.idx_components_bad = np.sort(temp)
            estimates.idx_components = [_ for _ in estimates.idx_components if _ not in sel_comps]
            if verbose:
                print('num est comp after:', len(estimates.idx_components))
                print('est comp after:', estimates.idx_components)

            # The rows now show the full list.
            storage.shown = [int(c) for c in estimates.idx_components]
            # No select-all: with every row selected a repeated click acts on all neurons,
            # and a click in toggle mode only clears the selection.
            after_change(storage)

        def merge_callback(event, storage=None):
            sel_inds = selected_positions(storage)
            if len(sel_inds) == 0:
                # Before the record: an empty click must not overwrite what Revert change restores.
                return
            estimates = storage.estimates
            if verbose:
                print('               Merge in progress...')
            sel_comps = [ind for i, ind in enumerate(storage.shown) if i in sel_inds]
            if verbose:
                print('sel_inds:', sel_inds)
                print('num est comp before:', len(estimates.idx_components))
                print('est comp before:', estimates.idx_components)
                print('shown before:', storage.shown)
                print('sel_comps:', sel_comps)
            remember(storage)
            before = storage.undo.fields
            estimates.manual_merge([sel_comps], params=cnmf_params(estimates))
            good_neurons = np.setdiff1d(list(range(before['C'].shape[0])), np.array(sel_comps))
            cmapping = dict(zip(good_neurons, np.arange(len(good_neurons))))
            estimates.SNR_comp = _merged_quality(before['SNR_comp'], cmapping, sel_comps)
            estimates.r_values = _merged_quality(before['r_values'], cmapping, sel_comps)

            use_numbering(storage, merged_numbering(storage, cmapping))
            storage.shown = [int(c) for c in estimates.idx_components]
            # Same as in del_callback: no select-all.
            after_change(storage)

        def show_callback(event, storage=None):
            sel_inds = selected_positions(storage)
            if verbose:
                print('               Zoom in progress...')
                print('sel inds:', sel_inds)

            if len(sel_inds) != 0:
                # Row numbers refer to the rows on screen, i.e. to the zoomed list, not to the full one.
                storage.shown = [ind for i, ind in enumerate(storage.shown) if i in sel_inds]
                if verbose:
                    print('est comp:', storage.estimates.idx_components)
                    print('shown:', storage.shown)
                show_rows(storage)
                src.selected.indices = [storage.slot_of(c) for c in storage.shown]

        def restore_callback(event, storage=None):
            if verbose:
                print('            Reset in progress...')
            storage.shown = [int(c) for c in storage.estimates.idx_components]
            show_rows(storage)

        def revert_callback(event, storage=None):
            rec = storage.undo
            estimates = storage.estimates
            for f, v in rec.fields.items():
                setattr(estimates, f, v)
            # the rows on screen at the change: a zoomed view comes back zoomed
            storage.shown = list(rec.shown)
            use_numbering(storage, rec.numbering)
            # no selection, as after Delete and Merge: a neuron left highlighted under a cleared panel could not
            # be selected again (no change of the selection), and a click would act on it unseen
            after_change(storage)

        def discard_callback(event, storage=None):
            if verbose:
                print('Discard in progress...')
            # attributes of the working object are replaced, never changed in place: a shallow copy is enough
            storage.estimates = copy.copy(estimates0)
            storage.shown = [int(c) for c in data['idx']]
            use_numbering(storage, numbering0)
            # as before, every row is selected after Discard all
            after_change(storage, select=[storage.slot_of(c) for c in storage.shown])
            if verbose:
                print('est comp:', storage.estimates.idx_components)
                print('num est comp:', len(storage.estimates.idx_components))
                print('num est comp bad:', len(storage.estimates.idx_components_bad))

        def seed_callback(event):
            seeds = [[pts_src.data['x']], [pts_src.data['y']]]
            seeds_fname = extract_name_with_pattern(estimates0.name) + '_seeds.pickle'
            with open(seeds_fname, "wb") as f:
                pickle.dump(seeds, f)
                print(f'Seeds saved to {seeds_fname}\n')

        # The server's save functions report in English text; markup in it is shown, not interpreted.
        status = (Div(text="", name="status")
                  if final_save is not None or progress_save is not None else None)

        def run_save_hook(save, storage):
            # a failed save must not end the callback: the page says so and the button stays usable
            try:
                text = save(storage.estimates)
            except Exception as e:
                logging.getLogger(__name__).exception('save failed for %s', fname)
                first = ''.join(traceback.format_exception_only(type(e), e)).strip().splitlines()[0]
                text = f'Save failed: {first}'
            status.text = html.escape(text, quote=False)

        def save_callback(event, storage=None):
            if progress_save is not None:
                run_save_hook(progress_save, storage)
                return
            dt = get_datetime()
            base_name = extract_name_with_pattern(estimates0.name)

            # remove previous date if it exists
            #if '-' in base_name:
            #    base_name = base_name[:2+1+2+1+4+1 + 2+1+2+1+2]
            out_name = base_name + dt.replace(':', '-') + '_estimates.pickle'
            save_processed_estimates(storage.estimates, out_name, compress=compress_estimates)
            print(f'Intermediate results for {title} saved to {out_name}\n')

        def final_save_callback(event, storage=None):
            if final_save is not None:
                run_save_hook(final_save, storage)
                return
            base_name = extract_name_with_pattern(estimates0.name)
            out_name = base_name + 'final_estimates.pickle'
            save_processed_estimates(storage.estimates, out_name, compress=compress_estimates)
            print(f'Final results for {title} saved to {out_name}\n')

            # now save to .mat file
            SaveResults(storage.estimates)
            print(f'Results for {title} saved in folder {os.path.dirname(fname)}\n')


        def mark_feedback_callback(feedback_type, event, storage=None):
            """Mark current neuron with FP or FN feedback (toggle behavior)"""
            if len(src.selected.indices) == 0:
                print("No neuron selected. Please select a neuron first.")
                return

            selected_idx = src.selected.indices[0]
            neuron_idx = src.data['idx'][selected_idx]

            # Toggle behavior: if already marked with same type, remove it
            if neuron_idx in storage.feedback_dict:
                if storage.feedback_dict[neuron_idx]['feedback_type'] == feedback_type:
                    del storage.feedback_dict[neuron_idx]
                    print(f"Removed {feedback_type} feedback for neuron #{neuron_idx}")
                    # Trigger metrics refresh to remove badge
                    on_selection_change('indices', [], src.selected.indices)
                    return

            # Capture ALL relevant metrics for this neuron
            # Exclude: display fields, dummy columns, decision logic, artifact flags, merge info
            exclude_keys = {'xs', 'ys', 'times', 'traces', 'colors', 'traces_recon',
                          'decision', 'is_corner_artifact', 'merge'}
            neuron_metrics = {}

            for key in src.data.keys():
                # Skip excluded keys and any failed_* or dummy_* columns
                if (key in exclude_keys or
                    key.startswith('failed_') or
                    key.startswith('dummy_')):
                    continue

                try:
                    value = src.data[key][selected_idx]
                    # Convert numpy types to native Python types for JSON serialization
                    if hasattr(value, 'item'):
                        value = value.item()
                    neuron_metrics[key] = value
                except (IndexError, KeyError):
                    neuron_metrics[key] = None

            # Add feedback metadata
            neuron_metrics['feedback_type'] = feedback_type
            neuron_metrics['session_name'] = estimates0.name if hasattr(estimates0, 'name') else 'unknown'
            neuron_metrics['timestamp'] = get_datetime()

            # Store complete record (overwrites if different type)
            storage.feedback_dict[neuron_idx] = neuron_metrics

            print(f"Marked neuron #{neuron_idx} as {feedback_type}")

            # Trigger metrics refresh to show badge
            on_selection_change('indices', [], src.selected.indices)


        def save_feedback_callback(event, storage=None):
            """Export all feedback to CSV with complete metrics"""
            if not storage.feedback_dict:
                print("No feedback recorded yet. Mark some neurons first.")
                return

            try:
                # Convert to DataFrame (all metrics already in feedback_dict)
                feedback_data = []
                for neuron_idx, metrics in storage.feedback_dict.items():
                    # Add neuron_idx to the record (it's the dict key, not in values)
                    record = {'neuron_idx': neuron_idx}
                    record.update(metrics)
                    feedback_data.append(record)

                df = pd.DataFrame(feedback_data)

                # Sort by neuron_idx for easier reading
                if 'idx' in df.columns:
                    df = df.sort_values('idx')
                else:
                    df = df.sort_values('neuron_idx')

                # Reorder columns: put key identifiers first, then all metrics
                priority_cols = ['neuron_idx', 'feedback_type', 'session_name', 'timestamp',
                               'ml_keep_probability', 'delete']
                other_cols = [c for c in df.columns if c not in priority_cols]
                ordered_cols = [c for c in priority_cols if c in df.columns] + sorted(other_cols)
                df = df[ordered_cols]

                # Generate filename and find/create artifacts folder
                from pathlib import Path

                # Extract clean session prefix (e.g., "LNOF_J01_1D" or "HOS_D04_1D_2T") for consistent naming
                raw_name = estimates0.name if hasattr(estimates0, 'name') else 'session'
                session_prefix = extract_session_id(raw_name) or extract_name_with_pattern(raw_name) or 'session'

                # Determine artifacts folder location
                estimates_path = Path(raw_name) if raw_name != 'session' else None
                parent_dir = estimates_path.parent if estimates_path else Path('.')

                # Step 1: Check if we're already inside an inspection_artifacts folder (by name)
                if parent_dir.name.startswith('inspection_artifacts_'):
                    artifacts_folder = parent_dir if parent_dir.exists() else parent_dir.resolve()
                    print(f"Using current artifacts folder: {artifacts_folder}")

                # Step 2: Search in parent directory
                elif parent_dir.exists():
                    artifacts_pattern = f'inspection_artifacts_{session_prefix}*'
                    artifacts_folders = sorted([p for p in parent_dir.glob(artifacts_pattern) if _same_session_folder(p, session_prefix)], key=lambda p: p.stat().st_mtime, reverse=True)
                    if artifacts_folders:
                        artifacts_folder = artifacts_folders[0]
                        print(f"Found artifacts folder: {artifacts_folder}")
                    else:
                        artifacts_folder = parent_dir / f'inspection_artifacts_{session_prefix}'
                        artifacts_folder.mkdir(exist_ok=True)
                        print(f"Created artifacts folder: {artifacts_folder}")

                # Step 3: Fallback to CWD
                else:
                    artifacts_pattern = f'inspection_artifacts_{session_prefix}*'
                    artifacts_folders = sorted([p for p in Path('.').glob(artifacts_pattern) if _same_session_folder(p, session_prefix)], key=lambda p: p.stat().st_mtime, reverse=True)
                    if artifacts_folders:
                        artifacts_folder = artifacts_folders[0]
                        print(f"Found artifacts folder in CWD: {artifacts_folder}")
                    else:
                        artifacts_folder = Path(f'inspection_artifacts_{session_prefix}')
                        artifacts_folder.mkdir(exist_ok=True)
                        print(f"Created artifacts folder in CWD: {artifacts_folder}")

                # Save feedback CSV in artifacts folder with session prefix
                feedback_csv = artifacts_folder / f'{session_prefix}_feedback.csv'

                # Save with verification
                df.to_csv(feedback_csv, index=False)

                # Verify the file was actually saved
                if not feedback_csv.exists():
                    print(f"ERROR: Failed to save feedback - file not created at {feedback_csv}")
                    return

                # Summary statistics
                fp_count = sum(1 for m in storage.feedback_dict.values() if m.get('feedback_type') == 'FP')
                fn_count = sum(1 for m in storage.feedback_dict.values() if m.get('feedback_type') == 'FN')
                num_metrics = len(df.columns) - 4  # exclude neuron_idx, feedback_type, session_name, timestamp

                print(f"SUCCESS: Saved {len(feedback_data)} feedback entries to {feedback_csv}")
                print(f"  FP count: {fp_count}")
                print(f"  FN count: {fn_count}")
                print(f"  Metrics per neuron: {num_metrics}")

            except Exception as e:
                import traceback
                print(f"ERROR saving feedback: {e}")
                print(f"Full traceback:\n{traceback.format_exc()}")


        # Sorting radiobutton
        if storage.mode == 'legacy':
            # Legacy mode: only basic metrics (no reconstruction metrics like R2, MAE, RMSE, SNR+)
            radio_button_group = RadioButtonGroup(labels=["XY", "SNR", "R-val", "H-val", "Area", "Corr"],
                                                  active=0, width=60)

            rb_js_callback = CustomJS(
                code="console.log('radio_button_group: active=' + this.origin.active, this.toString())")
            radio_button_group.js_on_event("button_click", rb_js_callback)
            radio_button_group.on_event("button_click", partial(sort_callback, storage=storage,
                                                                rb=radio_button_group))

            sorting_row = row(radio_button_group)

        elif storage.mode == 'capcan':
            # Show ALL metrics for sorting (not just active deletion metrics)
            auto_metric_names = [storage.metric_mapping[i] for i in range(len(storage.metric_mapping))]

            # Filter out non-numeric and duplicate metrics, separate special metrics
            exclude_metrics = {'decision', 'failed_corner_artefact'}
            special_metrics = []  # failed_* and ml_keep_probability go in last row
            regular_metrics = []

            for name in auto_metric_names:
                if name in exclude_metrics:
                    continue
                elif name.startswith('failed_'):
                    special_metrics.append(name)
                elif name == 'ml_keep_probability':
                    # Only show ML metric if ML model was loaded
                    if ml_model_path is not None:
                        special_metrics.append(name)
                else:
                    regular_metrics.append(name)

            # Build 4 rows: 3 for regular metrics + 1 for special metrics
            all_regular_labels = ["XY"] + regular_metrics
            n_regular = len(all_regular_labels)
            n_per_row = (n_regular + 2) // 3  # For 3 rows

            row1_labels = all_regular_labels[:n_per_row]
            row2_labels = all_regular_labels[n_per_row:2*n_per_row]
            row3_labels = all_regular_labels[2*n_per_row:]
            row4_labels = special_metrics  # failed_* + ml_keep_probability

            # Build mapping from display label to original metric index
            all_labels = all_regular_labels + special_metrics

            # Create GUI metric mapping: maps GUI index (excluding XY) to metric name
            # This is needed because we filter/reorder metrics for display
            gui_metric_mapping = {i: name for i, name in enumerate(regular_metrics + special_metrics)}
            storage.gui_metric_mapping = gui_metric_mapping

            # Create a virtual radio button group that tracks which is selected
            class SortingState:
                def __init__(self):
                    self.active = 0

            sorting_state = SortingState()

            # Create four button groups (3 regular + 1 special)
            radio_group1 = RadioButtonGroup(labels=row1_labels, active=0, width=60)
            radio_group2 = RadioButtonGroup(labels=row2_labels, active=-1, width=60) if row2_labels else None
            radio_group3 = RadioButtonGroup(labels=row3_labels, active=-1, width=60) if row3_labels else None
            radio_group4 = RadioButtonGroup(labels=row4_labels, active=-1, width=60) if row4_labels else None

            # Create a pseudo radio_button_group for compatibility with sort_callback
            class UnifiedRadioGroup:
                def __init__(self, state):
                    self.state = state

                @property
                def active(self):
                    return self.state.active

            radio_button_group = UnifiedRadioGroup(sorting_state)

            # Sync function to handle clicks
            def make_sync_callback(group_idx):
                def callback(attr, old, new):
                    if new == -1:
                        return

                    # Calculate global active index based on label position in all_labels
                    if group_idx == 1:
                        label = row1_labels[new]
                    elif group_idx == 2:
                        label = row2_labels[new]
                    elif group_idx == 3:
                        label = row3_labels[new]
                    elif group_idx == 4:
                        label = row4_labels[new]

                    # Find the label in all_labels to get proper index
                    global_active = all_labels.index(label)
                    sorting_state.active = global_active

                    # Deactivate other groups
                    if group_idx != 1:
                        radio_group1.active = -1
                    if group_idx != 2 and radio_group2:
                        radio_group2.active = -1
                    if group_idx != 3 and radio_group3:
                        radio_group3.active = -1
                    if group_idx != 4 and radio_group4:
                        radio_group4.active = -1

                    # Trigger sort
                    sort_callback(None, storage=storage, rb=radio_button_group)
                return callback

            radio_group1.on_change('active', make_sync_callback(1))
            if radio_group2:
                radio_group2.on_change('active', make_sync_callback(2))
            if radio_group3:
                radio_group3.on_change('active', make_sync_callback(3))
            if radio_group4:
                radio_group4.on_change('active', make_sync_callback(4))

            # Build layout with 4 rows
            rows_to_add = [radio_group1]
            if radio_group2:
                rows_to_add.append(radio_group2)
            if radio_group3:
                rows_to_add.append(radio_group3)
            if radio_group4:
                rows_to_add.append(radio_group4)

            sorting_row = column(*[row(rg) for rg in rows_to_add])

        else:
            raise NotImplementedError()

        # Buttons
        button_del = Button(label="Delete sel.", button_type="warning", width=bwidth, width_policy='fit')
        button_del.on_event('button_click', partial(del_callback, storage=storage),
                            partial(restore_callback, storage=storage),
                            partial(sort_callback, storage=storage, rb=radio_button_group))

        button_merge = Button(label="Merge sel.", button_type="warning", width=bwidth, width_policy='fit')
        button_merge.on_event('button_click', partial(merge_callback, storage=storage),
                              partial(restore_callback, storage=storage),
                              partial(sort_callback, storage=storage, rb=radio_button_group))

        button_show = Button(label="Show sel.", button_type="primary", width=bwidth, width_policy='fit')
        button_show.on_event('button_click', partial(show_callback, storage=storage),
                             partial(sort_callback, storage=storage, rb=radio_button_group))

        button_restore = Button(label="Reset view", button_type="primary", width=bwidth, width_policy='fit')
        button_restore.on_event('button_click', partial(restore_callback, storage=storage),
                                partial(sort_callback, storage=storage, rb=radio_button_group))

        button_revert = Button(label="Revert change", button_type="danger", width=bwidth, width_policy='fit')
        # no Reset view in this chain: Revert change returns the view of the last change, a zoomed one too
        button_revert.on_event('button_click', partial(revert_callback, storage=storage),
                               partial(sort_callback, storage=storage, rb=radio_button_group))

        button_discard = Button(label="Discard all", button_type="danger", width=bwidth, width_policy='fit')
        button_discard.on_event('button_click', partial(discard_callback, storage=storage),
                                partial(sort_callback, storage=storage, rb=radio_button_group))

        button_seed = Button(label="Save seeds", button_type="light", width=bwidth, width_policy='fit')
        button_seed.on_event('button_click', seed_callback)

        button_save = Button(label="Save progress", button_type="success", width=bwidth, width_policy='fit')
        button_save.on_event('button_click', partial(save_callback, storage=storage))

        button_save_final = Button(label="Save results", button_type="success", width=bwidth, width_policy='fit')
        button_save_final.on_event('button_click', partial(final_save_callback, storage=storage))

        # Feedback buttons (toggle behavior)
        button_mark_fp = Button(label="FP", button_type="warning", width=60, width_policy='fit')
        button_mark_fn = Button(label="FN", button_type="primary", width=60, width_policy='fit')
        button_save_feedback = Button(label="Save Feedback", button_type="success", width=110, width_policy='fit')

        # Reconstruction toggle checkbox (only visible if reconstructions available)
        has_reconstructions = recon_renderer is not None
        checkbox_recon = CheckboxGroup(
            labels=["Show reconstruction"],
            active=[],
            styles={'color': '#00AA00', 'font-weight': 'bold'},  # Green text
            visible=has_reconstructions
        )

        def recon_callback(attr, old, new):
            # Toggle reconstruction overlay visibility
            if recon_renderer is not None:
                recon_renderer.visible = (0 in new)

        checkbox_recon.on_change('active', recon_callback)

        # Wire feedback button callbacks
        button_mark_fp.on_event('button_click', partial(mark_feedback_callback, 'FP', storage=storage))
        button_mark_fn.on_event('button_click', partial(mark_feedback_callback, 'FN', storage=storage))
        button_save_feedback.on_event('button_click', partial(save_feedback_callback, storage=storage))

        doc.add_root(
            column(
                row(
                    button_del,
                    button_merge,
                    button_show,
                    button_restore,
                    button_revert,
                    button_discard,
                    Spacer(width=20),  # Visual separator
                    button_mark_fp,
                    button_mark_fn,
                    Spacer(width=20),  # Visual separator
                    button_save,
                    button_save_feedback,
                    button_save_final,
                    checkbox_recon
                ),
                *([status] if status is not None else []),
                sorting_row,
                row(p1, metrics_div, p2)
            )
        )

    return bkapp


def ExamineCells(fname, default_fps=20, bkapp_kwargs=None):
    """
    Interactive neuron examination GUI.

    Args:
        fname: Path to estimates pickle file
        default_fps: Default frames per second (default: 20)
        bkapp_kwargs: Dict with configuration options, see make_app.
    """
    # here, not at import: a server imports this module and must not switch Bokeh to notebook output
    output_notebook()
    show(make_app(fname, default_fps=default_fps, bkapp_kwargs=bkapp_kwargs))


def ManualSeeds(fname, size=600, cnmf_dict=None):
    def bkapp(doc):
        tools = ["pan", "tap", "box_select", "zoom_in", "zoom_out", "reset"]

        if cnmf_dict is not None:
            gsig = cnmf_dict['gSig'][0]
        else:
            gsig = 6

        imdata_ = build_average_image(fname, gsig, start_frame=0, end_frame=np.inf, step=5)
        imdata = np.flip(imdata_, axis=0)  # flip for reverting y-axis

        imwidth = size
        dims = imdata.shape
        height = int(imwidth * dims[0] / dims[1])

        title = get_session_name_from_path(fname)
        color_mapper = LinearColorMapper(palette="Greys256", low=1, high=256)
        p1 = figure(width=imwidth, height=height, tools=tools, toolbar_location='below', title=title)
        p1.image(image=[imdata], dh=dims[0], dw=dims[1], x=0, y=0, color_mapper=color_mapper)

        # this is for points addition
        pts_src = ColumnDataSource({'x': [], 'y': [], 'color': []})
        pts_renderer = p1.scatter(x='x', y='y', source=pts_src, color='color', size=3)
        draw_tool = PointDrawTool(renderers=[pts_renderer], empty_value='red')
        p1.add_tools(draw_tool)

        # Button callbscks

        def seed_callback(event):
            seeds = [[pts_src.data['x']], [pts_src.data['y']]]
            seeds_fname = fname.partition('_estimates')[0] + '_seeds.pickle'
            with open(seeds_fname, "wb") as f:
                pickle.dump(seeds, f)
                print(f'Seeds saved to {seeds_fname}\n')

        button_seed = Button(label="Save seeds", button_type="success", width=120)
        button_seed.on_event('button_click', seed_callback)

        doc.add_root(
            column(
                row(
                    button_seed,
                ),
                row(p1)
            )
        )

    output_notebook()  # show() of an app function needs notebook output (see ExamineCells)
    show(bkapp)


def build_average_image(fname, gsig, start_frame=0, end_frame=np.inf, step=5):
    tlen = len(tfl.TiffFile(fname).pages)
    data = tfl.imread(fname, key=range(start_frame, min(end_frame, tlen), step))

    _, pnr = cm.summary_images.correlation_pnr(data, gSig=gsig, swap_dim=False)
    pnr[np.where(pnr == np.inf)] = 0
    pnr[np.where(pnr > 70)] = 70
    pnr[np.isnan(pnr)] = 0
    imax = (pnr * 255 / np.max(pnr)).astype('uint8')
    return imax


def test_min_corr_and_pnr(fname, gsig, start_frame=0, end_frame=np.inf, step=5):
    tlen = len(tfl.TiffFile(fname).pages)
    data = tfl.imread(fname, key=range(start_frame, min(end_frame, tlen), step))

    correlation_image_pnr, pnr_image = cm.summary_images.correlation_pnr(data, gSig=gsig, swap_dim=False)
    pnr_image[np.where(pnr_image == np.inf)] = 0
    correlation_image_pnr[np.where(correlation_image_pnr == np.inf)] = 0
    pnr_image[np.isnan(pnr_image)] = 0
    correlation_image_pnr[np.isnan(correlation_image_pnr)] = 0

    fig = pl.figure(figsize=(10, 4))
    pl.axes([0.05, 0.2, 0.4, 0.7])
    im_cn = plt.imshow(correlation_image_pnr, cmap='jet')
    pl.title('correlation image')
    pl.colorbar()
    pl.axes([0.5, 0.2, 0.4, 0.7])
    im_pnr = pl.imshow(pnr_image, cmap='jet')
    pl.title('PNR')
    pl.colorbar()

    s_cn_max = Slider(pl.axes([0.05, 0.01, 0.35, 0.03]), 'vmax',
                      max(0, correlation_image_pnr.min()), min(1, correlation_image_pnr.max()),
                      valinit=min(1, correlation_image_pnr.max()))
    s_cn_min = Slider(pl.axes([0.05, 0.07, 0.35, 0.03]), 'vmin',
                      max(0, correlation_image_pnr.min()), min(1, correlation_image_pnr.max()),
                      valinit=max(0, correlation_image_pnr.min()))
    s_pnr_max = Slider(pl.axes([0.5, 0.01, 0.35, 0.03]), 'vmax',
                       max(0, pnr_image.min()), min(100, pnr_image.max()), valinit=min(100, pnr_image.max()))
    s_pnr_min = Slider(pl.axes([0.5, 0.07, 0.35, 0.03]), 'vmin',
                       max(0, pnr_image.min()), min(100, pnr_image.max()), valinit=max(0, pnr_image.min()))

    def update(val):
        im_cn.set_clim([s_cn_min.val, s_cn_max.val])
        im_pnr.set_clim([s_pnr_min.val, s_pnr_max.val])
        fig.canvas.draw_idle()

    s_cn_max.on_changed(update)
    s_cn_min.on_changed(update)
    s_pnr_max.on_changed(update)
    s_pnr_min.on_changed(update)

'''
def split_estimate(fname, default_fps=20, nparts=2):
    estimates0 = LoadEstimates(fname, default_fps=default_fps)
    chunks = np.array_split(estimates0.idx_components, nparts)
    for i, chunk in tqdm.tqdm(enumerate(chunks), total=len(chunks)):
        selected = chunk
        not_selected = np.array([comp for comp in estimates0.idx_components if comp not in chunk])

        estimates1 = copy.deepcopy(estimates0)
        estimates1.idx_components = list(selected)
        temp = list(estimates1.idx_components_bad) + list(not_selected)
        estimates1.idx_components_bad = np.sort(temp)

        base_name = fname.partition('_estimates')[0]
        out_name = base_name + f'_part_{i + 1}_out_of_{nparts}_estimates.pickle'
        with open(out_name, "wb") as f:
            pickle.dump(estimates1, f)

def merge_estimates(fnames, default_fps=20):
    all_estimates = []
    for fname in fnames:
        part_estimates = LoadEstimates(fname, default_fps=default_fps)
        all_estimates.append(part_estimates)

    all_good_comps = [est.idx_componentsa
                      chunks = np.array_split(estimates0.idx_components, nparts)
    for i, chunk in tqdm.tqdm(enumerate(chunks), total=len(chunks)):
        selected = chunk
    not_selected = np.array([comp for comp in estimates0.idx_components if comp not in chunk])

    estimates1 = copy.deepcopy(estimates0)
    estimates1.idx_components = list(selected)
    temp = list(estimates1.idx_components_bad) + list(not_selected)
    estimates1.idx_components_bad = np.sort(temp)

    base_name = fname.partition('_estimates')[0]
    out_name = base_name + f'_part_{i + 1}_out_of_{nparts}_estimates.pickle'
    with open(out_name, "wb") as f:
        pickle.dump(estimates1, f)
        
fname = askopenfilename(title = 'Select estimates file for examination',
                        initialdir = CONFIG['ROOT'],
                        filetypes = [('estimates files', '*estimates.pickle')])

print('estimates:', fname)
estimates = LoadEstimates(fname, default_fps=20)

# вот здесь надо руками вписать нужный тифф-файл, автоматизировать не нужно, т.к. структура папок везде разная
tifpath = "C:\\Users\\admin\\Projects\\H_mice\\HM_NOF_2D\\NOF_H04_4D_CR_MC.tif"

gsig=6
avim = build_average_image(tifpath, gsig)
estimates.imax = avim
out_name = fname.partition('_estimates')[0] + '_manual_imax_estimates.pickle'
print('edited estimates:', out_name)
with open(out_name, "wb") as f:
    pickle.dump(estimates, f)
'''
