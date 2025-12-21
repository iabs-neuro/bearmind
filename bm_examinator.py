# Stuff needed for plotting and widget callbacks
import copy
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
                          RadioButtonGroup, PointDrawTool, TapTool, LabelSet, Div, PreText, CheckboxGroup, Spacer)

from bokeh.layouts import column, row
from bokeh.events import Tap
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
from auto_inspector import estimates_to_metrics, save_processed_estimates
from polygon import get_contours
import matplotlib.colors as mcolors

output_notebook()


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


def LoadEstimates(name, default_fps=20):
    with open(name, "rb") as f:
        estimates = pickle.load(f, )
    estimates.name = name
    '''
    if not hasattr(estimates, 'imax'):  #temporal hack; normally, imax should be loaded from image simultaneously with estimates
        estimates.imax = LoadImaxFromResults(estimates.name.partition('estimates')[0] + 'results.pickle')
    '''
    estimates.time = get_timestamps(extract_name_with_pattern(estimates.name),
                                    estimates.C.shape[1],
                                    default_fps=default_fps)

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


def EstimatesToSrcFast(estimates, comps_to_select=[], cthr=0.3, corr_thr=0.6,
                       sf=None, ef=None, ds=1, fps=20,
                       detect_corner_artifacts=True, correlation_method='pearson'):

    if len(comps_to_select) == 0:
        comps_to_select = estimates.idx_components

    n_cells = len(comps_to_select)
    if n_cells == 0:
        return {}

    if sf is None:
        sf = 0
    if ef is None:
        ef = estimates.C.shape[1]

    traces = [(tr-min(tr)) / (np.max(tr)-np.min(tr)) + i for i, tr in enumerate(estimates.C[comps_to_select, sf:ef][:, ::ds])]
    traces_flat = [(tr - min(tr)) / (np.max(tr) - np.min(tr)) for i, tr in
              enumerate(estimates.C[comps_to_select, sf:ef][:, ::ds])]

    times = [estimates.time[sf:ef][::ds] for _ in range(n_cells)]
    colors = [colornum_Metro(i) for i in range(n_cells)]

    hvals = []
    for tr in traces:
        med = np.median(tr)
        meddev = median_abs_deviation(tr)
        hval = np.round(1.0 * len(np.where(tr >= med + 4 * meddev)[0]) / len(tr), 4)
        hvals.append(hval)

    estimates_data = estimates.A[:, comps_to_select]
    dims = estimates.imax.shape
    cm_conts = cm.utils.visualization.get_contours(estimates_data,
                                                   dims=estimates.imax.shape,
                                                   thr=cthr)

    contours = []
    areas = []
    for i, comp in enumerate(comps_to_select):
        coors = cm_conts[i]["coordinates"]
        area = calculate_polygon_area(coors)
        contours.append(coors[~np.isnan(coors).any(axis=1)])
        areas.append(area)

    xs = [[pt[0] for pt in c] for c in contours]
    ys = [[dims[0] - pt[1] for pt in c] for c in contours]  # flip for y-axis inversion

    # building correlation matrix and assigning corr scores to neurons
    trace_data = estimates.C[comps_to_select, sf:ef]
    if correlation_method == 'pearson':
        CM = np.corrcoef(trace_data)
    elif correlation_method == 'spearman':
        if len(comps_to_select) == 1:
            CM = np.array([[1.0]])
        else:
            CM, _ = spearmanr(trace_data, axis=1)
    else:
        raise ValueError(f"Unknown correlation method: {correlation_method}. Use 'pearson' or 'spearman'")
    np.fill_diagonal(CM, 0)
    CM[np.isnan(CM)] = 0

    TCM = CM.copy()
    TCM[np.where(TCM < corr_thr)] = 0

    nontrivial_ccs = [comp for comp in list(get_ccs_from_adj(TCM)) if len(comp) > 1]

    group_corr_scores = np.zeros(len(nontrivial_ccs))
    corr_scores = np.zeros(len(comps_to_select))
    corr_groups = np.zeros(len(comps_to_select))
    for i, group in enumerate(nontrivial_ccs):
        ordered = np.array(sorted(list(group)))
        subnetwork = CM[ordered, :][:, ordered]  # we take corr values from initial corr matrix
        nc = len(ordered)
        group_density = np.sum(subnetwork) / (nc ** 2 - nc)
        group_corr_scores[i] = group_density
        #group_av_nnz = np.mean(subnetwork[np.where(subnetwork != 0)])

    sorted_nontrivial_ccs = [nontrivial_ccs[i] for i in np.argsort(group_corr_scores)[::-1]] # sort components from highest to lowest score
    sorted_group_corr_scores = sorted(group_corr_scores)
    for i, group in enumerate(sorted_nontrivial_ccs):
        for comp in group:
            corr_scores[comp] = sorted_group_corr_scores[i]
            corr_groups[comp] = len(sorted_group_corr_scores) - i + 1 # big group number = high corr score

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
        mdf, _, _, _, _, _ = estimates_to_metrics(est, fps, comps_to_select=comps_to_select, cthr=cthr, contours=contours,
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

    technical = dict(idx=comps_to_select, xs=xs, ys=ys, times=times, traces=traces, colors=colors,
                     traces_recon=traces_recon)
    metrics = {k: v for k, v in mdf.to_dict(orient='list').items() if k not in ['component_idx', 'center']}
    return {**technical, **metrics}, {i: mname for i, mname in enumerate(list(metrics.keys()))}


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


def ExamineCells(fname, default_fps=20, bkapp_kwargs=None):
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
            - Other options: size, cthr, downsampling, corr_thr, etc.
    """
    if bkapp_kwargs is None:
        bkapp_kwargs = {}
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

    def slice_cds(cds, comps_to_leave):
        overall_data = dict(cds.data)
        show_data = dict()
        all_comps = overall_data['idx']
        indices_to_leave = np.array([i for i, comp in enumerate(all_comps) if comp in comps_to_leave])
        index_mapping = dict(zip(indices_to_leave, range(len(indices_to_leave))))

        for key in overall_data.keys():
            if key in ('traces', 'traces_recon'):
                # subtract id vals from trace vals and add new ids
                if overall_data[key] is not None:
                    new_traces = [val - i + index_mapping[i] for i, val in enumerate(overall_data[key]) if
                                  i in indices_to_leave]
                    show_data.update({key: new_traces})
                else:
                    show_data.update({key: None})
            else:
                data_part = [val for i, val in enumerate(overall_data[key]) if i in indices_to_leave]
                show_data.update({key: data_part})

        return show_data

    def sort_cds(cds, metric, order='up'):
        overall_data = dict(cds.data)
        if order == 'up':
            indices = np.argsort(metric)
        elif order == 'down':
            indices = np.argsort(metric)[::-1]
        else:
            raise ValueError('Wrong order! Only "up" and "down" are supported')
        #all_comps = overall_data['idx']

        show_data = copy.deepcopy(overall_data)

        #print('indies:',indices)
        new_traces = [None for _ in range(len(metric))]
        new_traces_recon = [None for _ in range(len(metric))] if overall_data.get('traces_recon') is not None else None
        new_ids = np.zeros(len(metric))
        for i, ind in enumerate(indices):  # we iterate over rows of CDS in the order given by sorted metric
            # ind = row number in cds
            # i = index of this row in sorted order
            current_id = overall_data['dummy_id'][ind]  # current id = current trace height
            current_trace = np.array(overall_data['traces'][ind])
            new_id = i  # new height is simply the index of the current row in sorting
            new_trace = current_trace - current_id + i  # subtract old height and add new one
            #new_trace = np.full(fill_value=i, shape=1000)
            new_traces[ind] = new_trace  # write new trace data to the current row in CDS
            new_ids[ind] = new_id  # write new height to the current row in CDS

            # Handle reconstruction traces with same offset adjustment
            if new_traces_recon is not None:
                current_recon = np.array(overall_data['traces_recon'][ind])
                new_traces_recon[ind] = current_recon - current_id + i

        # actually update our copy of CDS
        show_data.update({'traces': new_traces,
                          'dummy_id': new_ids,
                          'metric': [np.round(x, 2) for x in metric]
                          })
        if new_traces_recon is not None:
            show_data['traces_recon'] = new_traces_recon

        return show_data, indices

    def add_dummy_data(cds, ordering=None):
        ctraces = dict(cds.data)['traces']
        #time = dict(cds.data)['times'][0]
        if ordering is None:
            hdata = np.arange(len(ctraces))
        else:
            hdata = np.arange(len(ctraces))[ordering]
        #xdata = [-0.05*max(time) for _ in range(len(ctraces))]
        xdata = [-10 for _ in range(len(ctraces))]

        cds.add(hdata, 'dummy_id')
        cds.add(xdata, 'dummy_x')
        cds.add(hdata, 'metric')

    def bkapp(doc):

        class Storage:
            def __init__(self):
                self.estimates = None
                self.estimates_partial = None
                self.prev_estimates = None
                self.prev_estimates_partial = None
                self.prev_data = None
                self.prev_data_partial = None
                self.feedback_dict = {}  # User feedback: {neuron_idx: {feedback_type, ml_prob, ...}}

        operation_mode = bkapp_kwargs.get('mode', 'legacy')
        size = bkapp_kwargs.get('size', 500)
        cthr = bkapp_kwargs.get('cthr', 0.3)
        ds = bkapp_kwargs.get('downsampling', 1)
        corr_thr = bkapp_kwargs.get('corr_thr', 0.6)
        num_sessions = bkapp_kwargs.get('num_sessions', 1)
        match_threshold = bkapp_kwargs.get('match_threshold', 1)
        include_event_based = bkapp_kwargs.get('include_event_based', True)
        include_heavy = bkapp_kwargs.get('include_heavy', False)
        n_iter = bkapp_kwargs.get('n_iter', 2)
        color_by_ml_probability = bkapp_kwargs.get('color_by_ml_probability', False) if operation_mode == 'capcan' else False
        ml_threshold = bkapp_kwargs.get('ml_threshold', 0.5)
        detect_corner_artifacts = bkapp_kwargs.get('detect_corner_artifacts', True)
        corner_artifact_params = bkapp_kwargs.get('corner_artifact_params', None)
        correlation_method = bkapp_kwargs.get('correlation_method', 'pearson')

        sort_order = bkapp_kwargs.get('sort_order', 'up')
        verbose = bkapp_kwargs.get('verbose', False)

        fill_alpha = bkapp_kwargs.get('fill_alpha', 0.5)
        nonselection_alpha = bkapp_kwargs.get('ns_alpha', 0.2)
        line_width = bkapp_kwargs.get('line_width', 1)
        line_alpha = bkapp_kwargs.get('line_alpha', 1)
        trace_line_width = bkapp_kwargs.get('trace_line_width', 1)
        trace_alpha = bkapp_kwargs.get('trace_alpha', 1)
        bwidth = bkapp_kwargs.get('button_width', 110)

        start_frame = bkapp_kwargs.get('start_frame', 0)
        end_frame = bkapp_kwargs.get('end_frame', 0)
        emergency = bkapp_kwargs.get('oh_shit', False)

        if 'enable_gpu_backend' in bkapp_kwargs:
            backend = "webgl" if bool(bkapp_kwargs.get('enable_gpu_backend')) else "canvas"
        else:
            backend = "canvas"

        # for future resetting
        estimates0 = LoadEstimates(fname, default_fps=default_fps)

        # Extract active deletion metrics for GUI filtering
        active_deletion_metrics = getattr(estimates0, 'active_deletion_metrics', None)

        if operation_mode == 'legacy':
            est_data0 = EstimatesToSrcFast(estimates0,
                                           cthr=cthr,
                                           sf=start_frame,
                                           ef=end_frame,
                                           ds=ds,
                                           corr_thr=corr_thr,
                                           detect_corner_artifacts=detect_corner_artifacts,
                                           correlation_method=correlation_method)

        elif operation_mode == 'capcan':
            est_data0, metric_mapping = EstimatesToSrcFull(estimates0, default_fps,
                                                           comps_to_select=[], cthr=cthr,
                                           corr_thr=corr_thr, num_sessions=num_sessions,
                                           match_threshold=match_threshold,
                                           sf=start_frame, ef=end_frame, ds=ds,
                                           include_event_based=include_event_based,
                                           include_heavy=include_heavy,
                                           color_by_ml_probability=color_by_ml_probability,
                                           ml_threshold=ml_threshold,
                                           detect_corner_artifacts=detect_corner_artifacts,
                                           corner_artifact_params=corner_artifact_params,
                                           correlation_method=correlation_method,
                                           n_iter=n_iter)
        else:
            raise ValueError('wrong operation mode!')

        estimates = copy.deepcopy(estimates0)

        src = ColumnDataSource(data=copy.deepcopy(est_data0))  # for main view
        src_partial = ColumnDataSource(data=copy.deepcopy(est_data0))  # for plotting

        storage = Storage()
        storage.estimates = copy.deepcopy(estimates0)
        storage.estimates_partial = copy.deepcopy(estimates0)
        storage.prev_estimates = copy.deepcopy(estimates0)
        storage.prev_estimates_partial = copy.deepcopy(estimates0)
        storage.prev_data = copy.deepcopy(est_data0)
        storage.prev_data_partial = copy.deepcopy(est_data0)
        storage.mode = operation_mode

        if operation_mode == 'capcan':
            storage.metric_mapping = metric_mapping
            storage.active_deletion_metrics = active_deletion_metrics

        n_traces0 = len(est_data0['traces'])
        #storage.ordering = np.arange(n_traces0)

        dims = estimates.imax.shape
        title = fname.rpartition('/')[-1].partition('_estimates')[0]
        curr_traces = src_partial.data['traces']
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
        imdata = np.flip(estimates.imax, axis=0)  # flip for reverting y-axis

        # main plots, p1 is for image on the left, p2 is for traces on the right
        p1 = figure(width=imwidth, height=height, tools=tools1, toolbar_location='below', title=title,
                    output_backend=backend, background_fill_color='black', border_fill_color='black')
        p1.xgrid.grid_line_color = None
        p1.ygrid.grid_line_color = None
        p1.image(image=[imdata], color_mapper=color_mapper, dh=dims[0], dw=dims[1], x=0, y=0, syncable=False)

        p2 = figure(width=trwidth, height=height, tools=tools2, toolbar_location='below', output_backend=backend)

        if not emergency:
            p1.patches('xs',
                       'ys',
                       fill_alpha=fill_alpha,
                       nonselection_alpha=nonselection_alpha,
                       color='colors',
                       selection_line_color="yellow",
                       line_width=line_width,
                       line_alpha=line_alpha,
                       source=src_partial)

            # null_source = ColumnDataSource({'times': [], 'traces': [], 'colors': []})

            p2.multi_line('times',
                          'traces',
                          line_color='colors',
                          line_alpha=trace_alpha,
                          selection_line_width=trace_line_width,
                          source=src_partial)

            # Reconstruction overlay (dark grey, hidden by default)
            recon_renderer = p2.multi_line('times',
                          'traces_recon',
                          line_color='#404040',  # Dark grey
                          line_alpha=0.9,
                          line_width=1.5,
                          source=src_partial,
                          visible=False)

            # add dummy height property to ColumnDataSource to make traces selectable
            # (since multi_line does not support box selection, we have to plot additional scatter)

            add_dummy_data(src, ordering=None)
            add_dummy_data(src_partial, ordering=None)

            p2.scatter('dummy_x',
                       'dummy_id',
                       source=src_partial,
                       color='black',
                       size=5)

            p2.text(x='dummy_x', y='dummy_id', text='metric',
                    x_offset=5, y_offset=5, anchor="bottom_left",
                    source=src_partial, text_font_size='8pt')

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
            neuron_idx = src_partial.data['idx'][selected_idx]

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
            exclude_keys = ['idx', 'xs', 'ys', 'times', 'traces', 'colors', 'dummy_x', 'dummy_id', 'metric']
            metrics_data = {}

            for key in src_partial.data.keys():
                if key not in exclude_keys:
                    value = src_partial.data[key][selected_idx]
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
        src_partial.selected.on_change('indices', on_selection_change)

        # Button callbacks

        def sort_callback(event, storage=None, rb=None):
            # estimates = copy.deepcopy(storage.estimates)

            regime = storage.mode
            estimates_partial = copy.deepcopy(storage.estimates_partial)
            old_sel_indices = src_partial.selected.indices
            if len(old_sel_indices) == 0:
                old_sel_indices = np.arange(len(estimates_partial.idx_components))
                
            mode = rb.active

            if regime == 'capcan':
                # automated metric
                if mode == 0:
                    metric = np.arange(len(estimates_partial.idx_components))#[old_sel_indices]
                else:
                    # Use GUI metric mapping (filtered/reordered metrics)
                    mname = storage.gui_metric_mapping[mode-1]
                    metric = np.array(src_partial.data[mname])

            elif regime == 'legacy':
                if mode == 0:
                    metric = np.arange(len(estimates_partial.idx_components))#[old_sel_indices]
                elif mode == 1:
                    # trace SNR for each component
                    metric = np.array(estimates_partial.SNR_comp)[estimates_partial.idx_components]#[old_sel_indices]
                elif mode == 2:
                    # space correlation for each component
                    metric = np.array(estimates_partial.r_values)[estimates_partial.idx_components]#[old_sel_indices]
                elif mode == 3:
                    # % of high values (>median + 4*MAD) for each component
                    metric = np.array(src_partial.data['hvals'])#[estimates_partial.idx_components]
                elif mode == 4:
                    # area of each component
                    metric = np.array(src_partial.data['areas'])
                elif mode == 5:
                    # correlation with other components from corr matrix (belonging to the same connected component)
                    metric = np.array(src_partial.data['corr_groups'])
                elif mode == 6:
                    # reconstruction: r2
                    metric = np.array(src_partial.data['r2'])
                elif mode == 7:
                    # reconstruction: mae
                    metric = np.array(src_partial.data['mae'])
                elif mode == 8:
                    # reconstruction: rmse
                    metric = np.array(src_partial.data['rmse'])
                elif mode == 9:
                    # reconstruction: snr
                    metric = np.array(src_partial.data['snr_rec'])
                else:
                    raise ValueError('wrong RadioButton value')

            else:
                raise NotImplementedError()


            #print('mode=', mode)
            #print(metric[indices])

            #print(dict(src.data)['dummy_id'])
            #old_to_new_mapping = dict(zip(np.arange(len(indices)), indices))
            sorted_data, indices = sort_cds(src_partial, metric, order=sort_order)
            #storage.ordering = indices

            src_partial.data = sorted_data

            #src_partial.selected.update(indices=[old_to_new_mapping[ind] for ind in old_sel_indices])
            #add_dummy_data(src_partial, ordering=indices)
            #print(dict(src_partial.data)['dummy_id'])


        def del_callback(event, storage=None):
            estimates = copy.deepcopy(storage.estimates)
            estimates_partial = copy.deepcopy(storage.estimates_partial)

            # save previous state
            storage.prev_estimates = copy.deepcopy(estimates)
            storage.prev_estimates_partial = copy.deepcopy(estimates_partial)
            storage.prev_data = dict(src.data)
            storage.prev_data_partial = dict(src_partial.data)

            if verbose:
                print('               Delete in progress...')
            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(
                src_partial.selected.indices)
            sel_inds = np.array(sel_inds)
            sel_comps = np.array([ind for i, ind in enumerate(estimates_partial.idx_components) if i in sel_inds])
            if verbose:
                print('sel_inds:', sel_inds)
                print('num est comp before:', len(estimates.idx_components))
                print('est comp before:', estimates.idx_components)
                print('est partial before:', estimates_partial.idx_components)
                print('sel_comps:', sel_comps)
                print('new bad comps:', estimates_partial.idx_components[sel_inds].tolist())
            temp = estimates.idx_components_bad.tolist() + sel_comps.tolist()
            estimates.idx_components_bad = np.sort(temp)
            # print('all bad comps', len(temp))
            estimates.idx_components = [_ for _ in estimates.idx_components if _ not in sel_comps]
            if verbose:
                print('num est comp after:', len(estimates.idx_components))
                print('est comp after:', estimates.idx_components)

            # src.data = EstimatesToSrc(estimates, cthr=cthr)
            src.data = slice_cds(src, estimates.idx_components)
            add_dummy_data(src)
            src_partial.data = dict(src.data)
            src_partial.selected.indices = np.arange(len(estimates.idx_components))
            storage.estimates = copy.deepcopy(estimates)

        def merge_callback(event, storage=None):
            estimates = copy.deepcopy(storage.estimates)
            estimates_partial = copy.deepcopy(storage.estimates_partial)

            # save previous state
            storage.prev_estimates = copy.deepcopy(estimates)
            storage.prev_estimates_partial = copy.deepcopy(estimates_partial)
            storage.prev_data = dict(src.data)
            storage.prev_data_partial = dict(src_partial.data)

            if verbose:
                print('               Merge in progress...')
            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(
                src_partial.selected.indices)
            sel_inds = np.array(sel_inds)
            sel_comps = [ind for i, ind in enumerate(estimates_partial.idx_components) if i in sel_inds]
            not_sel_comps = [ind for i, ind in enumerate(estimates_partial.idx_components) if i not in sel_inds]
            if verbose:
                print('sel_inds:', sel_inds)
                print('num est comp before:', len(estimates.idx_components))
                print('est comp before:', estimates.idx_components)
                print('est partial before:', estimates_partial.idx_components)
                print('sel_comps:', sel_comps)

            # print('before:', [c for c in estimates.idx_components if c in sel_comps])
            if len(sel_inds) != 0:
                estimates.manual_merge([sel_comps],
                                       params=params.CNMFParams(params_dict=estimates.cnmf_dict))
                #estimates.evaluate_components()
                oest = storage.prev_estimates
                def get_unmerged_comp_mapping():
                    nr = oest.C.shape[0]
                    good_neurons = np.setdiff1d(list(range(nr)), np.array(sel_comps))
                    mapping = dict(zip(good_neurons, np.arange(len(good_neurons))))
                    return mapping

                def reassign_quality_metrics():
                    cmapping = get_unmerged_comp_mapping()

                    old_snr = oest.SNR_comp
                    new_snr = np.zeros(len(cmapping)+1)

                    # for untouched components
                    for oi, ni in cmapping.items():
                        new_snr[ni] = old_snr[oi]
                    # manual for merged components
                    merged_snrs = old_snr[sel_comps]
                    new_snr[-1] = np.mean(merged_snrs[~np.isinf(merged_snrs)])
                    estimates.SNR_comp = new_snr

                    old_r = oest.r_values
                    new_r = np.zeros(len(cmapping) + 1)

                    # for untouched components
                    for oi, ni in cmapping.items():
                        new_r[ni] = old_r[oi]
                    # manual for merged components
                    merged_rs = old_r[sel_comps]
                    new_r[-1] = np.mean(merged_rs[~np.isinf(merged_rs)])
                    estimates.r_values = new_r

                reassign_quality_metrics()

                # print('after', [c for c in estimates.idx_components if c in sel_comps])
                '''
                merged_data = EstimatesToSrcFast(estimates, cthr=cthr, comps_to_select=[estimates.idx_components[-1]])
                new_to_old_not_sel_comp_mapping = dict(zip(estimates.idx_components[:-1], not_sel_comps))
                not_touched_data = slice_cds(src, not_sel_comps)
                #print(merged_data)
                #print()
                #print(not_touched_data)
                n_not_touched = len(not_touched_data['xs'])

                # put merged data at the top of traces diagram:
                for i, data in enumerate(merged_data['traces']):
                    data += n_not_touched + i

                aggregated_data = copy.deepcopy(not_touched_data)
                # concatenate contents of both dicts
                for key in not_touched_data.keys():
                    aggregated_data[key].extend(merged_data[key])
                
                #print(aggregated_data)
                src.data = aggregated_data
                '''
                src.data = EstimatesToSrcFast(estimates,
                                              cthr=cthr,
                                              sf=start_frame,
                                              ef=end_frame,
                                              ds=ds,
                                              corr_thr=corr_thr)

                add_dummy_data(src)
                src_partial.data = dict(src.data)
                src_partial.selected.indices = np.arange(len(estimates.idx_components))

                storage.estimates = copy.deepcopy(estimates)

        def show_callback(event, storage=None):
            estimates = copy.deepcopy(storage.estimates)
            estimates_partial = copy.deepcopy(storage.estimates_partial)

            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(
                src_partial.selected.indices)
            #sel_inds = np.array(sel_inds)
            if verbose:
                print('               Zoom in progress...')
                print('sel inds:', sel_inds)

            if len(sel_inds) != 0:

                estimates_partial.idx_components = np.array(
                    [ind for i, ind in enumerate(estimates.idx_components) if i in sel_inds])

                part_to_total_mapping = {i: ind for i, ind in enumerate(estimates.idx_components) if i in sel_inds}
                if verbose:
                    print('est comp num:', len(estimates.idx_components))
                    print('est comp:', estimates.idx_components)
                    print('est part:', estimates_partial.idx_components)

                storage.estimates_partial = copy.deepcopy(estimates_partial)
                #print(estimates_partial.idx_components)
                #print(src_partial.selected.indices)
                #show_data = slice_cds(src, estimates_partial.idx_components[np.array(src_partial.selected.indices)])
                show_data = slice_cds(src, estimates_partial.idx_components)
                #show_data = slice_cds(src, src_partial.selected.indices)
                src_partial.data = show_data
                add_dummy_data(src_partial, ordering=None)
                # src_partial.data = EstimatesToSrc(estimates_partial, cthr=cthr)
                src_partial.selected.indices = np.arange(len(estimates_partial.idx_components))

        def restore_callback(event, storage=None):
            estimates = copy.deepcopy(storage.estimates)
            if verbose:
                print('            Reset in progress...')

            overall_data = dict(src.data)
            src_partial.data = copy.deepcopy(overall_data)
            #add_dummy_data(src_partial, ordering=None)
            if verbose:
                print('est comp:', estimates.idx_components)
                print('num est comp:', len(estimates.idx_components))
            storage.estimates_partial = copy.deepcopy(estimates)

        def revert_callback(event, storage=None):
            # prev_estimates = copy.deepcopy(storage.prev_estimates)
            # prev_estimates_partial = copy.deepcopy(storage.prev_estimates)
            prev_data = storage.prev_data
            prev_data_partial = storage.prev_data

            storage.estimates = copy.deepcopy(storage.prev_estimates)
            storage.estimates_partial = copy.deepcopy(storage.prev_estimates_partial)
            # src.data = EstimatesToSrc(prev_estimates, cthr=cthr)
            # src.data = slice_cds(src, prev_estimates.idx_components)
            # src_partial.data = EstimatesToSrc(prev_estimates_partial, cthr=cthr)
            # src_partial.data = slice_cds(src, prev_estimates_partial.idx_components)

            src.data = copy.deepcopy(prev_data)
            src_partial.data = copy.deepcopy(prev_data_partial)
            #add_dummy_data(src, ordering=storage.ordering)
            #add_dummy_data(src_partial, ordering=storage.ordering)

        def discard_callback(event, storage=None):
            if verbose:
                print('Discard in progress...')

            storage.estimates = copy.deepcopy(estimates0)
            storage.estimates_partial = copy.deepcopy(estimates0)
            src.data = copy.deepcopy(est_data0)
            src_partial.data = copy.deepcopy(est_data0)
            add_dummy_data(src)
            add_dummy_data(src_partial)
            src_partial.selected.indices = np.arange(len(storage.estimates.idx_components))
            # src.data = EstimatesToSrc(estimates, cthr=cthr)
            # src_partial.data = EstimatesToSrc(estimates_partial, cthr=cthr)
            if verbose:
                print('est comp:', estimates.idx_components)
                print('num est comp:', len(estimates.idx_components))
                print('num est comp bad:', len(estimates.idx_components_bad))

        def seed_callback(event):
            seeds = [[pts_src.data['x']], [pts_src.data['y']]]
            seeds_fname = extract_name_with_pattern(estimates.name) + '_seeds.pickle'
            with open(seeds_fname, "wb") as f:
                pickle.dump(seeds, f)
                print(f'Seeds saved to {seeds_fname}\n')

        def save_callback(event, storage=None):
            dt = get_datetime()
            base_name = extract_name_with_pattern(estimates.name)

            # remove previous date if it exists
            #if '-' in base_name:
            #    base_name = base_name[:2+1+2+1+4+1 + 2+1+2+1+2]
            out_name = base_name + dt.replace(':', '-') + '_estimates.pickle'
            save_processed_estimates(storage.estimates, out_name, compress=compress_estimates)
            print(f'Intermediate results for {title} saved to {out_name}\n')

        def final_save_callback(event, storage=None):
            base_name = extract_name_with_pattern(estimates.name)
            out_name = base_name + 'final_estimates.pickle'
            save_processed_estimates(storage.estimates, out_name, compress=compress_estimates)
            print(f'Final results for {title} saved to {out_name}\n')

            # now save to .mat file
            SaveResults(storage.estimates)
            print(f'Results for {title} saved in folder {os.path.dirname(fname)}\n')


        def mark_feedback_callback(feedback_type, event, storage=None):
            """Mark current neuron with FP or FN feedback (toggle behavior)"""
            if len(src_partial.selected.indices) == 0:
                print("No neuron selected. Please select a neuron first.")
                return

            selected_idx = src_partial.selected.indices[0]
            neuron_idx = src_partial.data['idx'][selected_idx]

            # Toggle behavior: if already marked with same type, remove it
            if neuron_idx in storage.feedback_dict:
                if storage.feedback_dict[neuron_idx]['feedback_type'] == feedback_type:
                    del storage.feedback_dict[neuron_idx]
                    print(f"Removed {feedback_type} feedback for neuron #{neuron_idx}")
                    # Trigger metrics refresh to remove badge
                    on_selection_change('indices', [], src_partial.selected.indices)
                    return

            # Collect context
            ml_prob = src_partial.data.get('ml_keep_probability', [None])[selected_idx]
            delete_status = src_partial.data.get('delete', [0])[selected_idx]

            # Store feedback (overwrites if different type)
            storage.feedback_dict[neuron_idx] = {
                'feedback_type': feedback_type,
                'ml_keep_probability': ml_prob,
                'delete_status': delete_status,
                'session_name': estimates.name if hasattr(estimates, 'name') else 'unknown',
                'timestamp': get_datetime()
            }

            print(f"Marked neuron #{neuron_idx} as {feedback_type}")

            # Trigger metrics refresh to show badge
            on_selection_change('indices', [], src_partial.selected.indices)


        def save_feedback_callback(event, storage=None):
            """Export all feedback to CSV"""
            if not storage.feedback_dict:
                print("No feedback recorded yet. Mark some neurons first.")
                return

            # Convert to DataFrame
            feedback_data = []
            for neuron_idx, feedback in storage.feedback_dict.items():
                feedback_data.append({
                    'neuron_idx': neuron_idx,
                    'session_name': feedback['session_name'],
                    'feedback_type': feedback['feedback_type'],
                    'ml_keep_probability': feedback['ml_keep_probability'],
                    'delete_status': feedback['delete_status'],
                    'timestamp': feedback['timestamp']
                })

            df = pd.DataFrame(feedback_data)
            df = df.sort_values('neuron_idx')

            # Generate filename
            dt = get_datetime().replace(':', '-')
            base_name = extract_name_with_pattern(estimates.name) if hasattr(estimates, 'name') else 'session'
            feedback_csv = f'{base_name}_feedback_{dt}.csv'

            # Save
            df.to_csv(feedback_csv, index=False)

            print(f"Saved {len(feedback_data)} feedback entries to {feedback_csv}")
            print(f"  FP count: {sum(1 for f in feedback_data if f['feedback_type'] == 'FP')}")
            print(f"  FN count: {sum(1 for f in feedback_data if f['feedback_type'] == 'FN')}")


        # Sorting radiobutton
        if storage.mode == 'legacy':
            radio_button_group = RadioButtonGroup(labels=["XY", "SNR", "R-val", "H-val", "Area", "Corr",
                                                          'R2', 'MAE', 'RMSE', 'SNR+'], active=0, width=60)

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
        button_revert.on_event('button_click', partial(revert_callback, storage=storage),
                               partial(restore_callback, storage=storage),
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
        has_reconstructions = est_data0.get('traces_recon') is not None
        checkbox_recon = CheckboxGroup(
            labels=["Show reconstruction"],
            active=[],
            styles={'color': '#00AA00', 'font-weight': 'bold'},  # Green text
            visible=has_reconstructions
        )

        def recon_callback(attr, old, new):
            # Toggle reconstruction overlay visibility
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
                sorting_row,
                row(p1, metrics_div, p2)
            )
        )

    show(bkapp)


def ManualSeeds(fname, size=600, cnmf_dict=None):
    def bkapp(doc):
        tools = ["pan", "tap", "box_select", "zoom_in", "zoom_out", "reset"]

        if cnmf_dict is not None:
            gsig = cnmf_dict['gSig'][0]
        else:
            gsig = 6

        imdata_ = build_average_image(fname, gsig, start_frame=0, end_frame=np.Inf, step=5)
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

    show(bkapp)


def build_average_image(fname, gsig, start_frame=0, end_frame=np.Inf, step=5):
    tlen = len(tfl.TiffFile(fname).pages)
    data = tfl.imread(fname, key=range(start_frame, min(end_frame, tlen), step))

    _, pnr = cm.summary_images.correlation_pnr(data, gSig=gsig, swap_dim=False)
    pnr[np.where(pnr == np.inf)] = 0
    pnr[np.where(pnr > 70)] = 70
    pnr[np.isnan(pnr)] = 0
    imax = (pnr * 255 / np.max(pnr)).astype('uint8')
    return imax


def test_min_corr_and_pnr(fname, gsig, start_frame=0, end_frame=np.Inf, step=5):
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
        estimates1.idx_components = selected.tolist()
        temp = estimates1.idx_components_bad.tolist() + not_selected.tolist()
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
    estimates1.idx_components = selected.tolist()
    temp = estimates1.idx_components_bad.tolist() + not_selected.tolist()
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
