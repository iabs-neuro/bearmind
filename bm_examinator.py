#Stuff needed for plotting and widget callbacks
import copy

from functools import partial
import tifffile as tfl
import caiman as cm
import pandas as pd
import numpy as np
import pickle
import os
from bokeh.plotting import figure, show, output_notebook
from bokeh.document.document import Document
from bokeh.models import LinearColorMapper, CDSView, ColumnDataSource, Plot, CustomJS, Button,PointDrawTool, TapTool
from bokeh.layouts import column, row
from bokeh.events import Tap
from bokeh.io import push_notebook
from glob import glob
from caiman.source_extraction.cnmf import params
from time import time
from scipy.ndimage import gaussian_filter
from scipy.io import savemat

from caiman.utils.visualization import nb_inspect_correlation_pnr, inspect_correlation_pnr
from config import get_session_name_from_path
from table_routines import *

output_notebook()


def colornum_Metro(num):
    #Returns color for each number as in Moscow Metro
    return {
    1:"red",        
    2:"green",      
    3:"mediumblue",        
    4:"cyan",        
    5:"sienna", 
    6:"darkorange",    
    7:"mediumvioletred",      
    8:"gold",   
    9:"grey",  
    0:"lawngreen"}.get(num%10)   



def LoadEstimates(name, default_fps=20):
    with open(name, "rb") as f:
        estimates = pickle.load(f,)
    estimates.name = name
    if not hasattr(estimates, 'imax'):  #temporal hack; normally, imax should be loaded from image simultaneously with estimates
        estimates.imax = LoadImaxFromResults(estimates.name.partition('estimates')[0] + 'results.pickle')
    #estimates.time = FindAndLoadTimestamp(estimates.name.partition('estimates')[0], estimates.C.shape[1])
    estimates.time = get_timestamps(estimates.name.partition('estimates')[0],
                                   estimates.C.shape[1],
                                   default_fps=default_fps)

    return estimates


def get_timestamps(name, n_frames, default_fps=20):
    #try to load timestamps, in case of failure use constant fps
    ts_files = glob(name + '*_timestamp.csv')
    if len(ts_files) == 0:
        return np.linspace(0, n_frames//default_fps, n_frames)
    else:
        ts_df = pd.read_csv(ts_files[0])
        time_col = find_time_column(ts_df)
        timeline = ts_df[time_col].values
        return timeline[:n_frames]


def get_fps_from_timestamps(name, default_fps=20, verbose=True):
    ts_files = glob(name + '*_timestamp.csv')
    if len(ts_files) == 0:
        if verbose:
            print('no timestamps found, reverting to default fps')
        return default_fps
    else:
        ts_df = pd.read_csv(ts_files[0])
        fps = get_fps(ts_df, verbose=verbose)
        return fps


def EstimatesToSrc(estimates, comps_to_select = [], cthr=0.3):
    n_cells = len(estimates.idx_components)
    if not n_cells:
        return {}
    traces = [tr/np.max(tr) + i for i, tr in enumerate(estimates.C[estimates.idx_components])]
    times = [estimates.time for i in range(n_cells)]
    colors = [colornum_Metro(i) for i in range(n_cells)]
    estimates_data = estimates.A
    dims = estimates.imax.shape
    cm_conts = cm.utils.visualization.get_contours(estimates_data,
                                                   dims=estimates.imax.shape,
                                                   thr=cthr)
    contours = []
    if len(comps_to_select) == 0:
        comps_to_select = estimates.idx_components

    for i in comps_to_select:
        coors = cm_conts[i]["coordinates"]
        contours.append(coors[~np.isnan(coors).any(axis=1)])
    xs = [[pt[0] for pt in c] for c in contours]
    ys = [[dims[0] - pt[1] for pt in c] for c in contours] # flip for y-axis inversion
    return dict(xs = xs, ys = ys, times = times, traces = traces, colors=colors, idx=comps_to_select)


def SaveResults(estimates, sigma = 3):
    #traces timestamping and writing
    stamped_traces = np.concatenate(([estimates.time], estimates.C[estimates.idx_components]), axis=0)
    pd.DataFrame(stamped_traces.T).to_csv(estimates.name.partition('estimates')[0] + 'traces.csv', index=False, header = ['time_s', *np.arange(len(estimates.idx_components))])

    #making directory and tiff writing
    fold = estimates.name.partition('estimates')[0] + 'filters'
    if not os.path.exists(fold):
        os.mkdir(fold)
    ims = []
    for i,sp in enumerate(estimates.A.T[estimates.idx_components]):
        im = sp.reshape(estimates.imax.shape[::-1]).todense()
        if sigma:    #gaussian smoothing of neural contours, omitted if sigma=0
            im = gaussian_filter(im, sigma=sigma)
        ims.append((im*255/np.max(im)).astype(np.uint8))
        tfl.imwrite(fold + f'\\filter_{i+1:03d}.tif', ims[-1])
    savemat(fold + '_session.mat', {"A":np.array(ims)})


def ExamineCells(fname, default_fps=20, bkapp_kwargs=None):
    #This is the main plotting functions which plots all images and traces and contains all button callbacks
    '''
    def count_selected():
        sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(
            src_partial.selected.indices)
        return len(sel_inds)
    '''

    def slice_cds(cds, comps_to_leave):
        overall_data = dict(cds.data)
        show_data = dict()
        all_comps = overall_data['idx']
        indices_to_leave = np.array([i for i, comp in enumerate(all_comps) if comp in comps_to_leave])
        index_mapping = dict(zip(indices_to_leave, range(len(indices_to_leave))))

        for key in overall_data.keys():
            if key == 'traces':
                # subtract id vals from trace vals and add new ids
                new_traces = [val-i+index_mapping[i] for i, val in enumerate(overall_data[key]) if i in indices_to_leave]
                show_data.update({'traces': new_traces})
            else:
                data_part = [val for i, val in enumerate(overall_data[key]) if i in indices_to_leave]
                show_data.update({key: data_part})

        return show_data

    def bkapp(doc):

        class Storage():
            def __init__(self):
                self.estimates=None
                self.estimates_partial=None
                self.prev_estimates=None
                self.prev_estimates_partial=None

        size = bkapp_kwargs.get('size') if 'size' in bkapp_kwargs else 500
        cthr = bkapp_kwargs.get('cthr') if 'cthr' in bkapp_kwargs else 0.3
        verbose = bkapp_kwargs.get('verbose') if 'verbose' in bkapp_kwargs else False
        fill_alpha = bkapp_kwargs.get('fill_alpha') if 'fill_alpha' in bkapp_kwargs else 0.5
        nonselection_alpha = bkapp_kwargs.get('ns_alpha') if 'ns_alpha' in bkapp_kwargs else 0.2
        line_width = bkapp_kwargs.get('line_width') if 'line_width' in bkapp_kwargs else 1
        line_alpha = bkapp_kwargs.get('line_alpha') if 'line_alpha' in bkapp_kwargs else 1
        trace_line_width = bkapp_kwargs.get('trace_line_width') if 'trace_line_width' in bkapp_kwargs else 1
        trace_alpha = bkapp_kwargs.get('trace_alpha') if 'trace_alpha' in bkapp_kwargs else 1

        if 'enable_gpu_backend' in bkapp_kwargs:
            backend = "webgl" if bool(bkapp_kwargs.get('enable_gpu_backend')) else "canvas"
        else:
            backend = "canvas"

        # for future resetting
        estimates0 = LoadEstimates(fname, default_fps=default_fps)
        est_data0 = EstimatesToSrc(estimates0, cthr=cthr)

        estimates = copy.deepcopy(estimates0)

        storage = Storage()
        storage.estimates = copy.deepcopy(estimates0)
        storage.estimates_partial = copy.deepcopy(estimates0)
        storage.prev_estimates = copy.deepcopy(estimates0)
        storage.prev_estimates_partial = copy.deepcopy(estimates0)

        src = ColumnDataSource(data=copy.deepcopy(est_data0))  # for main view
        src_partial = ColumnDataSource(data=copy.deepcopy(est_data0))  # for plotting

        dims = estimates.imax.shape
        title = fname.rpartition('/')[-1].partition('_estimates')[0]

        tools1 = ["pan", "tap", "box_select", "zoom_in", "zoom_out", "reset"]
        tools2 = ["pan", "tap", "box_zoom", "zoom_in", "zoom_out", "reset"]
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

        height = int(imwidth*dims[0]/dims[1])
        imdata = np.flip(estimates.imax, axis=0)  # flip for reverting y-axis
        #imdata = estimates.imax

        #main plots, p1 is for image on the left, p2 is for traces on the right
        p1 = figure(width = imwidth, height = height, tools = tools1, toolbar_location = 'below', title=title, output_backend=backend)
        p1.image(image=[imdata], color_mapper=color_mapper, dh = dims[0], dw = dims[1], x=0, y=0)
        p2 = figure(width = trwidth, height = height, tools = tools2, toolbar_location = 'below', output_backend=backend)

        p1.patches('xs',
                   'ys',
                   fill_alpha = fill_alpha,
                   nonselection_alpha = nonselection_alpha,
                   color = 'colors',
                   selection_line_color="yellow",
                   line_width=line_width,
                   line_alpha=line_alpha,
                   source=src_partial)

        null_source = ColumnDataSource({'times': [], 'traces': [], 'colors': []})

        p2.multi_line('times',
                      'traces',
                      line_color='colors',
                      line_alpha=trace_alpha,
                      selection_line_width=trace_line_width,
                      source=src_partial)

        #this is for points addition
        pts_src = ColumnDataSource({'x': [], 'y': [], 'color': []})
        pts_renderer = p1.scatter(x='x', y='y', source=pts_src, color = 'color',  size=5)
        draw_tool = PointDrawTool(renderers=[pts_renderer], empty_value='yellow')
        p1.add_tools(draw_tool)

        # image reload on tap
        def tap_callback(event):
            p1.image(image=[imdata], color_mapper=color_mapper, dh=dims[0], dw=dims[1], x=0, y=0)

            p1.patches('xs',
                       'ys',
                       fill_alpha=fill_alpha,
                       nonselection_alpha=nonselection_alpha,
                       color='colors',
                       selection_line_color="yellow",
                       line_width=line_width,
                       line_alpha=line_alpha,
                       source=src_partial)

        p1.add_tools(TapTool())
        p1.on_event(Tap, tap_callback)

        #Button callbacks
        def del_callback(event, storage=None):
            estimates = copy.deepcopy(storage.estimates)
            estimates_partial = copy.deepcopy(storage.estimates_partial)

            # save previous state
            storage.prev_estimates = copy.deepcopy(estimates)
            storage.prev_estimates_partial = copy.deepcopy(estimates_partial)

            if verbose:
                print('               Delete in progress...')
            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(src_partial.selected.indices)
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
            #print('all bad comps', len(temp))
            estimates.idx_components = [_ for _ in estimates.idx_components if _ not in sel_comps]
            if verbose:
                print('num est comp after:', len(estimates.idx_components))
                print('est comp after:', estimates.idx_components)

            #src.data = EstimatesToSrc(estimates, cthr=cthr)
            src.data = slice_cds(src, estimates.idx_components)
            storage.estimates = copy.deepcopy(estimates)

        def merge_callback(event, storage=None):
            estimates = copy.deepcopy(storage.estimates)
            estimates_partial = copy.deepcopy(storage.estimates_partial)

            # save previous state
            storage.prev_estimates = copy.deepcopy(estimates)
            storage.prev_estimates_partial = copy.deepcopy(estimates_partial)

            if verbose:
                print('               Merge in progress...')
            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(src_partial.selected.indices)
            sel_inds = np.array(sel_inds)
            sel_comps = np.array([ind for i, ind in enumerate(estimates_partial.idx_components) if i in sel_inds])
            if verbose:
                print('sel_inds:', sel_inds)
                print('num est comp before:', len(estimates.idx_components))
                print('est comp before:', estimates.idx_components)
                print('est partial before:', estimates_partial.idx_components)
                print('sel_comps:', sel_comps)

            #print([c for c in estimates.idx_components if c in sel_comps])
            if len(sel_inds) != 0:
                estimates.manual_merge([sel_comps],
                                       params=params.CNMFParams(params_dict=estimates.cnmf_dict))

                #print('after', [c for c in estimates.idx_components if c in sel_comps])
                src.data = EstimatesToSrc(estimates, cthr=cthr)
                storage.estimates = copy.deepcopy(estimates)

        def show_callback(event, storage=None):
            estimates = copy.deepcopy(storage.estimates)
            estimates_partial = copy.deepcopy(storage.estimates_partial)

            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(src_partial.selected.indices)
            sel_inds = np.array(sel_inds)
            if verbose:
                print('               Zoom in progress...')
                print('sel inds:', sel_inds)

            if len(sel_inds) != 0:
                estimates_partial.idx_components = np.array([ind for i, ind in enumerate(estimates.idx_components) if i in sel_inds])
                if verbose:
                    print('est comp num:', len(estimates.idx_components))
                    print('est comp:', estimates.idx_components)
                    print('est part:', estimates_partial.idx_components)

                storage.estimates_partial = copy.deepcopy(estimates_partial)
                show_data = slice_cds(src, estimates_partial.idx_components)
                src_partial.data = show_data
                #print(src_partial.__dict__)
                #src_partial.data = EstimatesToSrc(estimates_partial, cthr=cthr)
                #print(src_partial.__dict__)

        def restore_callback(event, storage=None):
            estimates = copy.deepcopy(storage.estimates)
            if verbose:
                print('            Reset in progress...')

            overall_data = dict(src.data)
            src_partial.data = copy.deepcopy(overall_data)
            if verbose:
                print('est comp:', estimates.idx_components)
                print('num est comp:', len(estimates.idx_components))
            storage.estimates_partial = copy.deepcopy(estimates)

        def revert_callback(event, storage=None):
            prev_estimates = copy.deepcopy(storage.prev_estimates)
            prev_estimates_partial = copy.deepcopy(storage.prev_estimates)

            storage.estimates = copy.deepcopy(storage.prev_estimates)
            storage.estimates_partial = copy.deepcopy(storage.prev_estimates_partial)
            src.data = EstimatesToSrc(prev_estimates, cthr=cthr)
            #src.data = slice_cds(src, prev_estimates.idx_components)
            src_partial.data = EstimatesToSrc(prev_estimates_partial, cthr=cthr)
            #src_partial.data = slice_cds(src, prev_estimates_partial.idx_components)

        def discard_callback(event, storage=None):
            if verbose:
                print('Discard in progress...')

            storage.estimates = copy.deepcopy(estimates0)
            storage.estimates_partial = copy.deepcopy(estimates0)
            src.data = copy.deepcopy(est_data0)
            src_partial.data = copy.deepcopy(est_data0)
            #src.data = EstimatesToSrc(estimates, cthr=cthr)
            #src_partial.data = EstimatesToSrc(estimates_partial, cthr=cthr)
            if verbose:
                print('est comp:', estimates.idx_components)
                print('num est comp:', len(estimates.idx_components))
                print('num est comp bad:', len(estimates.idx_components_bad))

        def seed_callback(event):
            seeds = [[pts_src.data['x']], [pts_src.data['y']]]
            seeds_fname = fname.partition('_estimates')[0] + '_seeds.pickle'
            with open(seeds_fname, "wb") as f:
                pickle.dump(seeds, f)
                print(f'Seeds saved to {seeds_fname}\n')

        def save_callback(event):
            SaveResults(storage.estimates)
            print(f'Results for {title} saved in folder {os.path.dirname(fname)}\n')

        #Buttons themselves
        button_del = Button(label="Delete selected", button_type="success", width = 120)
        button_del.on_event('button_click',partial(del_callback, storage=storage), partial(restore_callback, storage=storage))

        button_merge = Button(label="Merge selected", button_type="success", width = 120)
        button_merge.on_event('button_click',partial(merge_callback, storage=storage), partial(restore_callback, storage=storage))

        button_show = Button(label="Show selected", button_type="success", width = 120)
        button_show.on_event('button_click', partial(show_callback, storage=storage))

        button_restore = Button(label="Reset view", button_type="success", width = 120)
        button_restore.on_event('button_click', partial(restore_callback, storage=storage))

        button_revert = Button(label="Revert change", button_type="success", width = 120)
        button_revert.on_event('button_click', partial(revert_callback, storage=storage), partial(restore_callback, storage=storage))

        button_discard = Button(label="Discard changes", button_type="success", width = 120)
        button_discard.on_event('button_click', partial(discard_callback, storage=storage))

        button_seed = Button(label="Save seeds", button_type="success", width = 120)
        button_seed.on_event('button_click', seed_callback)

        button_save = Button(label="Save results", button_type="success", width = 120)
        button_save.on_event('button_click', save_callback)

        doc.add_root(
            column(
                row(
                    button_del,
                    button_merge,
                    button_show,
                    button_restore,
                    button_revert,
                    button_discard,
                    button_seed,
                    button_save,
                ),
                row(p1, p2)
            )
        )

    show(bkapp)


def build_average_image(fname, gsig, start_frame=0, end_frame=np.Inf, step=5):
    tlen = len(tfl.TiffFile(fname).pages)
    data = tfl.imread(fname, key=range(start_frame, min(end_frame, tlen), step))

    _, pnr = cm.summary_images.correlation_pnr(data, gSig=gsig, swap_dim=False)    
    imax = (pnr * 255 / np.max(pnr)).astype('uint16')
    return imax


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
        p1 = figure(width=imwidth, height = height, tools = tools, toolbar_location = 'below', title=title)
        p1.image(image=[imdata], dh = dims[0], dw = dims[1], x=0, y=0, color_mapper=color_mapper)

        #this is for points addition
        pts_src = ColumnDataSource({'x': [], 'y': [], 'color': []})
        pts_renderer = p1.scatter(x='x', y='y', source=pts_src, color = 'color',  size=5)
        draw_tool = PointDrawTool(renderers=[pts_renderer], empty_value='yellow')
        p1.add_tools(draw_tool)

        #Button callbscks

        def seed_callback(event):
            seeds = [[pts_src.data['x']], [pts_src.data['y']]]
            seeds_fname = fname.partition('_estimates')[0] + '_seeds.pickle'
            with open(seeds_fname, "wb") as f:
                pickle.dump(seeds, f)
                print(f'Seeds saved to {seeds_fname}\n')

        button_seed = Button(label="Save seeds", button_type="success", width = 120)
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
