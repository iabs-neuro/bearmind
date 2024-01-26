#Stuff needed for plotting and widget callbacks
import copy

import tifffile as tfl
import caiman as cm
import pandas as pd
import numpy as np
import pickle
import os
from bokeh.plotting import figure, show, output_notebook
from bokeh.document.document import Document
from bokeh.models import LinearColorMapper, CDSView, ColumnDataSource, Plot, CustomJS, Button, IndexFilter, BooleanFilter, PointDrawTool
from bokeh.layouts import column, row
from bokeh.io import push_notebook
from glob import glob
from caiman.source_extraction.cnmf import params
from time import time
from scipy.ndimage import gaussian_filter
from scipy.io import savemat

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


def FindAndLoadTimestamp_deprecated(name, n_frames, fps = 20):
    #try to load timestamp, in case of failure use constant fps
    tst = glob(name + '*_timestamp.csv')
    if not tst:
        return np.linspace(0, n_frames/fps, n_frames)
    else:
        time_s = np.genfromtxt(tst[0], delimiter = ',', skip_header = 1)[:,1]/1000
        return time_s[:n_frames]


def EstimatesToSrc(estimates, cthr=0.9):
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
    for i in estimates.idx_components:
        coors = cm_conts[i]["coordinates"]
        contours.append(coors[~np.isnan(coors).any(axis=1)])
    xs = [[pt[0] for pt in c] for c in contours]
    ys = [[dims[1] - pt[1] for pt in c] for c in contours] # flip for y-axis inversion
    return dict(xs = xs, ys = ys, times = times, traces = traces, colors=colors)


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
    def bkapp(doc):
        #print(doc.__dict__)
        #kwargs = doc._session_context
        #kwargs = doc._roots[0]
        # workaround since bokeh does not allow additional variables
        # for functions wrapped in show()
        #fname = kwargs.get('fname')
        cthr = bkapp_kwargs.get('cthr') if 'cthr' in bkapp_kwargs else 0.9

        estimates = LoadEstimates(fname, default_fps=default_fps)
        dims = estimates.imax.shape
        title = fname.rpartition('\\')[-1].partition('_estimates')[0]
        src = ColumnDataSource(data=EstimatesToSrc(estimates, cthr=cthr))

        tools1 = ["pan", "tap", "box_select", "zoom_in", "zoom_out", "reset"]
        tools2 = ["pan", "tap", "box_zoom", "zoom_in", "zoom_out", "reset"]
        color_mapper = LinearColorMapper(palette="Greys256", low=1, high=256)

        imwidth = 500
        trwidth = 500
        '''
        # TODO: fix resolution
        if 'pathway' in bkapp_kwargs:
            if bkapp_kwargs['pathway'] == 'bonsai':
                imwidth = 608
                trwidth = 608
        '''
        height = int(imwidth*dims[0]/dims[1])
        imdata = np.flip(estimates.imax, axis=0) # flip for reverting y-axis
        #imdata = estimates.imax
        #main plots, p1 is for image on the left, p2 is for traces on the right
        p1 = figure(width = imwidth, height = height, tools = tools1, toolbar_location = 'below', title = title)
        p1.image(image=[imdata], color_mapper=color_mapper, dh = dims[0], dw = dims[1], x=0, y=0)
        p2 = figure(width = trwidth, height = height, tools = tools2, toolbar_location = 'below')

        fill_alpha = bkapp_kwargs.get('fill_alpha') if 'fill_alpha' in bkapp_kwargs else 0.5
        nonselection_alpha = bkapp_kwargs.get('ns_alpha') if 'ns_alpha' in bkapp_kwargs else 0.2
        line_width = bkapp_kwargs.get('line_width') if 'line_width' in bkapp_kwargs else 2
        p1.patches('xs',
                   'ys',
                   fill_alpha = fill_alpha,
                   nonselection_alpha = nonselection_alpha,
                   color = 'colors',
                   selection_line_color="yellow",
                   line_width=line_width,
                   source=src)

        p2.multi_line('times',
                      'traces',
                      line_color='colors',
                      selection_line_width=line_width,
                      source=src)

        #this is for points addition
        pts_src = ColumnDataSource({'x': [], 'y': [], 'color': []})
        pts_renderer = p1.scatter(x='x', y='y', source=pts_src, color = 'color',  size=5)
        draw_tool = PointDrawTool(renderers=[pts_renderer], empty_value='yellow')
        p1.add_tools(draw_tool)

        #Button callbscks
        def del_callback(event):
            sel_inds = [src.selected.indices] if isinstance(src.selected.indices, int) else list(src.selected.indices)
            temp = estimates.idx_components_bad.tolist() + estimates.idx_components[sel_inds].tolist()
            estimates.idx_components_bad = np.sort(temp)
            estimates.idx_components = np.array([ind for i, ind in enumerate(estimates.idx_components) if i not in sel_inds])
            src.data = EstimatesToSrc(estimates)

        def merge_callback(event):
            sel_inds = [src.selected.indices] if isinstance(src.selected.indices, int) else list(src.selected.indices)
            if sel_inds:
                estimates.manual_merge([estimates.idx_components[sel_inds].tolist()], params = params.CNMFParams(params_dict = estimates.cnmf_dict))
                src.data = EstimatesToSrc(estimates)

        def seed_callback(event):
            seeds = [[pts_src.data['x']], [pts_src.data['y']]]
            seeds_fname = fname.partition('_estimates')[0] + '_seeds.pickle'
            with open(seeds_fname, "wb") as f:
                pickle.dump(seeds, f)
                print(f'Seeds saved to {seeds_fname}\n')

        def save_callback(event):
            SaveResults(estimates)
            print(f'Results for {title} saved in folder {os.path.dirname(fname)}\n')

        #Buttons themselves
        button_del = Button(label="Delete selected", button_type="success", width = 120)
        button_del.on_event('button_click', del_callback)

        button_merge = Button(label="Merge selected", button_type="success", width = 120)
        button_merge.on_event('button_click', merge_callback)

        button_seed = Button(label="Save seeds", button_type="success", width = 120)
        button_seed.on_event('button_click', seed_callback)

        button_save = Button(label="Save results", button_type="success", width = 120)
        button_save.on_event('button_click', save_callback)

        doc.add_root(column(row(button_del, button_merge, button_seed, button_save), row(p1, p2)))

    show(bkapp)


def ExamineCellsNew(fname, default_fps=20, bkapp_kwargs=None):
    #This is the main plotting functions which plots all images and traces and contains all button callbacks
    def bkapp(doc):
        #print(doc.__dict__)
        #kwargs = doc._session_context
        #kwargs = doc._roots[0]
        # workaround since bokeh does not allow additional variables
        # for functions wrapped in show()
        #fname = kwargs.get('fname')
        cthr = bkapp_kwargs.get('cthr') if 'cthr' in bkapp_kwargs else 0.9

        estimates = LoadEstimates(fname, default_fps=default_fps)
        dims = estimates.imax.shape
        title = fname.rpartition('\\')[-1].partition('_estimates')[0]
        est_data = EstimatesToSrc(estimates, cthr=cthr)
        src = ColumnDataSource(data=est_data)
        src_partial = ColumnDataSource(data=EstimatesToSrc(estimates, cthr=cthr))

        tools1 = ["pan", "tap", "box_select", "zoom_in", "zoom_out", "reset"]
        tools2 = ["pan", "tap", "box_zoom", "zoom_in", "zoom_out", "reset"]
        color_mapper = LinearColorMapper(palette="Greys256", low=1, high=256)

        imwidth = 500
        trwidth = 500
        '''
        # TODO: fix resolution
        if 'pathway' in bkapp_kwargs:
            if bkapp_kwargs['pathway'] == 'bonsai':
                imwidth = 608
                trwidth = 608
        '''
        height = int(imwidth*dims[0]/dims[1])
        imdata = np.flip(estimates.imax, axis=0)  # flip for reverting y-axis
        #imdata = estimates.imax
        #main plots, p1 is for image on the left, p2 is for traces on the right
        p1 = figure(width = imwidth, height = height, tools = tools1, toolbar_location = 'below', title = title)
        p1.image(image=[imdata], color_mapper=color_mapper, dh = dims[0], dw = dims[1], x=0, y=0)
        p2 = figure(width = trwidth, height = height, tools = tools2, toolbar_location = 'below')

        fill_alpha = bkapp_kwargs.get('fill_alpha') if 'fill_alpha' in bkapp_kwargs else 0.5
        nonselection_alpha = bkapp_kwargs.get('ns_alpha') if 'ns_alpha' in bkapp_kwargs else 0.2
        line_width = bkapp_kwargs.get('line_width') if 'line_width' in bkapp_kwargs else 2

        p1.patches('xs',
                   'ys',
                   fill_alpha = fill_alpha,
                   nonselection_alpha = nonselection_alpha,
                   color = 'colors',
                   selection_line_color="yellow",
                   line_width=line_width,
                   source=src_partial)

        p2.multi_line('times',
                      'traces',
                      line_color='colors',
                      selection_line_width=line_width,
                      source=src_partial)

        #this is for points addition
        pts_src = ColumnDataSource({'x': [], 'y': [], 'color': []})
        pts_renderer = p1.scatter(x='x', y='y', source=pts_src, color = 'color',  size=5)
        draw_tool = PointDrawTool(renderers=[pts_renderer], empty_value='yellow')
        p1.add_tools(draw_tool)

        #Button callbscks
        def del_callback(event):
            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(src_partial.selected.indices)
            temp = estimates.idx_components_bad.tolist() + estimates.idx_components[sel_inds].tolist()
            estimates.idx_components_bad = np.sort(temp)
            estimates.idx_components = np.array([ind for i, ind in enumerate(estimates.idx_components) if i not in sel_inds])
            src.data = EstimatesToSrc(estimates)

        def merge_callback(event):
            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(src_partial.selected.indices)
            if sel_inds:
                estimates.manual_merge([estimates.idx_components[sel_inds].tolist()],
                                       params = params.CNMFParams(params_dict = estimates.cnmf_dict))

                src.data = EstimatesToSrc(estimates)

        def show_callback(event):
            sel_inds = [src_partial.selected.indices] if isinstance(src_partial.selected.indices, int) else list(src_partial.selected.indices)
            if sel_inds:
                estimates_partial = copy.deepcopy(estimates)
                estimates_partial.idx_components = np.array([ind for i, ind in enumerate(estimates.idx_components) if i in sel_inds])
                src_partial.data = EstimatesToSrc(estimates_partial)

        def restore_callback(event):
            src_partial.data = src.data.copy()

        def revert_callback(event):
            src_partial.data = copy.deepcopy(est_data)
            src_partial.data = copy.deepcopy(est_data)

        def seed_callback(event):
            seeds = [[pts_src.data['x']], [pts_src.data['y']]]
            seeds_fname = fname.partition('_estimates')[0] + '_seeds.pickle'
            with open(seeds_fname, "wb") as f:
                pickle.dump(seeds, f)
                print(f'Seeds saved to {seeds_fname}\n')

        def save_callback(event):
            SaveResults(estimates)
            print(f'Results for {title} saved in folder {os.path.dirname(fname)}\n')

        #Buttons themselves
        button_del = Button(label="Delete selected", button_type="success", width = 120)
        button_del.on_event('button_click', del_callback, restore_callback)

        button_merge = Button(label="Merge selected", button_type="success", width = 120)
        button_merge.on_event('button_click', merge_callback, restore_callback)

        button_show = Button(label="Show selected", button_type="success", width = 120)
        button_show.on_event('button_click', show_callback)

        button_restore = Button(label="Reset view", button_type="success", width = 120)
        button_restore.on_event('button_click', restore_callback)

        button_revert = Button(label="Discard changes", button_type="success", width = 120)
        button_revert.on_event('button_click', revert_callback)

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
                    button_seed,
                    button_save,
                    button_revert
                ),
                row(p1, p2)
            )
        )

    show(bkapp)

    '''
    origins = bkapp_kwargs['origins']
    for origin in origins:
        print(origin[7:-1])
        os.environ["BOKEH_ALLOW_WS_ORIGIN"] = origin[7:-1]
        show(bkapp, notebook_url=origin[7:-1])
    '''
    '''
    origins = bkapp_kwargs['origins']
    url_ind = 0
    success = False
    while not success and url_ind < len(origins):
        try:
            print(f'Trying to connect through origin {origins[url_ind]}')
            os.environ["BOKEH_ALLOW_WS_ORIGIN"] = origins[url_ind][7:-1]
            x = show(bkapp)
            if x is not None:
                print('successfully connected')
                success = True
            else:
                print('connection error, switching...')
                url_ind += 1

        except Exception as e:
            print('failed:')
            print(repr(e))
            url_ind += 1

    if not success:
        print('Houston we have a problem')
    '''

'''
def ManualSeeds(fname, bkapp_kwargs=None):
    #This is the main plotting functions which plots all images and traces and contains all button callbacks
    def bkapp(doc):
        tools = ["pan", "tap", "box_select", "zoom_in", "zoom_out", "reset"]
        color_mapper = LinearColorMapper(palette="Greys256", low=1, high=256)

        imwidth = 500
        trwidth = 500

        height = int(imwidth*dims[0]/dims[1])
        imdata = np.flip(estimates.imax, axis=0)  # flip for reverting y-axis
        #imdata = estimates.imax
        #main plots, p1 is for image on the left, p2 is for traces on the right
        p1 = figure(width = imwidth, height = height, tools = tools1, toolbar_location = 'below', title = title)
        p1.image(image=[imdata], color_mapper=color_mapper, dh = dims[0], dw = dims[1], x=0, y=0)
        p2 = figure(width = trwidth, height = height, tools = tools2, toolbar_location = 'below')
        p3 = figure(width=trwidth, height=height, tools=tools2, toolbar_location='below')

        fill_alpha = bkapp_kwargs.get('fill_alpha') if 'fill_alpha' in bkapp_kwargs else 0.5
        nonselection_alpha = bkapp_kwargs.get('ns_alpha') if 'ns_alpha' in bkapp_kwargs else 0.2
        line_width = bkapp_kwargs.get('line_width') if 'line_width' in bkapp_kwargs else 2
        p1.patches('xs',
                   'ys',
                   fill_alpha = fill_alpha,
                   nonselection_alpha = nonselection_alpha,
                   color = 'colors',
                   selection_line_color="yellow",
                   line_width=line_width,
                   source=src)

        p2.multi_line('times',
                      'traces',
                      line_color='colors',
                      selection_line_width=line_width,
                      source=src)
                      #view=view)

        p3.multi_line('times',
                      'traces',
                      line_color='colors',
                      selection_line_width=line_width,
                      source=src_partial)

        #this is for points addition
        pts_src = ColumnDataSource({'x': [], 'y': [], 'color': []})
        pts_renderer = p1.scatter(x='x', y='y', source=pts_src, color = 'color',  size=5)
        draw_tool = PointDrawTool(renderers=[pts_renderer], empty_value='yellow')
        p1.add_tools(draw_tool)

        #Button callbscks
        def del_callback(event):
            sel_inds = [src.selected.indices] if isinstance(src.selected.indices, int) else list(src.selected.indices)
            temp = estimates.idx_components_bad.tolist() + estimates.idx_components[sel_inds].tolist()
            estimates.idx_components_bad = np.sort(temp)
            estimates.idx_components = np.array([ind for i, ind in enumerate(estimates.idx_components) if i not in sel_inds])
            src.data = EstimatesToSrc(estimates)

        def merge_callback(event):
            sel_inds = [src.selected.indices] if isinstance(src.selected.indices, int) else list(src.selected.indices)
            if sel_inds:
                estimates.manual_merge([estimates.idx_components[sel_inds].tolist()], params = params.CNMFParams(params_dict = estimates.cnmf_dict))
                src.data = EstimatesToSrc(estimates)

        def show_callback(event):
            sel_inds = [src.selected.indices] if isinstance(src.selected.indices, int) else list(src.selected.indices)
            if sel_inds:
                #view = CDSView(filter=IndexFilter(indices=sel_inds))
                estimates_partial = copy.deepcopy(estimates)
                estimates_partial.idx_components = np.array([ind for i, ind in enumerate(estimates.idx_components) if i in sel_inds])
                src_partial.data = EstimatesToSrc(estimates_partial)

        def seed_callback(event):
            seeds = [[pts_src.data['x']], [pts_src.data['y']]]
            seeds_fname = fname.partition('_estimates')[0] + '_seeds.pickle'
            with open(seeds_fname, "wb") as f:
                pickle.dump(seeds, f)
                print(f'Seeds saved to {seeds_fname}\n')

        def save_callback(event):
            SaveResults(estimates)
            print(f'Results for {title} saved in folder {os.path.dirname(fname)}\n')

        #Buttons themselves
        button_del = Button(label="Delete selected", button_type="success", width = 120)
        button_del.on_event('button_click', del_callback)

        button_merge = Button(label="Merge selected", button_type="success", width = 120)
        button_merge.on_event('button_click', merge_callback)

        button_show = Button(label="Show selected", button_type="success", width = 120)
        button_show.on_event('button_click', show_callback)

        #button_restore = Button(label="Reset view", button_type="success", width = 120)
        #button_show.on_event('button_click', restore_callback)

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
                    #button_restore,
                    button_seed,
                    button_save
                ),
                row(p1, p2, p3)
            )
        )

    show(bkapp)
'''