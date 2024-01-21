#Stuff needed for plotting and widget callbacks
import tifffile as tfl
import caiman as cm
import pandas as pd
import numpy as np
import pickle
import os
from bokeh.plotting import figure, show, output_notebook 
from bokeh.models import LinearColorMapper, CDSView, ColumnDataSource, Plot, CustomJS, Button, IndexFilter, PointDrawTool
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
        fps = get_fps(ts_df)
        return fps




def FindAndLoadTimestamp_deprecated(name, n_frames, fps = 20):
    #try to load timestamp, in case of failure use constant fps
    tst = glob(name + '*_timestamp.csv')
    if not tst:
        return np.linspace(0, n_frames/fps, n_frames)
    else:
        time_s = np.genfromtxt(tst[0], delimiter = ',', skip_header = 1)[:,1]/1000
        return time_s[:n_frames]


def EstimatesToSrc(estimates):
    n_cells = len(estimates.idx_components)
    if not n_cells:
        return {}
    traces = [tr/np.max(tr) + i for i, tr in enumerate(estimates.C[estimates.idx_components])]
    times = [estimates.time for i in range(n_cells)]
    colors = [colornum_Metro(i) for i in range(n_cells)] 
    cm_conts = cm.utils.visualization.get_contours(estimates.A, dims=estimates.imax.shape)
    contours = []
    for i in estimates.idx_components:
        coors = cm_conts[i]["coordinates"]
        contours.append(coors[~np.isnan(coors).any(axis=1)])
    xs = [[pt[0] for pt in c] for c in contours]
    ys = [[pt[1] for pt in c] for c in contours] 
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


def ExamineCells(fname, default_fps=20):
    #This is the main plotting functions which plots all images and traces and contains all button callbacks
    def bkapp(doc):
        estimates = LoadEstimates(fname, default_fps=default_fps)
        dims = estimates.imax.shape
        title = fname.rpartition('\\')[-1].partition('_estimates')[0]
        src = ColumnDataSource(data = EstimatesToSrc(estimates))
        tools1 = ["pan","tap","box_select","zoom_in","zoom_out","reset"]
        tools2 = ["pan","tap","box_zoom","zoom_in","zoom_out","reset"]
        color_mapper = LinearColorMapper(palette="Greys256", low=1, high=256)
        imwidth= 500
        trwidth = 500
        height = int(imwidth*dims[0]/dims[1])

        #main plots, p1 is for image on the left, p2 is for traces on the right
        p1 = figure(width = imwidth, height = height, tools = tools1, toolbar_location = 'below', title = title)
        p1.image(image=[estimates.imax], color_mapper=color_mapper, dh = dims[0], dw = dims[1], x=0, y=0)
        p2 = figure(width = trwidth,height = height, tools = tools2, toolbar_location = 'below')
        p1.patches('xs', 'ys', fill_alpha = 0.9, nonselection_alpha = 0.3, color = 'colors', selection_line_color="yellow", line_width=2, source = src)
        p2.multi_line('times', 'traces', line_color='colors', selection_line_width=2, source = src)
        
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
