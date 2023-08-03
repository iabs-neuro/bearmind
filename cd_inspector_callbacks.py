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

opts_dict = {'fr': 20,
             'decay_time': 2,
             'p': 1,
             'nb': 2,
             'rf': None,
             'only_init': False,
             'gSig': (5, 5),
             'gSig': (21, 21),
             'ssub': 1,
             'tsub': 1,
             'merge_thr': 0.85,
             'method_init': 'corr_pnr',
             'K': None,
             'low_rank_background': None,           # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0
             'update_background_components': True,  # sometimes setting to False improve the results
             'min_corr': .8,                        # min peak value from correlation image
             'min_pnr': 10,                         # min peak to noise ration from PNR image
             'normalize_init': False,               # just leave as is
             'center_psf': True,                    # leave as is for 1 photon
             'ring_size_factor': 1.4,               # radius of ring is gSiz*ring_size_factor
             'del_duplicates': True,                # whether to remove duplicates from initialization      
            }

opts = params.CNMFParams(params_dict = opts_dict)

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

def LoadEstimates(name, fps =20):
    with open(name, "rb") as f:
        estimates = pickle.load(f,)
    estimates.name = name   
    if not hasattr(estimates, 'imax'):  #temporal hack; normally, imax should be loaded from image simultaneously with estimates
        estimates.imax = LoadImaxFromResults(estimates.name.partition('estimates')[0] + 'results.pickle')
    estimates.time = FindAndLoadTimestamp(estimates.name.partition('estimates')[0], estimates.C.shape[1])
    return estimates

def LoadImaxFromResults(name):
#!!!to be deprecated
    #load imax from *_results.pickle containing [rois, traces, contours, imax] 
    with open(name, "rb") as f:
        [_,_,_,imax] = pickle.load(f,)
    return (imax*255/np.max(imax)).astype('uint8')

def FindAndLoadTimestamp(name, n_frames, fps = 20):
    #try to load timestamp, in case of failure use constant fps
    tst = glob(name + '*_timestamp.csv')
    if not tst:
        return np.linspace(0, n_frames/fps, n_frames)
    else:
        time_s = np.genfromtxt(tst[0], delimiter = ',', skip_header = 1)[:,1]/1000
        return time_s[:n_frames]

def EstimatesToSrc(estimates):
    n_cells = len(estimates.idx_components)
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

def DrawFigures(estimates):
    dims = estimates.imax.shape
    title = estimates.name.rpartition('\\')[-1].partition('_estimates')[0]
    src = ColumnDataSource(data = EstimatesToSrc(estimates))
    tools1 = ["pan","tap","box_select","zoom_in","zoom_out","reset"]
    tools2 = ["pan","tap","box_zoom","zoom_in","zoom_out","reset"]
    color_mapper = LinearColorMapper(palette="Greys256", low=1, high=256)
    imwidth= 500
    trwidth = 500
    height = int(imwidth*dims[0]/dims[1])
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
    
    return p1, p2, src, pts_src


def DeleteSelected(estimates, sel_inds = []):
    if isinstance(sel_inds, int):
        sel_inds = [sel_inds]
    if isinstance(sel_inds, tuple):
        sel_inds = list(sel_inds)
    temp = estimates.idx_components_bad.tolist() + estimates.idx_components[sel_inds].tolist()
    estimates.idx_components_bad = np.sort(temp)
    estimates.idx_components = np.array([ind for i, ind in enumerate(estimates.idx_components) if i not in sel_inds])
    return estimates

def MergeSelected(estimates, sel_inds, opts):
    if not sel_inds or isinstance(sel_inds, int):
        return estimates
    if isinstance(sel_inds, tuple):
        sel_inds = list(sel_inds)
    estimates.manual_merge([estimates.idx_components[sel_inds].tolist()], params = opts)
    return estimates

def SeedContours(estimates, seeded_pts):
    seeds = np.concatenate((estimates.center, np.array(seeded_pts).T))
    estimates = DoCNMF(estimates.tif_name, estimates.cnmf_dict, seeds = seeds) 
    return estimates

def SaveResults(estimates, sigma = 3):
    with open(estimates.name.partition('.')[0] + '_final.pickle', "wb") as f:
        pickle.dump(estimates, f) #just in case, may be it's excessive
    #traces timestamping and writing
    stamped_traces = np.concatenate(([estimates.time], estimates.C[estimates.idx_components]), axis=0)
    pd.DataFrame(stamped_traces.T).to_csv(estimates.name.partition('estimates')[0] + 'traces_test.csv', index=False, header = ['time_s', *np.arange(len(estimates.idx_components))])

    #making directory and tiff writing
    fold = estimates.name.partition('estimates')[0] + 'filters_test'
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
    return



#### Some deprecated stuff may be potentially useful as snippets

def UpdateContours(src, fname, gSig = (5, 5), gSiz = (21, 21)):
    start = time()
    Ain = np.array([(frame.T.flatten('F') > np.max(frame)/2).astype(bool) for frame in src.data['rois']]).T
    #Ain = np.array([frame.T.flatten('F') for frame in src.data['rois']], dtype = bool).T
    opts_dict = {'fnames': [fname],
                 'fr': 20,
                 'decay_time': 2,
                 'p': 1,
                 'nb': 2,
                 'rf': None,
                 'only_init': False,
                 'gSig': gSig,
                 'gSig': gSiz,
                 'ssub': 1,
                 'tsub': 1,
                 'merge_thr': 0.85,
                 'method_init': 'corr_pnr',
                 'K': None,
                 'low_rank_background': None,           # None leaves background of each patch intact, True performs global low-rank approximation if gnb>0
                 'update_background_components': True,  # sometimes setting to False improve the results
                 'min_corr': .8,                        # min peak value from correlation image
                 'min_pnr': 10,                         # min peak to noise ration from PNR image
                 'normalize_init': False,               # just leave as is
                 'center_psf': True,                    # leave as is for 1 photon
                 'ring_size_factor': 1.4,               # radius of ring is gSiz*ring_size_factor
                 'del_duplicates': True,                # whether to remove duplicates from initialization      
                }

    opts = params.CNMFParams(params_dict = opts_dict)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    cnm = cm.source_extraction.cnmf.CNMF(n_processes, dview=dview, Ain=Ain, params=opts)
    mem_fname = cm.save_memmap([fname], base_name=str(int(time())), order='C', border_to_0=0, dview=dview)
    Yr, dims, T = cm.load_memmap(mem_fname)
    images = Yr.T.reshape((T,) + dims, order='F')
    cnm.fit(images)
    with open(fname[:-4] + '_upd_estimates.pickle', "wb") as f:
        pickle.dump(cnm.estimates, f)

    cm.stop_server(dview=dview)
    dview.terminate()

    conts = cm.utils.visualization.get_contours(cnm.estimates.A, dims=dims)
    conts = [cont["coordinates"][~np.isnan(cont["coordinates"]).any(axis=1)] for cont in conts]
    src.data['xs'] = [[pt[0] for pt in c] for c in conts]
    src.data['ys'] = [[pt[1] for pt in c] for c in conts]
    src.data['traces'] = [tr/np.max(tr) + i for i, tr in enumerate(cnm.estimates.C)]
    print(f'{fname} updated in {time()-start} s')
    return src    