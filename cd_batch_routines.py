#Stuff needed for plotting and widget callbacks
import caiman as cm
import pandas as pd
import numpy as np
import pickle
import shutil
import os
import gc
from bokeh.plotting import figure, show, output_notebook 
from bokeh.models import LinearColorMapper, CDSView, ColumnDataSource, Plot, CustomJS, Button, IndexFilter, PointDrawTool
from bokeh.layouts import column, row
from bokeh.io import push_notebook
from glob import glob
from moviepy.editor import VideoFileClip
from tifffile import imwrite
from caiman.source_extraction.cnmf import params
from caiman.motion_correction import MotionCorrect
from time import time
from scipy.ndimage import gaussian_filter
from scipy.io import savemat

import warnings
warnings.filterwarnings('ignore')

def CleanMemmaps(name):
    mmap_files = glob(name.partition('.')[0] + '*.mmap')
    for mm in mmap_files:
        os.remove(mm)
        
def DoCropAndRewrite(root, name):
    #find, crop and rewrite .avi files as well as timestamps
    start = time()
    with open(name, 'rb') as f:
        cr_dict = pickle.load(f,)
    avi_names = glob(os.path.dirname(name) + '\\*.avi')
    avi_names.sort(key = len) 
    splt_path = os.path.normpath(name).split(os.sep)
    out_fname = root + '_'.join(splt_path[-5:-2]) + '_CR.tif'
    whole_data = []
    for av_name in avi_names:
        clip = VideoFileClip(av_name)
        data = np.array([frame[cr_dict['UP']:,cr_dict['LEFT']:,0] for frame in clip.iter_frames()])
        if cr_dict['DOWN']:
            data = data[:,:-cr_dict['DOWN'],:]
        if cr_dict['RIGHT']:
            data = data[:,:,:-cr_dict['RIGHT']]
        whole_data.append(data[:-1])
        #cropping per se
    imwrite(out_fname, np.concatenate(whole_data, axis=0), photometric='minisblack')
    print('_'.join(splt_path[-5:-2]) + f' cropped in {time() - start:.1f}s')
    
    #Extract and copy timestamp files
    tst_name = os.path.dirname(name) + '\\timeStamps.csv'
    out_fname = root + '_'.join(splt_path[-5:-2]) + '_timestamp.csv'
    try:
        shutil.copy(tst_name, out_fname)
    except:
        print('Timestamp not found!')
        
def DoMotionCorrection(name, mc_dict):
    start = time()
    #start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)
    
    opts = params.CNMFParams(params_dict=mc_dict)
    mc = MotionCorrect([name], dview=dview, **opts.get_group('motion'))
    mc.motion_correct(save_movie=True)
    fname_mc = mc.fname_tot_els if mc.pw_rigid else mc.fname_tot_rig
    if mc.pw_rigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)), np.max(np.abs(mc.y_shifts_els)))).astype(np.uint8)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.uint8)
    mc.bord_px = 0 if mc.border_nan == 'copy' else bord_px
    mov = mc.apply_shifts_movie([name])
    
    imwrite(name[:-4] + '_MC.tif', np.array(mov, dtype='uint8'), photometric='minisblack')
    print(os.path.split(name)[-1] + f' motion corrected in {time() - start:.1f}s')
    
    cm.stop_server(dview=dview)
    dview.terminate()   

    
def DoCNMF(name, cnmf_dict): 
    start = time()
    #cnmf option setting
    opts = params.CNMFParams(params_dict = cnmf_dict)
    
    #start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
    if 'dview' in locals():
        cm.stop_server(dview=dview)
        dview.terminate()
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

    #tif loading to memory
    mem_fname = cm.save_memmap([name], base_name=name, order='C', border_to_0=0, dview=dview)
    Yr, dims, T = cm.load_memmap(mem_fname)
    images = Yr.T.reshape((T,) + dims, order='F')
    
    #cnmf itself
    cnm = cm.source_extraction.cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
    cnm.fit(images)
    cnm.estimates.evaluate_components(images, params=opts, dview=dview)
    
    #addition of some fields to estimates object
    cnm.estimates.tif_name = name
    cnm.estimates.cnmf_dict = cnmf_dict
    _, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=cnmf_dict['gSig'][0], swap_dim=False)
    cnm.estimates.imax = (pnr*255/np.max(pnr)).astype('uint8')
    
    #estimates object saving 
    with open(name[:-4] + '_estimates.pickle', "wb") as f:
        pickle.dump(cnm.estimates, f)
              
    #cluster termination
    cm.stop_server(dview=dview)
    dview.terminate()
    print(os.path.split(name)[-1] + f' cnmf-ed in {time() - start:.1f}s')
    
    return cnm.estimates

def ReDoCNMF(s_name, e_name):
    start = time()
    #seeds construction
    with open(s_name, "rb") as f:
        seeded_pts = pickle.load(f,)
    with open(e_name, "rb") as f:
        estimates = pickle.load(f,)
    old_pts = FindMaxima(estimates)
    seeds = np.concatenate((old_pts.astype(np.double), np.array(seeded_pts).T))
    seeds = np.flip(seeds, axis = 1)
  
    #parameter adaptation for seesed cnmf
    params_dict = estimates.cnmf_dict
    params_dict['min_corr'] = 0
    params_dict['min_pnr'] = 0
    params_dict['seed_method'] = seeds
    params_dict['init_iter'] = 1
    params_dict['rf'] = None
    opts = params.CNMFParams(params_dict = params_dict)
    
    #tif loading to memory
    mem_fname = cm.save_memmap([estimates.tif_name], base_name = estimates.tif_name, order='C', border_to_0=0)
    Yr, dims, T = cm.load_memmap(mem_fname)
    images = Yr.T.reshape((T,) + dims, order='F')

    #cnmf itself
    cnm = cm.source_extraction.cnmf.CNMF(n_processes=1, params=opts)
    cnm.fit(images)
    cnm.estimates.evaluate_components(images, params=opts)
       
    #addition of some fields to estimates object
    cnm.estimates.tif_name = estimates.tif_name
    cnm.estimates.cnmf_dict = estimates.cnmf_dict
    _, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=estimates.cnmf_dict['gSig'][0], swap_dim=False)
    cnm.estimates.imax = (pnr*255/np.max(pnr)).astype('uint8')
    
    #estimates object saving
    with open(e_name, "wb") as f:
        pickle.dump(cnm.estimates, f) 
        
    print(f'cnmf-ed in {time() - start:.1f}s')
    return cnm.estimates


def FindMaxima(estimates):
    pts = []
    for i,sp in enumerate(estimates.A.T[estimates.idx_components]):
        im = sp.reshape(estimates.imax.shape[::-1]).todense()
        pts.append([np.where(im == np.amax(im))[0][0], np.where(im == np.amax(im))[1][0]])
    return np.array(pts)
    