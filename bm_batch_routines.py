#Stuff needed for plotting and widget callbacks
import caiman as cm
import pandas as pd
import numpy as np
import pickle
import os
import shutil
import tifffile as tfl
import ipywidgets as ipw
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tqdm
from bokeh.plotting import figure, show, output_notebook 
from bokeh.models import LinearColorMapper, CDSView, ColumnDataSource, Plot, CustomJS, Button, IndexFilter, PointDrawTool
from bokeh.layouts import column, row
from bokeh.io import push_notebook
from glob import glob
from moviepy.editor import VideoFileClip
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
        
    
def DrawFrameAndBox(data, x, left, right, up, down, dpi=200):
    plt.figure(dpi=dpi)
    plt.imshow(data[x,:,:])
    plt.gca().add_patch(Rectangle((left, up), data.shape[1]-left-right, data.shape[2]-up-down, fill = None, ec = 'r', lw = 1))     
        
    
def LoadSelectedVideos(fnames):
    fnames.sort(key = len)
    video = []
    for name in fnames:
        clip = VideoFileClip(name)
        for frame in clip.iter_frames():
            video.append(frame[:,:,0])
    return np.asarray(video)


def DrawCropper(data, dpi=200):
    x_slider = ipw.IntSlider(value=1,  min=0, max=data.shape[0]-1, layout=ipw.Layout(width='100%'))
    l_slider = ipw.IntSlider(value=50, min=0, max=data.shape[1]-1)
    r_slider = ipw.IntSlider(value=50, min=0, max=data.shape[1]-1)
    u_slider = ipw.IntSlider(value=50, min=0, max=data.shape[2]-1)
    d_slider = ipw.IntSlider(value=50, min=0, max=data.shape[2]-1)
    
    def update_right(*args):
        r_slider.max = data.shape[1] - l_slider.value - 1
    l_slider.observe(update_right, 'value')
    
    def update_down(*args):
        d_slider.max = data.shape[2] - u_slider.value - 1
    u_slider.observe(update_down, 'value')
    
    w = ipw.interactive(DrawFrameAndBox,
                        data=ipw.fixed(data),
                        x=x_slider,
                        left=l_slider,
                        right=r_slider,
                        up=u_slider,
                        down=d_slider,
                        dpi=ipw.fixed(200))
    
    display(w)
    
    return w

def SaveCrops(root, left, right, up, down):

        save_name = os.path.join(root, 'cropping.pickle')
        save_name = os.path.normpath(save_name)
        cropping_dict = {
            "LEFT": left,
            "RIGHT": right,
            "UP": up,
            "DOWN": down
        }
        with open(save_name, "wb") as f:
            pickle.dump(cropping_dict, f)

        print(cropping_dict)
        print(f'Crops saved to {save_name}\n')


def DoCropAndRewrite(root, name, pathway='legacy'):
    #find, crop and rewrite .avi files as well as timestamps
    start = time()
    with open(name, 'rb') as f:
        cr_dict = pickle.load(f,)

    splt_path = os.path.normpath(name).split(os.sep)
    if pathway == 'legacy':
        out_fname = '_'.join(splt_path[-5:-2]) + '_CR.tif'
    elif pathway == 'bonsai':
        out_fname = splt_path[-2] + '_CR.tif'
    else:
        raise ValueError('Wrong pathway!')

    whole_data = []

    avi_names = glob(os.path.join(os.path.dirname(name), '*.avi'))
    avi_names.sort(key=len)

    for av_name in tqdm.tqdm(avi_names, position=0, leave=True):
        clip = VideoFileClip(av_name)
        data = np.array([frame[cr_dict['UP']:, cr_dict['LEFT']:, 0] for frame in clip.iter_frames()])
        if cr_dict['DOWN']:
            data = data[:, :-cr_dict['DOWN'], :]
        if cr_dict['RIGHT']:
            data = data[:, :, :-cr_dict['RIGHT']]
        whole_data.append(data[:-1])

    #  cropping per se
    out_fpath = os.path.join(root, out_fname)
    tfl.imwrite(out_fpath, np.concatenate(whole_data, axis=0), photometric='minisblack')
    print(f'{out_fname} cropped in {time() - start:.1f}s')


def extract_and_copy_ts(root, name, pathway='legacy'):
    splt_path = os.path.normpath(name).split(os.sep)

    #Extract and copy timestamp files
    if pathway == 'legacy':
        tst_name = os.path.join(os.path.dirname(name), 'timeStamps.csv')
        out_fname = root + '_'.join(splt_path[-5:-2]) + '_timestamp.csv'
    elif pathway == 'bonsai':
        folder_name = splt_path[-2]
        tst_name = os.path.join(os.path.dirname(name), folder_name + '_Mini_TS.csv')
        out_fname = os.path.join(root, folder_name + '_timestamp.csv')
    else:
        raise ValueError('Wrong pathway!')

    try:
        shutil.copy(tst_name, out_fname)
    except Exception as e:
        print('Problem with timestamps!')
        print(repr(e))


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
    
    tfl.imwrite(name[:-4] + '_MC.tif', np.array(mov, dtype='uint8'), photometric='minisblack')
    print(os.path.split(name)[-1] + f' motion corrected in {time() - start:.1f}s')
    
    cm.stop_server(dview=dview)
    dview.terminate()   

    
def DoCNMF(name, cnmf_dict): 
    start = time()
    #cnmf option setting
    opts = params.CNMFParams(params_dict=cnmf_dict)
    
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
	#normalization in the case of ssub != 1
    seeds = seeds / estimates.cnmf_dict['ssub'] 
  
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


def Test_gSig_Range(fname, default_gsig = 6, maxframes = np.Inf, step = 5):
    tlen = len(tfl.TiffFile(fname).pages)
    data = tfl.imread(fname, key = range(0, min(maxframes, tlen), step))
    
    def DrawPnrImage(data, gSig, dpi = 200):
        _, pnr = cm.summary_images.correlation_pnr(data, gSig=gSig, swap_dim=False)
        plt.figure(dpi = dpi)
        plt.imshow(pnr)
        
    w = ipw.interactive(DrawPnrImage, data = ipw.fixed(data), gSig = ipw.BoundedIntText(value=default_gsig, min=0), dpi = ipw.fixed(200))
    display(w)    
    