#Stuff needed for plotting and widget callbacks
import caiman as cm
import pandas as pd
import numpy as np
import pickle
import os
import re
import shutil
import tifffile as tfl
import ipywidgets as ipw
from IPython.display import display
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tqdm
from bokeh.plotting import figure, show, output_notebook 
from bokeh.models import LinearColorMapper, CDSView, ColumnDataSource, Plot, CustomJS, Button, IndexFilter, PointDrawTool
from bokeh.layouts import column, row
from bokeh.io import push_notebook
from glob import glob
from moviepy.editor import VideoFileClip, concatenate_videoclips, ImageSequenceClip
from caiman.source_extraction.cnmf import params
from caiman.motion_correction import MotionCorrect
from time import time
from scipy.ndimage import gaussian_filter
from scipy.io import savemat

import warnings

from config import (CONFIG, read_config, get_mouse_config_path_from_fname,
                    update_config, get_session_name_from_path, get_session_config_path)


warnings.filterwarnings('ignore')


def CleanMemmaps(name):
    mmap_files = glob(name.partition('.')[0] + '*.mmap')
    for mm in mmap_files:
        os.remove(mm)


def DrawFrameAndBox(data, x, left, right, up, down, dpi=200, size=5, title=''):
    plt.figure(dpi=dpi, figsize=(size,size))
    plt.imshow(data[x,:,:])
    plt.title(title)
    plt.gca().add_patch(Rectangle((left, up), data.shape[2]-left-right, data.shape[1]-up-down, fill = None, ec = 'r', lw = 1))     


def LoadSelectedVideos(fnames, sort = True):
    if sort:
        fnames.sort(key=extract_number) #needed for proper sorting of Miniscope generated files like 1.avi, 22.avi etc
    video = []
    for name in fnames:
        if name[-4:] == '.avi':
            clip = VideoFileClip(name)
            for frame in clip.iter_frames():
                video.append(frame[:,:,0])
        elif name[-4:] == '.tif' or name[-5:] == '.tiff':
            with tfl.TiffFile(name) as tif:
                for page in tif.pages:
                    video.append(page.asarray())
        else:
            raise Exception('Unsupported input file type, only AVI and TIFF files allowed')
    return np.asarray(video)


def DrawCropper(data, dpi=200, fname=''):
    x_slider = ipw.IntSlider(value=1,  min=0, max=data.shape[0]-1, layout=ipw.Layout(width='100%'))
    l_slider = ipw.IntSlider(value=50, min=0, max=data.shape[1]-1)
    r_slider = ipw.IntSlider(value=50, min=0, max=data.shape[1]-1)
    u_slider = ipw.IntSlider(value=50, min=0, max=data.shape[2]-1)
    d_slider = ipw.IntSlider(value=50, min=0, max=data.shape[2]-1)
    s_slider = ipw.IntSlider(value=4, min=1, max=6)
    
    def update_right(*args):
        r_slider.max = data.shape[1] - l_slider.value - 1

    l_slider.observe(update_right, 'value')
    
    def update_down(*args):
        d_slider.max = data.shape[2] - u_slider.value - 1

    u_slider.observe(update_down, 'value')

    title = fname
    '''
    try:
        title = get_session_name_from_path(fname)
    except Exception:
        title = ''
    '''
    w = ipw.interactive(DrawFrameAndBox,
                        data=ipw.fixed(data),
                        x=x_slider,
                        left=l_slider,
                        right=r_slider,
                        up=u_slider,
                        down=d_slider,
                        size=s_slider,
                        dpi=ipw.fixed(dpi),
                        title=title)

    def on_load_button_clicked(b):
        with load_output:
            ms_config_name = get_mouse_config_path_from_fname(fname)
            session_config_name = get_session_config_path(title)
            crop_from_session_config = read_config(session_config_name).get('crop_params')

            if len(crop_from_session_config) != 0:
                l_slider.value = crop_from_session_config['LEFT']
                r_slider.value = crop_from_session_config['RIGHT']
                u_slider.value = crop_from_session_config['UP']
                d_slider.value = crop_from_session_config['DOWN']
                print(f'Crop loaded from session config {session_config_name}')

            else:
                print(f'Crop params not set in session config {session_config_name}!')
                crop_from_mouse_config = read_config(ms_config_name).get('crop_params')
                if len(crop_from_mouse_config) != 0:
                    l_slider.value = crop_from_mouse_config['LEFT']
                    r_slider.value = crop_from_mouse_config['RIGHT']
                    u_slider.value = crop_from_mouse_config['UP']
                    d_slider.value = crop_from_mouse_config['DOWN']
                    print(f'Crop loaded from mouse config {ms_config_name}')

                else:
                    print(f'Crop params not set in mouse config {ms_config_name}!')

    def on_session_config_button_clicked(b):
        with s_config_output:
            session_config_name = get_session_config_path(title)
            crop_to_config = {
                'crop_params': {
                    'LEFT': l_slider.value,
                    'RIGHT': r_slider.value,
                    'UP': u_slider.value,
                    'DOWN': d_slider.value
                }
            }

            update_config(crop_to_config, cpath=session_config_name)
            print(f'Session config {session_config_name} updated!')

    def on_mouse_config_button_clicked(b):
        with m_config_output:
            ms_config_name = get_mouse_config_path_from_fname(fname)
            crop_to_config = {
                'crop_params': {
                    'LEFT': l_slider.value,
                    'RIGHT': r_slider.value,
                    'UP': u_slider.value,
                    'DOWN': d_slider.value
                }
            }

            update_config(crop_to_config, cpath=ms_config_name)
            print(f'Session config {ms_config_name} updated!')

    def on_save_button_clicked(b):
        with save_output:
            SaveCrops(fname, w.kwargs['left'], w.kwargs['right'], w.kwargs['up'], w.kwargs['down'])

    common_layout = ipw.Layout(width='250px', height='40px')

    load_button = ipw.Button(description="Load crop from config", layout=common_layout)
    load_button.on_click(on_load_button_clicked)
    load_output = ipw.Output(layout=common_layout)

    s_config_button = ipw.Button(description="Save crop to session config", layout=common_layout)
    s_config_button.on_click(on_session_config_button_clicked)
    s_config_output = ipw.Output(layout=common_layout)

    m_config_button = ipw.Button(description="Save crop to mouse config", layout=common_layout)
    m_config_button.on_click(on_mouse_config_button_clicked)
    m_config_output = ipw.Output(layout=common_layout)

    save_button = ipw.Button(description="Save crop to file", layout=common_layout)
    save_button.on_click(on_save_button_clicked)
    save_output = ipw.Output(layout=common_layout)

    vbox1 = ipw.VBox([load_button, load_output])
    vbox2 = ipw.VBox([s_config_button, s_config_output])
    vbox3 = ipw.VBox([m_config_button, m_config_output])
    vbox4 = ipw.VBox([save_button, save_output])
    #display(ipw.HBox([vbox1, vbox2, vbox3, vbox4]))
    display(vbox4)
    display(w)
    
    return w


def SaveCrops(fname, left, right, up, down):
#    session_name = get_session_name_from_path(fname)
#    save_name = os.path.join(os.path.dirname(fname),
#                             session_name + f'_l={left}_r={right}_u={up}_d={down}'+'_cropping.pickle')

    save_name = os.path.normpath(base + f'_l={left}_r={right}_u={up}_d={down}'+'_cropping.pickle')
    cropping_dict = {
        "LEFT": left,
        "RIGHT": right,
        "UP": up,
        "DOWN": down
    }
    with open(save_name, "wb") as f:
        pickle.dump(cropping_dict, f)

    print('Crop params:', cropping_dict)
    print(f'Crops saved to {save_name}\n')


def get_file_num_id(name, pathway='bonsai'):
    if pathway == 'bonsai':
        num_id = name[:-4][-2:]
    elif pathway == 'legacy':
        num_id = name[:-4]
    else:
        raise Exception('Wrong pathway!')

    return int(num_id)

def extract_number(filename):
    match = re.search(r'(\d+)\.', os.path.basename(filename))
    if match:
        return int(match.group(1))
    return float('inf')

def DoCropAndRewrite(name, out_fname, sort = True):
    #find, crop and rewrite .avi files
    start = time()
    with open(name, 'rb') as f:
        cr_dict = pickle.load(f,)

    avi_names = glob(os.path.join(os.path.dirname(name), '*.avi'))
    if sort:
        avi_names.sort(key=extract_number)

    whole_data = []
    mp4_clips = []
    num_frames_whole = 0
    for av_name in tqdm.tqdm(avi_names, position=0, leave=True):

        clip = VideoFileClip(av_name)
        mp4_clips.append(clip)
        num_frames = clip.reader.nframes
        num_frames_whole = num_frames_whole + num_frames
        print(f"{av_name} - {num_frames} frames")
        
        #  cropping per se
        data = np.array([frame[cr_dict['UP']:, cr_dict['LEFT']:, 0] for frame in clip.iter_frames()])
        if cr_dict['DOWN']:
            data = data[:, :-cr_dict['DOWN'], :]
        if cr_dict['RIGHT']:
            data = data[:, :, :-cr_dict['RIGHT']]
        #whole_data.append(data[:-1])
        whole_data.append(data)
    mp4_clip = concatenate_videoclips(mp4_clips)


    tfl.imwrite(out_fname, np.concatenate(whole_data, axis=0), photometric='minisblack')
    mp4_clip.write_videofile(out_fname[:-4] + '.mp4')
    mp4_video_out = VideoFileClip(out_fname[:-4] + '.mp4')
    #tif_video_out = VideoFileClip(out_fpath)
    print(f"Original videos have {num_frames_whole} frames.mp4 video has {mp4_video_out.reader.nframes} frames.") # tif has {tif_video_out.reader.nframes} frames")
    print(f'{out_fname} cropped in {time() - start:.1f}s')

def extract_and_copy_ts(name):
    pathway = CONFIG['DATA_PATHWAY']
    root = CONFIG['ROOT']
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
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=14, single_thread=False)
    
    opts = params.CNMFParams(params_dict=mc_dict)

    mc = MotionCorrect([name], dview=dview, **opts.get_group('motion'))
    #mc = MotionCorrect([name], dview=None, **opts.get_group('motion'))
    print(f'Start of motion_correct {time() - start:.1f}s')
    mc.motion_correct(save_movie=True)
    print(f'End of motion_correct {time() - start:.1f}s')
    fname_mc = mc.fname_tot_els if mc.pw_rigid else mc.fname_tot_rig

    if mc.pw_rigid:
        bord_px = np.ceil(np.maximum(np.max(np.abs(mc.x_shifts_els)), np.max(np.abs(mc.y_shifts_els)))).astype(np.uint8)
    else:
        bord_px = np.ceil(np.max(np.abs(mc.shifts_rig))).astype(np.uint8)

    mc.bord_px = 0 if mc.border_nan == 'copy' else bord_px
    print(f'Start of apply_shifts_movie {time() - start:.1f}s')
    mov = mc.apply_shifts_movie([name])
    print(f'End of apply_shifts_movie {time() - start:.1f}s')
    tfl.imwrite(name[:-4] + '_MC.tif', np.array(mov, dtype='uint8'), photometric='minisblack')
    print(os.path.split(name)[-1] + f' motion corrected in {time() - start:.1f}s')

    # mp4-video creation
    tiff_frames = tfl.imread(name)
    frames_list = [np.stack((frame,) * 3, axis=-1) for frame in np.array(mov, dtype='uint8')]
    mp4_clip = ImageSequenceClip(frames_list, fps=30)
    mp4_clip.write_videofile(name[:-4] + '_MC.mp4', codec='libx264')
    
    cm.stop_server(dview=dview)
    dview.terminate()   

    
def DoCNMF(name, cnmf_dict, out_name=None, start_frame=None, end_frame=None, verbose=False):
    try:
        # cropping according to user preferences
        if start_frame is None:
            start_frame = 0
            sf_text = '--'
        else:
            sf_text = str(start_frame)

        if end_frame is None:
            end_frame = 10**10
            ef_text = '--'
        else:
            ef_text = str(end_frame)

        time_crop = slice(start_frame, end_frame)

        # output name construction
        frames_txt = f'_sf={sf_text}_ef={ef_text}'
        if out_name is None:
            out_name = name[:-4] + frames_txt + '_estimates.pickle'
        else:
            if '_estimates.pickle' not in out_name:
                out_name = out_name + frames_txt + '_estimates.pickle'

        start = time()
        #cnmf option setting
        opts = params.CNMFParams(params_dict=cnmf_dict)

        #start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
        if 'dview' in locals():
            cm.stop_server(dview=dview)
            dview.terminate()

        c, dview, n_processes = cm.cluster.setup_cluster(backend='local', n_processes=None, single_thread=False)

        if verbose:
            print('loading tif to memory...')
        # tif loading to memory
        mem_fname = cm.save_memmap([name],
                                   base_name=name[:-4],
                                   order='C',
                                   border_to_0=0,
                                   dview=dview,
                                   slices=[time_crop])
        print('cm.save_memmap ended...')
        Yr, dims, T = cm.load_memmap(mem_fname)
        print('cm.load_memmap ended...')
        images = Yr.T.reshape((T,) + dims, order='F')
        print('Yr.T.reshape ended...')

        if verbose:
            print('Performing CNMF...')
        # cnmf itself
        cnm = cm.source_extraction.cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
        print('source_extraction.cnmf')
        cnm.fit(images)
        print('cnm.fit')
        cnm.estimates.evaluate_components(images, params=opts, dview=dview)
        print('cnm.estimates.evaluate_components')

        if verbose:
            print('Computing imax...')
        #  addition of some fields to estimates object
        cnm.estimates.tif_name = name
        cnm.estimates.cnmf_dict = cnmf_dict
        _, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=cnmf_dict['gSig'][0], swap_dim=False)
        pnr[np.where(pnr == np.inf)] = 0
        pnr[np.isnan(pnr)] = 0
        pnr[np.where(pnr > 70)] = 70
        cnm.estimates.imax = (pnr*255/np.max(pnr)).astype('uint8')

        if verbose:
            print('Saving result...')
        #estimates object saving
        with open(out_name, "wb") as f:
            pickle.dump(cnm.estimates, f)

        #cluster termination
        cm.stop_server(dview=dview)
        dview.terminate()
        print(os.path.split(name)[-1] + f' cnmf-ed in {time() - start:.1f}s')

        return cnm.estimates

    except Exception as e:
        print(f'Problem with {out_name}, computation aborted:')
        print(repr(e))


def ReDoCNMF(s_name, e_name=None, cnmf_dict=None, tif_name=None):
    start = time()

    #seeds construction
    with open(s_name, "rb") as f:
        seeded_pts = pickle.load(f,)

    if e_name is not None:
        with open(e_name, "rb") as f:
            estimates = pickle.load(f,)
        old_pts = FindMaxima(estimates)
        seeds = np.concatenate((old_pts.astype(np.double), np.array(seeded_pts).T[:,0,:]))
    else:
        seeds = np.array(seeded_pts).T[:, 0, :]

    seeds = np.flip(seeds, axis = 1)

    if e_name is not None:
        params_dict = estimates.cnmf_dict
    else:
        params_dict = cnmf_dict.copy()

    # normalization in the case of ssub != 1
    seeds = seeds / params_dict['ssub']

    #  parameter adaptation for seeded cnmf


    opts = params.CNMFParams(params_dict = params_dict)
    params_dict['min_corr'] = 0
    params_dict['min_pnr'] = 0
    params_dict['seed_method'] = seeds
    params_dict['init_iter'] = 0
    params_dict['rf'] = None
    
    #tif loading to memory
    if tif_name is None:
        tif_name = estimates.tif_name
    mem_fname = cm.save_memmap([tif_name], base_name=tif_name, order='C', border_to_0=0)
    Yr, dims, T = cm.load_memmap(mem_fname)
    images = Yr.T.reshape((T,) + dims, order='F')

    #cnmf itself
    cnm = cm.source_extraction.cnmf.CNMF(n_processes=1, params=opts)
    cnm.fit(images)
    cnm.estimates.evaluate_components(images, params=opts)
       
    #addition of some fields to estimates object
    cnm.estimates.tif_name = tif_name
    #cnm.estimates.cnmf_dict = estimates.cnmf_dict
    cnm.estimates.cnmf_dict = params_dict.copy()
    _, pnr = cm.summary_images.correlation_pnr(images[::5], gSig=cnm.estimates.cnmf_dict['gSig'][0], swap_dim=False)
    pnr[np.where(pnr == np.inf)] = 0
    pnr[np.isnan(pnr)] = 0
    pnr[np.where(pnr > 70)] = 70
    cnm.estimates.imax = (pnr*255/np.max(pnr)).astype('uint8')
    
    #estimates object saving
    base_name = s_name.partition('_seeds')[0]
    out_name = base_name + '_redo_cnmf' + '_estimates.pickle'

    with open(out_name, "wb") as f:
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
        pnr[np.where(pnr == np.inf)] = 0
        pnr[np.isnan(pnr)] = 0

        plt.figure(dpi = dpi)
        plt.imshow(pnr)
        
    w = ipw.interactive(DrawPnrImage,
                        data = ipw.fixed(data),
                        gSig = ipw.BoundedIntText(value=default_gsig, min=0),
                        dpi = ipw.fixed(200))
    display(w)
