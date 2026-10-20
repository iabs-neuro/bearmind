"""
FFV1 (lossless) vs MPEG-4 (as written by Bonsai) : effect of acquisition compression on CNMF-E results.

For every session recorded in FFV1 (UCLA Miniscope DAQ layout: <session>/<date>/<time>/Miniscope/N.avi):
  1. MPEG-4 copy of every avi, encoded exactly as Bonsai does it: OpenCV VideoWriter, FourCC FMP4
     (mpeg4 Simple Profile, GOP 12, no B-frames, fixed quantizer 3 = OpenCV qmin; verified on LT1/LNOF files).
  2. Both versions go through the same pipeline: avi -> tif (no crop) -> DoMotionCorrection (mc_dict of the
     notebook) -> DoCNMF (cnmf_dict of the notebook), same thresholds for both versions:
       'auto' : bm_qc mirror-FDR suggestion computed on the FFV1 movie,
       'user' : gSig 4, min_corr 0.9, min_pnr 7.
  3. QC (video + residual maps) for the 'auto' runs.
Everything is cached: re-running skips finished steps.

Usage (from the bearmind env, cwd with config.json):
    python run_compression_test.py D:\\Projects\\Test_FFV1 [session ...]
"""
import os
import sys
import json
import glob
import time

import cv2
import numpy as np
import pandas as pd
import tifffile as tfl

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)

from bm_batch_routines import DoMotionCorrection, DoCNMF  # noqa: E402
import bm_qc  # noqa: E402

VERSIONS = ('ffv1', 'mpeg4')
USER_THR = {'gSig': 4, 'min_corr': 0.9, 'min_pnr': 7}

MC_DICT = {
    'pw_rigid': False, 'max_shifts': (35, 35), 'gSig_filt': (8, 8), 'strides': (48, 48), 'overlaps': (24, 24),
    'max_deviation_rigid': 15, 'border_nan': 'copy', 'use_cuda': True, 'memory_fact': 192 / 16.0,
    'niter_rig': 1, 'splits_rig': 20, 'num_splits_to_process_rig': None, 'write_mp4': False,
}


def cnmf_dict(gsig, min_corr, min_pnr, fps):
    """cnmf_dict of the pipeline notebook (Module 3), incl. its fr = fps/2, tsub = 2 convention."""
    gsiz = gsig * 4 + 1
    return {'fr': fps / 2, 'decay_time': 1, 'method_init': 'corr_pnr', 'K': None,
            'gSig': (gsig, gsig), 'gSiz': (gsiz, gsiz), 'merge_thr': 0.8, 'p': 1, 'tsub': 2, 'ssub': 1,
            'rf': 40, 'stride': 25, 'only_init': True, 'nb': 0, 'nb_patch': 0, 'method_deconvolution': 'oasis',
            'low_rank_background': None, 'update_background_components': True,
            'min_corr': min_corr, 'min_pnr': min_pnr, 'normalize_init': False, 'center_psf': True,
            'ssub_B': 2, 'ring_size_factor': 1.5, 'del_duplicates': True, 'border_pix': 5,
            'min_SNR': 2.5, 'rval_thr': 0.95, 'use_cnn': False}


def log(*args):
    print(time.strftime('%H:%M:%S'), *args, flush=True)


def session_dirs(root):
    out = {}
    for d in sorted(glob.glob(os.path.join(root, '*', '*', '*', 'Miniscope'))):
        out[os.path.normpath(d).split(os.sep)[-4]] = d
    return out


def avi_list(d):
    return sorted(glob.glob(os.path.join(d, '*.avi')), key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))


def check_complete(d):
    meta = json.load(open(os.path.join(d, 'metaData.json')))
    ts = pd.read_csv(os.path.join(d, 'timeStamps.csv'))
    avis = avi_list(d)
    nums = [int(os.path.splitext(os.path.basename(f))[0]) for f in avis]
    expected = int(np.ceil(len(ts) / meta['framesPerFile']))
    ok = nums == list(range(expected))
    return ok, f'{len(avis)} avi (numbers {nums[0]}..{nums[-1]}), timestamps {len(ts)} -> expected {expected} files'


def get_fps(d):
    ts = pd.read_csv(os.path.join(d, 'timeStamps.csv'))
    return 1000. / np.median(np.diff(ts['Time Stamp (ms)'].values))


def encode_mpeg4(src, dst, fps):
    """MPEG-4 exactly as Bonsai/OpenCV writes it (FMP4, q=3, GOP 12)."""
    cap = cv2.VideoCapture(src)
    w = None
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if w is None:
            w = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*'FMP4'), fps, (im.shape[1], im.shape[0]), isColor=False)
        w.write(im[:, :, 0])
    w.release()


def avis_to_tif(avis, out_tif):
    frames = []
    for f in avis:
        cap = cv2.VideoCapture(f)
        while True:
            ok, im = cap.read()
            if not ok:
                break
            frames.append(im[:, :, 0])
    tfl.imwrite(out_tif, np.stack(frames), photometric='minisblack')
    return len(frames)


def clean_mmaps(folder):
    for f in glob.glob(os.path.join(folder, '*.mmap')):
        try:
            os.remove(f)
        except OSError:
            pass


def run_session(root, name, d):
    out_root = os.path.join(root, 'analysis')
    fps = get_fps(d)
    meta = json.load(open(os.path.join(d, 'metaData.json')))
    log(f'== {name}: fps {fps:.2f}')
    info = {'session': name, 'fps': fps}

    # 1. MPEG-4 copies
    mp_dir = os.path.join(out_root, 'mpeg4_avi', name)
    os.makedirs(mp_dir, exist_ok=True)
    for f in avi_list(d):
        dst = os.path.join(mp_dir, os.path.basename(f))
        if not os.path.exists(dst):
            encode_mpeg4(f, dst, meta['frameRate'])
    info['size_ffv1_MB'] = sum(os.path.getsize(f) for f in avi_list(d)) / 1e6
    info['size_mpeg4_MB'] = sum(os.path.getsize(f) for f in avi_list(mp_dir)) / 1e6
    log(f'sizes: FFV1 {info["size_ffv1_MB"]:.0f} MB, MPEG-4 {info["size_mpeg4_MB"]:.0f} MB')

    srcs = {'ffv1': avi_list(d), 'mpeg4': avi_list(mp_dir)}
    mc_tifs = {}
    for ver in VERSIONS:
        vdir = os.path.join(out_root, ver)
        os.makedirs(vdir, exist_ok=True)
        cr = os.path.join(vdir, f'{name}_CR.tif')
        mc = cr[:-4] + '_MC.tif'
        mc_tifs[ver] = mc
        # 2. tif + motion correction
        if not os.path.exists(mc):
            if not os.path.exists(cr):
                n = avis_to_tif(srcs[ver], cr)
                log(f'{ver}: {n} frames -> {cr}')
            DoMotionCorrection(cr, dict(MC_DICT))
            clean_mmaps(vdir)
            log(f'{ver}: motion corrected')

    # 3. thresholds from the FFV1 movie (same for both versions)
    thr_path = os.path.join(out_root, f'{name}_thresholds.json')
    if not os.path.exists(thr_path):
        thr = {}
        for ver in VERSIONS:
            maps = bm_qc.compute_init_maps(mc_tifs[ver], USER_THR['gSig'], tsub=2, max_mem_gb=16, verbose=False)
            sugg, grid = bm_qc.suggest_init_thresholds(maps)
            bm_qc.plot_threshold_suggestions(maps, sugg, grid, title=f'{name} {ver}',
                                             manual=(USER_THR['min_corr'], USER_THR['min_pnr']),
                                             out_png=os.path.join(out_root, f'{name}_{ver}_thresholds.png'))
            np.savez_compressed(os.path.join(out_root, f'{name}_{ver}_init_maps.npz'),
                                **{k: v for k, v in maps.items() if isinstance(v, np.ndarray)})
            thr[ver] = {'suggestions': sugg,
                        'user': bm_qc.evaluate_thresholds(maps, USER_THR['min_corr'], USER_THR['min_pnr'])}
        s = thr['ffv1']['suggestions']
        pick = s['mirror_fdr'] if s['mirror_fdr'].get('min_corr') is not None else s['robust_bg']
        thr['auto'] = {'gSig': USER_THR['gSig'], 'min_corr': pick['min_corr'], 'min_pnr': pick['min_pnr']}
        # the same pair evaluated on the MPEG-4 maps
        maps_mp = dict(np.load(os.path.join(out_root, f'{name}_mpeg4_init_maps.npz')), gSig=USER_THR['gSig'])
        thr['auto_on_mpeg4'] = bm_qc.evaluate_thresholds(maps_mp, pick['min_corr'], pick['min_pnr'])
        json.dump(thr, open(thr_path, 'w'), indent=2, default=float)
    thr = json.load(open(thr_path))
    log(f'thresholds auto: {thr["auto"]}, user: {USER_THR}')

    # 4. CNMF-E with equal parameters
    for tag, t in (('auto', thr['auto']), ('user', USER_THR)):
        for ver in VERSIONS:
            vdir = os.path.join(out_root, ver)
            out_name = os.path.join(vdir, f'{name}_{tag}_estimates.pickle')
            if os.path.exists(out_name):
                continue
            log(f'CNMF {ver} {tag}: {t}')
            DoCNMF(mc_tifs[ver], cnmf_dict(t['gSig'], t['min_corr'], t['min_pnr'], fps), out_name=out_name,
                   verbose=False, noise_ampl=1e-5, qc=(tag == 'auto'),
                   qc_kwargs={'fps': fps, 'max_mem_gb': 16, 'tmp_dir': vdir, 'verbose': False})
            clean_mmaps(vdir)
            log(f'CNMF {ver} {tag}: done')
    json.dump(info, open(os.path.join(out_root, f'{name}_info.json'), 'w'), indent=2, default=float)


if __name__ == '__main__':
    root = sys.argv[1]
    wanted = sys.argv[2:]
    for name, d in session_dirs(root).items():
        if wanted and name not in wanted:
            continue
        ok, msg = check_complete(d)
        if not ok:
            log(f'SKIP {name}: incomplete ({msg})')
            continue
        run_session(root, name, d)
    log('all done')
