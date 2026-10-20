"""
Control for the FFV1 vs MPEG-4 comparison: how much do CNMF-E results change under a tiny perturbation that has
nothing to do with compression?  FFV1 motion-corrected movie + Gaussian noise (SD 0.5 DN, rounded to uint8;
about a third of the pixels change by +-1), then the same CNMF-E runs (same thresholds) and the same comparison
against the unperturbed FFV1 result.  If FFV1 vs FFV1+noise disagrees about as much as FFV1 vs MPEG-4, the
disagreement is CNMF-E instability, not compression.

Usage (bearmind env, cwd with config.json):
    python control_noise.py D:\\Projects\\Test_FFV1\\analysis RFC_F37_3D [more sessions]
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import tifffile as tfl

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run_compression_test import cnmf_dict, USER_THR, log, clean_mmaps  # noqa: E402
from compare_results import compare_pair, load  # noqa: E402
from bm_batch_routines import DoCNMF  # noqa: E402

NOISE_SD = 0.5


def perturb(src, dst, sd=NOISE_SD, chunk=500, seed=0):
    mov = tfl.memmap(src, mode='r')
    rng = np.random.default_rng(seed)

    def frames():
        for t0 in range(0, mov.shape[0], chunk):
            x = mov[t0:t0 + chunk].astype(np.float32) + rng.normal(0, sd, (min(chunk, mov.shape[0] - t0),) + mov.shape[1:])
            yield from np.clip(np.round(x), 0, 255).astype(np.uint8)

    tfl.imwrite(dst, frames(), shape=mov.shape, dtype=np.uint8, photometric='minisblack')


def main(analysis, names):
    cdir = os.path.join(analysis, 'ctrl_noise')
    os.makedirs(cdir, exist_ok=True)
    out_dir = os.path.join(analysis, 'report_ctrl')
    os.makedirs(out_dir, exist_ok=True)
    summaries = []
    for name in names:
        info = json.load(open(os.path.join(analysis, f'{name}_info.json')))
        thr = json.load(open(os.path.join(analysis, f'{name}_thresholds.json')))
        fps = info['fps']
        src = os.path.join(analysis, 'ffv1', f'{name}_CR_MC.tif')
        mc = os.path.join(cdir, f'{name}_CR_MC.tif')
        if not os.path.exists(mc):
            perturb(src, mc)
            log(f'{name}: perturbed movie written')
        dims = tfl.memmap(src, mode='r').shape[1:]
        for tag, t in (('auto', thr['auto']), ('user', USER_THR)):
            out_name = os.path.join(cdir, f'{name}_{tag}_estimates.pickle')
            if not os.path.exists(out_name):
                log(f'CNMF ctrl {name} {tag}: {t}')
                DoCNMF(mc, cnmf_dict(t['gSig'], t['min_corr'], t['min_pnr'], fps), out_name=out_name,
                       verbose=False, noise_ampl=1e-5)
                clean_mmaps(cdir)
            s, _ = compare_pair(load(os.path.join(analysis, 'ffv1', f'{name}_{tag}_estimates.pickle')),
                                load(out_name), dims, fps, name, tag, out_dir)
            summaries.append(s)
    df = pd.DataFrame(summaries)
    df.to_csv(os.path.join(out_dir, 'summary_ctrl.csv'), index=False)
    with pd.option_context('display.width', 250, 'display.max_columns', 50):
        print(df.T)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
