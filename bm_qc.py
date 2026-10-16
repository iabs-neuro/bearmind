"""
Quality-control tools for the 1p (CNMF-E) part of the pipeline.

1. Seed maps exactly as CNMF-E computes them + automatic suggestion of min_corr / min_pnr
   (compute_init_maps, count_seeds, suggest_init_thresholds, batch_suggest_thresholds).
2. Post-CNMF QC: 4-panel video (raw | raw - background | A*C | residual) and
   PNR / correlation maps of the residual, which show neurons the model missed (cnmf_qc).
3. dF/F (and CNMF-like spatially high-passed) videos for visual inspection of motion corrected movies
   (make_dff_video).

All heavy steps work on chunks (frames) or strips (rows), so full sessions do not have to fit in RAM.
Videos are for the eyes only: they are written with lossy H.264 and must never be used for analysis.
"""
import os
import json
import pickle
import tempfile

import cv2
import numpy as np
import tifffile as tfl
import tqdm
import matplotlib.pyplot as plt
import imageio_ffmpeg
from scipy.ndimage import percentile_filter

from caiman.source_extraction.cnmf.pre_processing import get_noise_fft
from caiman.summary_images import local_correlations_fft
from caiman.source_extraction.cnmf.utilities import decimation_matrix


# ----------------------------------------------------------------------------------------------------------------------
# movie access
# ----------------------------------------------------------------------------------------------------------------------

def open_movie(movie):
    """Return a (T, d1, d2) array-like without loading the whole movie into RAM (when possible)."""
    if isinstance(movie, (str, os.PathLike)):
        try:
            return tfl.memmap(str(movie), mode='r')
        except ValueError:  # compressed or non-contiguous tiff
            print(f'{movie}: cannot memory-map, loading into RAM')
            return tfl.imread(str(movie))
    return movie


def _tsub_mean(block, tsub):
    """Temporal block averaging, same as caiman downscale (last incomplete block averaged over existing frames)."""
    block = np.asarray(block, dtype=np.float32)
    if tsub == 1:
        return block
    T = block.shape[0]
    n_full = T // tsub
    out = block[:n_full * tsub].reshape((n_full, tsub) + block.shape[1:]).mean(1)
    if T % tsub:
        out = np.concatenate([out, block[n_full * tsub:].mean(0, keepdims=True)])
    return out


# ----------------------------------------------------------------------------------------------------------------------
# 1. CNMF-E seed maps and automatic thresholds
# ----------------------------------------------------------------------------------------------------------------------

def _psf_kernel(gSig):
    """Zero-mean gaussian kernel used by CNMF-E (center_psf=True, background_filter='disk')."""
    ksize = int(2 * gSig) * 2 + 1
    g = cv2.getGaussianKernel(ksize, gSig, cv2.CV_32F)
    psf = g.dot(g.T)
    ind_nonzero = psf >= psf[0].max()
    psf -= psf[ind_nonzero].mean()
    psf[~ind_nonzero] = 0
    return psf


def _maps_from_filtered(filt, thresh_init):
    """PNR and correlation images from spatially filtered, mean-subtracted data (T, h, w)."""
    T, h, w = filt.shape
    sn = get_noise_fft(filt.reshape(T, -1).T, noise_method='mean')[0].reshape(h, w)
    sn[sn <= 0] = np.inf
    pnr = filt.max(0) / sn
    tmp = filt.copy()
    tmp[tmp < thresh_init * sn] = 0
    cn = local_correlations_fft(tmp, swap_dim=False)
    return np.nan_to_num(cn).astype(np.float32), np.nan_to_num(pnr).astype(np.float32)


def compute_init_maps(movie, gSig, tsub=2, pre_tsub=1, frame_slice=None, thresh_init=2,
                      mirror=True, max_mem_gb=4, verbose=True):
    """
    Correlation and PNR images exactly as CNMF-E initialization (greedyROI_corr / init_neurons_corr_pnr) sees them.

    Differences from cm.summary_images.correlation_pnr used before:
      - the movie is block-averaged in time by tsub (as CNMF does), not decimated (every n-th frame);
      - the correlation image uses thresh_init*noise (CNMF uses 2), not 3;
      - CNMF compares PNR of the downsampled movie with min_pnr*sqrt(tsub). Here PNR is divided by sqrt(tsub),
        so the returned 'pnr' is directly comparable with min_pnr you put into cnmf_dict.

    Args:
        movie: path to tif or array-like (T, d1, d2)
        gSig: neuron half-size in pixels (as in cnmf_dict)
        tsub: temporal downsampling used in CNMF (cnmf_dict['tsub'])
        pre_tsub: if the movie was already averaged in time (e.g. residual memmap), its factor
        frame_slice: slice of frames to use (default: whole movie, as CNMF does)
        mirror: also compute maps of the sign-inverted movie. Calcium transients are positive, so peaks of the
            inverted movie are noise only; they give an empirical null distribution for both maps.
        max_mem_gb: approximate memory budget; the movie is processed in horizontal strips

    Returns:
        dict with 'cn', 'pnr' (and 'cn_neg', 'pnr_neg' if mirror), plus metadata
    """
    mov = open_movie(movie)
    T, d1, d2 = mov.shape
    fs = frame_slice if frame_slice is not None else slice(0, T)
    T_used = len(range(*fs.indices(T)))
    T_ds = int(np.ceil(T_used / tsub))

    psf = _psf_kernel(gSig)
    margin = psf.shape[0] // 2 + 1
    # ~6 float32 copies of a strip live at the same time (data, filtered, thresholded, fft buffers)
    rows = int(max_mem_gb * 1e9 / (6 * 4 * T_ds * d2)) - 2 * margin
    rows = int(np.clip(rows, 8, d1))

    out = {'cn': np.zeros((d1, d2), np.float32), 'pnr': np.zeros((d1, d2), np.float32),
           'mean': np.zeros((d1, d2), np.float32)}
    if mirror:
        out['cn_neg'] = np.zeros((d1, d2), np.float32)
        out['pnr_neg'] = np.zeros((d1, d2), np.float32)

    pnr_div = np.sqrt(tsub * pre_tsub)
    strips = range(0, d1, rows)
    for r0 in tqdm.tqdm(strips, desc='seed maps', disable=not verbose):
        r1 = min(d1, r0 + rows)
        a0, a1 = max(0, r0 - margin), min(d1, r1 + margin)
        blk = _tsub_mean(mov[fs, a0:a1, :], tsub)
        out['mean'][r0:r1] = blk[:, r0 - a0:r1 - a0].mean(0)
        filt = np.empty_like(blk)
        for t in range(blk.shape[0]):
            filt[t] = cv2.filter2D(blk[t], -1, psf, borderType=1)
        del blk
        filt -= filt.mean(0)

        cn, pnr = _maps_from_filtered(filt, thresh_init)
        out['cn'][r0:r1] = cn[r0 - a0:r1 - a0]
        out['pnr'][r0:r1] = pnr[r0 - a0:r1 - a0] / pnr_div
        if mirror:
            np.negative(filt, out=filt)
            cn, pnr = _maps_from_filtered(filt, thresh_init)
            out['cn_neg'][r0:r1] = cn[r0 - a0:r1 - a0]
            out['pnr_neg'][r0:r1] = pnr[r0 - a0:r1 - a0] / pnr_div
        del filt

    out.update({'gSig': gSig, 'tsub': tsub, 'pre_tsub': pre_tsub, 'n_frames': T_used,
                'thresh_init': thresh_init, 'pnr_units': 'min_pnr (already divided by sqrt(tsub))'})
    return out


def count_seeds(cn, pnr, min_corr, min_pnr, gSiz, bd=1, valid=None, min_dist=None):
    """
    Candidate seed pixels as in init_neurons_corr_pnr (local maxima of medianBlur(cn*pnr) above both thresholds).
    pnr must be in min_pnr units (see compute_init_maps).
    min_dist=None: all CNMF-E candidates (several per cell are common).
    min_dist=r: keep only candidates that are the strongest within r px, similar to CNMF-E removing the area of
                an initialized neuron; the count then approximates the number of initialized neurons.
    """
    d1, d2 = cn.shape
    v = (cn * pnr).astype(np.float32)
    v[(cn < min_corr) | (pnr < min_pnr)] = 0
    ind_search = v <= 0
    if valid is not None:
        ind_search |= ~valid
    if bd > 0:
        ind_search[:bd, :] = True
        ind_search[-bd:, :] = True
        ind_search[:, :bd] = True
        ind_search[:, -bd:] = True
    ii, jj = np.meshgrid(range(d2), range(d1))
    pixel_v = ((ii * 10 + jj) * 1e-5).astype(np.float32)
    v = cv2.medianBlur(v, 3) + pixel_v
    v[ind_search] = 0
    k = max(1, int(round(gSiz / 4.)))
    v_max = cv2.dilate(v, np.ones((k, k), np.uint8))
    seeds = (v == v_max) & (v >= min_corr * min_pnr) & ~ind_search
    if min_dist:
        r = int(min_dist)
        disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        seeds &= v == cv2.dilate(v, disk)  # strongest candidate within min_dist
    return np.argwhere(seeds)


def valid_mask(maps, border_pix=5, dark_frac=0.25):
    """
    Pixels usable for seeds: not within border_pix of the edge and not in dark regions (outside the lens,
    black corners after cropping/motion correction), where PNR/correlation are meaningless.
    """
    mean = maps.get('mean')
    valid = np.ones(maps['cn'].shape, bool)
    if mean is not None:
        dark = mean < dark_frac * np.median(mean)
        k = 2 * int(2 * maps['gSig']) + 1
        dark = cv2.dilate(dark.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
        valid &= ~dark
    if border_pix > 0:
        valid[:border_pix, :] = valid[-border_pix:, :] = False
        valid[:, :border_pix] = valid[:, -border_pix:] = False
    return valid


def _robust(x):
    """median and robust SD (1.4826 * MAD)"""
    med = np.median(x)
    return med, 1.4826 * np.median(np.abs(x - med))


def halo_mask(maps, valid, excl_radius, k_pnr=5.0, k_corr=3.0):
    """
    Neighbourhood (excl_radius) of clearly active cells. The zero-mean CNMF-E kernel turns real transients into
    a negative halo around cells, so the inverted movie is not a clean null there. The mask is fixed
    (independent of the tested thresholds) so that null counts stay comparable across the threshold grid.
    """
    mc, sc = _robust(maps['cn'][valid])
    mp, sp = _robust(maps['pnr'][valid])
    strong = valid & (maps['pnr'] > mp + k_pnr * sp) & (maps['cn'] > mc + k_corr * sc)
    r = int(excl_radius)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(strong.astype(np.uint8), disk).astype(bool)


def threshold_seeds(maps, min_corr, min_pnr, gSiz=None, valid=None, halo=None):
    """
    Real seeds and (if mirror maps exist) null seeds for a threshold pair, at most one per neuron-sized
    neighbourhood (min_dist = 2*gSig), so counts approximate numbers of initialized neurons.
    Null seeds are searched outside the halo mask; returns (seeds, null_seeds, area_factor), where
    area_factor rescales the null count to the whole valid area.
    """
    gSiz = gSiz if gSiz is not None else 4 * maps['gSig'] + 1
    min_dist = 2 * maps['gSig']
    valid = valid if valid is not None else valid_mask(maps)
    s = count_seeds(maps['cn'], maps['pnr'], min_corr, min_pnr, gSiz, valid=valid, min_dist=min_dist)
    if 'cn_neg' not in maps:
        return s, None, 1.0
    halo = halo if halo is not None else halo_mask(maps, valid, gSiz)
    null_area = valid & ~halo
    s0 = count_seeds(maps['cn_neg'], maps['pnr_neg'], min_corr, min_pnr, gSiz, valid=null_area,
                     min_dist=min_dist)
    return s, s0, valid.sum() / max(null_area.sum(), 1)


def evaluate_thresholds(maps, min_corr, min_pnr, gSiz=None, valid=None, halo=None):
    """Number of seeds, null seeds and estimated FDR for one (min_corr, min_pnr) pair, e.g. picked by eye."""
    s, s0, area_factor = threshold_seeds(maps, min_corr, min_pnr, gSiz, valid, halo)
    res = {'min_corr': round(float(min_corr), 3), 'min_pnr': round(float(min_pnr), 2), 'n_seeds': len(s)}
    if s0 is not None:
        n0 = len(s0) * area_factor
        res.update({'n_null_seeds': round(float(n0), 1), 'fdr': round(n0 / max(len(s), 1), 3)})
    return res


def suggest_init_thresholds(maps, gSiz=None, target_fdr=0.1, marginal_fdr=0.5, border_pix=5, excl_radius=None,
                            corr_grid=None, pnr_grid=None,
                            bg_k=3.0, pct=95, pct_pnr_margin=1.0, pct_corr_margin=0.02):
    """
    Automatic min_corr / min_pnr suggestions, three independent methods:

    'robust_bg'  : background level + bg_k robust SD (median + k*1.4826*MAD) of each map.
                   Background pixels dominate the field of view, so the median/MAD describe the noise floor.
    'pct_margin' : pct-th percentile of the map + fixed margin (simple rule of thumb).
    'mirror_fdr' : for every (min_corr, min_pnr) on a grid, count seeds in the real maps and in the maps of the
                   sign-inverted movie (null: inverted movie has no calcium transients). Estimated false discovery
                   rate = n_null / n_real. Picks the most permissive pair with FDR <= target_fdr, provided that
                   loosening to it from the neighbouring stricter grid point adds seeds of which at most
                   marginal_fdr are null-like.
                   Null seeds are not counted within excl_radius of clearly active cells (halo_mask): the zero-mean
                   CNMF-E kernel turns real transients into a negative halo around cells. The null count is
                   rescaled to the whole valid area.
                   Caveat: slow drifts and dark moving structures (vessels) also produce negative peaks,
                   which makes this estimate conservative.
    Dark regions and border_pix at the edges are excluded (valid_mask), as CNMF does with border_pix.

    Returns dict {method: {'min_corr', 'min_pnr', 'n_seeds', 'n_null_seeds', 'fdr'}} + the FDR grid.
    """
    cn, pnr = maps['cn'], maps['pnr']
    gSiz = gSiz if gSiz is not None else 4 * maps['gSig'] + 1
    mirror = 'cn_neg' in maps
    valid = valid_mask(maps, border_pix=border_pix)
    halo = halo_mask(maps, valid, excl_radius if excl_radius is not None else gSiz) if mirror else None
    cn_i, pnr_i = cn[valid], pnr[valid]

    def summary(min_corr, min_pnr):
        return evaluate_thresholds(maps, float(np.clip(min_corr, 0, 0.99)), float(max(min_pnr, 0)),
                                   gSiz, valid, halo)

    out = {}
    m_c, s_c = _robust(cn_i)
    m_p, s_p = _robust(pnr_i)
    out['robust_bg'] = summary(m_c + bg_k * s_c, m_p + bg_k * s_p)
    out['pct_margin'] = summary(np.percentile(cn_i, pct) + pct_corr_margin,
                                np.percentile(pnr_i, pct) + pct_pnr_margin)

    grid = None
    if mirror:
        corr_grid = np.round(np.arange(0.5, 0.981, 0.02), 3) if corr_grid is None else np.asarray(corr_grid)
        pnr_grid = np.round(np.arange(2, 20.01, 0.5), 2) if pnr_grid is None else np.asarray(pnr_grid)
        n_real = np.zeros((len(corr_grid), len(pnr_grid)), int)
        n_null = np.zeros(n_real.shape)
        for i, c in enumerate(corr_grid):
            for j, p in enumerate(pnr_grid):
                s, s0, area_factor = threshold_seeds(maps, c, p, gSiz, valid, halo)
                n_real[i, j], n_null[i, j] = len(s), len(s0) * area_factor
        fdr = n_null / np.maximum(n_real, 1)
        # marginal FDR: share of null among the seeds added by loosening one threshold by one grid step
        # (stops the global criterion from buying a few extra cells with a pile of noise, e.g. at vignetting edges)
        mfdr = np.zeros_like(fdr)
        for ax_ in (0, 1):
            d_real = n_real - np.roll(n_real, -1, axis=ax_)
            d_null = n_null - np.roll(n_null, -1, axis=ax_)
            m = np.where(d_real > 0, d_null / np.maximum(d_real, 1), 0)
            if ax_ == 0:
                m[-1, :] = 0
            else:
                m[:, -1] = 0
            mfdr = np.maximum(mfdr, m)
        # loosening must be admissible all the way from the strict corner: once a step adds mostly noise,
        # everything looser along that threshold is excluded too
        ok_path = ((fdr <= target_fdr) | (n_real == 0)) & (mfdr <= marginal_fdr)
        for i in range(len(corr_grid) - 2, -1, -1):
            ok_path[i, :] &= ok_path[i + 1, :]
        for j in range(len(pnr_grid) - 2, -1, -1):
            ok_path[:, j] &= ok_path[:, j + 1]
        ok = ok_path & (fdr <= target_fdr) & (n_real > 0)
        if ok.any():
            # most permissive admissible pair = max number of real seeds; ties -> stricter thresholds
            ii, jj = np.meshgrid(np.arange(len(corr_grid)), np.arange(len(pnr_grid)), indexing='ij')
            score = np.where(ok, n_real - fdr + 1e-6 * (ii + jj), -np.inf)
            i, j = np.unravel_index(np.argmax(score), score.shape)
            out['mirror_fdr'] = summary(corr_grid[i], pnr_grid[j])
        else:
            out['mirror_fdr'] = {'min_corr': None, 'min_pnr': None, 'n_seeds': 0,
                                 'note': f'no threshold pair reaches FDR <= {target_fdr}'}
        out['mirror_fdr']['target_fdr'] = target_fdr
        grid = {'corr_grid': corr_grid, 'pnr_grid': pnr_grid, 'n_real': n_real, 'n_null': n_null, 'fdr': fdr,
                'marginal_fdr': mfdr}

    return out, grid


def plot_threshold_suggestions(maps, suggestions, grid=None, title='', gSiz=None, out_png=None, dpi=150,
                               manual=None):
    """Summary figure: maps, mirror maps, histograms with thresholds, seed overlays, FDR grid."""
    cn, pnr = maps['cn'], maps['pnr']
    gSiz = gSiz if gSiz is not None else 4 * maps['gSig'] + 1
    valid = valid_mask(maps)
    halo = halo_mask(maps, valid, gSiz) if 'cn_neg' in maps else None
    methods = [m for m in suggestions if suggestions[m].get('min_corr') is not None]
    if manual is not None:
        methods = methods + ['manual']
        suggestions = dict(suggestions, manual=evaluate_thresholds(maps, *manual, gSiz, valid, halo))
    colors = {'robust_bg': 'tab:orange', 'pct_margin': 'tab:green', 'mirror_fdr': 'tab:red', 'manual': 'tab:cyan'}

    ncols = max(3, len(methods))
    fig, axs = plt.subplots(3, ncols, figsize=(5 * ncols, 14), dpi=dpi, squeeze=False)
    vmax = np.percentile(pnr[valid], 99.9)

    axs[0, 0].imshow(pnr, cmap='magma', vmin=0, vmax=vmax)
    axs[0, 0].set_title('PNR (min_pnr units)')
    axs[0, 1].imshow(cn, cmap='magma', vmin=0, vmax=1)
    axs[0, 1].set_title('Correlation (as in CNMF init)')
    if 'pnr_neg' in maps:
        axs[0, 2].imshow(maps['pnr_neg'], cmap='magma', vmin=0, vmax=vmax)
        axs[0, 2].set_title('PNR of inverted movie (null), same scale')

    ax = axs[1, 0]
    bins = np.linspace(0, vmax, 120)
    ax.hist(pnr[valid], bins=bins, histtype='step', color='k', label='movie', log=True)
    if 'pnr_neg' in maps:
        ax.hist(maps['pnr_neg'][valid], bins=bins, histtype='step', color='gray', ls='--', label='inverted', log=True)
    for m in methods:
        ax.axvline(suggestions[m]['min_pnr'], color=colors[m], label=m)
    ax.set_xlabel('PNR')
    ax.legend(fontsize=8)

    ax = axs[1, 1]
    bins = np.linspace(0, 1, 120)
    ax.hist(cn[valid], bins=bins, histtype='step', color='k', label='movie', log=True)
    if 'cn_neg' in maps:
        ax.hist(maps['cn_neg'][valid], bins=bins, histtype='step', color='gray', ls='--', label='inverted', log=True)
    for m in methods:
        ax.axvline(suggestions[m]['min_corr'], color=colors[m], label=m)
    ax.set_xlabel('correlation')
    ax.legend(fontsize=8)

    if grid is not None:
        ax = axs[1, 2]
        im = ax.imshow(np.clip(grid['fdr'], 0, 1), origin='lower', aspect='auto', cmap='RdYlGn_r', vmin=0, vmax=0.5,
                       extent=[grid['pnr_grid'][0], grid['pnr_grid'][-1], grid['corr_grid'][0], grid['corr_grid'][-1]])
        ax.contour(grid['pnr_grid'], grid['corr_grid'], grid['n_real'], levels=[10, 30, 100, 300, 1000],
                   colors='k', linewidths=0.5)
        for m in methods:
            ax.plot(suggestions[m]['min_pnr'], suggestions[m]['min_corr'], 'o', color=colors[m], mec='k')
        ax.set_xlabel('min_pnr')
        ax.set_ylabel('min_corr')
        ax.set_title('estimated FDR (color), n seeds (contours)')
        plt.colorbar(im, ax=ax, fraction=0.046)

    for k in range(3, ncols):
        axs[0, k].axis('off')
        axs[1, k].axis('off')

    v_img = cn * pnr
    for k in range(ncols):
        ax = axs[2, k]
        if k >= len(methods):
            ax.axis('off')
            continue
        m = methods[k]
        s = suggestions[m]
        seeds, null, _ = threshold_seeds(maps, s['min_corr'], s['min_pnr'], gSiz, valid, halo)
        ax.imshow(v_img, cmap='gray', vmin=0, vmax=np.percentile(v_img[valid], 99.9))
        ax.contour(valid, levels=[0.5], colors='y', linewidths=0.4)
        if halo is not None:
            ax.contour(halo, levels=[0.5], colors='tab:blue', linewidths=0.4)
        if len(seeds):
            ax.plot(seeds[:, 1], seeds[:, 0], 'o', mfc='none', ms=4, mew=0.8, color=colors[m])
        if null is not None and len(null):
            ax.plot(null[:, 1], null[:, 0], 'x', ms=3, color='w', alpha=0.7)
        txt = f"{m}: corr {s['min_corr']}, pnr {s['min_pnr']}\n{len(seeds)} seeds"
        if 'fdr' in s:
            txt += f", null (x) {s['n_null_seeds']}, est. FDR {s['fdr']}"
        ax.set_title(txt, fontsize=9)

    for ax in axs[0, :3]:
        ax.axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    if out_png is not None:
        fig.savefig(out_png, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
    return fig


def batch_suggest_thresholds(fnames, gsig_range=(4,), tsub=2, frame_slice=None, target_fdr=0.1,
                             out_dir=None, manual=None, max_mem_gb=4, save_maps=True, verbose=True):
    """
    Suggest CNMF-E thresholds for every movie and every gSig. Saves per-movie figures/json into out_dir
    (default: <movie folder>/param_images) and returns a summary DataFrame (also saved as csv).

    manual: optional dict {session_name_or_fname_part: (min_corr, min_pnr)} with your eye-picked values,
            they are drawn on the figures and their seed counts / FDR are added to the table.
    """
    import pandas as pd
    rows = []
    for fname in fnames:
        base = os.path.splitext(os.path.basename(fname))[0]
        odir = out_dir or os.path.join(os.path.dirname(fname), 'param_images')
        os.makedirs(odir, exist_ok=True)
        man = None
        if manual:
            for key, val in manual.items():
                if key in base:
                    man = tuple(val)
        for gsig in gsig_range:
            if verbose:
                print(f'{base}: gSig={gsig}')
            maps = compute_init_maps(fname, gsig, tsub=tsub, frame_slice=frame_slice,
                                     max_mem_gb=max_mem_gb, verbose=verbose)
            sugg, grid = suggest_init_thresholds(maps, target_fdr=target_fdr)
            tag = f'{base}_gSig={gsig}_tsub={tsub}'
            plot_threshold_suggestions(maps, sugg, grid, title=tag, manual=man,
                                       out_png=os.path.join(odir, tag + '_auto_thresholds.png'))
            if save_maps:
                np.savez_compressed(os.path.join(odir, tag + '_init_maps.npz'),
                                    **{k: v for k, v in maps.items() if isinstance(v, np.ndarray)})
            with open(os.path.join(odir, tag + '_auto_thresholds.json'), 'w') as f:
                json.dump({'movie': fname, 'gSig': gsig, 'tsub': tsub, 'n_frames': maps['n_frames'],
                           'suggestions': sugg}, f, indent=2, default=float)

            for m, s in sugg.items():
                rows.append({'movie': base, 'gSig': gsig, 'method': m, **{k: s.get(k) for k in
                             ('min_corr', 'min_pnr', 'n_seeds', 'n_null_seeds', 'fdr')}})
            if man is not None:
                rows.append({'movie': base, 'gSig': gsig, 'method': 'manual', **evaluate_thresholds(maps, *man)})
            del maps

    df = pd.DataFrame(rows)
    if len(fnames):
        csv_dir = out_dir or os.path.join(os.path.dirname(fnames[0]), 'param_images')
        df.to_csv(os.path.join(csv_dir, 'auto_thresholds_summary.csv'), index=False)
    return df


# ----------------------------------------------------------------------------------------------------------------------
# video writing helpers
# ----------------------------------------------------------------------------------------------------------------------

def _lut(cmap):
    """uint8 -> RGB lookup table for a matplotlib colormap."""
    return (plt.get_cmap(cmap)(np.arange(256))[:, :3] * 255).astype(np.uint8)


def _to_rgb(img, vmin, vmax, lut):
    x = (np.clip((img - vmin) / max(vmax - vmin, 1e-9), 0, 1) * 255).astype(np.uint8)
    return lut[x]


def _label(rgb, text, y=18):
    y = y if y >= 0 else rgb.shape[0] + y
    cv2.putText(rgb, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(rgb, text, (6, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return rgb


def _grid(panels, ncols):
    h, w = panels[0].shape[:2]
    nrows = int(np.ceil(len(panels) / ncols))
    H, W = nrows * h + 2 * (nrows - 1), ncols * w + 2 * (ncols - 1)
    H, W = H + H % 2, W + W % 2  # yuv420p needs even dimensions
    canvas = np.full((H, W, 3), 40, np.uint8)
    for k, p in enumerate(panels):
        r, c = divmod(k, ncols)
        canvas[r * (h + 2):r * (h + 2) + h, c * (w + 2):c * (w + 2) + w] = p
    return canvas


class _VideoWriter:
    def __init__(self, path, size_wh, fps, crf=18):
        self.gen = imageio_ffmpeg.write_frames(path, size_wh, fps=fps, codec='libx264', macro_block_size=1,
                                               pix_fmt_out='yuv420p', output_params=['-crf', str(crf)])
        self.gen.send(None)

    def write(self, rgb):
        self.gen.send(np.ascontiguousarray(rgb))

    def close(self):
        self.gen.close()


def _sample_frames(T, n=40, span=None):
    span = span if span is not None else (0, T)
    return np.unique(np.linspace(span[0], span[1] - 1, n).astype(int))


# ----------------------------------------------------------------------------------------------------------------------
# 2. post-CNMF QC: video + residual maps
# ----------------------------------------------------------------------------------------------------------------------

def _to_dense(x):
    return x.toarray() if hasattr(x, 'toarray') else np.asarray(x)


def _footprint_contours(A, idx, dims, thr=0.3):
    conts = []
    for i in idx:
        a = _to_dense(A[:, i]).reshape(dims, order='F')
        if a.max() <= 0:
            continue
        m = (a >= thr * a.max()).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        conts.extend(cs)
    return conts


class _ModelChunker:
    """Computes Y, B, A*C (selected) and residual for chunks of frames from a CaImAn estimates object."""

    def __init__(self, est, mov, components='accepted'):
        self.mov = mov
        T_mov, d1, d2 = mov.shape
        self.dims = (d1, d2)
        self.A = est.A.tocsc().astype(np.float32)
        if self.A.shape[0] != d1 * d2:
            raise ValueError(f'Footprints have {self.A.shape[0]} pixels, movie frame has {d1 * d2}: wrong movie?')
        self.C = np.asarray(est.C, dtype=np.float32)
        self.T = self.C.shape[1]
        if self.T > T_mov:
            raise ValueError(f'Estimates have {self.T} frames, movie only {T_mov}')

        n = self.A.shape[1]
        if components == 'accepted' and getattr(est, 'idx_components', None) is not None:
            self.sel = np.asarray(est.idx_components, dtype=int)
        elif components == 'all' or getattr(est, 'idx_components', None) is None:
            self.sel = np.arange(n)
        else:
            self.sel = np.asarray(components, dtype=int)
        self.A_sel = self.A[:, self.sel]
        self.C_sel = self.C[self.sel]

        self.W = getattr(est, 'W', None)
        self.b0 = getattr(est, 'b0', None)
        self.b, self.f = getattr(est, 'b', None), getattr(est, 'f', None)
        self.ds_mat, self.ssub_B = None, 1
        if self.W is not None:
            self.W = self.W.tocsr().astype(np.float32)
            self.b0 = np.asarray(self.b0, dtype=np.float32).ravel()
            self.ssub_B = int(np.round(np.sqrt(d1 * d2 / self.W.shape[0])))
            if self.ssub_B > 1:
                self.ds_mat = decimation_matrix(self.dims, self.ssub_B).astype(np.float32)
        elif self.b is None or self.f is None:
            raise ValueError('Estimates contain neither ring-model (W, b0) nor low-rank (b, f) background')

    def _frames_to_pix(self, Y):
        n = Y.shape[0]
        return np.asarray(Y, dtype=np.float32).transpose(0, 2, 1).reshape(n, -1).T  # (d, n), F-order pixels

    def _pix_to_frames(self, X):
        d1, d2 = self.dims
        return X.T.reshape(-1, d2, d1).transpose(0, 2, 1)

    def chunk(self, t0, t1):
        """Return (Y, B, AC_sel, residual) as (n, d1, d2) float32 arrays."""
        Yr = self._frames_to_pix(self.mov[t0:t1])
        AC_all = self.A.dot(self.C[:, t0:t1])
        if self.W is not None:
            X = Yr - AC_all - self.b0[:, None]
            d1, d2 = self.dims
            if self.ssub_B > 1:
                X = self.ds_mat.dot(X)
                Bd = self.W.dot(X).reshape(((d1 - 1) // self.ssub_B + 1, (d2 - 1) // self.ssub_B + 1, -1), order='F')
                Bd = np.repeat(np.repeat(Bd, self.ssub_B, 0), self.ssub_B, 1)[:d1, :d2].reshape((-1, Yr.shape[1]),
                                                                                                 order='F')
            else:
                Bd = self.W.dot(X)
            B = self.b0[:, None] + Bd
        else:
            B = _to_dense(self.b).dot(_to_dense(self.f)[:, t0:t1])
        AC = AC_all if len(self.sel) == self.A.shape[1] else self.A_sel.dot(self.C_sel[:, t0:t1])
        R = Yr - AC - B
        return tuple(self._pix_to_frames(np.asarray(x, dtype=np.float32)) for x in (Yr, B, AC, R))


def cnmf_qc(estimates, movie=None, out_base=None, fps=None, components='accepted',
            make_video=True, residual_maps=True, compare_raw_maps=True,
            frame_range=None, video_tbin=2, video_speed=2, gain_res=3, crf=20,
            tsub=2, chunk=500, tmp_dir=None, max_mem_gb=4, verbose=True):
    """
    Visual and quantitative check of a CNMF-E result.

    Video (out_base + '_qc.mp4'), 2x2 panels:
        raw Y | Y - B (raw minus background) | A*C of selected components | residual Y - A*C - B (x gain_res)
      Y-B and A*C share one color scale. Residual is shown on a diverging scale around 0. Contours of the selected
      components are drawn on the residual panel. Anything structured left in the residual (blinking blobs)
      is signal the model did not capture.

    Residual maps (out_base + '_qc_residual_maps.png/.npz/.json): PNR and correlation images of the residual,
    computed exactly as CNMF-E seed maps (see compute_init_maps). Seeds that pass the session thresholds in the
    residual are candidates for missed neurons ('outside' footprints) or poorly fitted ones ('inside').

    Args:
        estimates: CaImAn Estimates object or path to *_estimates.pickle
        movie: movie used for CNMF (path or (T, d1, d2) array). Default: estimates.tif_name
        fps: real imaging rate (for video timing). Default: 20
        components: 'accepted' (estimates.idx_components), 'all', or an index list
        frame_range: (start, stop) frames for the video (maps always use the whole recording)
        video_tbin: frames averaged per video frame; video_speed: playback speed relative to real time
        tsub: temporal averaging for residual maps (use the CNMF value)
    """
    if isinstance(estimates, (str, os.PathLike)):
        if out_base is None:
            out_base = str(estimates).replace('_estimates.pickle', '').replace('.pickle', '')
        with open(estimates, 'rb') as f:
            est = pickle.load(f)
    else:
        est = estimates
    if movie is None:
        movie = getattr(est, 'tif_name', None)
        if movie is None or not os.path.exists(str(movie)):
            raise FileNotFoundError(f'Movie {movie} not found, pass movie= explicitly')
    if out_base is None:
        out_base = os.path.splitext(str(movie))[0] if isinstance(movie, (str, os.PathLike)) else 'cnmf_qc'
    if fps is None:
        print('fps not given, using 20 for video timing')
        fps = 20

    cnmf_dict = getattr(est, 'cnmf_dict', {}) or {}
    gSig = cnmf_dict.get('gSig', [4, 4])
    gSig = gSig[0] if np.ndim(gSig) else gSig

    mov = open_movie(movie)
    model = _ModelChunker(est, mov, components=components)
    T, (d1, d2) = model.T, model.dims
    conts = _footprint_contours(model.A, model.sel, model.dims)
    summary = {'n_components_total': int(model.A.shape[1]), 'n_components_shown': int(len(model.sel)),
               'components': components if isinstance(components, str) else 'custom'}

    # ---- display scales from a sample of frames
    span = frame_range if frame_range is not None else (0, T)
    s_raw, s_sig, s_res = [], [], []
    for t in _sample_frames(T, 30, span):
        Y, B, AC, R = model.chunk(t, t + 1)
        s_raw.append(Y.ravel()[::7])
        s_sig.append(AC.ravel()[::7])
        s_res.append(R.ravel()[::7])
    raw_lo, raw_hi = np.percentile(np.concatenate(s_raw), [1, 99.9])
    sig_hi = max(np.percentile(np.concatenate(s_sig), 99.9), 1e-3)
    res_lim = sig_hi / gain_res
    lut_gray, lut_sig, lut_res = _lut('gray'), _lut('inferno'), _lut('RdBu_r')

    writer = None
    if make_video:
        vid_path = out_base + '_qc.mp4'
        out_fps = fps / video_tbin * video_speed
    res_mm = None
    if residual_maps:
        tmp_dir = tmp_dir or tempfile.gettempdir()
        os.makedirs(tmp_dir, exist_ok=True)
        T_ds = int(np.ceil(T / tsub))
        res_path = os.path.join(tmp_dir, os.path.basename(out_base) + '_residual_tsub.f16')
        res_mm = np.memmap(res_path, dtype=np.float16, mode='w+', shape=(T_ds, d1, d2))

    chunk = max(tsub * video_tbin, chunk - chunk % (tsub * video_tbin))  # chunks aligned to both binnings
    for t0 in tqdm.tqdm(range(0, T, chunk), desc='cnmf qc', disable=not verbose):
        t1 = min(T, t0 + chunk)
        Y, B, AC, R = model.chunk(t0, t1)
        if res_mm is not None:
            res_mm[t0 // tsub:t0 // tsub + int(np.ceil((t1 - t0) / tsub))] = _tsub_mean(R, tsub)
        if make_video and span[0] < t1 and t0 < span[1]:
            a, b_ = max(t0, span[0]) - t0, min(t1, span[1]) - t0
            Yb, YBb, ACb, Rb = (_tsub_mean(x[a:b_], video_tbin) for x in (Y, Y - B, AC, R))
            for k in range(Yb.shape[0]):
                res_rgb = _to_rgb(Rb[k], -res_lim, res_lim, lut_res)
                cv2.drawContours(res_rgb, conts, -1, (90, 90, 90), 1)
                panels = [_label(_to_rgb(Yb[k], raw_lo, raw_hi, lut_gray), 'raw'),
                          _label(_to_rgb(YBb[k], 0, sig_hi, lut_sig), 'raw - background'),
                          _label(_to_rgb(ACb[k], 0, sig_hi, lut_sig), f'A*C ({len(model.sel)} comps)'),
                          _label(res_rgb, f'residual x{gain_res}')]
                frame = _grid(panels, 2)
                t_sec = (max(t0, span[0]) + k * video_tbin) / fps
                _label(frame, f't = {t_sec:7.1f} s', y=-8)
                if writer is None:
                    writer = _VideoWriter(vid_path, (frame.shape[1], frame.shape[0]), out_fps, crf=crf)
                writer.write(frame)
    if writer is not None:
        writer.close()
        summary['video'] = vid_path
        if verbose:
            print(f'QC video saved: {vid_path}')

    if residual_maps:
        res_mm.flush()
        res_maps = compute_init_maps(res_mm, gSig, tsub=1, pre_tsub=tsub, mirror=False,
                                     max_mem_gb=max_mem_gb, verbose=verbose)
        del res_mm
        try:
            os.remove(res_path)
        except OSError:
            print(f'Could not remove temporary file {res_path}')
        raw_maps = compute_init_maps(mov, gSig, tsub=tsub,
                                     frame_slice=slice(0, T), mirror=False, max_mem_gb=max_mem_gb,
                                     verbose=verbose) if compare_raw_maps else None
        summary.update(_residual_report(res_maps, raw_maps, model, cnmf_dict, conts, out_base))
        if verbose:
            print(f"Residual maps saved: {out_base}_qc_residual_maps.png | candidate missed neurons: "
                  f"{summary['residual_seeds_outside']} outside, {summary['residual_seeds_inside']} inside footprints")

    with open(out_base + '_qc.json', 'w') as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


def _residual_report(res_maps, raw_maps, model, cnmf_dict, conts, out_base):
    d1, d2 = model.dims
    gSig = res_maps['gSig']
    gSiz = cnmf_dict.get('gSiz', [4 * gSig + 1])
    gSiz = gSiz[0] if np.ndim(gSiz) else gSiz
    min_corr, min_pnr = cnmf_dict.get('min_corr', 0.8), cnmf_dict.get('min_pnr', 8)

    # union of selected footprints (pixels above 0.2 of each footprint max)
    A = model.A_sel
    amax = np.asarray(A.max(0).todense()).ravel() if A.shape[1] else np.zeros(0)
    fp = np.zeros(d1 * d2, bool)
    for i in range(A.shape[1]):
        col = A[:, i]
        fp[col.indices[col.data >= 0.2 * amax[i]]] = True
    fp = fp.reshape((d1, d2), order='F')

    # residual has ~zero mean, so dark regions are taken from the movie maps
    valid = valid_mask(raw_maps if raw_maps is not None else {'cn': res_maps['cn'], 'gSig': gSig})
    seeds = count_seeds(res_maps['cn'], res_maps['pnr'], min_corr, min_pnr, gSiz, valid=valid, min_dist=2 * gSig)
    inside = fp[seeds[:, 0], seeds[:, 1]] if len(seeds) else np.zeros(0, bool)

    ncols = 3 if raw_maps is not None else 2
    fig, axs = plt.subplots(2, ncols, figsize=(5.5 * ncols, 10), dpi=150, squeeze=False)
    vmax = np.percentile(raw_maps['pnr'] if raw_maps is not None else res_maps['pnr'], 99.9)
    col = 0
    if raw_maps is not None:
        axs[0, 0].imshow(raw_maps['pnr'], cmap='magma', vmin=0, vmax=vmax)
        axs[0, 0].set_title('PNR: movie')
        axs[1, 0].imshow(raw_maps['cn'], cmap='magma', vmin=0, vmax=1)
        axs[1, 0].set_title('Correlation: movie')
        col = 1
    axs[0, col].imshow(res_maps['pnr'], cmap='magma', vmin=0, vmax=vmax)
    axs[0, col].set_title('PNR: residual (same scale)')
    axs[1, col].imshow(res_maps['cn'], cmap='magma', vmin=0, vmax=1)
    axs[1, col].set_title('Correlation: residual')

    v = res_maps['cn'] * res_maps['pnr']
    for r, ax in enumerate(axs[:, col + 1]):
        ax.imshow(v if r == 0 else fp, cmap='gray', vmin=0, vmax=np.percentile(v, 99.9) if r == 0 else 1)
        for c in conts:
            ax.plot(c[:, 0, 0], c[:, 0, 1], color='tab:blue', lw=0.4)
        if len(seeds):
            ax.plot(seeds[~inside, 1], seeds[~inside, 0], 'o', mfc='none', mec='tab:red', ms=6)
            ax.plot(seeds[inside, 1], seeds[inside, 0], 'x', color='tab:orange', ms=4)
    axs[0, col + 1].set_title(f'residual corr*PNR, thresholds corr {min_corr} / pnr {min_pnr}\n'
                              f'red o: {int((~inside).sum())} outside footprints, '
                              f'orange x: {int(inside.sum())} inside')
    axs[1, col + 1].set_title('footprints (selected components)')
    for ax in axs.ravel():
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(out_base + '_qc_residual_maps.png', bbox_inches='tight')
    plt.close(fig)

    np.savez_compressed(out_base + '_qc_residual_maps.npz', cn_res=res_maps['cn'], pnr_res=res_maps['pnr'],
                        seeds=seeds, seeds_inside=inside,
                        **({'cn_raw': raw_maps['cn'], 'pnr_raw': raw_maps['pnr']} if raw_maps is not None else {}))

    rep = {'residual_seeds_outside': int((~inside).sum()), 'residual_seeds_inside': int(inside.sum()),
           'residual_thresholds': {'min_corr': min_corr, 'min_pnr': min_pnr},
           'residual_pnr_p99': float(np.percentile(res_maps['pnr'], 99)),
           'residual_maps_png': out_base + '_qc_residual_maps.png'}
    if raw_maps is not None:
        rep['raw_pnr_p99'] = float(np.percentile(raw_maps['pnr'], 99))
        rep['raw_seeds'] = int(len(count_seeds(raw_maps['cn'], raw_maps['pnr'], min_corr, min_pnr, gSiz,
                                               valid=valid, min_dist=2 * gSig)))
    return rep


def batch_cnmf_qc(estimates_files, fps=None, **kwargs):
    """Run cnmf_qc for several *_estimates.pickle files. fps: number or dict {session part of name: fps}."""
    results = {}
    for fname in estimates_files:
        f = fps
        if isinstance(fps, dict):
            f = next((v for k, v in fps.items() if k in os.path.basename(fname)), None)
        try:
            results[fname] = cnmf_qc(fname, fps=f, **kwargs)
        except Exception as e:
            print(f'[FAILED] {fname}: {type(e).__name__}: {e}')
            results[fname] = None
    return results


# ----------------------------------------------------------------------------------------------------------------------
# 3. dF/F video for visual inspection
# ----------------------------------------------------------------------------------------------------------------------

def compute_f0(movie, fps, f0_window_s=60, f0_percentile=10, f0_bin_s=1.0, f0_ssub=4, chunk=2000, verbose=True):
    """
    Slow per-pixel baseline: running low percentile over f0_window_s, computed on a movie binned to f0_bin_s
    in time and f0_ssub in space (baseline is smooth, so this is cheap and accurate enough for display).
    Returns (F0 binned array (n_bins, h, w), frames per bin).
    """
    mov = open_movie(movie)
    T, d1, d2 = mov.shape
    fpb = max(1, int(round(fps * f0_bin_s)))
    h, w = int(np.ceil(d1 / f0_ssub)), int(np.ceil(d2 / f0_ssub))
    chunk = max(fpb, chunk - chunk % fpb)
    bins = []
    for t0 in tqdm.tqdm(range(0, T, chunk), desc='F0', disable=not verbose):
        blk = _tsub_mean(mov[t0:min(T, t0 + chunk)], fpb)
        bins.extend(cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA) for b in blk)
    Fb = np.stack(bins)
    win = max(3, int(round(f0_window_s / f0_bin_s)))
    F0b = percentile_filter(Fb, f0_percentile, size=(win, 1, 1), mode='nearest')
    return F0b, fpb


def make_dff_video(movie, out_path=None, fps=20, panels=('raw', 'dff', 'hp'), gSig=4,
                   f0_window_s=60, f0_percentile=10, smooth_sigma=1.0,
                   tbin=2, speed=2, frame_range=None, dff_max=None, chunk=1000, crf=20, verbose=True):
    """
    Video for visual inspection of a (motion corrected) movie. Panels, side by side:
        'raw' : movie as is
        'dff' : (F - F0) / F0, F0 = running f0_percentile-th percentile over f0_window_s per pixel,
                F slightly smoothed in space (smooth_sigma). Fixed color scale for the whole session.
        'hp'  : spatial high-pass with the CNMF-E kernel for gSig: what the seed search 'sees'
                (background removed, neurons stand out).
    Use on motion corrected data, otherwise motion shows up as edges.

    Args:
        movie: path to tif (e.g. *_MC.tif) or array (T, d1, d2)
        fps: real imaging rate
        tbin: frames averaged per video frame (noise reduction); speed: playback speed vs real time
        dff_max: upper limit of the dF/F color scale (default: 99.7th percentile of sampled frames)
    """
    mov = open_movie(movie)
    T, d1, d2 = mov.shape
    if out_path is None:
        base = os.path.splitext(str(movie))[0] if isinstance(movie, (str, os.PathLike)) else 'movie'
        out_path = base + '_' + '-'.join(panels) + '.mp4'
    span = frame_range if frame_range is not None else (0, T)

    F0b, fpb = (compute_f0(mov, fps, f0_window_s, f0_percentile, verbose=verbose)
                if 'dff' in panels else (None, None))
    f0_floor = max(1.0, float(np.percentile(F0b, 5))) if F0b is not None else 1.0
    dark = None
    if F0b is not None:  # outside the lens / black corners: F0 ~ 0 makes dF/F meaningless, show as 0
        f0_med = cv2.resize(np.median(F0b, 0), (d2, d1), interpolation=cv2.INTER_LINEAR)
        dark = f0_med < 0.25 * np.median(f0_med)
    psf = _psf_kernel(gSig)

    def f0_for(t):
        b = min(F0b.shape[0] - 1, t // fpb)
        return np.maximum(cv2.resize(F0b[b], (d2, d1), interpolation=cv2.INTER_LINEAR), f0_floor)

    def render(frame, t):
        out = {}
        if 'raw' in panels:
            out['raw'] = frame
        if 'dff' in panels:
            F = cv2.GaussianBlur(frame, (0, 0), smooth_sigma) if smooth_sigma else frame
            F0 = f0_for(t)
            dff = (F - F0) / F0
            dff[dark] = 0
            out['dff'] = dff
        if 'hp' in panels:
            out['hp'] = cv2.filter2D(frame, -1, psf, borderType=1)
        return out

    # fixed scales from sampled frames
    samples = {p: [] for p in panels}
    for t in _sample_frames(T, 40, span):
        for p, img in render(_tsub_mean(mov[t:t + tbin], tbin)[0], t).items():
            samples[p].append(img.ravel()[::5])
    scales = {}
    for p in panels:
        s = np.concatenate(samples[p])
        if p == 'raw':
            scales[p] = tuple(np.percentile(s, [1, 99.9]))
        elif p == 'dff':
            scales[p] = (0.0, dff_max if dff_max is not None else max(float(np.percentile(s, 99.7)), 0.05))
        else:
            scales[p] = (0.0, max(float(np.percentile(s, 99.9)), 1e-3))
    luts = {'raw': _lut('gray'), 'dff': _lut('inferno'), 'hp': _lut('gray')}
    names = {'raw': 'raw', 'dff': f'dF/F (0..{scales.get("dff", (0, 0))[1]:.2f})', 'hp': f'high-pass gSig={gSig}'}

    writer = None
    chunk = max(tbin, chunk - chunk % tbin)
    for t0 in tqdm.tqdm(range(span[0], span[1], chunk), desc='dF/F video', disable=not verbose):
        t1 = min(span[1], t0 + chunk)
        blk = _tsub_mean(mov[t0:t1], tbin)
        for k in range(blk.shape[0]):
            t = t0 + k * tbin
            imgs = render(blk[k], t)
            rgbs = [_label(_to_rgb(imgs[p], *scales[p], luts[p]), names[p]) for p in panels]
            frame = _grid(rgbs, len(rgbs))
            _label(frame, f't = {t / fps:7.1f} s', y=-8)
            if writer is None:
                writer = _VideoWriter(out_path, (frame.shape[1], frame.shape[0]), fps / tbin * speed, crf=crf)
            writer.write(frame)
    if writer is not None:
        writer.close()
    if verbose:
        print(f'Video saved: {out_path}')
    return out_path
