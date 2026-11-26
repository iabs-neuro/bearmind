"""
Threshold suggestion for CNMF-E seeding (min_corr / min_pnr) from the seed maps.

Kept apart from bm_qc because the Thresholds tab computes the suggestion of its methods and must not load CaImAn,
which bm_qc imports at module level. bm_qc imports these functions back, so bm_qc.<name> keeps working.
Needs only cv2, numpy and bearmind.seed_mask.
"""
import cv2
import numpy as np

from bearmind.seed_mask import valid_mask


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
    cn, pnr = maps['cn'], maps['pnr']
    valid = valid & np.isfinite(cn) & np.isfinite(pnr)
    if not np.any(valid):
        return np.zeros(cn.shape, dtype=bool)
    correlations = cn[valid]
    mc, sc = _robust(correlations)
    mp, sp = _robust(pnr[valid])
    # A broad correlation distribution can make median + k*MAD exceed 1,
    # losing every halo even where the strongest cell transients are present.
    corr_cut = min(mc + k_corr * sc, float(np.percentile(correlations, 99)))
    strong = valid & (pnr > mp + k_pnr * sp) & (cn > corr_cut)
    r = int(excl_radius)
    disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    return cv2.dilate(strong.astype(np.uint8), disk).astype(bool)


def _mirrored_background(maps, valid, halo):
    return valid & ~halo & np.isfinite(maps['cn_neg']) & np.isfinite(maps['pnr_neg'])


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
    if 'cn_neg' not in maps or 'pnr_neg' not in maps:
        return s, None, 1.0
    halo = halo if halo is not None else halo_mask(maps, valid, gSiz)
    null_area = _mirrored_background(maps, valid, halo)
    n_background = np.count_nonzero(null_area)
    if not n_background:
        return s, None, 1.0
    s0 = count_seeds(maps['cn_neg'], maps['pnr_neg'], min_corr, min_pnr, gSiz, valid=null_area,
                     min_dist=min_dist)
    return s, s0, valid.sum() / n_background


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

    'robust_bg'  : median + bg_k * 1.4826*MAD of the mirrored maps outside the cell halos.
                   When no mirrored background is available, estimate from the usable real maps.
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
    mirror = 'cn_neg' in maps and 'pnr_neg' in maps
    valid = valid_mask(maps, border_pix=border_pix) & np.isfinite(cn) & np.isfinite(pnr)
    if not np.any(valid):
        raise ValueError("no finite pixels in the usable seed-map area")
    halo = halo_mask(maps, valid, excl_radius if excl_radius is not None else gSiz) if mirror else None
    cn_i, pnr_i = cn[valid], pnr[valid]
    cn_bg, pnr_bg = cn_i, pnr_i
    if mirror:
        cn_neg, pnr_neg = maps['cn_neg'], maps['pnr_neg']
        background = _mirrored_background(maps, valid, halo)
        if np.any(background):
            cn_bg, pnr_bg = cn_neg[background], pnr_neg[background]

    def summary(min_corr, min_pnr):
        return evaluate_thresholds(maps, float(np.clip(min_corr, 0, 0.99)), float(max(min_pnr, 0)),
                                   gSiz, valid, halo)

    out = {}
    m_c, s_c = _robust(cn_bg)
    m_p, s_p = _robust(pnr_bg)
    out['robust_bg'] = summary(m_c + bg_k * s_c, m_p + bg_k * s_p)
    out['pct_margin'] = summary(np.percentile(cn_i, pct) + pct_corr_margin,
                                np.percentile(pnr_i, pct) + pct_pnr_margin)

    grid = None
    if mirror and not np.any(background):
        out['mirror_fdr'] = {'min_corr': None, 'min_pnr': None, 'n_seeds': 0,
                             'target_fdr': target_fdr, 'note': 'no usable mirrored background'}
    elif mirror:
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
