"""compute_init_maps against the algorithm it replaced (kept below as the reference).

The faster compute_init_maps cuts each filtered strip to the kept rows plus the correlation neighbours and
computes the noise once for both signs. With the same strips the maps must be bit for bit the same; with other
strips (the new memory model allows taller ones) they may differ only by float32 rounding.
"""
import os
import sys
import tracemalloc

import cv2
import numpy as np
import pytest
from caiman.source_extraction.cnmf.pre_processing import get_noise_fft
from caiman.summary_images import local_correlations_fft

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bm_qc  # noqa: E402
from test_bm_qc import synthetic_movie  # noqa: E402


# ----------------------------------------------------------------------------------------------------------------------
# reference: compute_init_maps as of commit 5e6d382 (helpers copied too, so later edits of bm_qc cannot move it)
# ----------------------------------------------------------------------------------------------------------------------

def _ref_tsub_mean(block, tsub):
    block = np.asarray(block, dtype=np.float32)
    if tsub == 1:
        return block
    T = block.shape[0]
    n_full = T // tsub
    out = block[:n_full * tsub].reshape((n_full, tsub) + block.shape[1:]).mean(1)
    if T % tsub:
        out = np.concatenate([out, block[n_full * tsub:].mean(0, keepdims=True)])
    return out


def _ref_psf_kernel(gSig):
    ksize = int(2 * gSig) * 2 + 1
    g = cv2.getGaussianKernel(ksize, gSig, cv2.CV_32F)
    psf = g.dot(g.T)
    ind_nonzero = psf >= psf[0].max()
    psf -= psf[ind_nonzero].mean()
    psf[~ind_nonzero] = 0
    return psf


def _ref_maps_from_filtered(filt, thresh_init):
    T, h, w = filt.shape
    sn = get_noise_fft(filt.reshape(T, -1).T, noise_method='mean')[0].reshape(h, w)
    sn[sn <= 0] = np.inf
    pnr = filt.max(0) / sn
    tmp = filt.copy()
    tmp[tmp < thresh_init * sn] = 0
    cn = local_correlations_fft(tmp, swap_dim=False)
    return np.nan_to_num(cn).astype(np.float32), np.nan_to_num(pnr).astype(np.float32)


def ref_strip_rows(max_mem_gb, T_ds, d1, d2, gSig):
    margin = _ref_psf_kernel(gSig).shape[0] // 2 + 1
    rows = int(max_mem_gb * 1e9 / (6 * 4 * T_ds * d2)) - 2 * margin
    return int(np.clip(rows, 8, d1))


def ref_compute_init_maps(mov, gSig, tsub=2, pre_tsub=1, frame_slice=None, thresh_init=2, mirror=True,
                          max_mem_gb=4):
    T, d1, d2 = mov.shape
    fs = frame_slice if frame_slice is not None else slice(0, T)
    T_used = len(range(*fs.indices(T)))
    T_ds = int(np.ceil(T_used / tsub))

    psf = _ref_psf_kernel(gSig)
    margin = psf.shape[0] // 2 + 1
    rows = ref_strip_rows(max_mem_gb, T_ds, d1, d2, gSig)

    out = {'cn': np.zeros((d1, d2), np.float32), 'pnr': np.zeros((d1, d2), np.float32),
           'mean': np.zeros((d1, d2), np.float32)}
    if mirror:
        out['cn_neg'] = np.zeros((d1, d2), np.float32)
        out['pnr_neg'] = np.zeros((d1, d2), np.float32)

    pnr_div = np.sqrt(tsub * pre_tsub)
    for r0 in range(0, d1, rows):
        r1 = min(d1, r0 + rows)
        a0, a1 = max(0, r0 - margin), min(d1, r1 + margin)
        blk = _ref_tsub_mean(mov[fs, a0:a1, :], tsub)
        out['mean'][r0:r1] = blk[:, r0 - a0:r1 - a0].mean(0)
        filt = np.empty_like(blk)
        for t in range(blk.shape[0]):
            filt[t] = cv2.filter2D(blk[t], -1, psf, borderType=1)
        del blk
        filt -= filt.mean(0)

        cn, pnr = _ref_maps_from_filtered(filt, thresh_init)
        out['cn'][r0:r1] = cn[r0 - a0:r1 - a0]
        out['pnr'][r0:r1] = pnr[r0 - a0:r1 - a0] / pnr_div
        if mirror:
            np.negative(filt, out=filt)
            cn, pnr = _ref_maps_from_filtered(filt, thresh_init)
            out['cn_neg'][r0:r1] = cn[r0 - a0:r1 - a0]
            out['pnr_neg'][r0:r1] = pnr[r0 - a0:r1 - a0] / pnr_div
        del filt
    return out


# ----------------------------------------------------------------------------------------------------------------------

MAP_KEYS = ('cn', 'pnr', 'cn_neg', 'pnr_neg', 'mean')
TINY_GB = 1e-6  # both memory models clip the strip to 8 rows: the same strips, each with full filter margins


def _movie(T, d=(64, 80), seed=0):
    return synthetic_movie(T=T, d=d, seed=seed)[0]


def _assert_same(new, ref, keys):
    for k in keys:
        assert new[k].dtype == np.float32 and new[k].shape == ref[k].shape, k
        assert np.array_equal(new[k], ref[k]), \
            f"{k}: {np.count_nonzero(new[k] != ref[k])} pixels differ, max {np.abs(new[k] - ref[k]).max()}"


@pytest.mark.parametrize("gSig", [3, 5, 7])
@pytest.mark.parametrize("mirror", [True, False])
@pytest.mark.parametrize("tsub", [1, 2])
def test_same_strips_bit_identical(gSig, mirror, tsub):
    """8-row strips with margins on both sides (and the frame edges): the cut strips change nothing."""
    mov = _movie(T=601)  # odd: tsub 2 leaves an incomplete last block; not a power of two
    T_ds = int(np.ceil(601 / tsub))
    assert bm_qc.init_maps_strip_rows(TINY_GB, T_ds, 64, 80, gSig, tsub) == 8
    assert ref_strip_rows(TINY_GB, T_ds, 64, 80, gSig) == 8
    new = bm_qc.compute_init_maps(mov, gSig=gSig, tsub=tsub, mirror=mirror, max_mem_gb=TINY_GB, verbose=False)
    ref = ref_compute_init_maps(mov, gSig=gSig, tsub=tsub, mirror=mirror, max_mem_gb=TINY_GB)
    _assert_same(new, ref, MAP_KEYS if mirror else ('cn', 'pnr', 'mean'))
    assert ('cn_neg' in new) is mirror


def test_whole_frame_strip_bit_identical():
    mov = _movie(T=500)
    new = bm_qc.compute_init_maps(mov, gSig=4, tsub=1, max_mem_gb=10, verbose=False)
    ref = ref_compute_init_maps(mov, gSig=4, tsub=1, max_mem_gb=10)
    _assert_same(new, ref, MAP_KEYS)


def test_long_movie_noise_path_bit_identical():
    """More than 3072 frames: get_noise_fft takes three windows of the time series instead of all of it."""
    mov = _movie(T=3337, d=(40, 48), seed=3)
    new = bm_qc.compute_init_maps(mov, gSig=3, tsub=1, max_mem_gb=TINY_GB, verbose=False)
    ref = ref_compute_init_maps(mov, gSig=3, tsub=1, max_mem_gb=TINY_GB)
    _assert_same(new, ref, MAP_KEYS)


def test_frame_slice_and_pre_tsub_bit_identical():
    mov = _movie(T=700, seed=5)
    kw = dict(gSig=4, tsub=2, pre_tsub=3, frame_slice=slice(37, 650), max_mem_gb=TINY_GB)
    _assert_same(bm_qc.compute_init_maps(mov, verbose=False, **kw), ref_compute_init_maps(mov, **kw), MAP_KEYS)


def test_other_strips_differ_by_rounding_only():
    """A budget at which the new model gives taller strips than the old one: float32 rounding of the filter
    (OpenCV filters these kernels via DFT tiles sized by the strip), nothing more; the suggestion is the same."""
    mov = _movie(T=601, d=(96, 80), seed=1)
    gb = 6 * 4 * 601 * 80 * 40 / 1e9  # the old model: 40 rows with margins
    assert bm_qc.init_maps_strip_rows(gb, 601, 96, 80, 3, 1) != ref_strip_rows(gb, 601, 96, 80, 3)
    new = bm_qc.compute_init_maps(mov, gSig=3, tsub=1, max_mem_gb=gb, verbose=False)
    ref = ref_compute_init_maps(mov, gSig=3, tsub=1, max_mem_gb=gb)
    assert np.array_equal(new['mean'], ref['mean'])
    for k in ('cn', 'pnr', 'cn_neg', 'pnr_neg'):
        np.testing.assert_allclose(new[k], ref[k], rtol=1e-4, atol=1e-4, err_msg=k)
    ref.update(gSig=3)
    s_new, _ = bm_qc.suggest_init_thresholds(new, gSiz=13)
    s_ref, _ = bm_qc.suggest_init_thresholds(ref, gSiz=13)
    assert s_new == s_ref


@pytest.mark.parametrize("gSig,tsub,row_copies", [
    (3, 1, 111),   # strips of 20 rows, the maps step binds
    (7, 1, 111),   # wide filter margins, the maps step still binds
    (4, 2, 111),   # tsub 2 with an odd frame count: the block averaging binds
    (5, 1, 1),     # below the 8-row floor: the budget is exceeded and the model says by how much
])
def test_peak_model_covers_the_measured_peak(gSig, tsub, row_copies):
    """init_maps_peak_bytes bounds what compute_init_maps allocates (numpy reports its buffers to tracemalloc),
    and not loosely: the strip height and the number of map workers rely on it."""
    T, d = 1501, (120, 96)
    mov = _movie(T=T, d=d, seed=2)
    T_ds = int(np.ceil(T / tsub))
    gb = row_copies * 4 * T_ds * d[1] / 1e9
    tracemalloc.start()
    try:
        bm_qc.compute_init_maps(mov, gSig=gSig, tsub=tsub, max_mem_gb=gb, verbose=False)
        measured = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    model = bm_qc.init_maps_peak_bytes(gb, T_ds, *d, gSig, tsub)
    slack = 1e6  # Python objects and per-pixel vectors, not modelled: a few MB at real sizes, against GBs
    assert measured <= model + slack, (measured, model)
    assert model <= 1.3 * measured, (measured, model)
    if row_copies > 1:
        assert model <= gb * 1e9 + 4 * 5 * d[0] * d[1]  # the strips fill the budget, the output maps come on top
