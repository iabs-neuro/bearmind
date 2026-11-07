"""Tests for bm_qc on small synthetic movies. Run: python -m pytest tests/test_bm_qc.py"""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import bm_qc  # noqa: E402


def synthetic_movie(T=600, d=(64, 80), n_cells=6, amp=40., noise=4., seed=0, return_model=False):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:d[0], :d[1]]
    centers = rng.uniform(12, np.array(d) - 12, size=(n_cells, 2))
    bg = 60 + 20 * np.exp(-((yy - d[0] / 2) ** 2 + (xx - d[1] / 2) ** 2) / (2 * 30 ** 2))  # smooth background
    mov = np.repeat(bg[None].astype(np.float32), T, 0)
    kernel = np.exp(-np.arange(60) / 20.)
    fps, traces = [], []
    for cy, cx in centers:
        spikes = (rng.random(T) < 0.01).astype(float)
        trace = amp * np.convolve(spikes, kernel)[:T]
        fp = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 3. ** 2))
        mov += trace[:, None, None] * fp[None]
        fps.append(fp)
        traces.append(trace)
    mov += rng.normal(0, noise, mov.shape)
    mov = np.clip(mov, 0, 255).astype(np.uint8)
    if return_model:
        return mov, centers, np.array(fps), np.array(traces), bg
    return mov, centers


def test_strips_equal_full_frame():
    mov, _ = synthetic_movie()
    full = bm_qc.compute_init_maps(mov, gSig=3, tsub=2, max_mem_gb=10, verbose=False)
    # tiny memory budget forces many strips
    strips = bm_qc.compute_init_maps(mov, gSig=3, tsub=2, max_mem_gb=1e-5, verbose=False)
    for k in ('cn', 'pnr', 'cn_neg', 'pnr_neg', 'mean'):
        np.testing.assert_allclose(full[k], strips[k], rtol=1e-4, atol=1e-4)


def test_seeds_find_cells_and_null_is_clean():
    mov, centers = synthetic_movie()
    maps = bm_qc.compute_init_maps(mov, gSig=3, tsub=2, verbose=False)
    sugg, grid = bm_qc.suggest_init_thresholds(maps, border_pix=3)
    s = sugg['mirror_fdr']
    assert s['min_pnr'] is not None
    seeds, null, _ = bm_qc.threshold_seeds(maps, s['min_corr'], s['min_pnr'], valid=bm_qc.valid_mask(maps, 3))
    dist = np.sqrt(((seeds[:, None, :] - centers[None]) ** 2).sum(-1))
    # every simulated cell is found, and almost every seed lies on a cell
    assert np.all(dist.min(0) < 3)
    assert np.sum(dist.min(1) > 5) <= 1
    assert s['fdr'] <= 0.1


def test_pure_noise_gives_high_fdr_at_low_thresholds():
    mov, _ = synthetic_movie(n_cells=0)
    maps = bm_qc.compute_init_maps(mov, gSig=3, tsub=2, verbose=False)
    res = bm_qc.evaluate_thresholds(maps, 0.3, 2.0)
    assert res['n_seeds'] > 10 and res['fdr'] > 0.5


def test_count_seeds_matches_caiman_rule():
    yy, xx = np.mgrid[:40, :40]
    blob = lambda y, x: np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * 3. ** 2)).astype(np.float32)
    cn = 0.95 * (blob(10, 10) + blob(30, 30))
    pnr = 12 * blob(10, 10) + 5 * blob(30, 30)  # second blob fails min_pnr
    seeds = bm_qc.count_seeds(cn, pnr, 0.5, 6, gSiz=13, min_dist=6)
    assert [tuple(s) for s in seeds] == [(10, 10)]


def test_cnmf_qc_finds_omitted_cell(tmp_path):
    from types import SimpleNamespace
    import scipy.sparse as sp
    mov, centers, fps, traces, bg = synthetic_movie(T=600, n_cells=5, amp=60., seed=3, return_model=True)
    T, d1, d2 = mov.shape
    A = sp.csc_matrix(np.stack([f.ravel(order='F') for f in fps], 1))
    est = SimpleNamespace(A=A, C=traces.astype(np.float32), W=None, b0=None,
                          b=bg.ravel(order='F')[:, None], f=np.ones((1, T)),
                          idx_components=np.arange(4),  # the 5th cell is left out on purpose
                          cnmf_dict={'gSig': [3, 3], 'gSiz': [13, 13], 'min_corr': 0.8, 'min_pnr': 6})
    out = bm_qc.cnmf_qc(est, movie=mov, out_base=str(tmp_path / 'qc'), fps=20, tmp_dir=str(tmp_path),
                        video_tbin=2, verbose=False)
    assert os.path.getsize(out['video']) > 0
    res = np.load(str(tmp_path / 'qc_qc_residual_maps.npz'))
    seeds = res['seeds'][~res['seeds_inside']]
    # the omitted cell shows up in the residual, outside the accepted footprints
    assert np.sqrt(((seeds - centers[4]) ** 2).sum(1)).min() < 3
    assert out['residual_excess_outside'] >= 1


def test_dff_video_written(tmp_path):
    mov, _ = synthetic_movie(T=200)
    out = bm_qc.make_dff_video(mov, out_path=str(tmp_path / 'v.mp4'), fps=20, f0_window_s=5, verbose=False)
    assert os.path.getsize(out) > 0


def test_block_mean():
    a = np.arange(35, dtype=np.float32).reshape(5, 7)
    b = bm_qc.block_mean(a, 2)
    assert b.shape == (2, 3) and b[0, 0] == a[:2, :2].mean() and b[1, 2] == a[2:4, 4:6].mean()
    assert np.array_equal(bm_qc.block_mean(a, 1), a)


def _video_frame_shapes(monkeypatch):
    """Records the (h, w) of the first frame each _VideoWriter receives, while still writing."""
    seen = []
    orig = bm_qc._VideoWriter

    class Spy(orig):
        def __init__(self, path, size_wh, fps, crf=18):
            seen.append((size_wh[1], size_wh[0]))
            super().__init__(path, size_wh, fps, crf=crf)

    monkeypatch.setattr(bm_qc, '_VideoWriter', Spy)
    return seen


def test_dff_video_ssub(tmp_path, monkeypatch):
    seen = _video_frame_shapes(monkeypatch)
    mov, _ = synthetic_movie(T=40, d=(64, 80))
    for ssub in (1, 2):
        bm_qc.make_dff_video(mov, out_path=str(tmp_path / f'v{ssub}.mp4'), fps=20, f0_window_s=5,
                             ssub=ssub, verbose=False)
    assert seen == [(64, 3 * 80 + 4), (32, 3 * 40 + 4)]


def test_cnmf_qc_video_ssub(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import scipy.sparse as sp
    seen = _video_frame_shapes(monkeypatch)
    mov, centers, fps, traces, bg = synthetic_movie(T=600, n_cells=5, amp=60., seed=3, return_model=True)
    T = mov.shape[0]
    A = sp.csc_matrix(np.stack([f.ravel(order='F') for f in fps], 1))
    est = SimpleNamespace(A=A, C=traces.astype(np.float32), W=None, b0=None,
                          b=bg.ravel(order='F')[:, None], f=np.ones((1, T)), idx_components=np.arange(4),
                          cnmf_dict={'gSig': [3, 3], 'gSiz': [13, 13], 'min_corr': 0.8, 'min_pnr': 6})
    for ssub in (1, 2):
        out = bm_qc.cnmf_qc(est, movie=mov, out_base=str(tmp_path / f'qc{ssub}'), fps=20, tmp_dir=str(tmp_path),
                            residual_maps=False, video_ssub=ssub, verbose=False)
        assert os.path.getsize(out['video']) > 0
    assert seen == [(130, 162), (66, 82)]


def test_block_mean_is_the_video_module_one():
    from bearmind import video
    assert bm_qc.block_mean is video.block_mean


def test_videos_have_even_sides_after_ssub(tmp_path):
    """66 x 90 with ssub 2 gives 33 x 45 panels: the files must still be written (yuv420p) with even sides."""
    import av
    mov, _ = synthetic_movie(T=40, d=(66, 90))
    out = bm_qc.make_dff_video(mov, out_path=str(tmp_path / 'v.mp4'), fps=20, f0_window_s=5, ssub=2, verbose=False)
    with av.open(out) as c:
        s = c.streams.video[0]
        assert s.codec_context.height % 2 == 0 and s.codec_context.width % 2 == 0
    w = bm_qc._VideoWriter(str(tmp_path / 'w.mp4'), (45, 33), 20)  # an odd frame goes in
    for _ in range(3):
        w.write(np.zeros((33, 45, 3), np.uint8))
    w.close()
    with av.open(str(tmp_path / 'w.mp4')) as c:
        s = c.streams.video[0]
        assert (s.codec_context.height, s.codec_context.width) == (32, 44)


@pytest.mark.parametrize("dims", [(456, 435), (437, 521)])
@pytest.mark.parametrize("ssub_B", [3, 4])
def test_model_chunker_reads_ssub_B_from_W_when_frame_does_not_divide(dims, ssub_B):
    """The QC runs before the estimates are saved: a size mismatch here would throw away a whole fit."""
    from types import SimpleNamespace

    import scipy.sparse as sp

    d1, d2 = dims
    rng = np.random.default_rng(0)
    n_ds = ((d1 - 1) // ssub_B + 1) * ((d2 - 1) // ssub_B + 1)
    T, K = 12, 3
    est = SimpleNamespace(A=sp.random(d1 * d2, K, density=0.001, format="csc", random_state=1),
                          C=rng.random((K, T)), W=sp.eye(n_ds, format="csr") * 0.1,
                          b0=rng.random(d1 * d2), b=None, f=None, idx_components=None)
    mov = rng.integers(0, 255, (T, d1, d2)).astype(np.uint8)
    m = bm_qc._ModelChunker(est, mov)
    assert m.ssub_B == ssub_B
    Y, B, AC, R = m.chunk(0, 5)
    assert Y.shape == B.shape == AC.shape == R.shape == (5, d1, d2)
    assert np.isfinite(B).all() and np.isfinite(R).all()
    # the ring part is one value per ssub_B x ssub_B block (nearest-neighbour upsampling), cut at the frame edge
    ring = B[0] - est.b0.reshape((d1, d2), order="F")
    assert np.allclose(ring[:ssub_B, :ssub_B], ring[0, 0], atol=1e-4)
