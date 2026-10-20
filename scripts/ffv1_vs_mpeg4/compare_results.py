"""
Compare CNMF-E results of the FFV1 (lossless) and MPEG-4 versions of the same sessions
(outputs of run_compression_test.py).

Usage: python compare_results.py D:\\Projects\\Test_FFV1\\analysis
Writes <analysis>/report/: summary csv files, per-session figures, report.json
"""
import os
import sys
import json
import glob
import pickle

import numpy as np
import pandas as pd
import tifffile as tfl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks
from scipy.stats import spearmanr

MATCH_MIN_COS = 0.5      # footprint cosine similarity needed to call two components the same neuron
EVENT_TOL_S = 0.5        # tolerance for event matching
EVENT_K = 4              # event = peak of C+YrA with prominence > EVENT_K * robust noise SD


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def robust_sd(x, axis=-1):
    med = np.median(x, axis=axis, keepdims=True)
    return 1.4826 * np.median(np.abs(x - med), axis=axis)


def centroids(A, dims):
    d1, d2 = dims
    yy, xx = np.meshgrid(np.arange(d1), np.arange(d2), indexing='ij')
    yy, xx = yy.ravel(order='F'), xx.ravel(order='F')
    s = np.asarray(A.sum(0)).ravel() + 1e-12
    return np.stack([A.T.dot(yy) / s, A.T.dot(xx) / s], 1)


def match(A1, A2):
    """Hungarian matching on footprint cosine similarity. Returns pairs (i, j, cos)."""
    n1 = np.sqrt(np.asarray(A1.power(2).sum(0)).ravel()) + 1e-12
    n2 = np.sqrt(np.asarray(A2.power(2).sum(0)).ravel()) + 1e-12
    cos = (A1.T.dot(A2)).toarray() / n1[:, None] / n2[None, :]
    r, c = linear_sum_assignment(-cos)
    keep = cos[r, c] >= MATCH_MIN_COS
    return r[keep], c[keep], cos[r[keep], c[keep]]


def events(trace, noise, fps):
    p, _ = find_peaks(trace, prominence=EVENT_K * noise, distance=max(1, int(0.5 * fps)))
    return p


def event_f1(e1, e2, tol):
    if len(e1) == 0 and len(e2) == 0:
        return np.nan, np.nan, np.nan
    used = np.zeros(len(e2), bool)
    tp = 0
    for t in e1:
        if len(e2) == 0:
            break
        d = np.abs(e2 - t).astype(float)
        d[used] = np.inf
        k = np.argmin(d)
        if d[k] <= tol:
            used[k] = True
            tp += 1
    prec = tp / max(len(e2), 1)   # e2 = MPEG-4 events, reference = FFV1 events e1
    rec = tp / max(len(e1), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return f1, prec, rec


def accepted(est):
    idx = getattr(est, 'idx_components', None)
    return np.arange(est.A.shape[1]) if idx is None else np.asarray(idx, int)


def snr(est, idx):
    s = getattr(est, 'SNR_comp', None)
    return np.asarray(s)[idx] if s is not None else np.full(len(idx), np.nan)


def compare_pair(est_f, est_m, dims, fps, name, tag, out_dir):
    acc_f, acc_m = accepted(est_f), accepted(est_m)
    A_f, A_m = est_f.A.tocsc()[:, acc_f], est_m.A.tocsc()[:, acc_m]
    i, j, cos = match(A_f, A_m)
    gi, gj = acc_f[i], acc_m[j]

    raw_f = est_f.C[gi] + est_f.YrA[gi]
    raw_m = est_m.C[gj] + est_m.YrA[gj]
    noise_f, noise_m = robust_sd(est_f.YrA[gi]), robust_sd(est_m.YrA[gj])
    amp_f, amp_m = np.percentile(est_f.C[gi], 99.5, axis=1), np.percentile(est_m.C[gj], 99.5, axis=1)

    rows = []
    for k in range(len(gi)):
        ef = events(raw_f[k], noise_f[k], fps)
        em = events(raw_m[k], noise_m[k], fps)
        f1, prec, rec = event_f1(ef, em, EVENT_TOL_S * fps)
        rows.append({'session': name, 'thr': tag, 'idx_ffv1': gi[k], 'idx_mpeg4': gj[k], 'footprint_cos': cos[k],
                     'corr_C': np.corrcoef(est_f.C[gi[k]], est_m.C[gj[k]])[0, 1],
                     'corr_raw': np.corrcoef(raw_f[k], raw_m[k])[0, 1],
                     'noise_ffv1': noise_f[k], 'noise_mpeg4': noise_m[k],
                     'amp_ffv1': amp_f[k], 'amp_mpeg4': amp_m[k],
                     'snr_ffv1': snr(est_f, [gi[k]])[0], 'snr_mpeg4': snr(est_m, [gj[k]])[0],
                     'n_events_ffv1': len(ef), 'n_events_mpeg4': len(em),
                     'event_f1': f1, 'event_precision': prec, 'event_recall': rec})
    pairs = pd.DataFrame(rows)

    # population structure: pairwise correlations of matched neurons, by distance
    cen = centroids(est_f.A.tocsc()[:, gi], dims)
    dist = np.sqrt(((cen[:, None] - cen[None]) ** 2).sum(-1))
    iu = np.triu_indices(len(gi), 1)
    cc_f = np.corrcoef(raw_f)[iu] if len(gi) > 2 else np.array([])
    cc_m = np.corrcoef(raw_m)[iu] if len(gi) > 2 else np.array([])
    dd = dist[iu]
    near = dd < 25

    unmatched_f = np.setdiff1d(np.arange(len(acc_f)), i)
    unmatched_m = np.setdiff1d(np.arange(len(acc_m)), j)
    summ = {
        'session': name, 'thr': tag,
        'n_total_ffv1': est_f.A.shape[1], 'n_total_mpeg4': est_m.A.shape[1],
        'n_accepted_ffv1': len(acc_f), 'n_accepted_mpeg4': len(acc_m),
        'n_matched': len(gi), 'only_ffv1': len(unmatched_f), 'only_mpeg4': len(unmatched_m),
        'median_footprint_cos': np.median(cos) if len(cos) else np.nan,
        'median_corr_C': pairs['corr_C'].median() if len(pairs) else np.nan,
        'median_corr_raw': pairs['corr_raw'].median() if len(pairs) else np.nan,
        'frac_corr_raw_below_0.9': (pairs['corr_raw'] < 0.9).mean() if len(pairs) else np.nan,
        'median_noise_ratio_m/f': np.median(noise_m / noise_f) if len(gi) else np.nan,
        'median_amp_ratio_m/f': np.median(amp_m / amp_f) if len(gi) else np.nan,
        'median_snr_ffv1': np.nanmedian(pairs['snr_ffv1']) if len(pairs) else np.nan,
        'median_snr_mpeg4': np.nanmedian(pairs['snr_mpeg4']) if len(pairs) else np.nan,
        'median_event_f1': np.nanmedian(pairs['event_f1']) if len(pairs) else np.nan,
        'median_event_recall': np.nanmedian(pairs['event_recall']) if len(pairs) else np.nan,
        'median_event_precision': np.nanmedian(pairs['event_precision']) if len(pairs) else np.nan,
        'pairwise_corr_agreement_r': np.corrcoef(cc_f, cc_m)[0, 1] if len(cc_f) > 2 else np.nan,
        'mean_pair_corr_near_ffv1': cc_f[near].mean() if near.any() else np.nan,
        'mean_pair_corr_near_mpeg4': cc_m[near].mean() if near.any() else np.nan,
        'mean_pair_corr_far_ffv1': cc_f[~near].mean() if (~near).any() else np.nan,
        'mean_pair_corr_far_mpeg4': cc_m[~near].mean() if (~near).any() else np.nan,
        'snr_only_ffv1_median': np.nanmedian(snr(est_f, acc_f[unmatched_f])) if len(unmatched_f) else np.nan,
        'snr_only_mpeg4_median': np.nanmedian(snr(est_m, acc_m[unmatched_m])) if len(unmatched_m) else np.nan,
    }

    # ---- figure
    fig = plt.figure(figsize=(18, 13), dpi=110)
    gs = fig.add_gridspec(3, 4)
    ax = fig.add_subplot(gs[0:2, 0:2])
    img = getattr(est_f, 'imax', None)
    if img is not None:
        ax.imshow(img, cmap='gray')
    cf = centroids(est_f.A.tocsc()[:, acc_f], dims)
    cm_ = centroids(est_m.A.tocsc()[:, acc_m], dims)
    ax.plot(cf[i, 1], cf[i, 0], 'o', mfc='none', mec='lime', ms=6, label=f'matched ({len(i)})')
    ax.plot(cf[unmatched_f, 1], cf[unmatched_f, 0], 's', mfc='none', mec='tab:blue', ms=6,
            label=f'only FFV1 ({len(unmatched_f)})')
    ax.plot(cm_[unmatched_m, 1], cm_[unmatched_m, 0], 'x', color='tab:red', ms=6,
            label=f'only MPEG-4 ({len(unmatched_m)})')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_title(f'{name} [{tag}] accepted components')
    ax.axis('off')

    def sc(ax, x, y, lab, logscale=False):
        ax.plot(x, y, '.', alpha=0.6)
        lo, hi = np.nanmin([x, y]), np.nanmax([x, y])
        ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8)
        if logscale:
            ax.set_xscale('log')
            ax.set_yscale('log')
        ax.set_xlabel(f'{lab} FFV1')
        ax.set_ylabel(f'{lab} MPEG-4')

    if len(pairs):
        sc(fig.add_subplot(gs[0, 2]), pairs['snr_ffv1'], pairs['snr_mpeg4'], 'SNR')
        sc(fig.add_subplot(gs[0, 3]), pairs['noise_ffv1'], pairs['noise_mpeg4'], 'noise SD', True)
        sc(fig.add_subplot(gs[1, 2]), pairs['amp_ffv1'], pairs['amp_mpeg4'], 'amplitude', True)
        ax = fig.add_subplot(gs[1, 3])
        ax.hist(pairs['corr_raw'], bins=np.linspace(0, 1, 41), alpha=0.7, label='C+YrA')
        ax.hist(pairs['corr_C'], bins=np.linspace(0, 1, 41), alpha=0.7, label='C')
        ax.set_xlabel('trace correlation FFV1 vs MPEG-4')
        ax.legend()
        ax = fig.add_subplot(gs[2, 0])
        if len(cc_f):
            ax.plot(cc_f[~near], cc_m[~near], '.', ms=2, alpha=0.3, label='far')
            ax.plot(cc_f[near], cc_m[near], '.', ms=3, color='tab:red', alpha=0.6, label='< 25 px')
            ax.plot([-0.2, 1], [-0.2, 1], 'k--', lw=0.8)
            ax.set_xlabel('pairwise corr FFV1')
            ax.set_ylabel('pairwise corr MPEG-4')
            ax.legend(fontsize=8)
        # example traces: median and worst raw-trace agreement
        order = np.argsort(pairs['corr_raw'].values)
        picks = [order[len(order) // 2], order[max(0, len(order) // 10)], order[0]]
        t = np.arange(raw_f.shape[1]) / fps
        span = slice(0, min(raw_f.shape[1], int(120 * fps)))
        for k, p in enumerate(picks):
            ax = fig.add_subplot(gs[2, 1 + k])
            ax.plot(t[span], raw_f[p, span], lw=0.6, label='FFV1')
            ax.plot(t[span], raw_m[p, span], lw=0.6, alpha=0.8, label='MPEG-4')
            ax.set_title(f'pair {p}: corr {pairs["corr_raw"].iloc[p]:.2f}', fontsize=9)
            ax.set_xlabel('s')
            if k == 0:
                ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_{tag}_comparison.png'))
    plt.close(fig)
    return summ, pairs


def pixel_level(cr_f, cr_m, name, out_dir, n=600):
    """Same frames before motion correction: compression error, its structure, temporal smoothing."""
    mf, mm = tfl.memmap(cr_f, mode='r'), tfl.memmap(cr_m, mode='r')
    T = min(mf.shape[0], mm.shape[0])
    idx = np.linspace(1, T - 1, n).astype(int)
    f = mf[idx].astype(np.float32)
    m = mm[idx].astype(np.float32)
    fprev, mprev = mf[idx - 1].astype(np.float32), mm[idx - 1].astype(np.float32)
    err = m - f
    # spatial structure of the error: power on 8-px block boundaries vs elsewhere
    gx = np.abs(np.diff(err, axis=2)).mean((0, 1))
    on = gx[7::8].mean()
    off = np.delete(gx, np.arange(7, len(gx), 8)).mean()
    # temporal noise: frame-to-frame differences (P-frames tend to keep blocks unchanged)
    td_f = robust_sd((f - fprev).reshape(n, -1), axis=1).mean() / np.sqrt(2)
    td_m = robust_sd((m - mprev).reshape(n, -1), axis=1).mean() / np.sqrt(2)
    res = {'session': name, 'mean_ffv1': float(f.mean()), 'mean_mpeg4': float(m.mean()),
           'bias': float(err.mean()), 'rmse': float(np.sqrt((err ** 2).mean())),
           'psnr_dB': float(20 * np.log10(255 / np.sqrt((err ** 2).mean()))),
           'block_edge_ratio': float(on / off),
           'temporal_noise_ffv1': float(td_f), 'temporal_noise_mpeg4': float(td_m),
           'n_unchanged_pixels_frac_ffv1': float(np.mean(f == fprev)),
           'n_unchanged_pixels_frac_mpeg4': float(np.mean(m == mprev))}
    fig, axs = plt.subplots(1, 3, figsize=(16, 5.5), dpi=110)
    k = n // 2
    axs[0].imshow(f[k], cmap='gray')
    axs[0].set_title('FFV1 frame')
    axs[1].imshow(m[k], cmap='gray')
    axs[1].set_title('MPEG-4 frame (q=3, as Bonsai)')
    lim = np.percentile(np.abs(err[k]), 99)
    axs[2].imshow(err[k], cmap='RdBu_r', vmin=-lim, vmax=lim)
    axs[2].set_title(f'MPEG-4 - FFV1, RMSE {res["rmse"]:.2f}, PSNR {res["psnr_dB"]:.1f} dB')
    for ax in axs:
        ax.axis('off')
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_pixels.png'))
    plt.close(fig)
    return res


def main(analysis):
    out_dir = os.path.join(analysis, 'report')
    os.makedirs(out_dir, exist_ok=True)
    summaries, all_pairs, pix, extra = [], [], [], []
    for info_p in sorted(glob.glob(os.path.join(analysis, '*_info.json'))):
        info = json.load(open(info_p))
        name, fps = info['session'], info['fps']
        thr = json.load(open(os.path.join(analysis, f'{name}_thresholds.json')))
        cr_f = os.path.join(analysis, 'ffv1', f'{name}_CR.tif')
        cr_m = os.path.join(analysis, 'mpeg4', f'{name}_CR.tif')
        dims = tfl.memmap(cr_f, mode='r').shape[1:]
        print(name, flush=True)
        pix.append(pixel_level(cr_f, cr_m, name, out_dir))
        for tag in ('auto', 'user'):
            pf = os.path.join(analysis, 'ffv1', f'{name}_{tag}_estimates.pickle')
            pm = os.path.join(analysis, 'mpeg4', f'{name}_{tag}_estimates.pickle')
            if not (os.path.exists(pf) and os.path.exists(pm)):
                continue
            s, p = compare_pair(load(pf), load(pm), dims, fps, name, tag, out_dir)
            t = thr['auto'] if tag == 'auto' else {'min_corr': 0.9, 'min_pnr': 7}
            s.update({'min_corr': t['min_corr'], 'min_pnr': t['min_pnr']})
            summaries.append(s)
            all_pairs.append(p)
        e = {'session': name, 'size_ffv1_MB': info['size_ffv1_MB'], 'size_mpeg4_MB': info['size_mpeg4_MB']}
        for ver in ('ffv1', 'mpeg4'):
            q = os.path.join(analysis, ver, f'{name}_auto_qc.json')
            if os.path.exists(q):
                qq = json.load(open(q))
                for k in ('residual_seeds_outside', 'residual_null_outside', 'residual_excess_outside',
                          'raw_pnr_p99', 'residual_pnr_p99'):
                    e[f'{k}_{ver}'] = qq.get(k)
            e[f'auto_thr_suggestions_{ver}'] = thr[ver]['suggestions']
        e['auto_thr_seeds_on_mpeg4'] = thr.get('auto_on_mpeg4')
        extra.append(e)

    pd.DataFrame(summaries).to_csv(os.path.join(out_dir, 'summary.csv'), index=False)
    pd.DataFrame(pix).to_csv(os.path.join(out_dir, 'pixels.csv'), index=False)
    if all_pairs:
        pd.concat(all_pairs).to_csv(os.path.join(out_dir, 'matched_pairs.csv'), index=False)
    json.dump(extra, open(os.path.join(out_dir, 'extra.json'), 'w'), indent=2, default=float)
    with pd.option_context('display.width', 250, 'display.max_columns', 50):
        print(pd.DataFrame(summaries).T)
        print(pd.DataFrame(pix).T)


if __name__ == '__main__':
    main(sys.argv[1])
