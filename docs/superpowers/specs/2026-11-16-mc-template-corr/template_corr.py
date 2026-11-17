"""MC template correlation over the 84 LT1 sessions: how many sessions each candidate threshold flags.

Run from the worktree root: conda run -n bearmind python -I docs/superpowers/specs/2026-11-16-mc-template-corr/template_corr.py
Reads H: only (registry, 3_MC shifts, threshold maps). Writes only into its own folder.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from caiman.motion_correction import high_pass_filter_space
from scipy.stats import spearmanr

ROOT = Path('H:/Projects/LT1/CalciumData')
ABS = [0.50, 0.55, 0.60, 0.65, 0.70]
REL = [0.05, 0.10]
MIN_SESS = 3
DARK_FRAC = 0.25  # same dark-zone rule as bearmind.seed_mask.valid_mask (without its border and dilation)

out = []


def p(s=''):
    out.append(s)
    print(s)


rows = []
for f in sorted((ROOT / '_bearmind' / 'registry').glob('*.json')):
    sid = f.stem
    reg = json.loads(f.read_text(encoding='utf-8'))
    corr = reg['stages']['mc']['metrics']['mean_template_corr']
    sh = np.load(ROOT / '3_MC' / f'{sid}_MC_shifts.npy')
    p99 = float(np.percentile(np.abs(sh).max(axis=1), 99))
    z = np.load(ROOT / '_bearmind' / 'thresholds' / f'{sid}_maps.npz')
    mean = z['mean']
    med = np.median(mean)
    bright = float(np.median(mean[mean >= DARK_FRAC * med]))
    keep = mean >= DARK_FRAC * med
    pnr = float(np.median(z['pnr_5'][keep]))
    cn = float(np.median(z['cn_5'][keep]))
    # texture proxy: contrast of the mean image after the high-pass filter that MC uses (gSig_filt 8)
    hp = high_pass_filter_space(mean.astype(np.float32), (8, 8))
    tex = float(np.std(hp[keep]) / bright)
    fm = np.abs(sh).max(axis=1)
    rows.append(dict(sid=sid, mouse=reg.get('mouse') or sid.split('_')[1], corr=float(corr), p99=p99, bright=bright,
                     pnr=pnr, cn=cn, tex=tex, msp=float(reg['stages']['mc']['metrics']['max_shift_px']),
                     mc_level=[c['level'] for c in reg['stages']['mc']['checks'] if c['name'] == 'mc_max_shift_px'][0],
                     big=[int(i) for i in np.where(fm > 10)[0]]))

sids = [r['sid'] for r in rows]
corr = np.array([r['corr'] for r in rows])
p99 = np.array([r['p99'] for r in rows])
bright = np.array([r['bright'] for r in rows])
mice = sorted({r['mouse'] for r in rows})
mmed = {m: float(np.median([r['corr'] for r in rows if r['mouse'] == m])) for m in mice}
mcount = {m: sum(r['mouse'] == m for r in rows) for m in mice}

p(f'sessions: {len(rows)}; mice: {len(mice)}')
p(f'corr: min {corr.min():.3f}, median {np.median(corr):.3f}, max {corr.max():.3f}')
p(f'p99 of per-frame max |shift| (px): min {p99.min():.2f}, median {np.median(p99):.2f}, max {p99.max():.2f}')
p(f'field brightness (median of mean image outside dark zone): min {bright.min():.1f}, median {np.median(bright):.1f}, max {bright.max():.1f}')
p()

flags = {}
p('== Candidates ==')
for t in ABS:
    m = corr < t
    flags[f'abs<{t:.2f}'] = m
    p(f'corr < {t:.2f}: {int(m.sum())} of {len(rows)} flagged: ' + (', '.join(s for s, k in zip(sids, m) if k) or '-'))
for d in REL:
    m = np.array([mcount[r['mouse']] >= MIN_SESS and r['corr'] < mmed[r['mouse']] - d for r in rows])
    flags[f'rel>{d:.2f}'] = m
    p(f'corr below mouse median by more than {d:.2f} (mice with >= {MIN_SESS} sessions): {int(m.sum())} flagged: '
      + (', '.join(f'{s} ({c:.3f})' for s, k, c in zip(sids, m, corr) if k) or '-'))
fm = sorted({r['mouse'] for r, k in zip(rows, flags['abs<0.70']) if k})
p(f'corr < 0.70 touches {len(fm)} of {len(mice)} mice: ' + ', '.join(fm))
p('mice with every session below 0.70: ' + ', '.join(m_ for m_ in fm if all(r['corr'] < 0.70 for r in rows if r['mouse'] == m_)))
p()
p('== Overlap of flagged sets (n) ==')
keys = list(flags)
p('rule x rule: ' + ' | '.join(keys))
for a in keys:
    p(f'{a}: ' + ' '.join(str(int((flags[a] & flags[b]).sum())) for b in keys))
p()
p('== Mouse medians of corr (n sessions, min corr, median corr, median p99 shift px, median brightness) ==')
for m in sorted(mice, key=lambda m: mmed[m]):
    rr = [r for r in rows if r['mouse'] == m]
    p(f'{m}: n={len(rr)}, min {min(r["corr"] for r in rr):.3f}, median {mmed[m]:.3f}, '
      f'p99 {np.median([r["p99"] for r in rr]):.2f}, brightness {np.median([r["bright"] for r in rr]):.1f}')
p()
r1 = spearmanr(corr, bright)
r2 = spearmanr(corr, p99)
p('== Spearman rank correlation of corr ==')
p(f'with field brightness: rho = {r1.statistic:.3f}, p = {r1.pvalue:.2g}')
p(f'with p99 shift: rho = {r2.statistic:.3f}, p = {r2.pvalue:.2g}')
low = corr < 0.70
p()
p('== Sessions below 0.70 (sid, corr, p99 shift px, brightness) ==')
for i in np.argsort(corr):
    if low[i]:
        p(f'{sids[i]}: {corr[i]:.3f}, {p99[i]:.2f}, {bright[i]:.1f}')
if low.any():
    p(f'median p99 shift: flagged {np.median(p99[low]):.2f} px vs others {np.median(p99[~low]):.2f} px')
    p(f'median brightness: flagged {np.median(bright[low]):.1f} vs others {np.median(bright[~low]):.1f}')
    p(f'flagged sessions with p99 shift > 10 px (p99, NOT the quantity of the shift check): {int((p99[low] > 10).sum())}')
p(f'all sessions with p99 shift > 10 px: {int((p99 > 10).sum())}')


p()
p('== Mouse reference median: two definitions ==')
p('A (above): median of all sessions of the mouse, the session itself included')
p('B: leave-one-out median (the session is left out of its own reference; mice with >= 4 sessions)')
loo = np.array([np.median([o['corr'] for o in rows if o['mouse'] == r['mouse'] and o is not r])
                if mcount[r['mouse']] - 1 >= MIN_SESS else np.nan for r in rows])
for d in REL:
    m = corr < loo - d
    p(f'B, below the reference by more than {d:.2f}: {int(m.sum())} flagged: '
      + (', '.join(f'{sid_} ({c:.3f}, ref {l:.3f})' for sid_, k, c, l in zip(sids, m, corr, loo) if k) or '-'))
mB = corr < loo - 0.05
p(f'B, rule 0.05 flags sessions with p99 shift (px): ' + ', '.join(f'{sid_} {x:.2f}' for sid_, x, k in zip(sids, p99, mB) if k))
for sid_ in ['LT1_J06_2D', 'LT1_J18_1D', 'LT1_J57_1D']:
    i = sids.index(sid_)
    mm = rows[i]['mouse']
    p(f'{sid_}: corr {corr[i]:.3f}, p99 shift {p99[i]:.2f} px; p99 of the 4 sessions of the mouse: '
      + ', '.join(f'{r["p99"]:.2f}' for r in rows if r['mouse'] == mm))
mrec = flags['abs<0.55'] | (corr < loo - 0.10)
p('recommended rule (corr < 0.55 OR below the leave-one-out mouse median by more than 0.10): '
  f'{int(mrec.sum())} flagged: ' + (', '.join(sid_ for sid_, k in zip(sids, mrec) if k) or '-'))
p(f'mice with fewer than {MIN_SESS + 1} sessions (no leave-one-out reference): {sum(mcount[m_] < MIN_SESS + 1 for m_ in mice)}')
p('These flagged sessions were NOT checked by eye (no video frames read).')

p()
p('== What tracks corr: maps (gSig 5, valid area) and texture proxy ==')
for key, name in [('pnr', 'median PNR'), ('cn', 'median local correlation (cn)'), ('tex', 'high-pass contrast of mean image / brightness')]:
    v = np.array([r[key] for r in rows])
    rs = spearmanr(corr, v)
    mv = np.array([np.median([r[key] for r in rows if r['mouse'] == m_]) for m_ in mice])
    rm = spearmanr(np.array([mmed[m_] for m_ in mice]), mv)
    p(f'{name}: sessions rho = {rs.statistic:.3f} (p = {rs.pvalue:.2g}); mouse level (n={len(mice)}) rho = {rm.statistic:.3f} (p = {rm.pvalue:.2g})')
pnr = np.array([r['pnr'] for r in rows])
p(f'median PNR: sessions flagged at 0.70 {np.median(pnr[low]):.2f} vs others {np.median(pnr[~low]):.2f}')
mid = np.array([mice.index(r['mouse']) for r in rows])
dc = corr - np.array([corr[mid == k].mean() for k in mid])
dp = p99 - np.array([p99[mid == k].mean() for k in mid])
rw = spearmanr(dc, dp)
p(f'within a mouse (both centred on the mouse mean): corr vs p99 shift rho = {rw.statistic:.3f}, p = {rw.pvalue:.2g}')
p(f'medians of p99 shift: flagged at 0.70 {np.median(p99[low]):.2f} px, others {np.median(p99[~low]):.2f} px')

p()
p('== Shift check mc_max_shift_px (registry max_shift_px = maximum over ALL frames, not p99) ==')
for lvl in sorted({r['mc_level'] for r in rows} - {'ok'}):
    sel = [r for r in rows if r['mc_level'] == lvl]
    p(f'level {lvl}: {len(sel)} sessions')
    for r in sorted(sel, key=lambda r: r['msp']):
        fr = r['big']
        p(f'  {r["sid"]}: max {r["msp"]:.1f} px, frames above 10 px: {len(fr)}, indices {fr[:6]}; only frame 0: {fr == [0]}')
sf = [r for r in rows if r['msp'] > 10]
p(f'of the {len(sf)} shift-flagged sessions, those with corr < 0.70: {sum(r["corr"] < 0.70 for r in sf)} ('
  + ', '.join(f'{r["sid"]} {r["corr"]:.3f}' for r in sf if r['corr'] < 0.70) + ')')
p(f'sessions with registry max_shift_px > 10: {sum(r["msp"] > 10 for r in rows)}; largest {max(r["msp"] for r in rows):.1f} px')

p()
p('== Calibration (docs/superpowers/specs/2026-11-01-e2e-report.md, table of 6 real sessions; these numbers are copied from it) ==')
CAL = {'NOF_H01_1D': (0.94, [0.88, 0.75, 0.61, 0.41]), 'LT1_K02_1D': (0.82, [0.73, 0.57, 0.43, 0.25])}
for k, (c0, v) in CAL.items():
    p(f'{k}: corr {c0:.2f}; whole movie shifted by 1/2/3/5 px -> ' + '/'.join(f'{x:.2f}' for x in v)
      + f'; drop at 1 px {c0 - v[0]:.2f}, at 2 px {c0 - v[1]:.2f}')
for t in ABS:
    p(f'floor {t:.2f}: catches a 2 px misalignment on ' + (', '.join(k for k, (c0, v) in CAL.items() if v[1] < t) or 'neither calibration session'))
p('Calibration sessions: 6 real sessions, corr 0.815-0.939 (LT1_K02_1D 0.815, NOF_H01_1D 0.939, NOF_H03_3D 0.935, NOF_H03_4D 0.921, RFC_F01_1D 0.906, RFC_F20_1D 0.915); shifts were a constant offset of the whole movie.')
p('Only LT1_K02_1D (corr 0.815) of the 6 calibration sessions is from LT1.')

(HERE / 'template_corr_out.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')

# figure
rec = flags['abs<0.55'] | (corr < loo - 0.10)  # recommended: floor 0.55 + leave-one-out median, 0.10
fig, ax = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
a = ax[0, 0]
a.hist(corr, bins=24, color='0.6')
for t in ABS:
    a.axvline(t, color='C3' if t == 0.70 else 'C0', lw=1.2, ls='-' if t == 0.70 else '--')
    a.text(t, a.get_ylim()[1], f'{t:.2f}', rotation=90, va='top', ha='right', fontsize=8)
a.set_xlabel('mean template corr'); a.set_ylabel('sessions'); a.set_title('Histogram; lines = candidate thresholds')
a = ax[0, 1]
order = sorted(mice, key=lambda m: mmed[m])
for i, m in enumerate(order):
    v = [r['corr'] for r in rows if r['mouse'] == m]
    a.scatter([i] * len(v), v, s=14, color='C0')
    rv = [r['corr'] for r in rows if r['mouse'] == m and rec[sids.index(r['sid'])]]
    a.scatter([i] * len(rv), rv, s=60, facecolors='none', edgecolors='C1', linewidths=1.8)
    a.hlines(mmed[m], i - 0.35, i + 0.35, color='k')
a.axhline(0.70, color='C3', lw=1)
a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=90, fontsize=8)
a.set_ylabel('mean template corr'); a.set_title('By mouse (black = median, red = 0.70)\n' + 'ring = recommended rule', fontsize=10)
a = ax[1, 0]
a.scatter(bright, corr, s=14)
a.axhline(0.70, color='C3', lw=1)
a.set_xlabel('field brightness (median of mean image outside dark zone)'); a.set_ylabel('mean template corr')
a.set_title(f'Corr vs brightness (Spearman rho {r1.statistic:.2f})')
a = ax[1, 1]
a.scatter(p99, corr, s=14)
a.axhline(0.70, color='C3', lw=1)
a.set_xlabel('p99 of per-frame max |shift|, px'); a.set_ylabel('mean template corr')
a.set_title(f'Corr vs shift (Spearman rho {r2.statistic:.2f})')
fig.savefig(HERE / 'template_corr.png', dpi=110)
print('[OK] done')
