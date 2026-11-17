"""MC template correlation over the 84 LT1 sessions: how many sessions each candidate threshold flags.

Run from the worktree root WITHOUT -I (the repo root is put on sys.path below; the package does not import under -I):
  PYTHONDONTWRITEBYTECODE=1 conda run -n bearmind python docs/superpowers/specs/2026-11-16-mc-template-corr/template_corr.py
Reads H: only (registry, 3_MC shifts, threshold maps). Writes only into its own folder.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))  # worktree root, so the repo package is importable

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
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
    rows.append(dict(sid=sid, mouse=reg.get('mouse') or sid.split('_')[1], corr=float(corr), p99=p99, bright=bright))

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
m = flags['abs<0.55'] | flags['rel>0.05']
p(f'combined: corr < 0.55 OR below mouse median by more than 0.05: {int(m.sum())} flagged: '
  + (', '.join(s for s, k in zip(sids, m) if k) or '-'))
m = flags['abs<0.60'] | flags['rel>0.05']
p(f'combined: corr < 0.60 OR below mouse median by more than 0.05: {int(m.sum())} flagged: '
  + (', '.join(s for s, k in zip(sids, m) if k) or '-'))
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
    p(f'flagged sessions with p99 shift > 10 px (the existing shift check): {int((p99[low] > 10).sum())}')
p(f'all sessions with p99 shift > 10 px: {int((p99 > 10).sum())}')
p('Calibration reference (defaults.yaml): 6 bright sessions, corr 0.82-0.94; a 2 px misalignment gives 0.57-0.75')

(HERE / 'template_corr_out.txt').write_text('\n'.join(out) + '\n', encoding='utf-8')

# figure
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
    a.hlines(mmed[m], i - 0.35, i + 0.35, color='k')
a.axhline(0.70, color='C3', lw=1)
a.set_xticks(range(len(order))); a.set_xticklabels(order, rotation=90, fontsize=8)
a.set_ylabel('mean template corr'); a.set_title('By mouse (black = mouse median, red = 0.70)')
a = ax[1, 0]
a.scatter(bright, corr, s=14)
a.axhline(0.70, color='C3', lw=1)
a.set_xlabel('field brightness (median of mean image)'); a.set_ylabel('mean template corr')
a.set_title(f'Corr vs brightness (Spearman rho {r1.statistic:.2f})')
a = ax[1, 1]
a.scatter(p99, corr, s=14)
a.axhline(0.70, color='C3', lw=1)
a.set_xlabel('p99 of per-frame max |shift|, px'); a.set_ylabel('mean template corr')
a.set_title(f'Corr vs shift (Spearman rho {r2.statistic:.2f})')
fig.savefig(HERE / 'template_corr.png', dpi=110)
print('[OK] done')
