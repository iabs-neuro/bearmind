"""Sequentially run batch_export over CC, RFC, FOF inspection-artifact roots.

Each dataset is processed independently (errors in one don't abort the next).
Per-session export goes to the SAME folder as its `_processed.pickle`
(matching what batch_export does standalone). A separate log line per
dataset is printed with success/fail counts and total size.

Usage:
    python batch_export_all.py [--datasets CC,RFC,FOF] [--ml-threshold X]
"""
from __future__ import annotations
import argparse
import time
import traceback
from pathlib import Path

from batch_export import discover_pickles, export_one

ROOT = Path(__file__).parent

DATASETS = {
    'CC':  ROOT / 'output' / 'CC',
    'RFC': ROOT / 'output' / 'RFC',
    'FOF': ROOT / 'output' / 'FOF',
}


def run_dataset(name: str, root: Path, ml_threshold: float | None):
    pickles = discover_pickles([str(root)], recurse=True)
    print(f'\n========================================', flush=True)
    print(f'[{name}] root={root}  pickles={len(pickles)}', flush=True)
    print(f'========================================', flush=True)
    if not pickles:
        print(f'[{name}] nothing to export', flush=True)
        return dict(ok=0, fail=0, neurons=0, mb=0.0, sec=0.0)

    started = time.time()
    n_ok = n_fail = total_neurons = 0
    total_mb = 0.0
    for i, pkl in enumerate(pickles, 1):
        print(f'\n[{name}][{i}/{len(pickles)}] {pkl.name}', flush=True)
        try:
            result = export_one(pkl, ml_threshold_override=ml_threshold)
            n_ok += 1
            total_neurons += result['neurons']
            total_mb += result['size_mb']
        except Exception as e:
            n_fail += 1
            print(f'  [ERROR] {type(e).__name__}: {e}', flush=True)
            traceback.print_exc()

    sec = time.time() - started
    print(f'\n[{name}] DONE: ok={n_ok}, fail={n_fail}, '
          f'neurons={total_neurons}, size={total_mb:.1f} MB, '
          f'time={sec:.1f}s', flush=True)
    return dict(ok=n_ok, fail=n_fail, neurons=total_neurons, mb=total_mb, sec=sec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--datasets', default='CC,RFC,FOF',
                   help='Comma-separated datasets in order (default: CC,RFC,FOF)')
    p.add_argument('--ml-threshold', type=float, default=None,
                   help='Override ML threshold (default: use config / DEFAULT_ML_THRESHOLD)')
    p.add_argument('--dry-run', action='store_true',
                   help='Just list pickles per dataset; do not export')
    args = p.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            print(f'unknown dataset: {d}; known: {list(DATASETS)}', flush=True)
            return

    if args.dry_run:
        print('DRY RUN — listing pickle counts only', flush=True)
        for d in datasets:
            root = DATASETS[d]
            pkls = discover_pickles([str(root)], recurse=True)
            print(f'  {d}: {len(pkls)} pickles in {root}')
        return

    print(f'BATCH EXPORT ALL  datasets={datasets}  ml_threshold={args.ml_threshold}',
          flush=True)
    t0 = time.time()
    summary = {}
    for d in datasets:
        summary[d] = run_dataset(d, DATASETS[d], args.ml_threshold)
    print(f'\n==================== SUMMARY ====================')
    for d, s in summary.items():
        print(f'  {d}: ok={s["ok"]}, fail={s["fail"]}, '
              f'neurons={s["neurons"]}, size={s["mb"]:.1f} MB, time={s["sec"]:.1f}s')
    print(f'ALL DONE in {(time.time() - t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
