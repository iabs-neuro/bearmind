"""Run capcan autoinspection on CC, RFC, FOF in sequence using notebook
parameters (incl. updated pxlthr_distance_boundary=15).

Designed for unattended overnight runs. Streams per-session progress to a log
file. On failure of a single session, keeps going.

Usage:
    python batch_autoinspect_all.py [--datasets CC,RFC,FOF] [--skip-existing]
"""
from __future__ import annotations
import argparse
import time
import traceback
from pathlib import Path

import naming
from ae_launch import run_auto_inspection

# Notebook capcan parameters (one-to-one with cell 27 of BEARMiND_full_pipeline.ipynb)
CAPCAN_PARAMS = dict(
    fps=None,
    session_name=None,
    comps_to_select=None,
    cthr=0.35,
    corr_thr=0.4,
    num_sessions=1,
    match_threshold=1,
    sf=None,
    ef=None,
    ds=1,
    include_event_based=True,
    include_heavy=True,
    detect_corner_artifacts=False,
    corner_artifact_params=None,
    event_method='wavelet',
    n_iter=3,
    correlation_method='spearman',
    wavelet_backend='auto',
    hybrid_kinetics=True,
    brain='hybrid',
    ml_model_path='production_models/ebm_v9_iter8.pkl',
    ml_threshold=0.72,
    deletion_rules=None,
    pxlthr_distance_boundary=15,  # conservative; most genuine NMF duplicates fit within 10 px
    d_snr_thr=10,
    enable_merge=True,
    track_criteria_failures=True,
    save_artifacts=True,
    save_estimates=True,
    save_matrices=True,
    save_corner_detection=True,
    compress_estimates=True,
    verbose=True,
)

ROOT = Path(__file__).parent

DATASETS = {
    'CC':  dict(
        estimates_root=Path(r'W:\DATASET\CC\ClaciumData\Estimates_refined'),
        artifacts_root=ROOT / 'output' / 'CC',
    ),
    'RFC': dict(
        estimates_root=Path(r'W:\DATASET\RFC\ClaciumData\4_Estimates_refined'),
        artifacts_root=ROOT / 'output' / 'RFC',
    ),
    'FOF': dict(
        estimates_root=Path(r'W:\DATASET\FOF\CalciumData\4_Estimates_refined'),
        artifacts_root=ROOT / 'output' / 'FOF',
    ),
}


def extract_session_id(pickle_path: Path) -> str:
    """`<dataset>_<mouseId>_<dayId>[_<trial>]_<timestamp>_estimates.pickle` ->
    `<dataset>_<mouseId>_<dayId>[_<trial>]`. Falls back to first 3 underscore tokens."""
    stem = pickle_path.stem.replace('_estimates', '')
    sid = naming.extract_session_id(stem)
    if sid:
        return sid
    parts = stem.split('_')
    return '_'.join(parts[:3]) if len(parts) >= 3 else stem


def already_done(artifacts_root: Path, session_id: str) -> bool:
    art_dir = artifacts_root / f'inspection_artifacts_{session_id}'
    if not art_dir.exists():
        return False
    return any(art_dir.glob('*_processed.pickle'))


def run_dataset(dataset: str, *, skip_existing: bool):
    cfg = DATASETS[dataset]
    estimates_root: Path = cfg['estimates_root']
    artifacts_root: Path = cfg['artifacts_root']
    artifacts_root.mkdir(parents=True, exist_ok=True)

    if not estimates_root.exists():
        print(f'[{dataset}] estimates root not found: {estimates_root}', flush=True)
        return

    pickles = sorted(estimates_root.glob('*_estimates.pickle'))
    print(f'\n========================================', flush=True)
    print(f'[{dataset}] {len(pickles)} estimates pickles in {estimates_root}', flush=True)
    print(f'[{dataset}] output -> {artifacts_root}', flush=True)
    print(f'========================================', flush=True)

    started = time.time()
    n_ok = n_skip = n_fail = 0
    for i, pkl in enumerate(pickles, 1):
        sid = extract_session_id(pkl)
        if skip_existing and already_done(artifacts_root, sid):
            print(f'[{dataset}][{i}/{len(pickles)}] SKIP existing: {sid}', flush=True)
            n_skip += 1
            continue

        t0 = time.time()
        print(f'\n[{dataset}][{i}/{len(pickles)}] ===== {sid} =====', flush=True)
        try:
            result = run_auto_inspection(
                str(pkl),
                artifacts_path=str(artifacts_root),
                **CAPCAN_PARAMS,
            )
            n_kept = len(result['estimates'].idx_components)
            dt = time.time() - t0
            print(f'[{dataset}][{i}/{len(pickles)}] OK  {sid}: '
                  f'{n_kept} neurons kept in {dt:.1f}s', flush=True)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f'[{dataset}][{i}/{len(pickles)}] FAIL {sid}: {type(e).__name__}: {e}',
                  flush=True)
            traceback.print_exc()

    elapsed = time.time() - started
    print(f'\n[{dataset}] DONE: ok={n_ok}, skip={n_skip}, fail={n_fail} '
          f'in {elapsed/60:.1f} min', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', default='CC,RFC,FOF',
                        help='Comma-separated datasets in run order (default: CC,RFC,FOF)')
    parser.add_argument('--skip-existing', action='store_true',
                        help='Skip sessions that already have a *_processed.pickle')
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(',') if d.strip()]
    for d in datasets:
        if d not in DATASETS:
            print(f'unknown dataset: {d}; known: {list(DATASETS)}', flush=True)
            return

    print(f'BATCH AUTOINSPECT start  datasets={datasets}  skip_existing={args.skip_existing}',
          flush=True)
    t0 = time.time()
    for d in datasets:
        run_dataset(d, skip_existing=args.skip_existing)
    print(f'\nALL DONE in {(time.time() - t0)/60:.1f} min', flush=True)


if __name__ == '__main__':
    main()
