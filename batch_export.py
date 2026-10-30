"""
batch_export.py - Batch export of processed estimates to NPZ/JSON/MAT.

Mirrors the "Batch export data and metadata" cell in BEARMiND_full_pipeline.ipynb,
minus the tkinter file picker. Outputs are written next to each input pickle, so
results land in the same locations as the notebook produces.

Usage:
    # Recursively find *_processed.pickle under one or more roots
    python batch_export.py path/to/folder [more/folders ...]

    # Pass explicit pickle files
    python batch_export.py session1_processed.pickle session2_processed.pickle

    # Only look at top-level of each directory
    python batch_export.py path/to/folder --no-recurse

    # Override ML threshold (otherwise taken from est.autoinspection_config or default)
    python batch_export.py path/to/folder --ml-threshold 0.7

Output per session (in same folder as the input pickle):
    {session}_data.npz       - C (traces), asp, reconstructions, component_indices
    {session}_metadata.json  - FPS, session info, ML filter stats, feedback summary
    {session}_filters.mat    - Spatial footprints (MATLAB format)
"""
import argparse
import time
import traceback
from pathlib import Path

import numpy as np

from export_estimates_data import (
    DEFAULT_ML_THRESHOLD,
    apply_feedback,
    build_metadata,
    export,
    export_filters_mat,
    extract_data,
    extract_session_name,
    find_feedback_file,
    get_fps,
    load_estimates,
    load_feedback,
)


def discover_pickles(paths, recurse=True):
    """Resolve input arguments into a sorted list of *_processed.pickle files."""
    pattern = '**/*_processed.pickle' if recurse else '*_processed.pickle'
    seen = set()
    files = []
    for raw in paths:
        p = Path(raw)
        if p.is_file():
            if p.suffix == '.pickle' and p not in seen:
                files.append(p)
                seen.add(p)
        elif p.is_dir():
            for hit in sorted(p.glob(pattern)):
                if hit not in seen:
                    files.append(hit)
                    seen.add(hit)
        else:
            print(f"[WARN] skipping (not found): {raw}")
    return files


def export_one(est_path: Path, ml_threshold_override=None):
    """Export a single processed-estimates pickle.

    Mirrors the notebook batch cell step-for-step. Returns a result dict.
    """
    folder = est_path.parent

    est = load_estimates(est_path)
    session_name = extract_session_name(est)
    fps = get_fps(session_name)

    n_frames = est.C.shape[1] if hasattr(est, 'C') and est.C is not None else 0
    duration_s = n_frames / fps if fps > 0 else 0
    duration_min = duration_s / 60

    print(f"  Session: {session_name}")
    print(f"  FPS: {fps:.1f}, Duration: {duration_min:.1f} min ({n_frames} frames)")

    component_indices = list(est.idx_components)
    n_before_ml_filter = len(component_indices)
    ml_filtered = False
    ml_threshold_used = None

    if hasattr(est, 'metrics_df') and est.metrics_df is not None:
        df = est.metrics_df
        if 'ml_keep_probability' in df.columns:
            if ml_threshold_override is not None:
                threshold = ml_threshold_override
            else:
                threshold = DEFAULT_ML_THRESHOLD
                if hasattr(est, 'autoinspection_config') and est.autoinspection_config:
                    threshold = est.autoinspection_config.get(
                        'ml_threshold', DEFAULT_ML_THRESHOLD
                    )

            ml_approved_mask = df['ml_keep_probability'] >= threshold
            ml_approved_indices = set(df.loc[ml_approved_mask, 'component_idx'].tolist())
            component_indices = np.array(
                [i for i in component_indices if i in ml_approved_indices]
            )

            ml_filtered = True
            ml_threshold_used = threshold
            n_ml_rejected = n_before_ml_filter - len(component_indices)
            print(f"  ML filter (>={threshold}): "
                  f"{n_before_ml_filter} -> {len(component_indices)} (-{n_ml_rejected})")

    feedback_path = find_feedback_file(est_path, session_name)
    feedback_applied = False
    feedback_summary = None

    if feedback_path:
        print(f"  Feedback: {feedback_path.name}")
        feedback_df = load_feedback(feedback_path)
        component_indices, feedback_summary = apply_feedback(component_indices, feedback_df)
        feedback_applied = True
        print(f"    FP removed: {feedback_summary['n_fp_removed']}, "
              f"FN added: {feedback_summary['n_fn_added']}")
    else:
        print(f"  Feedback: none")

    ml_filter_info = {
        'ml_filtered': ml_filtered,
        'threshold_used': ml_threshold_used,
        'n_before': n_before_ml_filter,
        'n_after': len(component_indices) if ml_filtered else None,
    }

    data = extract_data(est, component_indices)
    metadata = build_metadata(
        est, fps, session_name, feedback_applied, feedback_summary,
        ml_filter_info, component_indices,
    )
    npz_path, _ = export(data, metadata, session_name, folder)

    mat_path = folder / f'{session_name}_filters.mat'
    export_filters_mat(est, component_indices, mat_path)

    npz_size = npz_path.stat().st_size / (1024 ** 2) if npz_path.exists() else 0
    mat_size = mat_path.stat().st_size / (1024 ** 2) if mat_path.exists() else 0
    session_size = npz_size + mat_size
    n_exported = len(component_indices)

    print(f"  Exported: {n_exported} neurons, {session_size:.1f} MB "
          f"(npz: {npz_size:.1f}, mat: {mat_size:.1f})")

    return {
        'session': session_name,
        'neurons': n_exported,
        'duration_min': duration_min,
        'size_mb': session_size,
        'ml_filtered': ml_filtered,
        'feedback': feedback_applied,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Batch export of processed estimates (same behaviour as the "
                    "notebook 'Batch export data and metadata' cell)."
    )
    parser.add_argument(
        'paths', nargs='+',
        help='One or more *_processed.pickle files or directories to scan.'
    )
    parser.add_argument(
        '--no-recurse', action='store_true',
        help='Do not recurse into subdirectories when scanning a folder.'
    )
    parser.add_argument(
        '--ml-threshold', type=float, default=None,
        help='Override ML threshold for filtering (default: use '
             "est.autoinspection_config['ml_threshold'] or DEFAULT_ML_THRESHOLD)."
    )
    args = parser.parse_args()

    pickle_files = discover_pickles(args.paths, recurse=not args.no_recurse)

    if not pickle_files:
        print("No *_processed.pickle files found.")
        return

    print(f"Selected {len(pickle_files)} files for export")
    print("=" * 70)

    successful = 0
    failed = 0
    total_neurons = 0
    total_size_mb = 0
    export_results = []
    start_time = time.time()

    for i, est_path in enumerate(pickle_files, 1):
        print(f"\n[{i}/{len(pickle_files)}] {est_path.name}")
        try:
            result = export_one(est_path, ml_threshold_override=args.ml_threshold)
            export_results.append(result)
            total_neurons += result['neurons']
            total_size_mb += result['size_mb']
            successful += 1
        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            failed += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("BATCH EXPORT COMPLETE")
    print("-" * 70)
    print(f"  Files: {successful} succeeded, {failed} failed")
    print(f"  Total neurons exported: {total_neurons}")
    print(f"  Total size: {total_size_mb:.1f} MB")
    print(f"  Time elapsed: {elapsed:.1f}s")

    if export_results:
        print("\n  Per-session summary:")
        header = f"  {'Session':<30} {'Neurons':>8} {'Duration':>10} {'Size':>10} {'ML':>4} {'FB':>4}"
        print(header)
        print(f"  {'-'*30} {'-'*8} {'-'*10} {'-'*10} {'-'*4} {'-'*4}")
        for r in export_results:
            ml_flag = 'Y' if r['ml_filtered'] else 'N'
            fb_flag = 'Y' if r['feedback'] else 'N'
            print(f"  {r['session']:<30} {r['neurons']:>8} "
                  f"{r['duration_min']:>8.1f}m {r['size_mb']:>8.1f}MB "
                  f"{ml_flag:>4} {fb_flag:>4}")

    print("=" * 70)
    print("\nOutput files per session (next to input pickle):")
    print("  {session}_data.npz      - C (traces), asp, reconstructions, component_indices")
    print("  {session}_metadata.json - FPS, session info, ML filter stats, feedback summary")
    print("  {session}_filters.mat   - Spatial footprints (MATLAB format)")


if __name__ == '__main__':
    main()
