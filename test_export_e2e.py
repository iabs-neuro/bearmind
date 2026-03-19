"""
End-to-end test: run full autoinspection on a raw session, then compress and export.
Validates that recent fixes correctly handle index alignment throughout the pipeline.

Tests:
1. metrics_df index transformation during compression (0ebad3a)
2. asp_cache index transformation during compression (0f22b68)
3. Metadata filtering to match exported Calcium array (f67cdd7)

Usage:
    python test_export_e2e.py                            # uses default BOWL session
    python test_export_e2e.py path/to/estimates.pickle   # uses specified file
"""
import pickle
import tempfile
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ae_launch import run_auto_inspection
from estimates_compression import compress_estimates_ultra_lightweight
from export_estimates_data import (
    load_estimates, extract_session_name, get_fps_from_table,
    extract_data, build_metadata, export, export_filters_mat,
    DEFAULT_ML_THRESHOLD
)

# Raw BOWL session (not yet autoinspected)
DEFAULT_FILE = Path('data/BOWL_J19_5D_29-12-2025 18-31-25_estimates.pickle')


def check_alignment(est, label):
    """Validate all component-indexed structures are aligned. Returns list of errors."""
    errors = []
    n_components = len(est.idx_components)

    # C matrix rows
    if hasattr(est, 'C') and est.C is not None:
        if est.C.shape[0] != n_components:
            errors.append(f"C rows={est.C.shape[0]} != idx_components={n_components}")

    # metrics_df
    if hasattr(est, 'metrics_df') and est.metrics_df is not None and not est.metrics_df.empty:
        n_rows = len(est.metrics_df)
        if n_rows != n_components:
            errors.append(f"metrics_df rows={n_rows} != idx_components={n_components}")

        if 'component_idx' in est.metrics_df.columns:
            metrics_set = set(int(x) for x in est.metrics_df['component_idx'].values)
            expected_set = set(int(x) for x in est.idx_components)
            if metrics_set != expected_set:
                errors.append(
                    f"metrics_df component_idx mismatch: "
                    f"min={min(metrics_set)}, max={max(metrics_set)}, "
                    f"expected 0..{n_components-1}"
                )

    # reconstructions
    if hasattr(est, 'reconstructions') and est.reconstructions:
        recon_keys = set(est.reconstructions.keys())
        expected = set(int(x) for x in est.idx_components)
        extra = recon_keys - expected
        if extra:
            errors.append(f"reconstructions has extra keys: {sorted(extra)[:5]}...")
        missing = expected - recon_keys
        if len(missing) > n_components * 0.5:
            errors.append(f"reconstructions missing {len(missing)}/{n_components} keys")

    # asp_cache
    if hasattr(est, 'asp_cache') and est.asp_cache:
        asp_keys = set(est.asp_cache.keys())
        expected = set(int(x) for x in est.idx_components)
        extra = asp_keys - expected
        if extra:
            errors.append(f"asp_cache has extra keys: {sorted(extra)[:5]}...")
        missing = expected - asp_keys
        if len(missing) > n_components * 0.5:
            errors.append(f"asp_cache missing {len(missing)}/{n_components} keys")

    # Contiguity check
    expected_indices = set(range(n_components))
    actual_indices = set(int(i) for i in est.idx_components)
    if actual_indices != expected_indices:
        errors.append(
            f"idx_components not contiguous 0..{n_components-1}: "
            f"min={min(actual_indices)}, max={max(actual_indices)}"
        )

    status = "PASS" if not errors else "FAIL"
    print(f"\n[{status}] Alignment check: {label}")
    print(f"  Components: {n_components}")
    has_mdf = hasattr(est, 'metrics_df') and est.metrics_df is not None
    has_recon = hasattr(est, 'reconstructions') and bool(getattr(est, 'reconstructions', None))
    has_asp = hasattr(est, 'asp_cache') and bool(getattr(est, 'asp_cache', None))
    has_ml = has_mdf and 'ml_keep_probability' in est.metrics_df.columns
    print(f"  Has: metrics_df={has_mdf}, ml_proba={has_ml}, reconstructions={has_recon}, asp_cache={has_asp}")
    if has_ml:
        ml = est.metrics_df['ml_keep_probability']
        above = (ml >= 0.72).sum()
        below = (ml < 0.72).sum()
        print(f"  ML probabilities: {above} above 0.72, {below} below (will be filtered in export)")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print(f"  All structures aligned")
    return errors


def test_export_pipeline(est, tmpdir, session_name):
    """Test the full export pipeline and validate outputs."""
    errors = []

    fps = get_fps_from_table(session_name, default_fps=30)
    print(f"\n--- Export pipeline test ---")
    print(f"  Session: {session_name}, FPS: {fps}")

    # Apply ML threshold (same logic as notebook cell-32)
    component_indices = est.idx_components.copy()
    n_before = len(component_indices)
    ml_filtered = False

    if hasattr(est, 'metrics_df') and est.metrics_df is not None:
        df = est.metrics_df
        if 'ml_keep_probability' in df.columns:
            threshold = DEFAULT_ML_THRESHOLD
            if hasattr(est, 'autoinspection_config') and est.autoinspection_config:
                threshold = est.autoinspection_config.get('ml_threshold', DEFAULT_ML_THRESHOLD)

            ml_approved = set(df.loc[df['ml_keep_probability'] >= threshold, 'component_idx'].tolist())
            component_indices = np.array([i for i in component_indices if i in ml_approved])
            ml_filtered = True
            print(f"  ML filter (>={threshold}): {n_before} -> {len(component_indices)}")

    n_exported = len(component_indices)
    if n_exported == 0:
        errors.append("No components passed ML filter!")
        return errors

    # Extract data
    data = extract_data(est, component_indices)

    # Validate C shape
    if data['C'].shape[0] != n_exported:
        errors.append(f"Exported C rows={data['C'].shape[0]} != n_exported={n_exported}")

    # Validate ASP shape
    if 'asp' in data:
        if data['asp'].shape[0] != n_exported:
            errors.append(f"Exported ASP rows={data['asp'].shape[0]} != n_exported={n_exported}")
        zero_rows = np.sum(np.all(data['asp'] == 0, axis=1))
        if zero_rows > 0:
            errors.append(f"ASP has {zero_rows} all-zero rows (missing asp_cache entries)")
    else:
        print(f"  (no asp_cache)")

    # Validate reconstructions shape
    if 'reconstructions' in data:
        if data['reconstructions'].shape[0] != n_exported:
            errors.append(f"Exported recon rows={data['reconstructions'].shape[0]} != n_exported={n_exported}")
    else:
        print(f"  (no reconstructions)")

    # Build metadata
    ml_filter_info = {
        'ml_filtered': ml_filtered,
        'threshold_used': threshold if ml_filtered else None,
        'n_before': n_before,
        'n_after': n_exported if ml_filtered else None
    }
    metadata = build_metadata(est, fps, session_name,
                               ml_filter_info=ml_filter_info,
                               component_indices=component_indices)

    # Validate metadata metrics_df count matches exported count
    if 'metrics_df' in metadata and metadata['metrics_df']:
        n_metadata_rows = len(metadata['metrics_df'].get('component_idx', []))
        if n_metadata_rows != n_exported:
            errors.append(f"Metadata metrics_df rows={n_metadata_rows} != n_exported={n_exported}")

        meta_indices = metadata['metrics_df'].get('component_idx', [])
        if meta_indices:
            max_idx = max(meta_indices)
            if max_idx >= est.C.shape[0]:
                errors.append(f"Metadata component_idx max={max_idx} >= C rows={est.C.shape[0]}")

    # Export to disk
    output_dir = Path(tmpdir)
    npz_path, json_path = export(data, metadata, session_name, output_dir)

    # Re-load and validate
    loaded = np.load(npz_path)
    if loaded['C'].shape[0] != n_exported:
        errors.append(f"Reloaded C shape mismatch: {loaded['C'].shape[0]} != {n_exported}")

    if 'asp' in loaded and loaded['asp'].shape[0] != n_exported:
        errors.append(f"Reloaded ASP shape mismatch: {loaded['asp'].shape[0]} != {n_exported}")

    # Export filters
    mat_path = output_dir / f'{session_name}_filters.mat'
    export_filters_mat(est, component_indices, mat_path)

    if mat_path.exists():
        from scipy.io import loadmat
        mat_data = loadmat(mat_path)
        if mat_data['A'].shape[0] != n_exported:
            errors.append(f"MAT filters shape={mat_data['A'].shape[0]} != n_exported={n_exported}")

    status = "PASS" if not errors else "FAIL"
    print(f"\n[{status}] Export pipeline")
    print(f"  Exported {n_exported} components")
    if errors:
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        outputs = ["C", "metadata"]
        if 'asp' in data:
            outputs.append("ASP")
        if 'reconstructions' in data:
            outputs.append("reconstructions")
        if mat_path.exists():
            outputs.append("MAT filters")
        print(f"  All outputs aligned: {', '.join(outputs)}")
    return errors


def main():
    # Pick test file
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
    elif DEFAULT_FILE.exists():
        test_file = DEFAULT_FILE
    else:
        print("No test file found. Provide path as argument.")
        sys.exit(1)

    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        sys.exit(1)

    all_errors = []

    # =====================================================
    # TEST 1: Run full autoinspection (metrics + ML + compression)
    # =====================================================
    print("=" * 60)
    print("TEST 1: Full autoinspection pipeline")
    print("=" * 60)
    print(f"Input: {test_file.name} ({test_file.stat().st_size / 1024**2:.0f} MB)")

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmpdir:
        result = run_auto_inspection(
            str(test_file),
            fps=None,
            session_name=None,

            # Metrics extraction (same as notebook cell-27)
            comps_to_select=None,
            cthr=0.35,
            corr_thr=0.6,
            num_sessions=1,
            match_threshold=3,
            sf=None,
            ef=None,
            ds=1,
            include_event_based=True,
            include_heavy=True,
            detect_corner_artifacts=False,
            corner_artifact_params=None,
            event_method='wavelet',
            n_iter=3,
            correlation_method='pearson',
            wavelet_backend='auto',
            hybrid_kinetics=True,

            # ML model
            brain='hybrid',
            ml_model_path='production_models/ebm_v9_iter8.pkl',
            ml_threshold=0.72,

            # Decision parameters
            deletion_rules=None,
            pxlthr_distance_boundary=5,
            d_snr_thr=10,
            enable_merge=True,

            # Tracking
            track_criteria_failures=True,

            # Save artifacts to temp dir
            save_artifacts=True,
            artifacts_path=tmpdir,
            save_estimates=True,
            save_matrices=True,
            save_corner_detection=True,
            compress_estimates=True,

            verbose=True
        )

        elapsed = time.time() - t0
        est = result['estimates']
        summary = result.get('summary', {})
        print(f"\n  Autoinspection completed in {elapsed:.1f}s")
        print(f"  Summary: {summary}")

        # =====================================================
        # TEST 2: Check alignment after autoinspection + compression
        # =====================================================
        print("\n" + "=" * 60)
        print("TEST 2: Alignment after autoinspection + compression")
        print("=" * 60)
        errs = check_alignment(est, "post-autoinspection")
        all_errors.extend(errs)

        # =====================================================
        # TEST 3: Check the saved processed pickle too
        # =====================================================
        print("\n" + "=" * 60)
        print("TEST 3: Alignment of saved processed pickle")
        print("=" * 60)
        # Find the saved processed pickle in tmpdir
        saved_pickles = list(Path(tmpdir).rglob('*_processed.pickle'))
        if saved_pickles:
            saved_path = saved_pickles[0]
            print(f"  Loading saved: {saved_path.name} ({saved_path.stat().st_size / 1024**2:.0f} MB)")
            est_saved = load_estimates(saved_path)
            errs = check_alignment(est_saved, "saved processed pickle")
            all_errors.extend(errs)
        else:
            print("  No saved processed pickle found in artifacts dir")

        # =====================================================
        # TEST 4: Full export pipeline
        # =====================================================
        print("\n" + "=" * 60)
        print("TEST 4: Full export pipeline")
        print("=" * 60)
        session_name = extract_session_name(test_file)
        with tempfile.TemporaryDirectory() as export_dir:
            errs = test_export_pipeline(est, export_dir, session_name)
            all_errors.extend(errs)

    # =====================================================
    # SUMMARY
    # =====================================================
    print("\n" + "=" * 60)
    if all_errors:
        print(f"FAILED: {len(all_errors)} errors found")
        for e in all_errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        print("  Full pipeline: autoinspect -> compress -> export -> validate")
        sys.exit(0)


if __name__ == '__main__':
    main()
