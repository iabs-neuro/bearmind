"""
ae_launch.py - Primary user interface for BEARMiND auto-inspection pipeline.

This module provides the main entry point for running automated neuron quality
inspection on CaImAn estimates files.

Example usage:
    from ae_launch import run_auto_inspection

    result = run_auto_inspection(
        'path/to/estimates.pickle',
        fps=20,
        save_artifacts=True,
        artifacts_path='./output'
    )

    est = result['estimates']  # Modified estimates object
    print(f"Kept {len(est.idx_components)} neurons")
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse

from bm_examinator import LoadEstimates
from naming import extract_session_id, extract_base_session
from bearmind.config import load_defaults
from bearmind.timestamps import FPS_TABLE, fps_from_table

# The parameters the autoinspect stage passes (the autoinspect section of bearmind/defaults.yaml, the one source)
# default to the same values here, so a direct call runs as the batch does. The parameters that section does not
# hold keep their own defaults below: the stage runs with them and its hash does not include them, so changing one
# would change autoinspection without re-running it (audit row N22).
_AUTOINSPECT = load_defaults()["autoinspect"]
# model_path in defaults.yaml is relative to the repo root, as in the stage; not to the folder Python started in
_DEFAULT_MODEL = str(Path(__file__).resolve().parent / _AUTOINSPECT["model_path"])


def get_fps_from_table(session_name: str, default_fps: int | None = None) -> int:
    """
    FPS of a session from fps_data.csv (bearmind.timestamps.fps_from_table), rounded to an integer.

    Rounded on purpose: the ML model's features and the paper data were built with integer fps (the autoinspect
    stage rounds the timestamps' rate the same way). Export reads the same table without rounding.

    Args:
        session_name: Session ID, file name or path containing it (CODE_MOUSE_DAY[_TRIAL], e.g. LNOF_J53_3D,
            3DM_D17_1D_1T, path/NOF_H01_1D.pickle)
        default_fps: Returned when the session is not in the table. None (default): that is an error, since a
            silent default runs autoinspection at a rate the session may not have.

    Examples:
        >>> get_fps_from_table('NOF_H01_1D')
        20
        >>> get_fps_from_table('3DM_D17_1D_1T')
        30
    """
    fps = fps_from_table(session_name)
    if fps is None:
        if default_fps is None:
            raise ValueError(f"FPS not found for session {session_name!r} in {FPS_TABLE}; pass fps= or add the "
                             f"session to the table")
        return default_fps
    return round(fps)


from auto_inspector import (
    estimates_to_metrics,
    metrics_to_decision,
    implement_decision,
    save_processed_estimates,
    transform_metrics_df_indices
)
from ae_utils import visualize_corner_artifacts
from export_estimates_data import MERGE_GROUPING


def _save_inspection_artifacts(
    session_name: str,
    decision_df: pd.DataFrame,
    match_mtx: np.ndarray,
    FCD: np.ndarray,
    FBD: np.ndarray,
    edge_info: dict,
    est_processed=None,
    artifacts_path: str = None,
    save_matrices: bool = True,
    save_corner_detection: bool = True,
    compress_estimates: bool = False,
    artifacts_dir: str = None
) -> Path:
    """
    Save all inspection artifacts to a folder.

    Args:
        session_name: Name for the session (used in folder name)
        decision_df: DataFrame with metrics and decision columns
        match_mtx: Correlation match matrix
        FCD: Footprint center distance matrix
        FBD: Footprint boundary distance matrix
        edge_info: Corner artifact detection info dict
        est_processed: Processed estimates object (None to skip saving)
        artifacts_path: Base path for artifacts folder
        save_matrices: Whether to save FCD, FBD, match_mtx as .npy files
        save_corner_detection: Whether to save corner artifact visualization
        compress_estimates: Whether to compress estimates before saving
        artifacts_dir: If given, write directly into this folder (no
            inspection_artifacts_<session> subfolder; artifacts_path is ignored)

    Returns:
        Path to the created artifacts folder
    """
    base = Path(artifacts_path) if artifacts_path else Path('.')

    # Extract session name (without timestamp, trial suffix kept) for folder and filenames
    # E.g., "LNOF_J01_1D_22-12-2025 14-20-16" -> "LNOF_J01_1D"
    session_prefix = extract_session_id(session_name) or session_name

    folder = Path(artifacts_dir) if artifacts_dir else base / f'inspection_artifacts_{session_prefix}'
    folder.mkdir(parents=True, exist_ok=True)

    # Save full metrics with decisions
    decision_df.to_csv(folder / f'{session_prefix}_metrics_with_decisions.csv', index=False)

    # Save rejected neurons summary
    rejected = decision_df[decision_df['delete'] == 1]
    if len(rejected) > 0:
        rejected.to_csv(folder / f'{session_prefix}_rejected_neurons.csv', index=False)

    # Save distance/correlation matrices
    if save_matrices:
        np.save(folder / f'{session_prefix}_FCD.npy', FCD)
        np.save(folder / f'{session_prefix}_FBD.npy', FBD)
        np.save(folder / f'{session_prefix}_match_mtx.npy', match_mtx)

    # Save corner artifact visualization if requested and available
    if save_corner_detection and edge_info is not None and 'is_corner_artifact' in decision_df.columns:
        try:
            visualize_corner_artifacts(decision_df, edge_info, folder, session_name)
        except Exception as e:
            print(f"[WARNING] Failed to create corner artifact visualization: {e}")
            # Visualization is optional, continue without it

    # Save processed estimates
    if est_processed is not None:
        save_processed_estimates(est_processed, folder, session_name, compress=compress_estimates)

    return folder


def run_auto_inspection(
    estimates_path: str,
    fps: int = None,
    *,
    # --- Session naming ---
    session_name: str = None,

    # --- Metrics extraction parameters ---
    comps_to_select: list = None,
    cthr: float = _AUTOINSPECT['cthr'],
    corr_thr: float = _AUTOINSPECT['corr_thr'],
    num_sessions: int = 1,
    match_threshold: int = 3,
    sf: int = None,
    ef: int = None,
    ds: int = 1,
    include_event_based: bool = _AUTOINSPECT['include_event_based'],
    include_heavy: bool = _AUTOINSPECT['include_heavy'],
    detect_corner_artifacts: bool = _AUTOINSPECT['detect_corner_artifacts'],
    corner_artifact_params: dict = None,
    event_method: str = _AUTOINSPECT['event_method'],
    correlation_method: str = _AUTOINSPECT['correlation_method'],
    n_iter: int = _AUTOINSPECT['n_iter'],
    hybrid_kinetics: bool = _AUTOINSPECT['hybrid_kinetics'],
    wavelet_backend: str = 'auto',
    seed: int | None = _AUTOINSPECT['seed'],

    # --- Decision parameters ---
    deletion_rules: list = None,  # Use DEFAULT_DELETION_RULES if None
    pxlthr_distance_boundary: float = _AUTOINSPECT['pxlthr_distance_boundary'],  # For merge detection
    d_snr_thr: float = _AUTOINSPECT['d_snr_thr'],  # For merge detection
    enable_merge: bool = _AUTOINSPECT['enable_merge'],  # Renamed from use_corr_check

    # --- Brain selection ---
    brain: str = _AUTOINSPECT['brain'],
    ml_model_path: str = _DEFAULT_MODEL,
    ml_threshold: float = _AUTOINSPECT['ml_threshold'],

    # --- Tracking ---
    track_criteria_failures: bool = True,

    # --- Artifact saving ---
    save_artifacts: bool = True,
    artifacts_path: str = './output',
    save_estimates: bool = True,
    save_matrices: bool = True,
    save_corner_detection: bool = True,
    compress_estimates: bool = True,
    artifacts_dir: str = None,

    # --- Verbosity ---
    verbose: bool = False
) -> dict:
    """
    Run the full auto-inspection pipeline on a CaImAn estimates file.

    This is the primary user interface for the BEARMiND auto-inspection system.
    It loads estimates, extracts metrics, makes deletion/merge decisions, applies
    them to the estimates, and optionally saves all artifacts.

    Args:
        estimates_path: Path to the CaImAn estimates pickle file
        fps: Imaging frame rate in Hz (None: get_fps_from_table, an error when the session is not in fps_data.csv)

        session_name: Name for the session (auto-derived from filename if None)

        comps_to_select: Component indices to process (None = all idx_components)
        cthr: Contour threshold for footprint extraction
        corr_thr: Correlation threshold for multi-session matching
        num_sessions: Number of sessions for correlation analysis
        match_threshold: Correlation match threshold (adjusted by num_sessions)
        sf: Start frame for trace analysis (None = 0)
        ef: End frame for trace analysis (None = end of recording)
        ds: Downsample factor for traces
        include_event_based: Compute event-based temporal metrics
        include_heavy: Compute reconstruction quality metrics (slow)
        detect_corner_artifacts: Enable corner artifact detection
        corner_artifact_params: Parameters for corner detection (None = defaults)
        event_method: Event detection method ('threshold' or 'wavelet')
        seed: Seed of driada's tie-breaking noise (1e-8 per frame); the same seed gives the same
            decisions on the same data (None: unseeded, results vary from run to run)

        circ_thr: Maximum circularity (elongation) threshold
        maxedge_thr: Maximum edge length threshold
        convex_thr: Maximum inverse convexity threshold
        pxlthr_area: Minimum footprint area in pixels
        pxlthr_distance_boundary: Distance threshold for merge detection
        d_snr_thr: SNR difference threshold for merge decisions
        t_rise_min: Minimum event rise time in seconds
        caiman_r_score_min: Minimum CaImAn spatial correlation score
        caiman_snr_min: Minimum CaImAn signal-to-noise ratio
        t_off_min: Minimum event decay time in seconds

        use_circularity_check: Enable circularity threshold check
        use_area_check: Enable area threshold check
        use_max_edge_check: Enable max edge threshold check
        use_convexity_check: Enable convexity threshold check
        use_t_rise_check: Enable rise time threshold check
        use_caiman_r_score_check: Enable CaImAn r-score check
        use_caiman_snr_check: Enable CaImAn SNR check
        use_t_off_check: Enable decay time check
        use_corr_check: Enable correlation-based merge detection

        brain: Decision brain type ('thresholds', 'ml' or 'hybrid')
        ml_model_path: Path to ML model pickle (required if brain is 'ml' or 'hybrid')
        ml_threshold: P(KEEP) threshold for ML brain

        Defaults of the parameters held by the autoinspect section of bearmind/defaults.yaml come from it
        (model_path relative to the repo root).

        track_criteria_failures: Track which criteria each neuron fails

        save_artifacts: Save all artifacts to folder
        artifacts_path: Base path for artifacts (None = same dir as estimates)
        save_estimates: Save processed estimates pickle
        save_matrices: Save FCD, FBD, match_mtx as .npy files
        save_corner_detection: Save corner artifact visualization (default True)
        compress_estimates: Compress estimates before saving (removes bad components,
            converts to float32, sparse S matrix). Default: False

        verbose: Print progress messages

    Returns:
        dict with keys:
            'estimates': Modified estimates object with decisions applied
            'metrics_df': Full metrics DataFrame with decision columns
            'match_mtx': Correlation match matrix
            'FCD': Footprint center distance matrix
            'FBD': Footprint boundary distance matrix
            'edge_info': Corner artifact detection info
            'artifacts_folder': Path to saved artifacts (None if not saved)
            'summary': Dict with summary statistics:
                - n_initial: Initial neuron count
                - n_deleted: Neurons marked for deletion
                - n_merged: Neurons involved in merge groups
                - n_corner_artifacts: Corner artifacts detected
                - n_kept_before_merge: Neurons kept after deletion, before merging
                - n_final: Final neuron count after all operations

    Raises:
        FileNotFoundError: If estimates_path does not exist
        ValueError: If brain='ml' but ml_model_path is None
    """
    # --- Validate inputs ---
    estimates_path = Path(estimates_path)
    if not estimates_path.exists():
        raise FileNotFoundError(f"Estimates file not found: {estimates_path}")

    # Auto-lookup fps from table if not provided
    if fps is None:
        fps = get_fps_from_table(estimates_path.stem)

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    if brain in ('ml', 'hybrid') and ml_model_path is None:
        raise ValueError(f"brain='{brain}' requires ml_model_path to be specified")

    # --- Validate ML model path early (before expensive metrics computation) ---
    if brain in ('ml', 'hybrid'):
        ml_model_path = Path(ml_model_path)
        if not ml_model_path.exists():
            raise FileNotFoundError(
                f"ML model file not found: {ml_model_path}\n"
                f"Please check the path before running inspection."
            )
        # Verify it's a valid model with predict_proba
        try:
            import pickle
            with open(ml_model_path, 'rb') as f:
                model = pickle.load(f)
            if not hasattr(model, 'predict_proba'):
                raise ValueError(f"Model at {ml_model_path} does not have predict_proba method")
            del model  # Free memory
        except (pickle.UnpicklingError, EOFError) as e:
            raise ValueError(f"Invalid pickle file: {ml_model_path}\nError: {e}")

    # --- Derive session name from filename if not provided ---
    if session_name is None:
        session_name = estimates_path.stem
        # Remove common suffixes
        for suffix in ['_estimates', '_processed']:
            if session_name.endswith(suffix):
                session_name = session_name[:-len(suffix)]

    if verbose:
        print(f"[run_auto_inspection] Session: {session_name}")
        print(f"[run_auto_inspection] FPS: {fps}")
        print(f"[run_auto_inspection] Loading estimates from: {estimates_path}")

    # --- Step 1: Load estimates ---
    est = LoadEstimates(str(estimates_path), default_fps=fps)

    if verbose:
        print(f"[run_auto_inspection] Loaded {len(est.idx_components)} components")

    # Short-circuit on empty estimates: nothing to inspect, decision/merge pipeline
    # would fail on an empty metrics_df (missing columns like 'corr_groups').
    if len(est.idx_components) == 0:
        if verbose:
            print("[run_auto_inspection] 0 components — skipping metrics/decision pipeline.")
        est.metrics_df = pd.DataFrame()
        est.reconstructions = {}
        est.asp_cache = {}
        est.active_deletion_metrics = []
        est.autoinspection_config = {
            'fps': fps,
            'brain': brain,
            'ml_model_path': str(ml_model_path) if ml_model_path else None,
            'ml_threshold': ml_threshold,
            'deletion_rules': deletion_rules if deletion_rules else 'DEFAULT_DELETION_RULES',
            'seed': seed,
            'merge_grouping': MERGE_GROUPING,
        }
        return {
            'estimates': est,
            'metrics_df': pd.DataFrame(),
            'match_mtx': None,
            'FCD': np.array([]),
            'FBD': np.array([]),
            'edge_info': None,
            'artifacts_folder': None,
            'summary': {
                'n_initial': 0,
                'n_deleted': 0,
                'n_merged': 0,
                'n_corner_artifacts': 0,
                'n_kept_before_merge': 0,
                'n_final': 0,
            },
        }

    # --- Step 2: Extract metrics ---
    if verbose:
        print("[run_auto_inspection] Extracting metrics...")

    metrics_df, match_mtx, FCD, FBD, edge_info, reconstructions, asp_cache = estimates_to_metrics(
        est, fps,
        comps_to_select=comps_to_select if comps_to_select else [],
        cthr=cthr,
        corr_thr=corr_thr,
        num_sessions=num_sessions,
        match_threshold=match_threshold,
        sf=sf,
        ef=ef,
        ds=ds,
        include_event_based=include_event_based,
        include_heavy=include_heavy,
        detect_corner_artifacts_flag=detect_corner_artifacts,
        corner_artifact_params=corner_artifact_params,
        event_method=event_method,
        correlation_method=correlation_method,
        n_iter=n_iter,
        hybrid_kinetics=hybrid_kinetics,
        wavelet_backend=wavelet_backend,
        seed=seed
    )

    if verbose:
        print(f"[run_auto_inspection] Extracted metrics for {len(metrics_df)} neurons")
        if 'is_corner_artifact' in metrics_df.columns:
            n_corner = (metrics_df['is_corner_artifact'] == 1).sum()
            print(f"[run_auto_inspection] Detected {n_corner} corner artifacts")

    # --- Step 3: Make decisions ---
    if verbose:
        print(f"[run_auto_inspection] Making decisions with brain='{brain}'...")

    decision_df = metrics_to_decision(
        metrics_df.copy(),
        match_mtx,
        FCD,
        FBD,
        deletion_rules=deletion_rules,
        pxlthr_distance_boundary=pxlthr_distance_boundary,
        d_snr_thr=d_snr_thr,
        enable_merge=enable_merge,
        brain=brain,
        ml_model_path=ml_model_path,
        ml_threshold=ml_threshold,
        track_criteria_failures=track_criteria_failures
    )

    # Extract active metrics for GUI filtering (only for threshold/hybrid brains)
    if brain in ['thresholds', 'hybrid']:
        from auto_inspector import get_active_metrics_from_rules, DEFAULT_DELETION_RULES
        active_deletion_metrics = get_active_metrics_from_rules(
            deletion_rules if deletion_rules else DEFAULT_DELETION_RULES
        )
    else:
        active_deletion_metrics = []

    # --- Step 4: Add 'decision' column for implement_decision ---
    decision_df['decision'] = decision_df['delete'].apply(
        lambda x: 'delete' if x == 1 else 'ok'
    )

    n_deleted = (decision_df['delete'] == 1).sum()
    n_merged = (decision_df['merge'] > 0).sum()

    if verbose:
        print(f"[run_auto_inspection] Decisions: {n_deleted} deleted, {n_merged} in merge groups")

    # --- Step 5: Convert sparse S matrix to dense (required for manual_merge) ---
    if sparse.issparse(est.S):
        est.S = est.S.toarray()

    # --- Step 6: Apply decisions to estimates ---
    if verbose:
        print("[run_auto_inspection] Applying decisions to estimates...")
        print(f"  Before: {len(est.idx_components)} neurons in est.idx_components")
        print(f"  Marked for deletion: {n_deleted} neurons")
        print(f"  Neurons in merge groups: {n_merged}")

    est_processed, mapping_info = implement_decision(est, decision_df, return_index_mapping=True)

    if verbose:
        print(f"[run_auto_inspection] Final: {len(est_processed.idx_components)} components kept")
        print(f"  Index mapping: {mapping_info['nr_before']} -> {mapping_info['nr_after']} components")
        print(f"  Merged groups: {len(mapping_info['merged_groups'])}")
        print(f"  Deleted: {len(mapping_info['deleted'])} components")

    # Transform metrics_df indices using the mapping
    # This fixes the critical index mismatch bug where manual_merge renumbers all indices
    if verbose:
        print("[run_auto_inspection] Transforming metrics_df indices to match post-merge estimates...")

    transformed_df = transform_metrics_df_indices(
        decision_df,
        mapping_info,
        est_processed,
        fps=fps,
        cthr=cthr,
        include_event_based=include_event_based,
        event_method=event_method,
        seed=seed
    )

    est_processed.metrics_df = transformed_df

    # Attach active deletion metrics for GUI filtering
    est_processed.active_deletion_metrics = active_deletion_metrics

    # Attach reconstructions with transformed indices
    if include_heavy and reconstructions:
        # Transform reconstruction indices using the mapping
        transformed_recons = {}
        old_to_new = mapping_info.get('old_to_new', {})
        for old_idx, rec in reconstructions.items():
            if old_idx in old_to_new:
                new_idx = old_to_new[old_idx]
                transformed_recons[new_idx] = rec
        est_processed.reconstructions = transformed_recons
        if verbose:
            print(f"[run_auto_inspection] Attached {len(transformed_recons)} reconstructions")
    else:
        est_processed.reconstructions = {}

    # Attach ASP cache with transformed indices
    if asp_cache:
        transformed_asp = {}
        old_to_new = mapping_info.get('old_to_new', {})
        for old_idx, asp in asp_cache.items():
            if old_idx in old_to_new:
                new_idx = old_to_new[old_idx]
                transformed_asp[new_idx] = asp
        est_processed.asp_cache = transformed_asp
        if verbose:
            print(f"[run_auto_inspection] Attached {len(transformed_asp)} ASP arrays")
    else:
        est_processed.asp_cache = {}

    # Store autoinspection config for reproducibility and export
    est_processed.autoinspection_config = {
        'fps': fps,
        'brain': brain,
        'ml_model_path': str(ml_model_path) if ml_model_path else None,
        'ml_threshold': ml_threshold,
        'deletion_rules': deletion_rules if deletion_rules else 'DEFAULT_DELETION_RULES',
        'event_method': event_method,
        'n_iter': n_iter,
        'hybrid_kinetics': hybrid_kinetics,
        'cthr': cthr,
        'corr_thr': corr_thr,
        'include_event_based': include_event_based,
        'include_heavy': include_heavy,
        'detect_corner_artifacts': detect_corner_artifacts,
        'correlation_method': correlation_method,
        'wavelet_backend': wavelet_backend,
        'enable_merge': enable_merge,
        'pxlthr_distance_boundary': pxlthr_distance_boundary,
        'd_snr_thr': d_snr_thr,
        'seed': seed,
        'merge_grouping': MERGE_GROUPING,
    }

    # Verify metrics_df was attached and indices are valid
    if verbose:
        if hasattr(est_processed, 'metrics_df') and est_processed.metrics_df is not None:
            n_rows = len(est_processed.metrics_df)
            n_merged = (est_processed.metrics_df['decision'] == 'from_merge').sum()
            print(f"[run_auto_inspection] Attached transformed metrics_df with {n_rows} rows ({n_merged} merged)")

            # Validate ml_keep_probability values
            if 'ml_keep_probability' in est_processed.metrics_df.columns:
                prob_col = est_processed.metrics_df['ml_keep_probability']
                n_valid = prob_col.notna().sum()
                n_nan = prob_col.isna().sum()
                print(f"[run_auto_inspection] ml_keep_probability: {n_valid}/{n_rows} valid, {n_nan} NaN/None")

            # Validate indices
            max_idx = est_processed.metrics_df['component_idx'].max()
            if max_idx < est_processed.C.shape[0]:
                print(f"[run_auto_inspection] Index validation PASSED: max_idx={max_idx} < C.shape[0]={est_processed.C.shape[0]}")
            else:
                print(f"[run_auto_inspection] WARNING: Index validation FAILED: max_idx={max_idx} >= C.shape[0]={est_processed.C.shape[0]}")
        else:
            print("[run_auto_inspection] WARNING: Failed to attach metrics_df!")

    # --- Step 7: Save artifacts ---
    artifacts_folder = None
    if save_artifacts:
        if artifacts_path is None:
            artifacts_path = estimates_path.parent

        if verbose:
            print(f"[run_auto_inspection] Saving artifacts to: {artifacts_path}")

        artifacts_folder = _save_inspection_artifacts(
            session_name=session_name,
            decision_df=decision_df,
            match_mtx=match_mtx,
            FCD=FCD,
            FBD=FBD,
            edge_info=edge_info,
            est_processed=est_processed if save_estimates else None,
            artifacts_path=str(artifacts_path),
            save_matrices=save_matrices,
            save_corner_detection=save_corner_detection,
            compress_estimates=compress_estimates,
            artifacts_dir=artifacts_dir
        )

        if verbose:
            print(f"[run_auto_inspection] Artifacts saved to: {artifacts_folder}")

    # --- Step 8: Compute summary statistics ---
    n_corner_artifacts = 0
    if 'is_corner_artifact' in decision_df.columns:
        n_corner_artifacts = int((decision_df['is_corner_artifact'] == 1).sum())

    # Count initial neurons from original estimates (before any processing)
    n_initial_est = len(est.idx_components)
    n_initial_metrics = len(metrics_df)
    n_kept = (decision_df['delete'] == 0).sum()
    n_final = len(est_processed.idx_components)

    summary = {
        'n_initial': n_initial_est,
        'n_deleted': int(n_deleted),
        'n_merged': int(n_merged),
        'n_corner_artifacts': n_corner_artifacts,
        'n_kept_before_merge': int(n_kept),
        'n_final': n_final
    }

    if verbose:
        print(f"[run_auto_inspection] Summary: {summary}")

        # Sanity check
        expected_after_delete = n_initial_est - n_deleted
        if n_kept != expected_after_delete:
            print(f"[WARNING] Count mismatch: expected {expected_after_delete} after deletion, but decision_df shows {n_kept} kept")
        if n_initial_est != n_initial_metrics:
            print(f"[WARNING] Initial count mismatch: est.idx_components={n_initial_est}, metrics_df={n_initial_metrics}")

    # --- Return results ---
    return {
        'estimates': est_processed,
        'metrics_df': decision_df,
        'match_mtx': match_mtx,
        'FCD': FCD,
        'FBD': FBD,
        'edge_info': edge_info,
        'artifacts_folder': artifacts_folder,
        'summary': summary
    }


if __name__ == '__main__':
    # Example usage
    import argparse

    parser = argparse.ArgumentParser(description='Run auto-inspection on estimates file')
    parser.add_argument('estimates_path', help='Path to estimates pickle file')
    parser.add_argument('--fps', type=int, default=None, help='Imaging frame rate (auto from fps_data.csv if not provided)')
    # defaults: the autoinspect section of bearmind/defaults.yaml, as for the function
    parser.add_argument('--brain', choices=['thresholds', 'ml', 'hybrid'], default=_AUTOINSPECT['brain'],
                        help='Decision brain type (default: %(default)s)')
    parser.add_argument('--ml-model', type=str, default=_DEFAULT_MODEL,
                        help='Path to ML model (default: %(default)s)')
    parser.add_argument('--ml-threshold', type=float, default=_AUTOINSPECT['ml_threshold'],
                        help='ML classification threshold (default: %(default)s)')
    parser.add_argument('--n-iter', type=int, default=_AUTOINSPECT['n_iter'],
                        help='Number of iterations for event reconstruction (default: %(default)s)')
    parser.add_argument('--deletion-rules', type=str, nargs='+', default=None,
                        help='Deletion rules (e.g., "area<1" "circularity>4"). Default: use DEFAULT_DELETION_RULES')
    parser.add_argument('--enable-merge', dest='enable_merge', action='store_true', default=True,
                        help='Enable correlation-based merge detection (default: True)')
    parser.add_argument('--no-merge', dest='enable_merge', action='store_false',
                        help='Disable merge detection')
    parser.add_argument('--output', type=str, default='./output',
                        help='Output directory for artifacts (default: ./output)')
    parser.add_argument('--no-save', action='store_true', help='Disable artifact saving')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress')

    args = parser.parse_args()

    result = run_auto_inspection(
        args.estimates_path,
        fps=args.fps,
        deletion_rules=args.deletion_rules,
        enable_merge=args.enable_merge,
        brain=args.brain,
        ml_model_path=args.ml_model,
        ml_threshold=args.ml_threshold,
        n_iter=args.n_iter,
        artifacts_path=args.output,
        save_artifacts=not args.no_save,
        verbose=args.verbose
    )

    summary = result['summary']
    print(f"\nAuto-inspection complete:")
    print(f"  Initial neurons: {summary['n_initial']}")
    print(f"  Deleted: {summary['n_deleted']}")
    print(f"  Corner artifacts: {summary['n_corner_artifacts']}")
    print(f"  Final neurons: {summary['n_final']}")

    if result['artifacts_folder']:
        print(f"  Artifacts saved to: {result['artifacts_folder']}")
