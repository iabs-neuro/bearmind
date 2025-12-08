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
from auto_inspector import (
    estimates_to_metrics,
    metrics_to_decision,
    implement_decision,
    save_processed_estimates
)
from ae_utils import visualize_corner_artifacts


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
    compress_estimates: bool = False
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

    Returns:
        Path to the created artifacts folder
    """
    base = Path(artifacts_path) if artifacts_path else Path('.')
    folder = base / f'inspection_artifacts_{session_name}'
    folder.mkdir(parents=True, exist_ok=True)

    # Save full metrics with decisions
    decision_df.to_csv(folder / 'metrics_with_decisions.csv', index=False)

    # Save rejected neurons summary
    rejected = decision_df[decision_df['delete'] == 1]
    if len(rejected) > 0:
        rejected.to_csv(folder / 'rejected_neurons.csv', index=False)

    # Save distance/correlation matrices
    if save_matrices:
        np.save(folder / 'FCD.npy', FCD)
        np.save(folder / 'FBD.npy', FBD)
        np.save(folder / 'match_mtx.npy', match_mtx)

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
    fps: float,
    *,
    # --- Session naming ---
    session_name: str = None,

    # --- Metrics extraction parameters ---
    comps_to_select: list = None,
    cthr: float = 0.3,
    corr_thr: float = 0.6,
    num_sessions: int = 1,
    match_threshold: int = 3,
    sf: int = None,
    ef: int = None,
    ds: int = 1,
    include_event_based: bool = True,
    include_heavy: bool = False,
    detect_corner_artifacts: bool = True,
    corner_artifact_params: dict = None,
    event_method: str = 'threshold',

    # --- Decision parameters (threshold brain) ---
    circ_thr: float = 4,
    maxedge_thr: float = 42,
    convex_thr: float = 42,
    pxlthr_area: float = 6.9,
    pxlthr_distance_boundary: float = 5,
    d_snr_thr: float = 10,
    t_rise_min: float = 0.10,
    caiman_r_score_min: float = 0.05,
    caiman_snr_min: float = 2.9,
    t_off_min: float = 1.5,

    # --- Enable/disable checks ---
    use_circularity_check: bool = True,
    use_area_check: bool = True,
    use_max_edge_check: bool = True,
    use_convexity_check: bool = True,
    use_t_rise_check: bool = True,
    use_caiman_r_score_check: bool = True,
    use_caiman_snr_check: bool = True,
    use_t_off_check: bool = True,
    use_corr_check: bool = True,

    # --- Brain selection ---
    brain: str = 'thresholds',
    ml_model_path: str = None,
    ml_threshold: float = 0.5,

    # --- Tracking ---
    track_criteria_failures: bool = True,

    # --- Artifact saving ---
    save_artifacts: bool = True,
    artifacts_path: str = None,
    save_estimates: bool = True,
    save_matrices: bool = True,
    save_corner_detection: bool = True,
    compress_estimates: bool = False,

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
        fps: Imaging frame rate in Hz

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

        brain: Decision brain type ('thresholds' or 'ml')
        ml_model_path: Path to ML model pickle (required if brain='ml')
        ml_threshold: P(KEEP) threshold for ML brain (default 0.5)

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
        print(f"[run_auto_inspection] Loading estimates from: {estimates_path}")

    # --- Step 1: Load estimates ---
    est = LoadEstimates(str(estimates_path), default_fps=fps)

    if verbose:
        print(f"[run_auto_inspection] Loaded {len(est.idx_components)} components")

    # --- Step 2: Extract metrics ---
    if verbose:
        print("[run_auto_inspection] Extracting metrics...")

    metrics_df, match_mtx, FCD, FBD, edge_info = estimates_to_metrics(
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
        event_method=event_method
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
        circ_thr=circ_thr,
        maxedge_thr=maxedge_thr,
        convex_thr=convex_thr,
        pxlthr_area=pxlthr_area,
        pxlthr_distance_boundary=pxlthr_distance_boundary,
        d_snr_thr=d_snr_thr,
        t_rise_min=t_rise_min,
        caiman_r_score_min=caiman_r_score_min,
        caiman_snr_min=caiman_snr_min,
        t_off_min=t_off_min,
        use_circularity_check=use_circularity_check,
        use_area_check=use_area_check,
        use_max_edge_check=use_max_edge_check,
        use_convexity_check=use_convexity_check,
        use_t_rise_check=use_t_rise_check,
        use_caiman_r_score_check=use_caiman_r_score_check,
        use_caiman_snr_check=use_caiman_snr_check,
        use_t_off_check=use_t_off_check,
        use_corr_check=use_corr_check,
        brain=brain,
        ml_model_path=ml_model_path,
        ml_threshold=ml_threshold,
        track_criteria_failures=track_criteria_failures
    )

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

    est_processed = implement_decision(est, decision_df)

    if verbose:
        print(f"[run_auto_inspection] Final: {len(est_processed.idx_components)} components kept")
        print(f"  Deleted: {len(est.idx_components_bad)} total bad components")
        print(f"  Added by deletion: {len(est_processed.idx_components_bad) - len(est.idx_components_bad)} components")

    est_processed.metrics_df = decision_df

    # Verify metrics_df was attached
    if verbose:
        if hasattr(est_processed, 'metrics_df') and est_processed.metrics_df is not None:
            print(f"[run_auto_inspection] Attached metrics_df with {len(est_processed.metrics_df)} rows to estimates")
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
            compress_estimates=compress_estimates
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
    parser.add_argument('--fps', type=float, required=True, help='Imaging frame rate')
    parser.add_argument('--brain', choices=['thresholds', 'ml'], default='thresholds',
                        help='Decision brain type')
    parser.add_argument('--ml-model', type=str, help='Path to ML model (required if brain=ml)')
    parser.add_argument('--output', type=str, help='Output directory for artifacts')
    parser.add_argument('--no-save', action='store_true', help='Disable artifact saving')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print progress')

    args = parser.parse_args()

    result = run_auto_inspection(
        args.estimates_path,
        fps=args.fps,
        brain=args.brain,
        ml_model_path=args.ml_model,
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
