import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from auto_inspector import (
    estimates_to_metrics,
    metrics_to_decision,
    implement_decision,
    save_processed_estimates,
    compute_metrics,
    print_report
)
from ae_utils import save_validation_outputs

# Configuration
project_root = os.path.dirname(os.path.abspath(__file__))
val_path = os.path.join(project_root, 'data')

# Default paths (use compressed data)
init_path = os.path.join(val_path, 'raw_compressed')
gt_path = os.path.join(val_path, 'final_compressed')

# Validation deletion rules (stricter than defaults for validation)
# These match the old validation thresholds:
# - area > 3 (old) → area<=3 (delete if <=3, keep if >3)
# - circularity <= 1.7 (old) → circularity>1.7 (delete if >1.7, keep if <=1.7)
# - max_edge <= 1.45 (old) → max_edge>1.45 (delete if >1.45, keep if <=1.45)
# - convexity <= 42 (old) → convexity>42 (delete if >42, keep if <=42)
VALIDATION_DELETION_RULES = [
    'area<=3',           # Stricter: larger area required
    'circularity>1.7',   # Stricter: more circular required
    'max_edge>1.45',     # Stricter: shorter edges required
    'convexity>42'       # Same as default
]


def load_fps_data(fps_file='fps_data.csv'):
    """
    Load FPS data from CSV file.

    Args:
        fps_file: Path to FPS data CSV file

    Returns:
        dict: Mapping of session_id -> fps (rounded to nearest int)
    """
    fps_path = os.path.join(project_root, fps_file)
    fps_df = pd.read_csv(fps_path)

    # Create mapping: filename (first 10 chars) -> rounded FPS
    fps_map = {}
    for _, row in fps_df.iterrows():
        session_id = row['Filename'][:10]
        fps_rounded = int(round(row['FPS']))
        fps_map[session_id] = fps_rounded

    return fps_map


# Load FPS data
fps_map = load_fps_data()
print(f"Loaded FPS data for {len(fps_map)} sessions\n")


def build_session_mapping(init_path, gt_path):
    """
    Build session mapping from init and ground truth paths.

    Args:
        init_path: Path to raw/initial estimates
        gt_path: Path to final/ground truth estimates

    Returns:
        dict: Mapping of session_id -> [init_file, gt_file]
    """
    all_init_files = os.listdir(init_path)
    all_gt_files = os.listdir(gt_path)

    sessions = [name[:10] for name in all_gt_files]
    mapping = {}
    for session in sessions:
        init = [f for f in all_init_files if session in f][0]
        gt = [f for f in all_gt_files if session in f][0]
        mapping.update({session: [init, gt]})

    return mapping


# Build default session mapping (for module-level usage)
mapping = build_session_mapping(init_path, gt_path)
print(f"Found {len(mapping)} sessions to validate\n")


def load_estimates(filepath):
    """Load CaImAn estimates object from pickle file."""
    with open(filepath, 'rb') as f:
        estimates = pickle.load(f)
    return estimates


def validate_single_session(session_id, init_file, gt_file, fps=None, verbose=True):
    """
    Validate automated pipeline on a single session.

    Args:
        session_id: Session identifier
        init_file: Filename for raw estimates
        gt_file: Filename for final (ground truth) estimates
        fps: Frames per second (if None, will lookup from fps_map)
        verbose: Print detailed progress

    Returns:
        dict: Validation metrics
    """
    # Get FPS from map if not provided
    if fps is None:
        fps = fps_map.get(session_id, 20)  # Default to 20 if not found

    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATING SESSION: {session_id}")
        print(f"{'='*60}")
        print(f"Raw estimates: {init_file}")
        print(f"Final estimates: {gt_file}")
        print(f"FPS: {fps}\n")

    # === PART 1: Load estimates ===
    if verbose:
        print("[1/6] Loading estimates...")

    est_init = load_estimates(os.path.join(init_path, init_file))
    est_gt = load_estimates(os.path.join(gt_path, gt_file))

    n_init = len(est_init.idx_components)
    n_gt = len(est_gt.idx_components)

    if verbose:
        print(f"  Initial neurons: {n_init}")
        print(f"  Ground truth neurons: {n_gt}")
        print(f"  Manual curation removed: {n_init - n_gt} neurons\n")

    # === PART 2: Extract metrics from initial estimates ===
    if verbose:
        print("[2/6] Extracting metrics from raw estimates...")

    metrics_df_init, match_mtx_init, FCD_init, FBD_init, _, _ = estimates_to_metrics(
        est_init,
        fps=fps,
        include_event_based=True,
        include_heavy=False  # Skip reconstruction metrics for speed
    )

    if verbose:
        print(f"  Extracted {len(metrics_df_init)} neuron metrics\n")

    # === PART 3: Extract metrics from ground truth estimates ===
    if verbose:
        print("[3/6] Extracting metrics from ground truth estimates...")

    metrics_df_gt, _, _, _, _, _ = estimates_to_metrics(
        est_gt,
        fps=fps,
        include_event_based=False,
        include_heavy=False
    )

    if verbose:
        print(f"  Extracted {len(metrics_df_gt)} neuron metrics\n")

    # === PART 4: Run automated decision pipeline ===
    if verbose:
        print("[4/6] Running automated decision pipeline...")

    # Make decisions based on metrics
    metrics_df_auto = metrics_to_decision(
        metrics_df_init.copy(),
        match_mtx_init,
        FCD_init,
        FBD_init,
        deletion_rules=VALIDATION_DELETION_RULES,
        pxlthr_distance_boundary=5,
        d_snr_thr=10,
        enable_merge=True
    )

    # Convert decisions to 'decision' column format expected by implement_decision
    metrics_df_auto['decision'] = metrics_df_auto['delete'].apply(
        lambda x: 'delete' if x == 1 else 'ok'
    )

    n_delete = (metrics_df_auto['delete'] == 1).sum()
    n_merge_groups = len(metrics_df_auto[metrics_df_auto['merge'] != 0]['merge'].unique())
    n_corner = (metrics_df_auto.get('is_corner_artifact', 0) == 1).sum()

    if verbose:
        print(f"  Automated decisions:")
        print(f"    Delete: {n_delete} neurons")
        print(f"    Corner artifacts: {n_corner} neurons (will be deleted)")
        print(f"    Merge: {n_merge_groups} groups")
        print(f"    Keep: {len(metrics_df_auto) - n_delete} neurons\n")

    # Apply decisions to get automated estimates
    if verbose:
        print("[5/6] Implementing automated decisions...")

    # Convert S to dense if sparse (CaImAn's manual_merge doesn't handle sparse S)
    from scipy import sparse
    if sparse.issparse(est_init.S):
        if verbose:
            print("  Converting sparse S to dense for merging...")
        est_init.S = est_init.S.toarray()

    # Apply ALL decisions including corner artifacts
    # Corner artifacts are already marked for deletion via 'delete' column
    est_auto = implement_decision(est_init, metrics_df_auto)

    # After implementing decisions, identify which neurons were corner artifacts
    # for reporting purposes (exclude from validation metrics)
    if n_corner > 0:
        corner_indices = metrics_df_auto[metrics_df_auto.get('is_corner_artifact', 0) == 1]['component_idx'].values
        metrics_df_init_filtered = metrics_df_init[~metrics_df_init['component_idx'].isin(corner_indices)].copy()

        if verbose:
            print(f"  Corner artifacts detected: {n_corner}")
            print(f"  Validation will compare {len(metrics_df_init_filtered)} neurons (excluding corners)\n")
    else:
        metrics_df_init_filtered = metrics_df_init.copy()

    # Extract metrics from automated result
    metrics_df_auto_final, _, _, _, _, _ = estimates_to_metrics(
        est_auto,
        fps=fps,
        include_event_based=False,  # Only need positions for comparison
        include_heavy=False
    )

    n_auto = len(metrics_df_auto_final)

    if verbose:
        print(f"  Automated pipeline result: {n_auto} neurons")
        print(f"  Ground truth: {n_gt} neurons\n")

    # === PART 5: Compute validation metrics ===
    if verbose:
        print("[6/6] Computing validation metrics...\n")

    # Filter initial metrics to match (exclude corner artifacts if any were found)
    if n_corner > 0:
        metrics_df_init_filtered = metrics_df_init[~metrics_df_init['component_idx'].isin(corner_indices)].copy()
    else:
        metrics_df_init_filtered = metrics_df_init.copy()

    validation_metrics = compute_metrics(
        metrics_df_init_filtered,
        metrics_df_gt,
        metrics_df_auto_final,
        max_match_distance=50  # pixels
    )

    # Add session metadata
    validation_metrics['session_id'] = session_id
    validation_metrics['n_initial'] = n_init
    validation_metrics['n_initial_filtered'] = len(metrics_df_init_filtered)
    validation_metrics['n_ground_truth'] = n_gt
    validation_metrics['n_auto'] = n_auto
    validation_metrics['n_deleted'] = n_delete
    validation_metrics['n_corner_artifacts'] = n_corner
    validation_metrics['n_merge_groups'] = n_merge_groups

    return validation_metrics


def batch_validate(mapping, fps=None, include_heavy=True, output_file=None, save_artifacts=True,
                   artifacts_base_path='.', verbose=False, session_list=None,
                   save_estimates=False, estimates_output_path=None,
                   event_method='threshold', n_iter=2):
    """
    Run validation on all sessions or specific sessions.

    Args:
        mapping: Dict mapping session names to (init_file, gt_file)
        fps: Frames per second (if None, will lookup from fps_map for each session)
        include_heavy: Include heavy reconstruction metrics
        output_file: Where to save results CSV (optional)
        save_artifacts: Save detailed per-session artifacts to capcan_artifacts folders
        artifacts_base_path: Base path for artifact folders
        verbose: Print progress for each session
        session_list: Optional list of specific session IDs to validate
        save_estimates: Save processed estimates pickle files
        estimates_output_path: Directory to save processed estimates
        event_method: Event detection method ('threshold' or 'oasis')
        n_iter: Number of iterations for event reconstruction

    Returns:
        pd.DataFrame: Validation results
    """
    # Filter mapping if specific sessions requested
    if session_list is not None:
        mapping = {k: v for k, v in mapping.items() if k in session_list}
        if len(mapping) == 0:
            print("ERROR: None of the specified sessions found in mapping")
            return pd.DataFrame()
    print("="*60)
    print("BATCH VALIDATION - ALL SESSIONS")
    print("="*60)
    print(f"Sessions to validate: {len(mapping)}")
    print(f"Include heavy metrics: {include_heavy}")
    print(f"Save detailed artifacts: {save_artifacts}")
    if save_artifacts:
        print(f"Artifacts base path: {artifacts_base_path}")
    print(f"Save processed estimates: {save_estimates}")
    if save_estimates:
        print(f"Estimates output path: {estimates_output_path}")
    print()

    results = []
    artifact_folders = []
    successful = 0
    failed = 0

    from tqdm import tqdm

    for session_id, (init_file, gt_file) in tqdm(mapping.items(), desc="Validating"):
        try:
            # Get FPS for this session
            session_fps = fps if fps is not None else fps_map.get(session_id, 20)

            # Load estimates
            est_init = load_estimates(os.path.join(init_path, init_file))
            est_gt = load_estimates(os.path.join(gt_path, gt_file))

            # Extract metrics with heavy option
            metrics_df_init, match_mtx_init, FCD_init, FBD_init, corner_info, _ = estimates_to_metrics(
                est_init,
                fps=session_fps,
                include_event_based=True,
                include_heavy=include_heavy,
                event_method=event_method,
                n_iter=n_iter
            )

            metrics_df_gt, _, _, _, _, _ = estimates_to_metrics(
                est_gt,
                fps=session_fps,
                include_event_based=True,
                include_heavy=include_heavy,
                event_method=event_method,
                n_iter=n_iter
            )

            # Run automated decision pipeline
            metrics_df_auto = metrics_to_decision(
                metrics_df_init.copy(),
                match_mtx_init,
                FCD_init,
                FBD_init,
                deletion_rules=VALIDATION_DELETION_RULES,
                pxlthr_distance_boundary=5,
                d_snr_thr=10,
                enable_merge=True
            )

            n_corner = (metrics_df_auto.get('is_corner_artifact', 0) == 1).sum()

            # Convert 'delete' to 'decision' column for implement_decision
            metrics_df_auto['decision'] = metrics_df_auto['delete'].apply(
                lambda x: 'delete' if x == 1 else 'ok'
            )

            # Convert S to dense if sparse (CaImAn's manual_merge doesn't handle sparse S)
            from scipy import sparse
            if sparse.issparse(est_init.S):
                est_init.S = est_init.S.toarray()

            # Implement decisions (handles merges) - apply ALL decisions including corner artifacts
            est_auto = implement_decision(est_init, metrics_df_auto)

            # Save processed estimates if requested (inside artifacts folder)
            if save_estimates:
                try:
                    # Save inside the artifacts folder for this session
                    artifacts_folder = Path(artifacts_base_path) / f'capcan_artifacts_{session_id}'
                    artifacts_folder.mkdir(parents=True, exist_ok=True)
                    save_path = save_processed_estimates(est_auto, artifacts_folder, session_id)
                    if verbose:
                        print(f"  Saved: {save_path}")
                except Exception as e:
                    print(f"  WARNING: Failed to save estimates for {session_id}: {e}")

            # Filter corner artifacts for validation metrics AFTER implementing decisions
            if n_corner > 0:
                corner_indices = metrics_df_auto[metrics_df_auto.get('is_corner_artifact', 0) == 1]['component_idx'].values
                metrics_df_init_filtered = metrics_df_init[~metrics_df_init['component_idx'].isin(corner_indices)].copy()
            else:
                metrics_df_init_filtered = metrics_df_init.copy()

            # Extract metrics from automated result
            metrics_df_auto_final, _, _, _, _, _ = estimates_to_metrics(
                est_auto,
                fps=session_fps,
                include_event_based=False,
                include_heavy=False
            )

            # Compute validation metrics
            validation_metrics = compute_metrics(
                metrics_df_init_filtered,
                metrics_df_gt,
                metrics_df_auto_final,
                max_match_distance=3
            )

            # Add metadata
            validation_metrics['session_id'] = session_id
            validation_metrics['fps'] = session_fps
            validation_metrics['n_initial'] = len(est_init.idx_components)
            validation_metrics['n_initial_filtered'] = len(metrics_df_init_filtered)
            validation_metrics['n_ground_truth'] = len(est_gt.idx_components)
            validation_metrics['n_auto'] = len(metrics_df_auto_final)
            validation_metrics['n_deleted'] = (metrics_df_auto['delete'] == 1).sum()
            validation_metrics['n_corner_artifacts'] = n_corner
            validation_metrics['n_merge_groups'] = len(metrics_df_auto[metrics_df_auto['merge'] != 0]['merge'].unique())

            results.append(validation_metrics)
            successful += 1

            # Save detailed artifacts if requested
            if save_artifacts:
                try:
                    # Pass UNFILTERED metrics_df_init so visualization shows corner artifacts in red
                    artifact_folder = save_validation_outputs(
                        session_name=session_id,
                        metrics_df_init=metrics_df_init,  # UNFILTERED for visualization
                        metrics_df_gt=metrics_df_gt,
                        metrics_df_auto=metrics_df_auto_final,
                        decision_df=metrics_df_auto,
                        validation_metrics=validation_metrics,
                        corner_info=corner_info,
                        base_path=artifacts_base_path,
                        FCD=FCD_init,
                        FBD=FBD_init,
                        match_mtx=match_mtx_init
                    )
                    artifact_folders.append(str(artifact_folder))
                except Exception as e:
                    print(f"  WARNING: Failed to save artifacts for {session_id}: {e}")

            if verbose:
                print(f"  {session_id}: P={validation_metrics['precision']:.2%}, R={validation_metrics['recall']:.2%}, F1={validation_metrics['f1_score']:.2%}")

        except Exception as e:
            print(f"  ERROR {session_id}: {e}")
            failed += 1

    # Convert to DataFrame
    results_df = pd.DataFrame(results)

    # Save if output file specified
    if output_file:
        results_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    print(f"Total sessions: {len(mapping)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if save_artifacts and len(artifact_folders) > 0:
        print(f"\nDetailed artifacts saved to {len(artifact_folders)} folders:")
        print(f"  Base path: {artifacts_base_path}")
        print(f"  Folder pattern: capcan_artifacts_<session_name>")

    if len(results_df) > 0:
        print("\n" + "-"*60)
        print("AGGREGATE STATISTICS")
        print("-"*60)
        print(f"\nDetection Metrics (mean +/- std):")
        print(f"  Precision:    {results_df['precision'].mean():.2%} +/- {results_df['precision'].std():.2%}")
        print(f"  Recall:       {results_df['recall'].mean():.2%} +/- {results_df['recall'].std():.2%}")
        print(f"  F1 Score:     {results_df['f1_score'].mean():.2%} +/- {results_df['f1_score'].std():.2%}")

        print(f"\nPositional Accuracy (mean +/- std):")
        print(f"  Mean Error:   {results_df['mean_error'].mean():.2f} +/- {results_df['mean_error'].std():.2f} pixels")
        print(f"  Median Error: {results_df['median_error'].mean():.2f} +/- {results_df['median_error'].std():.2f} pixels")

    print("="*60)

    return results_df


# === RUN VALIDATION ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate automated inspection pipeline")
    parser.add_argument("--batch", action="store_true", help="Run on all sessions (default: single test session)")
    parser.add_argument("--session", type=str, help="Specific session to validate (default: first)")
    parser.add_argument("--sessions", type=str, help="Comma-separated list of specific sessions to validate in batch mode")
    parser.add_argument("--raw-path", type=str, help="Path to raw estimates folder (default: data/raw_compressed)")
    parser.add_argument("--final-path", type=str, help="Path to final estimates folder (default: data/final_compressed)")
    parser.add_argument("--fps", type=float, help="Override FPS for all sessions (default: use fps_data.csv lookup)")
    parser.add_argument("--include-heavy", action="store_true", help="Include heavy reconstruction metrics")
    parser.add_argument("--output", "-o", help="Output CSV file for batch mode")
    parser.add_argument("--save-artifacts", action="store_true", default=True, help="Save detailed artifacts to capcan_artifacts folders (default: True)")
    parser.add_argument("--no-save-artifacts", dest="save_artifacts", action="store_false", help="Disable saving artifacts")
    parser.add_argument("--artifacts-path", default=".", help="Base path for artifact folders (default: current directory)")
    parser.add_argument("--save-estimates", action="store_true", help="Save processed estimates pickle files")
    parser.add_argument("--estimates-path", default="data/processed_estimates", help="Directory to save processed estimates (default: data/processed_estimates)")
    parser.add_argument("--event-method", default="threshold", choices=["threshold", "oasis"], help="Event detection method (default: threshold)")
    parser.add_argument("--n-iter", type=int, default=2, help="Number of iterations for event reconstruction (default: 2)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

    # Override paths if specified
    if args.raw_path:
        init_path = args.raw_path
    if args.final_path:
        gt_path = args.final_path

    # Rebuild mapping with specified paths
    if args.raw_path or args.final_path:
        mapping = build_session_mapping(init_path, gt_path)
        print(f"Using custom paths:")
        print(f"  Raw estimates: {init_path}")
        print(f"  Final estimates: {gt_path}")
        print(f"  Sessions found: {len(mapping)}\n")

    if args.batch:
        # Batch mode - all sessions or specific sessions
        output_file = args.output or "data/batch_validation_heavy_results.csv"

        # Parse session list if provided
        session_list = None
        if args.sessions:
            session_list = [s.strip() for s in args.sessions.split(',')]
            print(f"Validating specific sessions: {session_list}\n")

        results_df = batch_validate(
            mapping,
            fps=args.fps,  # None by default, will use fps_map lookup
            include_heavy=args.include_heavy,
            output_file=output_file,
            save_artifacts=args.save_artifacts,
            artifacts_base_path=args.artifacts_path,
            verbose=args.verbose,
            session_list=session_list,
            save_estimates=args.save_estimates,
            estimates_output_path=args.estimates_path,
            event_method=args.event_method,
            n_iter=args.n_iter
        )
    else:
        # Single session mode
        if args.session:
            if args.session not in mapping:
                print(f"ERROR: Session {args.session} not found")
                exit(1)
            test_session = args.session
        else:
            test_session = list(mapping.keys())[0]

        init_file, gt_file = mapping[test_session]

        print(f"\nValidating session: {test_session}")
        print(f"="*60)

        metrics = validate_single_session(
            session_id=test_session,
            init_file=init_file,
            gt_file=gt_file,
            fps=args.fps,  # None by default, will use fps_map lookup
            verbose=True
        )

        # Print comprehensive report
        print_report(metrics)