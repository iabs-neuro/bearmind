import os
import pickle
import numpy as np
import pandas as pd
from auto_inspector import (
    estimates_to_metrics,
    metrics_to_decision,
    implement_decision,
    compute_metrics,
    print_report
)
from ae_utils import save_validation_outputs

# Configuration
project_root = os.path.dirname(os.path.abspath(__file__))
val_path = os.path.join(project_root, 'data')
init_path = os.path.join(val_path, '4.1_EstimatesRaw')
gt_path = os.path.join(val_path, '4.1_EstimatesFinal')

# Build session mapping
all_init_files = os.listdir(init_path)
all_gt_files = os.listdir(gt_path)

sessions = [name[:10] for name in all_gt_files]
mapping = {}
for session in sessions:
    init = [f for f in all_init_files if session in f][0]
    gt = [f for f in all_gt_files if session in f][0]
    mapping.update({session: [init, gt]})

print(f"Found {len(mapping)} sessions to validate\n")


def load_estimates(filepath):
    """Load CaImAn estimates object from pickle file."""
    with open(filepath, 'rb') as f:
        estimates = pickle.load(f)
    return estimates


def validate_single_session(session_id, init_file, gt_file, fps=20, verbose=True):
    """
    Validate automated pipeline on a single session.

    Args:
        session_id: Session identifier
        init_file: Filename for raw estimates
        gt_file: Filename for final (ground truth) estimates
        fps: Frames per second for calcium imaging
        verbose: Print detailed progress

    Returns:
        dict: Validation metrics
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"VALIDATING SESSION: {session_id}")
        print(f"{'='*60}")
        print(f"Raw estimates: {init_file}")
        print(f"Final estimates: {gt_file}\n")

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

    metrics_df_init, match_mtx_init, FCD_init, FBD_init, _ = estimates_to_metrics(
        est_init,
        fps=fps,
        include_wavelet=True,
        include_heavy=False  # Skip reconstruction metrics for speed
    )

    if verbose:
        print(f"  Extracted {len(metrics_df_init)} neuron metrics\n")

    # === PART 3: Extract metrics from ground truth estimates ===
    if verbose:
        print("[3/6] Extracting metrics from ground truth estimates...")

    metrics_df_gt, _, _, _, _ = estimates_to_metrics(
        est_gt,
        fps=fps,
        include_wavelet=False,
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
        # Default thresholds from function signature
        circ_thr=1.7,
        maxedge_thr=1.45,
        convex_thr=42,
        pxlthr_area=3,
        pxlthr_distance_boundary=5,
        d_snr_thr=42,
        use_circularity_check=True,
        use_area_check=True,
        use_max_edge_check=True,
        use_convexity_check=True,
        use_corr_check=True
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
    metrics_df_auto_final, _, _, _, _ = estimates_to_metrics(
        est_auto,
        fps=fps,
        include_wavelet=False,  # Only need positions for comparison
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


def batch_validate(mapping, fps=20, include_heavy=True, output_file=None, save_artifacts=True,
                   artifacts_base_path='.', verbose=False, session_list=None):
    """
    Run validation on all sessions or specific sessions.

    Args:
        mapping: Dict mapping session names to (init_file, gt_file)
        fps: Frames per second
        include_heavy: Include heavy reconstruction metrics
        output_file: Where to save results CSV (optional)
        save_artifacts: Save detailed per-session artifacts to capcan_artifacts folders
        artifacts_base_path: Base path for artifact folders
        verbose: Print progress for each session
        session_list: Optional list of specific session IDs to validate

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
    print()

    results = []
    artifact_folders = []
    successful = 0
    failed = 0

    from tqdm import tqdm

    for session_id, (init_file, gt_file) in tqdm(mapping.items(), desc="Validating"):
        try:
            # Load estimates
            est_init = load_estimates(os.path.join(init_path, init_file))
            est_gt = load_estimates(os.path.join(gt_path, gt_file))

            # Extract metrics with heavy option
            metrics_df_init, match_mtx_init, FCD_init, FBD_init, corner_info = estimates_to_metrics(
                est_init,
                fps=fps,
                include_wavelet=True,
                include_heavy=include_heavy
            )

            metrics_df_gt, _, _, _, _ = estimates_to_metrics(
                est_gt,
                fps=fps,
                include_wavelet=True,
                include_heavy=include_heavy
            )

            # Run automated decision pipeline
            metrics_df_auto = metrics_to_decision(
                metrics_df_init.copy(),
                match_mtx_init,
                FCD_init,
                FBD_init,
                circ_thr=1.7,
                maxedge_thr=1.45,
                convex_thr=42,
                pxlthr_area=3,
                pxlthr_distance_boundary=5,
                d_snr_thr=42,
                t_rise_min=0.10,
                caiman_r_score_min=0.05,
                caiman_snr_min=2.9,
                t_off_min=1.5,
                use_circularity_check=True,
                use_area_check=True,
                use_max_edge_check=True,
                use_convexity_check=True,
                use_corr_check=True
            )

            n_corner = (metrics_df_auto.get('is_corner_artifact', 0) == 1).sum()

            # Convert 'delete' to 'decision' column for implement_decision
            metrics_df_auto['decision'] = metrics_df_auto['delete'].apply(
                lambda x: 'delete' if x == 1 else 'ok'
            )

            # Implement decisions (handles merges) - apply ALL decisions including corner artifacts
            est_auto = implement_decision(est_init, metrics_df_auto)

            # Filter corner artifacts for validation metrics AFTER implementing decisions
            if n_corner > 0:
                corner_indices = metrics_df_auto[metrics_df_auto.get('is_corner_artifact', 0) == 1]['component_idx'].values
                metrics_df_init_filtered = metrics_df_init[~metrics_df_init['component_idx'].isin(corner_indices)].copy()
            else:
                metrics_df_init_filtered = metrics_df_init.copy()

            # Extract metrics from automated result
            metrics_df_auto_final, _, _, _, _ = estimates_to_metrics(
                est_auto,
                fps=fps,
                include_wavelet=False,
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
                    artifact_folder = save_validation_outputs(
                        session_name=session_id,
                        metrics_df_init=metrics_df_init_filtered,
                        metrics_df_gt=metrics_df_gt,
                        metrics_df_auto=metrics_df_auto_final,
                        decision_df=metrics_df_auto,
                        validation_metrics=validation_metrics,
                        corner_info=corner_info,
                        base_path=artifacts_base_path
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
    parser.add_argument("--fps", type=float, default=20, help="Frames per second")
    parser.add_argument("--include-heavy", action="store_true", help="Include heavy reconstruction metrics")
    parser.add_argument("--output", "-o", help="Output CSV file for batch mode")
    parser.add_argument("--save-artifacts", action="store_true", default=True, help="Save detailed artifacts to capcan_artifacts folders (default: True)")
    parser.add_argument("--no-save-artifacts", dest="save_artifacts", action="store_false", help="Disable saving artifacts")
    parser.add_argument("--artifacts-path", default=".", help="Base path for artifact folders (default: current directory)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()

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
            fps=args.fps,
            include_heavy=args.include_heavy,
            output_file=output_file,
            save_artifacts=args.save_artifacts,
            artifacts_base_path=args.artifacts_path,
            verbose=args.verbose,
            session_list=session_list
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
            fps=args.fps,
            verbose=True
        )

        # Print comprehensive report
        print_report(metrics)