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

    if verbose:
        print(f"  Automated decisions:")
        print(f"    Delete: {n_delete} neurons")
        print(f"    Merge: {n_merge_groups} groups")
        print(f"    Keep: {len(metrics_df_auto) - n_delete} neurons\n")

    # Apply decisions to get automated estimates
    if verbose:
        print("[5/6] Implementing automated decisions...")

    est_auto = implement_decision(est_init, metrics_df_auto)

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

    validation_metrics = compute_metrics(
        metrics_df_init,
        metrics_df_gt,
        metrics_df_auto_final,
        max_match_distance=50  # pixels
    )

    # Add session metadata
    validation_metrics['session_id'] = session_id
    validation_metrics['n_initial'] = n_init
    validation_metrics['n_ground_truth'] = n_gt
    validation_metrics['n_auto'] = n_auto
    validation_metrics['n_deleted'] = n_delete
    validation_metrics['n_merge_groups'] = n_merge_groups

    return validation_metrics


# === RUN VALIDATION ON FIRST SESSION ===
if __name__ == "__main__":
    # Test on first session
    test_session = list(mapping.keys())[0]
    init_file, gt_file = mapping[test_session]

    print(f"\nTesting validation pipeline on session: {test_session}")
    print(f"="*60)

    metrics = validate_single_session(
        session_id=test_session,
        init_file=init_file,
        gt_file=gt_file,
        fps=20,  # Typical calcium imaging frame rate
        verbose=True
    )

    # Print comprehensive report
    print_report(metrics)