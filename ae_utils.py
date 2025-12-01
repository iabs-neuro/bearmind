"""
Auto-Inspector Utilities

Helper functions for auto_inspector.py including output folder management,
visualization, and reporting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from pathlib import Path


def create_capcan_artifacts_folder(session_name, base_path='.'):
    """
    Create folder structure for auto-inspection outputs.

    Parameters:
        session_name: Name/ID of the session
        base_path: Directory where artifact folders will be created (default: '.' for current directory)
                   Example: base_path='./outputs' creates './outputs/capcan_artifacts_<session_name>/'

    Returns:
        folder_path: Path to created folder (capcan_artifacts_<session_name>)
    """
    folder_name = f'capcan_artifacts_{session_name}'
    folder_path = Path(base_path) / folder_name
    folder_path.mkdir(parents=True, exist_ok=True)

    return folder_path


def save_metrics_dataframe(metrics_df, output_folder):
    """
    Save metrics dataframe to CSV.

    Parameters:
        metrics_df: DataFrame with neuron metrics
        output_folder: Path to output folder
    """
    output_path = Path(output_folder) / 'metrics.csv'
    metrics_df.to_csv(output_path, index=False)
    print(f'Saved metrics to {output_path}')


def save_decision_dataframe(decision_df, output_folder):
    """
    Save decision dataframe with all columns including criteria failures.

    Parameters:
        decision_df: DataFrame with decisions and failure tracking
        output_folder: Path to output folder
    """
    output_path = Path(output_folder) / 'decisions_with_criteria.csv'
    decision_df.to_csv(output_path, index=False)
    print(f'Saved decisions to {output_path}')


def save_rejected_neurons_summary(decision_df, output_folder):
    """
    Save separate dataframe of rejected neurons with criteria failures.

    Parameters:
        decision_df: DataFrame with decisions
        output_folder: Path to output folder
    """
    # Get only rejected neurons
    rejected = decision_df[decision_df['delete'] == 1].copy()

    if len(rejected) == 0:
        print('No rejected neurons to save')
        return

    # Identify failure columns
    failure_cols = [col for col in rejected.columns if col.startswith('failed_')]

    # Create summary columns
    basic_cols = ['component_idx', 'area', 'circularity', 'max_edge', 'convexity',
                 'caiman_snr', 'caiman_r_score', 't_rise', 't_off']

    # Get columns that exist in dataframe
    existing_basic_cols = [col for col in basic_cols if col in rejected.columns]

    # Select relevant columns
    summary_cols = existing_basic_cols + failure_cols
    rejected_summary = rejected[summary_cols]

    output_path = Path(output_folder) / 'rejected_neurons.csv'
    rejected_summary.to_csv(output_path, index=False)
    print(f'Saved {len(rejected)} rejected neurons to {output_path}')


def visualize_corner_artifacts(metrics_df, edge_info, output_folder, session_name='session'):
    """
    Create visualization of edge artifact detection with CoM and ellipse boundary.

    Shows three-color scheme:
    - Blue: main neurons (not artifacts)
    - Yellow: corner-only artifacts
    - Black: ellipse-only artifacts
    - Red: detected by both methods

    Parameters:
        metrics_df: DataFrame with 'center' and 'is_corner_artifact' columns
        edge_info: Dict with edge detection information (from detect_edge_artifacts)
        output_folder: Path to output folder
        session_name: Session name for plot title
    """
    if 'is_corner_artifact' not in metrics_df.columns:
        print('No edge artifact information to visualize')
        return

    if 'center' not in metrics_df.columns:
        print('No position information to visualize')
        return

    # Extract positions
    positions = np.array([np.array(c) if not isinstance(c, np.ndarray) else c
                         for c in metrics_df['center']])

    # Compute FOV dimensions and center
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    y_min, y_max = positions[:, 1].min(), positions[:, 1].max()
    fov_width = x_max - x_min
    fov_height = y_max - y_min
    center_x, center_y = np.mean(positions[:, 0]), np.mean(positions[:, 1])

    # Get breakdown from edge_info if available
    n_corner_only = edge_info.get('n_corner_only', 0) if edge_info else 0
    n_ellipse_only = edge_info.get('n_ellipse_only', 0) if edge_info else 0
    n_both = edge_info.get('n_both', 0) if edge_info else 0

    # Compute corner and ellipse labels to determine colors
    # We need to recompute these to get the breakdown
    from corner_artifacts import detect_corner_artifacts_from_positions, detect_ellipse_artifacts_from_positions

    corner_labels, _ = detect_corner_artifacts_from_positions(positions, fov_width, fov_height)
    ellipse_labels, ellipse_info = detect_ellipse_artifacts_from_positions(positions, fov_width, fov_height, threshold=0.9)

    # Create masks for different categories
    corner_only = (corner_labels == 1) & (ellipse_labels == 0)
    ellipse_only = (corner_labels == 0) & (ellipse_labels == 1)
    both = (corner_labels == 1) & (ellipse_labels == 1)
    main_mask = (corner_labels == 0) & (ellipse_labels == 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot with colors: blue=main, yellow=corner-only, black=ellipse-only, red=both
    if main_mask.any():
        ax.scatter(positions[main_mask, 0], positions[main_mask, 1],
                  c='blue', alpha=0.6, s=20, label=f'Main ({main_mask.sum()})')

    if corner_only.any():
        ax.scatter(positions[corner_only, 0], positions[corner_only, 1],
                  c='yellow', edgecolors='orange', alpha=0.9, s=30,
                  label=f'Corner only ({corner_only.sum()})')

    if ellipse_only.any():
        ax.scatter(positions[ellipse_only, 0], positions[ellipse_only, 1],
                  c='black', alpha=0.8, s=30,
                  label=f'Ellipse only ({ellipse_only.sum()})')

    if both.any():
        ax.scatter(positions[both, 0], positions[both, 1],
                  c='red', alpha=0.9, s=35,
                  label=f'Both ({both.sum()})')

    # Draw ellipse boundary (r=0.9)
    ellipse_threshold = 0.9
    ellipse_width = fov_width * ellipse_threshold
    ellipse_height = fov_height * ellipse_threshold
    ellipse_patch = Ellipse((center_x, center_y), ellipse_width, ellipse_height,
                            fill=False, edgecolor='darkred', linestyle='--', linewidth=2)
    ax.add_patch(ellipse_patch)

    # Mark center of mass
    ax.scatter([center_x], [center_y], c='green', marker='+', s=200, linewidths=3,
              label='CoM', zorder=10)

    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')

    n_total = len(positions)
    n_artifacts = corner_only.sum() + ellipse_only.sum() + both.sum()
    ax.set_title(f'{session_name}\nArtifacts: {n_artifacts}/{n_total} ({n_artifacts/n_total*100:.1f}%)')

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    plt.tight_layout()

    output_path = Path(output_folder) / 'edge_artifacts.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Saved edge artifact visualization to {output_path}')


def create_inspection_summary(metrics_df, decision_df, corner_info, output_folder):
    """
    Create text summary of auto-inspection results.

    Parameters:
        metrics_df: DataFrame with metrics
        decision_df: DataFrame with decisions
        corner_info: Dict with corner information
        output_folder: Path to output folder
    """
    summary_lines = []
    summary_lines.append('='*60)
    summary_lines.append('AUTO-INSPECTION SUMMARY')
    summary_lines.append('='*60)
    summary_lines.append('')

    # Overall statistics
    n_total = len(decision_df)
    n_rejected = (decision_df['delete'] == 1).sum()
    n_kept = n_total - n_rejected
    n_merge_groups = len(decision_df[decision_df['merge'] != 0]['merge'].unique())

    summary_lines.append(f'Total neurons: {n_total}')
    summary_lines.append(f'Rejected: {n_rejected} ({n_rejected/n_total*100:.1f}%)')
    summary_lines.append(f'Kept: {n_kept} ({n_kept/n_total*100:.1f}%)')
    summary_lines.append(f'Merge groups: {n_merge_groups}')
    summary_lines.append('')

    # Corner artifact statistics
    if 'is_corner_artifact' in decision_df.columns:
        n_corner = (decision_df['is_corner_artifact'] == 1).sum()
        summary_lines.append(f'Corner artifacts detected: {n_corner} ({n_corner/n_total*100:.1f}%)')

        if corner_info:
            corners_detected = [name for name, info in corner_info.items()
                              if info.get('cluster_found', False)]
            if corners_detected:
                summary_lines.append(f'Corners with clusters: {", ".join(corners_detected)}')
                for corner_name in corners_detected:
                    info = corner_info[corner_name]
                    summary_lines.append(f'  {corner_name}: {info["n_neurons"]} neurons, '
                                       f'threshold={info["threshold"]:.1f}px')
        summary_lines.append('')

    # Rejection criteria breakdown
    failure_cols = [col for col in decision_df.columns if col.startswith('failed_')]
    if failure_cols:
        summary_lines.append('REJECTION CRITERIA BREAKDOWN:')
        rejected_df = decision_df[decision_df['delete'] == 1]

        for col in failure_cols:
            criterion_name = col.replace('failed_', '').replace('_', ' ').title()
            n_failed = rejected_df[col].sum()
            if n_failed > 0:
                summary_lines.append(f'  {criterion_name}: {n_failed} neurons '
                                   f'({n_failed/n_rejected*100:.1f}% of rejected)')
        summary_lines.append('')

    # Quality metric ranges for kept neurons
    kept_df = decision_df[decision_df['delete'] == 0]
    if len(kept_df) > 0:
        summary_lines.append('KEPT NEURONS - QUALITY METRICS:')

        metric_cols = ['area', 'circularity', 'max_edge', 'convexity',
                      'caiman_snr', 'caiman_r_score', 't_rise', 't_off', 'wavelet_snr']

        for col in metric_cols:
            if col in kept_df.columns:
                values = kept_df[col].dropna()
                if len(values) > 0:
                    summary_lines.append(f'  {col}: min={values.min():.3f}, '
                                       f'mean={values.mean():.3f}, max={values.max():.3f}')

    summary_lines.append('')
    summary_lines.append('='*60)

    # Save to file
    output_path = Path(output_folder) / 'summary.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f'Saved summary to {output_path}')

    # Also print to console
    print('\n'.join(summary_lines))


def save_auto_inspection_outputs(session_name, metrics_df, decision_df, corner_info=None, base_path='.'):
    """
    Complete workflow to save all auto-inspection outputs.

    Parameters:
        session_name: Name/ID of the session
        metrics_df: DataFrame with neuron metrics
        decision_df: DataFrame with decisions
        corner_info: Optional dict with corner detection info
        base_path: Directory where artifact folders will be created (default: '.' for current directory)
                   All capcan_artifacts_* folders will be created inside this directory

    Returns:
        output_folder: Path to created folder
    """
    # Create folder
    output_folder = create_capcan_artifacts_folder(session_name, base_path)

    # Save dataframes
    save_metrics_dataframe(metrics_df, output_folder)
    save_decision_dataframe(decision_df, output_folder)
    save_rejected_neurons_summary(decision_df, output_folder)

    # Create visualizations
    if corner_info is not None:
        visualize_corner_artifacts(metrics_df, corner_info, output_folder, session_name)

    # Create summary
    create_inspection_summary(metrics_df, decision_df, corner_info, output_folder)

    print(f'\nAll outputs saved to: {output_folder}')

    return output_folder


def save_validation_outputs(session_name, metrics_df_init, metrics_df_gt, metrics_df_auto,
                            decision_df, validation_metrics, corner_info=None, base_path='.',
                            FCD=None, FBD=None, match_mtx=None):
    """
    Save validation-specific outputs to capcan_artifacts folder.

    Parameters:
        session_name: Name/ID of the session
        metrics_df_init: Initial/raw metrics dataframe
        metrics_df_gt: Ground truth metrics dataframe
        metrics_df_auto: Automated result metrics dataframe
        decision_df: Decision dataframe with criteria
        validation_metrics: Dict with validation metrics (precision, recall, etc.)
        corner_info: Optional corner detection info
        base_path: Directory where artifact folders will be created

    Returns:
        output_folder: Path to created folder
    """
    # Create folder
    output_folder = create_capcan_artifacts_folder(session_name, base_path)

    # Save all metrics dataframes
    init_path = Path(output_folder) / 'metrics_init.csv'
    metrics_df_init.to_csv(init_path, index=False)
    print(f'Saved initial metrics to {init_path}')

    gt_path = Path(output_folder) / 'metrics_gt.csv'
    metrics_df_gt.to_csv(gt_path, index=False)
    print(f'Saved ground truth metrics to {gt_path}')

    auto_path = Path(output_folder) / 'metrics_auto_final.csv'
    metrics_df_auto.to_csv(auto_path, index=False)
    print(f'Saved automated final metrics to {auto_path}')

    # Save decision dataframe
    save_decision_dataframe(decision_df, output_folder)
    save_rejected_neurons_summary(decision_df, output_folder)

    # Save validation metrics
    val_path = Path(output_folder) / 'validation_metrics.csv'
    val_df = pd.DataFrame([validation_metrics])
    val_df.to_csv(val_path, index=False)
    print(f'Saved validation metrics to {val_path}')

    # Save FCD, FBD, and match_mtx if provided
    if FCD is not None:
        fcd_path = Path(output_folder) / 'FCD.npy'
        np.save(fcd_path, FCD)
        print(f'Saved FCD to {fcd_path}')

    if FBD is not None:
        fbd_path = Path(output_folder) / 'FBD.npy'
        np.save(fbd_path, FBD)
        print(f'Saved FBD to {fbd_path}')

    if match_mtx is not None:
        match_path = Path(output_folder) / 'match_mtx.npy'
        np.save(match_path, match_mtx)
        print(f'Saved match_mtx to {match_path}')

    # Create visualizations
    if corner_info is not None:
        visualize_corner_artifacts(metrics_df_init, corner_info, output_folder, session_name)

    # Create validation summary
    create_validation_summary(metrics_df_init, metrics_df_gt, metrics_df_auto,
                             decision_df, validation_metrics, corner_info, output_folder)

    print(f'\nAll validation outputs saved to: {output_folder}')

    return output_folder


def create_validation_summary(metrics_df_init, metrics_df_gt, metrics_df_auto,
                              decision_df, validation_metrics, corner_info, output_folder):
    """
    Create text summary of validation results.

    Parameters:
        metrics_df_init: Initial metrics
        metrics_df_gt: Ground truth metrics
        metrics_df_auto: Automated result metrics
        decision_df: Decision dataframe
        validation_metrics: Validation metrics dict
        corner_info: Corner detection info
        output_folder: Path to output folder
    """
    summary_lines = []
    summary_lines.append('='*60)
    summary_lines.append('VALIDATION SUMMARY')
    summary_lines.append('='*60)
    summary_lines.append('')

    # Neuron counts
    summary_lines.append(f'NEURON COUNTS:')
    summary_lines.append(f'  Initial (raw): {validation_metrics.get("n_initial", len(metrics_df_init))}')

    n_corner = validation_metrics.get('n_corner_artifacts', 0)
    if n_corner > 0:
        summary_lines.append(f'  Corner artifacts: {n_corner}')
        summary_lines.append(f'  Initial (after corner filter): {validation_metrics.get("n_initial_filtered", len(metrics_df_init))}')

    summary_lines.append(f'  Ground truth: {validation_metrics.get("n_ground_truth", len(metrics_df_gt))}')
    summary_lines.append(f'  Automated result: {validation_metrics.get("n_auto", len(metrics_df_auto))}')
    summary_lines.append(f'  Deleted by automation: {validation_metrics.get("n_deleted", 0)}')
    summary_lines.append('')

    # Validation metrics
    summary_lines.append(f'DETECTION PERFORMANCE:')
    summary_lines.append(f'  Precision: {validation_metrics.get("precision", 0):.2%}')
    summary_lines.append(f'  Recall: {validation_metrics.get("recall", 0):.2%}')
    summary_lines.append(f'  F1 Score: {validation_metrics.get("f1_score", 0):.2%}')
    summary_lines.append('')

    summary_lines.append(f'DETECTION DETAILS:')
    summary_lines.append(f'  True Positives: {validation_metrics.get("true_positives", 0)}')
    summary_lines.append(f'  False Positives: {validation_metrics.get("false_positives", 0)}')
    summary_lines.append(f'  False Negatives: {validation_metrics.get("false_negatives", 0)}')
    summary_lines.append('')

    # Positional accuracy
    summary_lines.append(f'POSITIONAL ACCURACY:')
    summary_lines.append(f'  Mean error: {validation_metrics.get("mean_error", 0):.2f} pixels')
    summary_lines.append(f'  Median error: {validation_metrics.get("median_error", 0):.2f} pixels')
    summary_lines.append(f'  Max error: {validation_metrics.get("max_error", 0):.2f} pixels')
    summary_lines.append('')

    # Decision statistics
    n_merge = validation_metrics.get('n_merge_groups', 0)
    if n_merge > 0:
        summary_lines.append(f'MERGE OPERATIONS:')
        summary_lines.append(f'  Merge groups: {n_merge}')
        summary_lines.append('')

    # Rejection criteria breakdown
    failure_cols = [col for col in decision_df.columns if col.startswith('failed_')]
    if failure_cols:
        n_rejected = validation_metrics.get('n_deleted', 0)
        summary_lines.append('REJECTION CRITERIA BREAKDOWN:')
        rejected_df = decision_df[decision_df['delete'] == 1]

        for col in failure_cols:
            criterion_name = col.replace('failed_', '').replace('_', ' ').title()
            n_failed = rejected_df[col].sum() if len(rejected_df) > 0 else 0
            if n_failed > 0:
                pct = n_failed/n_rejected*100 if n_rejected > 0 else 0
                summary_lines.append(f'  {criterion_name}: {n_failed} neurons ({pct:.1f}% of rejected)')
        summary_lines.append('')

    summary_lines.append('='*60)

    # Save to file
    output_path = Path(output_folder) / 'validation_summary.txt'
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f'Saved validation summary to {output_path}')

    # Also print to console
    print('\n'.join(summary_lines))
