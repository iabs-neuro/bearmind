"""
Auto-Inspector Utilities

Helper functions for auto_inspector.py including output folder management,
visualization, and reporting.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def visualize_corner_artifacts(metrics_df, corner_info, output_folder, session_name='session'):
    """
    Create visualization of corner artifact detection.

    Parameters:
        metrics_df: DataFrame with 'center' and 'is_corner_artifact' columns
        corner_info: Dict with corner detection information
        output_folder: Path to output folder
        session_name: Session name for plot title
    """
    if 'is_corner_artifact' not in metrics_df.columns:
        print('No corner artifact information to visualize')
        return

    if 'center' not in metrics_df.columns:
        print('No position information to visualize')
        return

    # Extract positions
    positions = np.array([np.array(c) if not isinstance(c, np.ndarray) else c
                         for c in metrics_df['center']])

    is_artifact = metrics_df['is_corner_artifact'].values == 1
    is_main = ~is_artifact

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot main cluster
    if is_main.any():
        ax.scatter(positions[is_main, 0], positions[is_main, 1],
                  c='blue', s=30, alpha=0.6, edgecolors='darkblue',
                  linewidths=0.5, label=f'Main ({is_main.sum()})')

    # Plot corner artifacts
    if is_artifact.any():
        ax.scatter(positions[is_artifact, 0], positions[is_artifact, 1],
                  c='red', s=30, alpha=0.6, edgecolors='darkred',
                  linewidths=0.5, label=f'Corner Artifacts ({is_artifact.sum()})')

    ax.set_xlabel('X position (pixels)')
    ax.set_ylabel('Y position (pixels)')
    ax.set_title(f'{session_name} - Corner Artifact Detection')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

    # Add corner info as text
    if corner_info:
        corners_detected = [name for name, info in corner_info.items()
                          if info.get('cluster_found', False)]
        if corners_detected:
            info_text = 'Detected corners:\n' + ', '.join(corners_detected)
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                   fontsize=9)

    plt.tight_layout()

    output_path = Path(output_folder) / 'corner_artifacts.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Saved corner artifact visualization to {output_path}')


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
