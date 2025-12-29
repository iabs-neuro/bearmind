"""
Compute ellipse_r for LNOF dataset.

Loads processed estimates pickle files and computes ellipse_r for all LNOF neurons
by extracting spatial footprint centers and computing normalized radial distance.

This fixes the bug where ellipse_r was not computed during LNOF autoinspection
because corner artifact detection was disabled.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from corner_artifacts import detect_ellipse_artifacts_from_positions

def compute_lnof_ellipse_r(
    dataset_path='ml/results/training_dataset_v9.csv',
    output_path=None,
    lnof_dir='data/LNOF'
):
    """
    Add ellipse_r column to LNOF neurons in v9 dataset.

    Args:
        dataset_path: Path to v9 dataset CSV
        output_path: Path for updated dataset (default: overwrite original)
        lnof_dir: Directory containing LNOF inspection_artifacts folders
    """
    print('='*80)
    print('ADDING ELLIPSE_R TO LNOF NEURONS IN V9 DATASET')
    print('='*80)

    # Load dataset
    print(f'\nLoading dataset: {dataset_path}')
    df = pd.read_csv(dataset_path)
    print(f'Loaded {len(df):,} neurons from {df["session_name"].nunique()} sessions')

    # Check current ellipse_r status
    lnof_mask = df['experiment'] == 'LNOF'
    lnof_count = lnof_mask.sum()
    lnof_missing = df.loc[lnof_mask, 'ellipse_r'].isna().sum()

    print(f'\nLNOF neurons: {lnof_count:,}')
    print(f'LNOF with missing ellipse_r: {lnof_missing:,} ({100*lnof_missing/lnof_count:.1f}%)')

    # Get unique LNOF sessions
    lnof_sessions = df[lnof_mask]['session_name'].unique()
    print(f'\nProcessing {len(lnof_sessions)} LNOF sessions...')

    lnof_path = Path(lnof_dir)
    sessions_processed = 0
    sessions_failed = 0
    neurons_processed = 0

    for session in tqdm(lnof_sessions, desc='Computing ellipse_r'):
        # Find _processed.pickle file
        artifact_folders = list(lnof_path.glob(f'inspection_artifacts_{session}*'))
        if not artifact_folders:
            print(f'\nWarning: No artifacts folder found for {session}')
            sessions_failed += 1
            continue

        folder = artifact_folders[0]
        pickle_files = list(folder.glob('*_processed.pickle'))
        if not pickle_files:
            print(f'\nWarning: No processed pickle found for {session}')
            sessions_failed += 1
            continue

        # Load processed estimates
        try:
            with open(pickle_files[0], 'rb') as f:
                est = pickle.load(f)
        except Exception as e:
            print(f'\nWarning: Failed to load pickle for {session}: {e}')
            sessions_failed += 1
            continue

        # Get metrics_df with centers
        if not hasattr(est, 'metrics_df') or est.metrics_df is None:
            print(f'\nWarning: No metrics_df in estimates for {session}')
            sessions_failed += 1
            continue

        if 'center' not in est.metrics_df.columns:
            print(f'\nWarning: No center column in metrics_df for {session}')
            sessions_failed += 1
            continue

        # Get neurons for this session from v9 dataset
        session_mask = (df['session_name'] == session) & lnof_mask
        session_neurons = df[session_mask]
        component_indices = session_neurons['component_idx'].values.astype(int)

        # Validate component indices
        if component_indices.max() >= len(est.metrics_df):
            print(f'\nWarning: Component indices out of range for {session}')
            print(f'  Max index: {component_indices.max()}, metrics_df len: {len(est.metrics_df)}')
            sessions_failed += 1
            continue

        # Extract positions from metrics_df
        # Match the indices from v9 dataset to the metrics_df
        metrics_for_session = est.metrics_df.iloc[component_indices]
        positions = np.array([
            np.array(c) if not isinstance(c, np.ndarray) else c
            for c in metrics_for_session['center']
        ])

        # Compute ellipse_r using corner_artifacts function
        try:
            _, ellipse_info = detect_ellipse_artifacts_from_positions(
                positions,
                fov_width=None,  # Auto-detect from positions
                fov_height=None,
                threshold=0.9
            )
            ellipse_r_values = ellipse_info['radial_dist']
        except Exception as e:
            print(f'\nWarning: Failed to compute ellipse_r for {session}: {e}')
            sessions_failed += 1
            continue

        # Assign to DataFrame
        df.loc[session_mask, 'ellipse_r'] = ellipse_r_values

        sessions_processed += 1
        neurons_processed += len(session_neurons)

    print(f'\n{"="*80}')
    print('SUMMARY')
    print('='*80)
    print(f'Sessions processed: {sessions_processed}/{len(lnof_sessions)}')
    print(f'Sessions failed: {sessions_failed}/{len(lnof_sessions)}')
    print(f'LNOF neurons with ellipse_r added: {neurons_processed:,}/{lnof_count:,}')
    print(f'LNOF neurons still missing ellipse_r: {df.loc[lnof_mask, "ellipse_r"].isna().sum():,}')

    # Overall statistics
    print(f'\nOverall dataset:')
    print(f'  Total neurons: {len(df):,}')
    print(f'  Neurons with ellipse_r: {df["ellipse_r"].notna().sum():,}')
    print(f'  Neurons missing ellipse_r: {df["ellipse_r"].isna().sum():,}')

    # Statistics by experiment
    print(f'\nellipse_r coverage by experiment:')
    for exp in ['NOF', 'RFC', 'FOF', 'LNOF']:
        exp_mask = df['experiment'] == exp
        exp_count = exp_mask.sum()
        exp_valid = df.loc[exp_mask, 'ellipse_r'].notna().sum()
        print(f'  {exp}: {exp_valid:,}/{exp_count:,} ({100*exp_valid/exp_count:.1f}%)')

    # ellipse_r statistics
    print(f'\nellipse_r Statistics:')
    print(df['ellipse_r'].describe())

    # Save
    if output_path is None:
        output_path = dataset_path

    df.to_csv(output_path, index=False)
    print(f'\nDataset saved to: {output_path}')
    print(f'Total columns: {len(df.columns)}')

    print('\n' + '='*80)
    print('COMPLETE')
    print('='*80)

    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add ellipse_r metric to LNOF neurons in v9 dataset')
    parser.add_argument('--dataset', default='ml/results/training_dataset_v9.csv')
    parser.add_argument('--output', default=None)
    parser.add_argument('--lnof-dir', default='data/LNOF')
    args = parser.parse_args()

    compute_lnof_ellipse_r(args.dataset, args.output, args.lnof_dir)
