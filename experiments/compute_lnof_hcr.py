"""
Compute half_crossing_rate for LNOF dataset.

Loads processed estimates pickle files and computes HCR for all LNOF neurons.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from auto_inspector import get_half_crossing_rates

def compute_lnof_hcr(
    dataset_path='LNOF_dataset_from_processed.csv',
    output_path=None,
    lnof_dir='data/LNOF',
    fps_csv='fps_data.csv'
):
    """
    Add half_crossing_rate column to LNOF dataset.

    Args:
        dataset_path: Path to LNOF dataset CSV
        output_path: Path for updated dataset (default: overwrite original)
        lnof_dir: Directory containing LNOF inspection_artifacts folders
        fps_csv: Path to FPS data CSV
    """
    print('='*80)
    print('ADDING HALF_CROSSING_RATE TO LNOF DATASET')
    print('='*80)

    # Load FPS data
    print(f'\nLoading FPS data from {fps_csv}')
    fps_df = pd.read_csv(fps_csv, delimiter=';')
    # Handle different column names
    if 'Filename' in fps_df.columns:
        fps_dict = dict(zip(fps_df['Filename'], fps_df['FPS'].round().astype(int)))
    elif 'filename' in fps_df.columns:
        fps_dict = dict(zip(fps_df['filename'], fps_df['fps'].round().astype(int)))
    else:
        # Assume columns are: index, session, fps
        fps_dict = dict(zip(fps_df.iloc[:, 1], fps_df.iloc[:, 2].round().astype(int)))

    print(f'Loaded FPS for {len(fps_dict)} sessions')
    print(f'LNOF sessions in FPS data: {len([k for k in fps_dict.keys() if "LNOF" in str(k)])}')

    # Load dataset
    print(f'\nLoading dataset: {dataset_path}')
    df = pd.read_csv(dataset_path)
    print(f'Loaded {len(df):,} neurons from {df["session_name"].nunique()} sessions')

    # Initialize HCR column
    df['half_crossing_rate'] = np.nan

    # Get unique sessions
    sessions = df['session_name'].unique()
    print(f'\nProcessing {len(sessions)} LNOF sessions...')

    lnof_path = Path(lnof_dir)
    sessions_processed = 0
    sessions_failed = 0
    neurons_processed = 0

    for session in tqdm(sessions, desc='Computing HCR'):
        # Get FPS for this session
        fps = fps_dict.get(session)
        if fps is None:
            print(f'\nWarning: No FPS data for {session}, skipping')
            sessions_failed += 1
            continue

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

        if not hasattr(est, 'C') or est.C is None:
            print(f'\nWarning: No traces in estimates for {session}')
            sessions_failed += 1
            continue

        # Get neurons for this session
        session_mask = df['session_name'] == session
        session_neurons = df[session_mask]
        component_indices = session_neurons['component_idx'].values.astype(int)

        # Validate component indices
        if component_indices.max() >= est.C.shape[0]:
            print(f'\nWarning: Component indices out of range for {session}')
            print(f'  Max index: {component_indices.max()}, est.C shape: {est.C.shape}')
            sessions_failed += 1
            continue

        # Extract traces for these components
        traces = est.C[component_indices, :]

        # Compute HCR
        try:
            hcr_values = get_half_crossing_rates(traces, fps=fps)
        except Exception as e:
            print(f'\nWarning: Failed to compute HCR for {session}: {e}')
            sessions_failed += 1
            continue

        # Assign to DataFrame
        df.loc[session_mask, 'half_crossing_rate'] = hcr_values

        sessions_processed += 1
        neurons_processed += len(session_neurons)

    print(f'\n{"="*80}')
    print('SUMMARY')
    print('='*80)
    print(f'Sessions processed: {sessions_processed}/{len(sessions)}')
    print(f'Sessions failed: {sessions_failed}/{len(sessions)}')
    print(f'Neurons with HCR: {neurons_processed:,}/{len(df):,}')
    print(f'Neurons missing HCR: {df["half_crossing_rate"].isna().sum():,}')

    # Statistics
    print(f'\nHCR Statistics (crossings per minute):')
    print(df['half_crossing_rate'].describe())

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
    parser = argparse.ArgumentParser(description='Add HCR metric to LNOF dataset')
    parser.add_argument('--dataset', default='LNOF_dataset_from_processed.csv')
    parser.add_argument('--output', default=None)
    parser.add_argument('--lnof-dir', default='data/LNOF')
    parser.add_argument('--fps-csv', default='fps_data.csv')
    args = parser.parse_args()

    compute_lnof_hcr(args.dataset, args.output, args.lnof_dir, args.fps_csv)
