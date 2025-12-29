"""
Add half_crossing_rate metric to existing training dataset.
Loads raw traces from pickle files and computes HCR post-facto.
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from auto_inspector import get_half_crossing_rates

def load_estimates_from_pickle(session_name, raw_dir='data/raw_compressed'):
    """Load CaImAn estimates from raw_compressed directory."""
    raw_path = Path(raw_dir)

    # Try multiple file patterns
    patterns = [
        f'{session_name}_estimates*.pickle',
        f'{session_name}_raw*.pickle',
        f'{session_name}_*.pickle',
        f'{session_name}.pickle'
    ]

    for pattern in patterns:
        files = list(raw_path.glob(pattern))
        if files:
            try:
                with open(files[0], 'rb') as f:
                    data = pickle.load(f)

                # Extract estimates object (handle different formats)
                if isinstance(data, dict):
                    if 'estimates' in data:
                        return data['estimates']
                    elif 'est' in data:
                        return data['est']
                    for key in ['cnmf', 'cnm', 'results']:
                        if key in data and hasattr(data[key], 'estimates'):
                            return data[key].estimates

                if hasattr(data, 'A') and hasattr(data, 'C'):
                    return data
                if hasattr(data, 'estimates'):
                    return data.estimates

                return data
            except Exception as e:
                print(f"Warning: Failed to load {files[0]}: {e}")
                continue

    return None

def add_hcr_to_dataset(
    dataset_path='ml/results/training_dataset_v8_corrected_iter3.csv',
    output_path=None,
    raw_dir='data/raw_compressed',
    fps_csv='fps_data.csv'
):
    """
    Add half_crossing_rate column to existing dataset.

    Args:
        dataset_path: Path to existing training dataset CSV
        output_path: Path for updated dataset (default: overwrite original)
        raw_dir: Directory containing raw estimate pickle files
        fps_csv: Path to FPS data CSV (columns: #, Filename, FPS)
    """
    print('='*80)
    print('ADDING HALF_CROSSING_RATE TO DATASET')
    print('='*80)

    # Load FPS data
    print(f'\nLoading FPS data from {fps_csv}')
    fps_df = pd.read_csv(fps_csv)
    fps_dict = dict(zip(fps_df['Filename'], fps_df['FPS'].round().astype(int)))
    print(f'Loaded FPS for {len(fps_dict)} sessions')

    # Load dataset
    print(f'\nLoading dataset: {dataset_path}')
    df = pd.read_csv(dataset_path)
    print(f'Loaded {len(df):,} neurons from {df["session"].nunique()} sessions')

    # Initialize HCR column
    df['half_crossing_rate'] = np.nan

    # Group by session
    sessions = df['session'].unique()
    print(f'\nProcessing {len(sessions)} sessions...')

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

        # Load raw estimates
        est = load_estimates_from_pickle(session, raw_dir)

        if est is None or not hasattr(est, 'C') or est.C is None:
            print(f'\nWarning: Could not load traces for {session}')
            sessions_failed += 1
            continue

        # Get neurons for this session
        session_mask = df['session'] == session
        session_neurons = df[session_mask]
        component_indices = session_neurons['component_idx'].values.astype(int)

        # Validate component indices
        if component_indices.max() >= est.C.shape[0]:
            print(f'\nWarning: Component indices out of range for {session}')
            sessions_failed += 1
            continue

        # Extract traces for these components
        traces = est.C[component_indices, :]

        # Compute HCR (rate per minute, not count)
        hcr_values = get_half_crossing_rates(traces, fps=fps)

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
    print(f'New column count: {len(df.columns)} (was {len(df.columns)-1})')

    return df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Add HCR metric to dataset')
    parser.add_argument('--dataset', default='ml/results/training_dataset_v8_corrected_iter3.csv')
    parser.add_argument('--output', default=None)
    parser.add_argument('--raw-dir', default='data/raw_compressed')
    parser.add_argument('--fps-csv', default='fps_data.csv')
    args = parser.parse_args()

    add_hcr_to_dataset(args.dataset, args.output, args.raw_dir, args.fps_csv)
