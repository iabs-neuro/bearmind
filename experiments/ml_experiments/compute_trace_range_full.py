"""
Compute trace_range for ALL neurons in the v9 dataset.

This processes all 187 sessions to compute trace_range = max - min
for every neuron in the dataset.
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

print('='*80)
print('COMPUTING trace_range FOR FULL v9 DATASET')
print('='*80)

# Load full dataset
dataset_path = 'ml/results/training_dataset_v9_corrected_iter7.csv'
print(f'\nLoading dataset: {dataset_path}')
df = pd.read_csv(dataset_path)
print(f'Total neurons: {len(df):,}')
print(f'Sessions: {df["session_name"].nunique()}')

# Storage for trace_range
trace_range_data = {
    'session_name': [],
    'component_idx': [],
    'trace_range': [],
}

# Get unique sessions
unique_sessions = df['session_name'].unique()
print(f'\nProcessing {len(unique_sessions)} sessions...')

# Track progress
sessions_found = 0
sessions_missing = 0
neurons_computed = 0

# Possible paths to check
def find_estimates_file(session):
    """Find estimates file for a session."""
    # Determine experiment type
    exp_type = session.split('_')[0]  # NOF, RFC, FOF, LNOF, etc.

    # LNOF data location (in timestamped inspection_artifacts folders)
    if exp_type == 'LNOF':
        lnof_dir = Path('data/LNOF')
        if lnof_dir.exists():
            # Look for inspection_artifacts_{session}_* folders
            for artifact_dir in lnof_dir.glob(f'inspection_artifacts_{session}_*'):
                # Files are named {session}_{timestamp}_processed.pickle
                for processed in artifact_dir.glob(f'{session}_*_processed.pickle'):
                    if processed.exists():
                        return processed
                for estimates in artifact_dir.glob(f'{session}_*_estimates.pickle'):
                    if estimates.exists():
                        return estimates

    # Other experiments in data/capcan_validation_99_v8 (in capcan_artifacts folders)
    capcan_dir = Path('data/capcan_validation_99_v8')
    if capcan_dir.exists():
        artifact_dir = capcan_dir / f'capcan_artifacts_{session}'
        if artifact_dir.exists():
            processed = artifact_dir / f'{session}_processed.pickle'
            if processed.exists():
                return processed
            estimates = artifact_dir / f'{session}_estimates.pickle'
            if estimates.exists():
                return estimates

    # Try processed output folder
    processed_path = Path(f'output/inspection_artifacts_{session}/{session}_processed.pickle')
    if processed_path.exists():
        return processed_path

    # Try raw compressed (with glob to handle parameter suffixes like _gsig4_mincorr0.92_minpnr7)
    raw_dir = Path('data/raw_compressed')
    if raw_dir.exists():
        patterns = [
            f'{session}_estimates*.pickle',
            f'{session}_raw*.pickle',
            f'{session}_*.pickle',
        ]
        for pattern in patterns:
            files = list(raw_dir.glob(pattern))
            if files:
                return files[0]

    return None

# Process all sessions
print('\nProcessing sessions:')
print('-'*80)

for session in tqdm(unique_sessions, desc='Sessions'):
    # Find estimates file
    estimates_path = find_estimates_file(session)

    if estimates_path is None:
        sessions_missing += 1
        continue

    sessions_found += 1

    try:
        # Load estimates
        with open(estimates_path, 'rb') as f:
            est = pickle.load(f)

        # Get all components from this session
        session_df = df[df['session_name'] == session]

        # Process each component
        for idx, row in session_df.iterrows():
            comp_idx = int(row['component_idx'])

            # Check if component exists in estimates
            if comp_idx >= est.C.shape[0]:
                continue

            # Get trace
            trace = est.C[comp_idx, :].copy()

            # Compute trace_range (raw: max - min)
            trace_range = np.max(trace) - np.min(trace)

            # Store
            trace_range_data['session_name'].append(session)
            trace_range_data['component_idx'].append(comp_idx)
            trace_range_data['trace_range'].append(trace_range)

            neurons_computed += 1

    except Exception as e:
        tqdm.write(f'  ERROR processing {session}: {e}')
        continue

print('\n' + '='*80)
print('COMPUTATION SUMMARY')
print('='*80)
print(f'Sessions found: {sessions_found} / {len(unique_sessions)}')
print(f'Sessions missing: {sessions_missing}')
print(f'Neurons processed: {neurons_computed:,} / {len(df):,} ({neurons_computed/len(df)*100:.1f}%)')

# Convert to DataFrame
range_df = pd.DataFrame(trace_range_data)
print(f'\ntrace_range computed for {len(range_df):,} neurons')

# Merge with original dataset
print('\nMerging with original dataset...')
merged_df = df.merge(range_df, on=['session_name', 'component_idx'], how='left')

# Check merge success
n_with_range = merged_df['trace_range'].notna().sum()
print(f'Neurons with trace_range: {n_with_range:,} / {len(merged_df):,} ({n_with_range/len(merged_df)*100:.1f}%)')

# Save augmented dataset
output_path = 'ml/results/training_dataset_v9_with_trace_range.csv'
merged_df.to_csv(output_path, index=False)
print(f'\nSaved augmented dataset to: {output_path}')

# Quick statistics
print('\n' + '='*80)
print('trace_range STATISTICS')
print('='*80)

valid_range = merged_df[merged_df['trace_range'].notna()]
keep_mask = valid_range['ground_truth'] == 1
delete_mask = valid_range['ground_truth'] == 0

print(f'\nKEEP neurons (n={keep_mask.sum():,}):')
print(f'  Mean:   {valid_range.loc[keep_mask, "trace_range"].mean():.2f}')
print(f'  Median: {valid_range.loc[keep_mask, "trace_range"].median():.2f}')
print(f'  Std:    {valid_range.loc[keep_mask, "trace_range"].std():.2f}')

print(f'\nDELETE neurons (n={delete_mask.sum():,}):')
print(f'  Mean:   {valid_range.loc[delete_mask, "trace_range"].mean():.2f}')
print(f'  Median: {valid_range.loc[delete_mask, "trace_range"].median():.2f}')
print(f'  Std:    {valid_range.loc[delete_mask, "trace_range"].std():.2f}')

# Effect size
from scipy import stats as sp_stats

keep_vals = valid_range.loc[keep_mask, 'trace_range'].dropna()
delete_vals = valid_range.loc[delete_mask, 'trace_range'].dropna()

pooled_std = np.sqrt((np.std(keep_vals)**2 + np.std(delete_vals)**2) / 2)
cohens_d = (np.mean(keep_vals) - np.mean(delete_vals)) / pooled_std

print(f'\nEffect size (Cohen\'s d): {cohens_d:+.3f}')

# Statistical test
stat, p_val = sp_stats.mannwhitneyu(keep_vals, delete_vals, alternative='two-sided')
print(f'Mann-Whitney U test: p={p_val:.2e}')

if p_val < 0.001:
    print('Result: HIGHLY SIGNIFICANT (p < 0.001)')
elif p_val < 0.05:
    print('Result: SIGNIFICANT (p < 0.05)')
else:
    print('Result: Not significant')

print('\n' + '='*80)
print('READY FOR MODEL TRAINING')
print('='*80)
print(f'\nNext step: Run CV experiment with trace_range')
print(f'Dataset: {output_path}')

print('\n' + '='*80)
print('COMPLETE')
print('='*80)
