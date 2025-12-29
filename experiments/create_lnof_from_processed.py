"""
Create LNOF dataset from _processed.pickle files.

This script extracts metrics from processed estimates (after component merging),
ensuring that indices match the expert feedback indices.

Key Advantage: _processed.pickle files have est.metrics_df with POST-MERGE indices
that match the expert feedback indices, eliminating index mismatch issues.
"""
import pandas as pd
import pickle
from pathlib import Path
import re

def extract_session_name(folder_name):
    """Extract session name like LNOF_J01_1D from folder name."""
    match = re.search(r'(LNOF_J\d+_\dD)', folder_name)
    return match.group(1) if match else None

def create_lnof_from_processed():
    """Create LNOF dataset from _processed.pickle files."""
    print('='*80)
    print('CREATING LNOF DATASET FROM PROCESSED ESTIMATES')
    print('='*80)
    print()

    lnof_dir = Path('data/LNOF')
    if not lnof_dir.exists():
        raise FileNotFoundError(f'LNOF directory not found: {lnof_dir}')

    # Find all inspection_artifacts folders
    artifact_folders = sorted([f for f in lnof_dir.iterdir()
                              if f.is_dir() and f.name.startswith('inspection_artifacts_LNOF_')])

    print(f'Found {len(artifact_folders)} LNOF session folders')
    print()

    all_sessions = []
    total_feedback_applied = 0
    fp_corrections = 0
    fn_corrections = 0
    sessions_with_feedback = 0
    sessions_without_metrics = []

    for folder in artifact_folders:
        session_name = extract_session_name(folder.name)
        if not session_name:
            print(f'[WARNING] Could not extract session name from: {folder.name}')
            continue

        # Find _processed.pickle file
        pickle_files = list(folder.glob('*_processed.pickle'))
        if not pickle_files:
            print(f'[WARNING] No processed pickle found for {session_name}')
            continue

        # Load processed estimates
        pickle_file = pickle_files[0]
        print(f'[{session_name}] Loading: {pickle_file.name}')

        try:
            with open(pickle_file, 'rb') as f:
                est = pickle.load(f)
        except Exception as e:
            print(f'[ERROR] Failed to load {pickle_file.name}: {e}')
            continue

        # Extract metrics_df
        if hasattr(est, 'metrics_df') and est.metrics_df is not None:
            df = est.metrics_df.copy()
            print(f'  Extracted metrics_df: {len(df)} neurons')
        else:
            print(f'[ERROR] metrics_df not attached to {session_name}')
            sessions_without_metrics.append(session_name)
            continue

        # Add session_name column
        df['session_name'] = session_name

        # Load and apply expert feedback (if exists)
        feedback_file = folder / f'{session_name}_feedback.csv'
        if feedback_file.exists():
            feedback_df = pd.read_csv(feedback_file)
            sessions_with_feedback += 1

            session_fp = 0
            session_fn = 0

            for _, fb_row in feedback_df.iterrows():
                # Get neuron index from feedback
                neuron_idx = fb_row.get('neuron_idx', fb_row.get('idx', None))
                if neuron_idx is None:
                    continue

                feedback_type = fb_row['feedback_type']

                # Find neuron in metrics_df (using component_idx)
                mask = df['component_idx'] == neuron_idx

                if mask.any():
                    if feedback_type == 'FP':
                        # False Positive: model should DELETE
                        df.loc[mask, 'delete'] = 1
                        session_fp += 1
                    elif feedback_type == 'FN':
                        # False Negative: model should KEEP
                        df.loc[mask, 'delete'] = 0
                        session_fn += 1

            total_feedback_applied += len(feedback_df)
            fp_corrections += session_fp
            fn_corrections += session_fn

            print(f'  Applied feedback: {len(feedback_df)} neurons ({session_fp} FP, {session_fn} FN)')
        else:
            print(f'  No feedback (model decisions assumed correct)')

        all_sessions.append(df)

    print()
    print('='*80)
    print('SUMMARY')
    print('='*80)
    print(f'Total sessions processed: {len(all_sessions)}')
    print(f'  With feedback: {sessions_with_feedback}')
    print(f'  Without feedback: {len(all_sessions) - sessions_with_feedback}')
    print()
    print(f'Expert corrections applied: {total_feedback_applied}')
    print(f'  FP corrections (model -> DELETE): {fp_corrections}')
    print(f'  FN corrections (model -> KEEP): {fn_corrections}')
    print()

    if sessions_without_metrics:
        print(f'[WARNING] Sessions without metrics_df: {sessions_without_metrics}')
        print()

    # Concatenate all sessions
    if all_sessions:
        lnof_dataset = pd.concat(all_sessions, ignore_index=True)
        print(f'Final dataset: {len(lnof_dataset)} neurons across {len(all_sessions)} sessions')
    else:
        print('[ERROR] No sessions were processed successfully')
        return None

    # Save dataset
    output_file = 'LNOF_dataset_from_processed.csv'
    lnof_dataset.to_csv(output_file, index=False)

    print()
    print(f'Saved dataset: {output_file}')
    print(f'  Total neurons: {len(lnof_dataset)}')
    print(f'  Sessions: {lnof_dataset["session_name"].nunique()}')
    print(f'  Columns: {len(lnof_dataset.columns)}')

    # Summary statistics
    if 'delete' in lnof_dataset.columns:
        keep_count = (lnof_dataset['delete'] == 0).sum()
        delete_count = (lnof_dataset['delete'] == 1).sum()
        print(f'  Decisions: KEEP={keep_count}, DELETE={delete_count}')

    print()
    print('='*80)
    print('COMPLETE')
    print('='*80)

    return lnof_dataset

if __name__ == '__main__':
    create_lnof_from_processed()
